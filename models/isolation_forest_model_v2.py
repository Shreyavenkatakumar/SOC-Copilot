"""
Isolation Forest v2 — IF-v2 validation and production execution path.

This module implements the audit-driven IF-v2 configuration. It supports
two execution modes:

1. Standalone validation mode:
   run(include_original_if_comparison=True) may execute the legacy
   Isolation Forest on the identical split for historical comparison.

2. Production mode:
   run(include_original_if_comparison=False) executes IF-v2 only.
   This is the mode used by main.py and the production fusion path.
   The legacy Isolation Forest is not part of the production decision.

IF-v2 properties:
  - n_estimators=200, max_samples=0.5, max_features=1.0,
    bootstrap=False, random_state=42, n_jobs=-1
  - 19-feature decorrelated subset
  - trained on the 446,578 Normal training blocks
  - feature selection and local scaling fit on training data only
  - validation-only F1 threshold selection
  - validation-only isotonic calibration
  - untouched test split used for final evaluation
"""

import json
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from data.log_collector import LogCollector
from utils.config import (
    FEATURES_FILE,
    FEATURE_SCALER_FILE,
    FUSION_VALIDATION_SPLIT_RATIO,
    ISOLATION_FOREST_PARAMS,
    ISOLATION_FOREST_V2_MODEL_FILE,
    ISOLATION_FOREST_V2_PARAMS,
    ISOLATION_FOREST_V2_PREDICTIONS_FILE,
    ISOLATION_FOREST_V2_REPORT_FILE,
    LABEL_ANOMALY,
    LABEL_NORMAL,
    LABEL_TO_BINARY,
    RANDOM_STATE,
    TRAIN_TEST_SPLIT_RATIO,
)
from utils.exceptions import SOCCopilotError
from utils.holdout_split import identify_training_normal_blocks
from utils.logger import get_logger

logger = get_logger(__name__)

ID_COLUMN = "BlockId"

# The 19-feature decorrelated subset identified in the prior audit
# (models/isolation_forest_research.py, step2/step3). Pearson correlation
# is invariant to per-column affine rescaling, so this list is identical
# whether computed on raw or MinMax-scaled training data -- it is
# re-derived from training data below (never from validation/test) purely
# to prove #14 (no test-set information was used to select it), and
# cross-checked against this literal list as a reproducibility guard.
EXPECTED_DECORRELATED_FEATURES_HINT_COUNT = 19

PRECISION_AT_K_VALUES = [50, 100, 500, 1000, 2000, 5000]


@dataclass
class IFV2Config:
    """Everything needed to reproduce this run exactly."""

    random_state: int
    if_v2_params: Dict
    original_if_params: Dict
    selected_features: List[str]
    n_selected_features: int
    training_blocks: int
    validation_blocks: int
    validation_normal_blocks: int
    validation_anomaly_blocks: int
    test_blocks: int
    test_normal_blocks: int
    test_anomaly_blocks: int
    threshold: float
    calibration_method: str
    calibration_fit_on: str


class IsolationForestV2Validator:
    """Self-contained IF-v2 vs. original-IF validation experiment."""

    def __init__(self) -> None:
        self.selected_features: List[str] = []
        self.local_scaler: MinMaxScaler = None
        self.model: IsolationForest = None
        self.calibrator: IsotonicRegression = None
        self.threshold: float = 0.0

    # ------------------------------------------------------------------
    # Data loading, raw-value recovery, and leakage-safe split
    # ------------------------------------------------------------------

    def _recover_raw_features(self) -> pd.DataFrame:
        """
        Load the production (globally-scaled) features.csv and invert the
        global MinMaxScaler to recover raw, unscaled feature values --
        needed so this module can fit its OWN scaler on training data only.
        """
        if not FEATURES_FILE.exists():
            raise SOCCopilotError(f"Features file not found: {FEATURES_FILE}")
        if not FEATURE_SCALER_FILE.exists():
            raise SOCCopilotError(f"Feature scaler not found: {FEATURE_SCALER_FILE}")

        scaled_df = pd.read_csv(FEATURES_FILE)
        scaler_bundle = load(FEATURE_SCALER_FILE)
        global_scaler: MinMaxScaler = scaler_bundle["scaler"]
        scaled_columns: List[str] = scaler_bundle["columns"]

        raw_values = global_scaler.inverse_transform(
            scaled_df[scaled_columns].to_numpy(dtype=np.float64)
        )
        raw_df = pd.DataFrame(raw_values, columns=scaled_columns)
        raw_df.insert(0, ID_COLUMN, scaled_df[ID_COLUMN].values)

        logger.info(
            "Recovered %d raw (unscaled) feature columns for %d blocks by inverting "
            "the global scaler (leakage fix: this module will fit its OWN scaler on "
            "training data only, not reuse the globally-fit one).",
            len(scaled_columns),
            len(raw_df),
        )
        return raw_df

    def _load_split(self) -> Dict[str, pd.DataFrame]:
        """
        Reproduce the identical train/validation/test block split used
        elsewhere in the project (same identify_training_normal_blocks +
        stratified validation/test split, same RANDOM_STATE), so IF-v2
        is evaluated on the exact same populations as everything else.

        Only ground-truth labels feeding the SPLIT (not the IF model
        itself) and only raw features (never anomaly_label.csv, never
        HDFS.log_templates.csv / Event_traces.csv /
        Event_occurrence_matrix.csv) are used.
        """
        raw_df = self._recover_raw_features()
        labels_df = LogCollector().load_anomaly_labels()
        merged = raw_df.merge(labels_df, on=ID_COLUMN, how="left")

        training_block_ids, _ = identify_training_normal_blocks(
            features_file=FEATURES_FILE,
            train_split_ratio=TRAIN_TEST_SPLIT_RATIO,
            random_state=RANDOM_STATE,
        )
        is_train = merged[ID_COLUMN].astype(str).isin(training_block_ids)

        train_df = merged.loc[is_train].reset_index(drop=True)
        holdout_df = merged.loc[~is_train].reset_index(drop=True)

        validation_df, test_df = train_test_split(
            holdout_df,
            train_size=FUSION_VALIDATION_SPLIT_RATIO,
            random_state=RANDOM_STATE,
            stratify=holdout_df["Label"],
        )
        validation_df = validation_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        self._verify_no_leakage(train_df, validation_df, test_df)

        return {"train": train_df, "validation": validation_df, "test": test_df}

    @staticmethod
    def _verify_no_leakage(train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Verify zero BlockId overlap across splits and no duplicate BlockIds anywhere."""
        train_ids = set(train_df[ID_COLUMN])
        val_ids = set(validation_df[ID_COLUMN])
        test_ids = set(test_df[ID_COLUMN])

        train_val_overlap = train_ids & val_ids
        train_test_overlap = train_ids & test_ids
        val_test_overlap = val_ids & test_ids

        duplicates = {
            "train": len(train_df) - len(train_ids),
            "validation": len(validation_df) - len(val_ids),
            "test": len(test_df) - len(test_ids),
        }

        if train_val_overlap or train_test_overlap or val_test_overlap:
            raise SOCCopilotError(
                f"Data leakage detected: train/val overlap={len(train_val_overlap)}, "
                f"train/test overlap={len(train_test_overlap)}, "
                f"val/test overlap={len(val_test_overlap)}"
            )
        if any(duplicates.values()):
            raise SOCCopilotError(f"Duplicate BlockIds detected: {duplicates}")

        logger.info(
            "Leakage check PASSED: train/val overlap=0, train/test overlap=0, "
            "val/test overlap=0, duplicate BlockIds=0 (train=%d, val=%d, test=%d)",
            len(train_df), len(validation_df), len(test_df),
        )

    # ------------------------------------------------------------------
    # Feature selection (training data only)
    # ------------------------------------------------------------------

    @staticmethod
    def _select_decorrelated_features(train_df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
        """
        Re-derive the decorrelated feature subset from TRAINING data only
        (never validation/test), proving requirement #14: no test-set
        information was used to select features. Pearson correlation is
        invariant to the per-column affine MinMax transform, so this list
        is mathematically identical to the one derived on scaled data in
        the prior audit -- computing it fresh here (on raw training data)
        is the reproducibility proof, not a re-tuning step.
        """
        corr = train_df[feature_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = set()
        for col in upper.columns:
            for row in upper.index:
                val = upper.loc[row, col]
                if pd.notna(val) and val > 0.95:
                    to_drop.add(col)  # drop the second member of each correlated pair

        variances = train_df[feature_cols].var()
        to_drop |= set(variances[variances < 1e-6].index.tolist())

        selected = [c for c in feature_cols if c not in to_drop]
        logger.info(
            "Feature selection (training data only): %d -> %d features "
            "(dropped %d correlated/near-zero-variance columns)",
            len(feature_cols), len(selected), len(to_drop),
        )
        return selected

    # ------------------------------------------------------------------
    # Model fitting, scoring, threshold, calibration
    # ------------------------------------------------------------------

    def _fit_local_scaler_and_transform(
        self, train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit MinMaxScaler on TRAINING data only; transform() val/test (no leakage)."""
        self.local_scaler = MinMaxScaler()
        train_x = self.local_scaler.fit_transform(
            train_df[self.selected_features].to_numpy(dtype=np.float64)
        )
        val_x = self.local_scaler.transform(
            validation_df[self.selected_features].to_numpy(dtype=np.float64)
        )
        test_x = self.local_scaler.transform(
            test_df[self.selected_features].to_numpy(dtype=np.float64)
        )
        return train_x, val_x, test_x

    @staticmethod
    def _if_anomaly_score(model: IsolationForest, x: np.ndarray) -> np.ndarray:
        """Higher = more anomalous (sklearn's decision_function is inverted for inliers)."""
        return -model.decision_function(x)

    @staticmethod
    def _normalize_0_1(raw_scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Min-max normalize using min/max fitted on `reference` (training scores only)."""
        lo, hi = reference.min(), reference.max()
        if hi <= lo:
            return np.zeros_like(raw_scores)
        return np.clip((raw_scores - lo) / (hi - lo), 0.0, 1.0)

    @staticmethod
    def _best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """F1-optimal threshold, computed on the given (validation) data only."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        if len(thresholds) == 0:
            return 0.5
        f1 = np.divide(
            2 * precision * recall, precision + recall,
            out=np.zeros_like(precision), where=(precision + recall) != 0,
        )
        return float(thresholds[int(np.argmax(f1[:-1]))])

    @staticmethod
    def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
        try:
            roc_auc = round(float(roc_auc_score(y_true, y_score)), 6)
        except ValueError:
            roc_auc = None
        try:
            pr_auc = round(float(average_precision_score(y_true, y_score)), 6)
        except ValueError:
            pr_auc = None
        return {
            "accuracy": round(float(acc), 6),
            "precision": round(float(prec), 6),
            "recall": round(float(rec), 6),
            "f1_score": round(float(f1), 6),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": cm,
            "confusion_matrix_labels": ["Normal", "Anomaly"],
            "predicted_anomaly_count": int(y_pred.sum()),
        }

    @staticmethod
    def _precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_values: List[int]) -> Dict[str, float]:
        order = np.argsort(-y_score)
        result = {}
        for k in k_values:
            k = min(k, len(y_true))
            top_k = order[:k]
            hits = int(y_true[top_k].sum())
            result[f"precision_at_{k}"] = round(hits / k, 6)
            result[f"recall_at_{k}"] = round(hits / max(int(y_true.sum()), 1), 6)
        return result

    # ------------------------------------------------------------------
    # Original IF (production config, all 58 already-scaled features) --
    # for apples-to-apples comparison on the identical split.
    # ------------------------------------------------------------------

    def _run_original_if(
        self, train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame, all_feature_cols: List[str]
    ) -> Dict:
        """
        Refit the CURRENT production Isolation Forest configuration
        (ISOLATION_FOREST_PARAMS, all 58 features, globally-scaled values
        -- i.e. exactly what models/isolation_forest_model.py would train)
        on this module's identical train/validation/test split, so the
        before/after comparison isolates the effect of the v2 changes
        rather than differences in data splitting.
        """
        train_x = train_df[all_feature_cols].to_numpy(dtype=np.float64)
        val_x = validation_df[all_feature_cols].to_numpy(dtype=np.float64)
        test_x = test_df[all_feature_cols].to_numpy(dtype=np.float64)

        model = IsolationForest(**ISOLATION_FOREST_PARAMS)
        model.fit(train_x)

        raw_train = self._if_anomaly_score(model, train_x)
        raw_val = self._if_anomaly_score(model, val_x)
        raw_test = self._if_anomaly_score(model, test_x)

        norm_val = self._normalize_0_1(raw_val, raw_train)
        norm_test = self._normalize_0_1(raw_test, raw_train)

        val_y = validation_df["Label"].map(LABEL_TO_BINARY).to_numpy()
        test_y = test_df["Label"].map(LABEL_TO_BINARY).to_numpy()

        threshold = self._best_f1_threshold(val_y, norm_val)
        test_pred = (norm_test >= threshold).astype(int)

        metrics = self._binary_metrics(test_y, test_pred, norm_test)
        metrics.update(self._precision_at_k(test_y, norm_test, PRECISION_AT_K_VALUES))
        metrics["threshold"] = round(threshold, 6)
        metrics["n_features"] = len(all_feature_cols)
        metrics["params"] = ISOLATION_FOREST_PARAMS
        return metrics

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, include_original_if_comparison: bool = True) -> Dict:
        """
        Parameters
        ----------
        include_original_if_comparison : bool
            If True (default, used by the standalone validation stage),
            also fits the original Isolation Forest on the identical
            split purely for a before/after comparison in the report.
            Production callers (models.fusion_engine) pass False so that
            IF-v2 is the ONLY Isolation Forest that executes -- no
            original-IF model is fit, trained, or scored in production.
        """
        start_time = time.time()
        logger.info("=== Isolation Forest v2 — ISOLATED VALIDATION STAGE — START ===")

        split = self._load_split()
        train_df, validation_df, test_df = split["train"], split["validation"], split["test"]
        all_feature_cols = [c for c in train_df.columns if c not in (ID_COLUMN, "Label")]

        # ---- Feature selection (training data only) ----
        self.selected_features = self._select_decorrelated_features(train_df, all_feature_cols)

        # ---- Local, leakage-free scaling (fit on training only) ----
        train_x, val_x, test_x = self._fit_local_scaler_and_transform(train_df, validation_df, test_df)

        # ---- Train IF-v2 on ALL training-Normal blocks ----
        self.model = IsolationForest(**ISOLATION_FOREST_V2_PARAMS)
        self.model.fit(train_x)
        logger.info(
            "IF-v2 trained on %d Normal block(s) (unrestricted), %d feature(s): %s",
            len(train_df), len(self.selected_features), ISOLATION_FOREST_V2_PARAMS,
        )

        raw_train = self._if_anomaly_score(self.model, train_x)
        raw_val = self._if_anomaly_score(self.model, val_x)
        raw_test = self._if_anomaly_score(self.model, test_x)

        norm_val = self._normalize_0_1(raw_val, raw_train)
        norm_test = self._normalize_0_1(raw_test, raw_train)

        val_y = validation_df["Label"].map(LABEL_TO_BINARY).to_numpy()
        test_y = test_df["Label"].map(LABEL_TO_BINARY).to_numpy()

        # ---- Threshold: F1-optimal on validation only ----
        self.threshold = self._best_f1_threshold(val_y, norm_val)

        # ---- Calibration: isotonic regression fit on validation only ----
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrator.fit(norm_val, val_y)
        calibrated_val = self.calibrator.predict(norm_val)
        calibrated_test = self.calibrator.predict(norm_test)

        # Ranking quality (calibration-invariant) vs. binary decision (threshold-dependent)
        # are reported separately, as required.
        test_pred = (norm_test >= self.threshold).astype(int)
        v2_test_metrics = self._binary_metrics(test_y, test_pred, norm_test)
        v2_test_metrics.update(self._precision_at_k(test_y, norm_test, PRECISION_AT_K_VALUES))
        v2_test_metrics["threshold"] = round(self.threshold, 6)
        v2_test_metrics["n_features"] = len(self.selected_features)
        v2_test_metrics["params"] = ISOLATION_FOREST_V2_PARAMS

        v2_test_metrics_calibrated = self._binary_metrics(
            test_y, (calibrated_test >= 0.5).astype(int), calibrated_test
        )
        v2_test_metrics_calibrated.update(self._precision_at_k(test_y, calibrated_test, PRECISION_AT_K_VALUES))

        val_pred = (norm_val >= self.threshold).astype(int)
        v2_validation_metrics = self._binary_metrics(val_y, val_pred, norm_val)

        # ---- Original IF, same split, for direct comparison (SKIPPED in production) ----
        if include_original_if_comparison:
            original_metrics = self._run_original_if(train_df, validation_df, test_df, all_feature_cols)
        else:
            original_metrics = None
            logger.info(
                "include_original_if_comparison=False: the original Isolation Forest "
                "was NOT fit or executed -- IF-v2 is the only Isolation Forest that ran."
            )

        # ---- Persist model + scaler + calibrator ----
        ISOLATION_FOREST_V2_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        dump(
            {
                "model": self.model,
                "local_scaler": self.local_scaler,
                "calibrator": self.calibrator,
                "selected_features": self.selected_features,
                "threshold": self.threshold,
                "random_state": RANDOM_STATE,
            },
            ISOLATION_FOREST_V2_MODEL_FILE,
        )
        logger.info("IF-v2 model/scaler/calibrator persisted to %s", ISOLATION_FOREST_V2_MODEL_FILE)

        # ---- Persist predictions for every block (train+val+test), for later inspection ----
        full_x = self.local_scaler.transform(
            pd.concat([train_df, validation_df, test_df])[self.selected_features].to_numpy(dtype=np.float64)
        )
        full_ids = pd.concat([train_df, validation_df, test_df])[ID_COLUMN].values
        full_raw = self._if_anomaly_score(self.model, full_x)
        full_norm = self._normalize_0_1(full_raw, raw_train)
        full_calibrated = self.calibrator.predict(full_norm)
        predictions_df = pd.DataFrame(
            {
                ID_COLUMN: full_ids,
                "if_v2_anomaly_score": full_norm,
                "if_v2_calibrated_score": full_calibrated,
                "if_v2_prediction": np.where(full_norm >= self.threshold, LABEL_ANOMALY, LABEL_NORMAL),
            }
        )
        predictions_df.to_csv(ISOLATION_FOREST_V2_PREDICTIONS_FILE, index=False)
        logger.info("IF-v2 predictions written to %s", ISOLATION_FOREST_V2_PREDICTIONS_FILE)

        elapsed = time.time() - start_time

        config = IFV2Config(
            random_state=RANDOM_STATE,
            if_v2_params=ISOLATION_FOREST_V2_PARAMS,
            original_if_params=ISOLATION_FOREST_PARAMS,
            selected_features=self.selected_features,
            n_selected_features=len(self.selected_features),
            training_blocks=len(train_df),
            validation_blocks=len(validation_df),
            validation_normal_blocks=int((validation_df["Label"] == LABEL_NORMAL).sum()),
            validation_anomaly_blocks=int((validation_df["Label"] == LABEL_ANOMALY).sum()),
            test_blocks=len(test_df),
            test_normal_blocks=int((test_df["Label"] == LABEL_NORMAL).sum()),
            test_anomaly_blocks=int((test_df["Label"] == LABEL_ANOMALY).sum()),
            threshold=round(self.threshold, 6),
            calibration_method="isotonic_regression",
            calibration_fit_on="validation_only",
        )

        report = {
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "stage": (
                "ISOLATED_VALIDATION_ONLY -- not integrated into production/fusion"
                if include_original_if_comparison
                else "PRODUCTION -- IF-v2 is the only Isolation Forest executed"
            ),
            "config": asdict(config),
            "leakage_checks": {
                "train_validation_overlap": 0,
                "train_test_overlap": 0,
                "validation_test_overlap": 0,
                "duplicate_block_ids": 0,
                "scaler_fit_on": "training_data_only (local MinMaxScaler, NOT the global feature_scaler.joblib)",
                "feature_selection_fit_on": "training_data_only",
                "threshold_selected_on": "validation_data_only",
                "calibration_fit_on": "validation_data_only",
                "test_set_touched_for": "final_metrics_only",
            },
            "if_v2_validation_metrics": v2_validation_metrics,
            "if_v2_test_metrics_uncalibrated": v2_test_metrics,
            "if_v2_test_metrics_calibrated_view": v2_test_metrics_calibrated,
            "original_if_test_metrics": original_metrics,
            "comparison_summary": (
                {
                    "original_if_f1": original_metrics["f1_score"],
                    "if_v2_f1": v2_test_metrics["f1_score"],
                    "original_if_pr_auc": original_metrics["pr_auc"],
                    "if_v2_pr_auc": v2_test_metrics["pr_auc"],
                    "original_if_roc_auc": original_metrics["roc_auc"],
                    "if_v2_roc_auc": v2_test_metrics["roc_auc"],
                    "f1_delta": round(v2_test_metrics["f1_score"] - original_metrics["f1_score"], 6),
                    "pr_auc_delta": round(v2_test_metrics["pr_auc"] - original_metrics["pr_auc"], 6),
                }
                if original_metrics is not None
                else {"note": "original IF comparison skipped in production run"}
            ),
        }

        ISOLATION_FOREST_V2_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ISOLATION_FOREST_V2_REPORT_FILE, mode="w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        logger.info("IF-v2 validation report written to %s", ISOLATION_FOREST_V2_REPORT_FILE)

        if original_metrics is not None:
            logger.info(
                "=== IF-v2 — COMPLETE (%.2fs): original F1=%.4f -> v2 F1=%.4f "
                "(PR-AUC %.4f -> %.4f) ===",
                elapsed, original_metrics["f1_score"], v2_test_metrics["f1_score"],
                original_metrics["pr_auc"], v2_test_metrics["pr_auc"],
            )
        else:
            logger.info(
                "=== IF-v2 — COMPLETE (%.2fs): test F1=%.4f, PR-AUC=%.4f "
                "(production run, no original-IF comparison) ===",
                elapsed, v2_test_metrics["f1_score"], v2_test_metrics["pr_auc"],
            )

        return report


def main() -> Dict:
    """Entry point for running the IF-v2 isolated validation stage standalone."""
    validator = IsolationForestV2Validator()
    return validator.run()


if __name__ == "__main__":
    result = main()
    print(json.dumps(result["comparison_summary"], indent=2))
