"""
Module 5, Part A: Fusion Engine — PRODUCTION (approved architecture).

Combines Isolation Forest v2 (models/isolation_forest_model_v2.py) and
Autoencoder (models/autoencoder_model.py) anomaly scores into a single
fused score per block, using the fixed, already-validated configuration:

    fusion_score = 0.10 * calibrated_IFv2_score + 0.90 * calibrated_AE_score
    fusion_prediction = Anomaly if fusion_score >= 0.323459 else Normal

This module does NOT perform a new grid search, weight sweep, or
threshold search. The weight (IF=0.10, AE=0.90) and threshold
(0.323459) were selected in a separate, already-completed experiment
(models/complementarity_analysis.py, validation-only weight/threshold
search; see outputs/reports/ae_ifv2_complementarity_report.json,
step6_score_level_fusion) and are applied here directly, as fixed
constants (utils.config.FUSION_PRODUCTION_IF_WEIGHT /
FUSION_PRODUCTION_AE_WEIGHT / FUSION_PRODUCTION_THRESHOLD).

The only thing this module still fits at runtime is score CALIBRATION
(isotonic regression mapping each detector's raw 0-1 score onto a
comparable scale) -- fit on VALIDATION data only, exactly as in the
validated experiment, never on test data and never using labels as
model input features (labels are used only to fit the calibrators and
to report validation/test metrics downstream).

Isolation Forest v2 supersedes the original Isolation Forest
(models/isolation_forest_model.py) as the sole production IF. That
original module is no longer invoked by this pipeline; it has not been
deleted (kept for reference/rollback) but its predictions are not read
here.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from data.log_collector import LogCollector
from utils.config import (
    AUTOENCODER_PREDICTIONS_FILE,
    FEATURES_FILE,
    FUSION_CONFIG_FILE,
    FUSION_PREDICTIONS_FILE,
    FUSION_PRODUCTION_AE_WEIGHT,
    FUSION_PRODUCTION_IF_WEIGHT,
    FUSION_PRODUCTION_THRESHOLD,
    FUSION_VALIDATION_SPLIT_RATIO,
    ISOLATION_FOREST_V2_PREDICTIONS_FILE,
    LABEL_ANOMALY,
    LABEL_NORMAL,
    RANDOM_STATE,
    TRAIN_TEST_SPLIT_RATIO,
)
from utils.exceptions import FusionError
from utils.holdout_split import identify_training_normal_blocks
from utils.logger import get_logger

logger = get_logger(__name__)

ID_COLUMN = "BlockId"


@dataclass
class FusionProductionConfig:
    """Everything needed to reproduce the production fusion run exactly."""

    random_state: int
    if_source: str
    if_weight: float
    ae_weight: float
    threshold: float
    calibration_method: str
    calibration_fit_on: str
    training_blocks: int
    validation_blocks: int
    test_blocks: int


class FusionEngine:
    """
    Applies the fixed, approved fusion configuration (IF-v2 weight=0.10,
    AE weight=0.90, threshold=0.323459) to every block, after calibrating
    both detectors' scores via isotonic regression fit on validation only.

    Parameters
    ----------
    iforest_v2_predictions_file : Path
        CSV produced by ``IsolationForestV2Validator.run()``.
    autoencoder_predictions_file : Path
        CSV produced by ``AutoencoderDetector.run()``.
    features_file : Path
        Feature matrix, used only to reproduce the shared training-block
        exclusion (identical split IF-v2 and the Autoencoder both use).
    predictions_file : Path
        Destination CSV for fused scores/predictions over every block.
    config_file : Path
        Destination joblib file for the fixed weights/threshold/seed and
        the fitted calibrators, for reproducibility.
    if_weight, ae_weight, threshold : float
        Fixed, approved production configuration. Not searched here.
    """

    def __init__(
        self,
        iforest_v2_predictions_file: Path = ISOLATION_FOREST_V2_PREDICTIONS_FILE,
        autoencoder_predictions_file: Path = AUTOENCODER_PREDICTIONS_FILE,
        features_file: Path = FEATURES_FILE,
        predictions_file: Path = FUSION_PREDICTIONS_FILE,
        config_file: Path = FUSION_CONFIG_FILE,
        if_weight: float = FUSION_PRODUCTION_IF_WEIGHT,
        ae_weight: float = FUSION_PRODUCTION_AE_WEIGHT,
        threshold: float = FUSION_PRODUCTION_THRESHOLD,
        train_split_ratio: float = TRAIN_TEST_SPLIT_RATIO,
        validation_split_ratio: float = FUSION_VALIDATION_SPLIT_RATIO,
    ) -> None:
        self.iforest_v2_predictions_file = Path(iforest_v2_predictions_file)
        self.autoencoder_predictions_file = Path(autoencoder_predictions_file)
        self.features_file = Path(features_file)
        self.predictions_file = Path(predictions_file)
        self.config_file = Path(config_file)
        self.if_weight = if_weight
        self.ae_weight = ae_weight
        self.threshold = threshold
        self.train_split_ratio = train_split_ratio
        self.validation_split_ratio = validation_split_ratio

        if round(self.if_weight + self.ae_weight, 6) != 1.0:
            raise FusionError(
                f"IF weight ({self.if_weight}) + AE weight ({self.ae_weight}) must sum to 1.0"
            )

    def _load_predictions(self) -> pd.DataFrame:
        """Load and inner-join Isolation Forest v2 and Autoencoder predictions."""
        if not self.iforest_v2_predictions_file.exists():
            raise FusionError(
                f"Isolation Forest v2 predictions not found: {self.iforest_v2_predictions_file}. "
                "Run models.isolation_forest_model_v2 first."
            )
        if not self.autoencoder_predictions_file.exists():
            raise FusionError(
                f"Autoencoder predictions not found: {self.autoencoder_predictions_file}. "
                "Run models.autoencoder_model first."
            )

        try:
            if_df = pd.read_csv(
                self.iforest_v2_predictions_file,
                usecols=[ID_COLUMN, "if_v2_anomaly_score"],
            )
            ae_df = pd.read_csv(
                self.autoencoder_predictions_file,
                usecols=[ID_COLUMN, "autoencoder_anomaly_score"],
            )
        except (OSError, pd.errors.ParserError, ValueError) as exc:
            raise FusionError(f"Failed to read prediction file(s): {exc}") from exc

        merged = if_df.merge(ae_df, on=ID_COLUMN, how="inner")
        if merged.empty:
            raise FusionError(
                "No overlapping blocks between IF-v2 and Autoencoder predictions -- "
                "were they scored from the same features.csv?"
            )

        dropped = (len(if_df) - len(merged)) + (len(ae_df) - len(merged))
        if dropped:
            logger.warning(
                "%d block(s) present in only one detector's predictions were "
                "excluded from fusion (BlockId mismatch).",
                dropped,
            )
        if len(if_df) != len(ae_df) or len(merged) != len(if_df):
            logger.warning(
                "Prediction row counts differ: IF-v2=%d, AE=%d, joined=%d -- "
                "verify both detectors scored the identical block population.",
                len(if_df), len(ae_df), len(merged),
            )

        return merged

    def _build_split(self, merged: pd.DataFrame):
        """
        Reproduce train / validation / test exactly as in the validated
        experiment: exclude the 446,578 training-Normal blocks (shared by
        IF-v2 and AE), then split the remaining holdout into validation
        (used here only to fit calibrators, never to select weight or
        threshold) and test (touched only by the caller's evaluation
        step, never here).
        """
        labels_df = LogCollector().load_anomaly_labels()
        merged = merged.merge(labels_df, on=ID_COLUMN, how="inner")

        training_block_ids, _ = identify_training_normal_blocks(
            features_file=self.features_file,
            train_split_ratio=self.train_split_ratio,
            random_state=RANDOM_STATE,
        )
        is_train = merged[ID_COLUMN].astype(str).isin(training_block_ids)
        train_df = merged.loc[is_train].reset_index(drop=True)
        holdout_df = merged.loc[~is_train].reset_index(drop=True)

        if holdout_df.empty:
            raise FusionError("No holdout blocks remain after excluding training blocks.")

        validation_df, test_df = train_test_split(
            holdout_df,
            train_size=self.validation_split_ratio,
            random_state=RANDOM_STATE,
            stratify=holdout_df["Label"],
        )
        validation_df = validation_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        self._verify_no_leakage(train_df, validation_df, test_df)
        return train_df, validation_df, test_df

    @staticmethod
    def _verify_no_leakage(train_df, validation_df, test_df) -> None:
        train_ids = set(train_df[ID_COLUMN])
        val_ids = set(validation_df[ID_COLUMN])
        test_ids = set(test_df[ID_COLUMN])

        tv = train_ids & val_ids
        tt = train_ids & test_ids
        vt = val_ids & test_ids
        dupes = {
            "train": len(train_df) - len(train_ids),
            "validation": len(validation_df) - len(val_ids),
            "test": len(test_df) - len(test_ids),
        }
        if tv or tt or vt:
            raise FusionError(
                f"Data leakage detected: train/val={len(tv)}, train/test={len(tt)}, val/test={len(vt)}"
            )
        if any(dupes.values()):
            raise FusionError(f"Duplicate BlockIds detected: {dupes}")

        logger.info(
            "Leakage check PASSED: train/val overlap=0, train/test overlap=0, "
            "val/test overlap=0, duplicate BlockIds=0 (train=%d, val=%d, test=%d)",
            len(train_df), len(validation_df), len(test_df),
        )

    def run(self) -> Dict:
        """
        Execute production fusion: load frozen IF-v2 + AE scores,
        calibrate both on validation only, apply the fixed approved
        weight/threshold to every block, and persist results.

        Returns
        -------
        dict
            Summary of the fusion run.
        """
        start_time = time.time()

        merged = self._load_predictions()
        train_df, validation_df, test_df = self._build_split(merged)

        # ---- Calibrate BOTH scores on validation only (never test) ----
        if_calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ae_calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

        val_y = validation_df["Label"].map({LABEL_NORMAL: 0, LABEL_ANOMALY: 1}).to_numpy()
        if_calibrator.fit(validation_df["if_v2_anomaly_score"].to_numpy(), val_y)
        ae_calibrator.fit(validation_df["autoencoder_anomaly_score"].to_numpy(), val_y)

        logger.info(
            "Calibrators fit on validation only (%d blocks: %d Normal, %d Anomaly).",
            len(validation_df),
            int((validation_df["Label"] == LABEL_NORMAL).sum()),
            int((validation_df["Label"] == LABEL_ANOMALY).sum()),
        )

        # ---- Apply calibration + fixed weights to EVERY block ----
        merged["if_v2_calibrated_score"] = if_calibrator.predict(
            merged["if_v2_anomaly_score"].to_numpy()
        )
        merged["autoencoder_calibrated_score"] = ae_calibrator.predict(
            merged["autoencoder_anomaly_score"].to_numpy()
        )
        merged["fusion_score"] = (
            self.if_weight * merged["if_v2_calibrated_score"]
            + self.ae_weight * merged["autoencoder_calibrated_score"]
        )
        merged["fusion_prediction"] = np.where(
            merged["fusion_score"] >= self.threshold, LABEL_ANOMALY, LABEL_NORMAL
        )

        output_columns = [
            ID_COLUMN,
            "if_v2_anomaly_score",
            "autoencoder_anomaly_score",
            "if_v2_calibrated_score",
            "autoencoder_calibrated_score",
            "fusion_score",
            "fusion_prediction",
        ]
        predictions_df = merged[output_columns]

        self.predictions_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            predictions_df.to_csv(self.predictions_file, index=False)
        except OSError as exc:
            raise FusionError(f"Failed to write fusion predictions: {exc}") from exc
        logger.info("Fusion predictions written to %s", self.predictions_file)

        # ---- Persist config + calibrators for reproducibility ----
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        config = FusionProductionConfig(
            random_state=RANDOM_STATE,
            if_source="isolation_forest_model_v2 (19 features, production)",
            if_weight=self.if_weight,
            ae_weight=self.ae_weight,
            threshold=self.threshold,
            calibration_method="isotonic_regression (fit independently for IF-v2 and AE)",
            calibration_fit_on="validation_only",
            training_blocks=len(train_df),
            validation_blocks=len(validation_df),
            test_blocks=len(test_df),
        )
        try:
            dump(
                {
                    "config": asdict(config),
                    "if_calibrator": if_calibrator,
                    "ae_calibrator": ae_calibrator,
                },
                self.config_file,
            )
        except OSError as exc:
            raise FusionError(f"Failed to persist fusion config: {exc}") from exc
        logger.info("Fusion config + calibrators persisted to %s", self.config_file)

        elapsed = time.time() - start_time
        anomaly_count = int((predictions_df["fusion_prediction"] == LABEL_ANOMALY).sum())
        normal_count = int((predictions_df["fusion_prediction"] == LABEL_NORMAL).sum())

        logger.info(
            "Production fusion complete: IF-v2 weight=%.2f AE weight=%.2f threshold=%.6f -> "
            "%d blocks scored (%d Anomaly, %d Normal), %.2fs elapsed",
            self.if_weight, self.ae_weight, self.threshold,
            len(predictions_df), anomaly_count, normal_count, elapsed,
        )

        return {
            "predictions_file": str(self.predictions_file),
            "config_file": str(self.config_file),
            "if_weight": self.if_weight,
            "ae_weight": self.ae_weight,
            "threshold": self.threshold,
            "training_blocks": len(train_df),
            "validation_blocks": len(validation_df),
            "test_blocks": len(test_df),
            "total_blocks": len(predictions_df),
            "predicted_anomaly_count": anomaly_count,
            "predicted_normal_count": normal_count,
            "elapsed_seconds": round(elapsed, 2),
        }


def main() -> Dict:
    """Entry point for running production fusion as a standalone step."""
    engine = FusionEngine()
    return engine.run()


if __name__ == "__main__":
    result = main()
    logger.info("Fusion summary: %s", json.dumps(result, indent=2))
