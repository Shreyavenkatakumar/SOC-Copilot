"""
Module 5, Part B: Fusion Evaluation.

Joins fused predictions (models/fusion_engine.py) against ground-truth
block labels and computes standard classification metrics, restricted
to the **test** slice of the holdout population -- the portion of
holdout data that ``FusionEngine`` reserved and never touched while
grid-searching the fusion weight/threshold.

This module never imports or modifies ``models/fusion_engine.py``.
Instead it independently reproduces, in order:

  1. the same Normal-only training-block exclusion used by both
     underlying detectors (via ``utils.holdout_split``), and
  2. the same validation/test split of the resulting holdout population
     (same ``RANDOM_STATE``, same stratification, same
     ``FUSION_VALIDATION_SPLIT_RATIO``) that ``FusionEngine`` used.

Both steps are deterministic given the same inputs and seed, so this
recovers exactly the test slice ``FusionEngine`` never scored during
weight selection -- keeping the final reported metrics free of any
leakage from the weight/threshold search itself.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from data.log_collector import LogCollector
from utils.config import (
    FEATURES_FILE,
    FUSION_EVALUATION_FILE,
    FUSION_PREDICTIONS_FILE,
    FUSION_VALIDATION_SPLIT_RATIO,
    LABEL_ANOMALY,
    LABEL_NORMAL,
    LABEL_TO_BINARY,
    RANDOM_STATE,
    TRAIN_TEST_SPLIT_RATIO,
)
from utils.exceptions import EvaluationError
from utils.holdout_split import identify_training_normal_blocks
from utils.logger import get_logger

logger = get_logger(__name__)

ID_COLUMN = "BlockId"


@dataclass
class FusionEvaluationReport:
    """Classification metrics comparing fused predictions against ground truth."""

    generated_at: str
    predictions_file: str
    features_file: str
    total_predicted_blocks: int
    unmatched_prediction_blocks: int
    labeled_normal_blocks: int
    training_blocks_excluded: int
    validation_blocks_excluded: int
    test_normal_blocks: int
    test_anomaly_blocks: int
    evaluated_blocks: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: List[List[int]]
    confusion_matrix_labels: List[str]
    elapsed_seconds: float = 0.0
    warnings: List[str] = None


class FusionEvaluator:
    """
    Computes evaluation metrics for fused predictions against the
    ground-truth HDFS block labels, restricted to the untouched test
    slice of the holdout population.

    Parameters
    ----------
    predictions_file : Path
        CSV produced by ``FusionEngine.run()``.
    features_file : Path
        Feature matrix, used only to reproduce the shared training-block
        exclusion.
    output_file : Path
        Destination JSON for the evaluation report.
    train_split_ratio : float
        Must match the ratio used by the underlying detectors.
    validation_split_ratio : float
        Must match the ratio used by ``FusionEngine`` when splitting
        holdout into validation/test.
    """

    def __init__(
        self,
        predictions_file: Path = FUSION_PREDICTIONS_FILE,
        features_file: Path = FEATURES_FILE,
        output_file: Path = FUSION_EVALUATION_FILE,
        train_split_ratio: float = TRAIN_TEST_SPLIT_RATIO,
        validation_split_ratio: float = FUSION_VALIDATION_SPLIT_RATIO,
    ) -> None:
        self.predictions_file = Path(predictions_file)
        self.features_file = Path(features_file)
        self.output_file = Path(output_file)
        self.train_split_ratio = train_split_ratio
        self.validation_split_ratio = validation_split_ratio

    def _load_predictions(self) -> pd.DataFrame:
        if not self.predictions_file.exists():
            raise EvaluationError(f"Predictions file not found: {self.predictions_file}")
        try:
            return pd.read_csv(self.predictions_file)
        except (OSError, pd.errors.ParserError) as exc:
            raise EvaluationError(f"Failed to read {self.predictions_file}: {exc}") from exc

    @staticmethod
    def _load_ground_truth() -> pd.DataFrame:
        return LogCollector().load_anomaly_labels()

    def _split_validation_test(self, holdout: pd.DataFrame) -> pd.DataFrame:
        """
        Reproduce ``FusionEngine._split_validation_test`` exactly and
        return only the test slice.
        """
        if len(holdout) < 4:
            # Matches FusionEngine's fallback for tiny smoke-test datasets.
            return holdout

        _validation, test = train_test_split(
            holdout,
            train_size=self.validation_split_ratio,
            random_state=RANDOM_STATE,
            stratify=holdout["Label"],
        )
        return test

    def evaluate(self) -> Dict:
        """
        Run the full Module 5, Part B evaluation on the test slice only.

        Returns
        -------
        dict
            Summary of computed metrics.

        Raises
        ------
        EvaluationError
            If predictions/labels/features cannot be loaded, joined, or
            the reproduced split does not yield a usable test set.
        """
        start_time = time.time()
        warnings: List[str] = []

        predictions_df = self._load_predictions()
        labels_df = self._load_ground_truth()

        merged = predictions_df.merge(labels_df, on=ID_COLUMN, how="inner")
        unmatched = len(predictions_df) - len(merged)
        if unmatched:
            warnings.append(
                f"{unmatched} predicted block(s) had no matching ground-truth "
                "label and were excluded from evaluation."
            )

        if merged.empty:
            raise EvaluationError(
                "No overlapping blocks between predictions and ground-truth labels."
            )

        unknown_labels = set(merged["Label"].unique()) - set(LABEL_TO_BINARY.keys())
        if unknown_labels:
            raise EvaluationError(f"Unexpected ground-truth label value(s): {unknown_labels}")

        training_block_ids, labeled_normal_count = identify_training_normal_blocks(
            features_file=self.features_file,
            train_split_ratio=self.train_split_ratio,
            random_state=RANDOM_STATE,
        )

        is_training_block = merged[ID_COLUMN].astype(str).isin(training_block_ids)
        holdout = merged.loc[~is_training_block].copy()
        training_blocks_excluded = int(is_training_block.sum())

        if holdout.empty:
            raise EvaluationError(
                "No blocks remain after excluding training blocks; cannot "
                "reproduce the validation/test split."
            )

        test = self._split_validation_test(holdout)
        validation_blocks_excluded = len(holdout) - len(test)

        if test.empty:
            raise EvaluationError("No blocks remain in the reproduced test slice.")

        test_normal_count = int((test["Label"] == LABEL_NORMAL).sum())
        test_anomaly_count = int((test["Label"] == LABEL_ANOMALY).sum())

        logger.info(
            "Excluded %d training block(s) and %d validation block(s) used for "
            "fusion weight selection. Evaluating fusion on %d test block(s) "
            "(%d Normal, %d Anomaly).",
            training_blocks_excluded,
            validation_blocks_excluded,
            len(test),
            test_normal_count,
            test_anomaly_count,
        )

        y_true = test["Label"].map(LABEL_TO_BINARY).to_numpy()
        y_pred = test["fusion_prediction"].map(LABEL_TO_BINARY).to_numpy()
        y_score = test["fusion_score"].to_numpy()

        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        except ValueError as exc:
            raise EvaluationError(f"Failed to compute classification metrics: {exc}") from exc

        try:
            roc_auc = roc_auc_score(y_true, y_score)
        except ValueError as exc:
            roc_auc = float("nan")
            warnings.append(f"ROC-AUC could not be computed: {exc}")

        try:
            pr_auc = average_precision_score(y_true, y_score)
        except ValueError as exc:
            pr_auc = float("nan")
            warnings.append(f"PR-AUC could not be computed: {exc}")

        elapsed = time.time() - start_time

        report = FusionEvaluationReport(
            generated_at=pd.Timestamp.now(tz="UTC").isoformat(),
            predictions_file=str(self.predictions_file),
            features_file=str(self.features_file),
            total_predicted_blocks=len(predictions_df),
            unmatched_prediction_blocks=unmatched,
            labeled_normal_blocks=labeled_normal_count,
            training_blocks_excluded=training_blocks_excluded,
            validation_blocks_excluded=validation_blocks_excluded,
            test_normal_blocks=test_normal_count,
            test_anomaly_blocks=test_anomaly_count,
            evaluated_blocks=len(test),
            accuracy=round(float(accuracy), 6),
            precision=round(float(precision), 6),
            recall=round(float(recall), 6),
            f1_score=round(float(f1), 6),
            roc_auc=round(float(roc_auc), 6) if roc_auc == roc_auc else None,  # NaN check
            pr_auc=round(float(pr_auc), 6) if pr_auc == pr_auc else None,  # NaN check
            confusion_matrix=cm.tolist(),
            confusion_matrix_labels=["Normal", "Anomaly"],
            elapsed_seconds=round(elapsed, 2),
            warnings=warnings,
        )
        self._write_report(report)

        logger.info(
            "Fusion evaluation complete (test-only): accuracy=%.4f precision=%.4f "
            "recall=%.4f f1=%.4f roc_auc=%s pr_auc=%s",
            report.accuracy,
            report.precision,
            report.recall,
            report.f1_score,
            report.roc_auc,
            report.pr_auc,
        )

        return asdict(report)

    def _write_report(self, report: FusionEvaluationReport) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.output_file, mode="w", encoding="utf-8") as handle:
                json.dump(asdict(report), handle, indent=2)
        except OSError as exc:
            raise EvaluationError(f"Failed to write evaluation report: {exc}") from exc
        logger.info("Fusion evaluation report written to %s", self.output_file)


def main() -> Dict:
    """Entry point for running evaluation as a standalone step."""
    evaluator = FusionEvaluator()
    return evaluator.evaluate()


if __name__ == "__main__":
    result = main()
    logger.info("Fusion evaluation summary: %s", json.dumps(result, indent=2))
