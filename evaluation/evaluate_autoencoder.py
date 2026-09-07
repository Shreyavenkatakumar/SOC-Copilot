"""
Module 4, Part B: Autoencoder Evaluation.

Joins Autoencoder predictions against ground-truth block labels
(``anomaly_label.csv``) and computes standard classification metrics,
restricted to blocks the Autoencoder never trained on (holdout Normal
blocks + all Anomaly blocks) -- the same leakage-free approach used for
the Isolation Forest evaluator (evaluation/evaluate_isolation_forest.py).
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

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
    AUTOENCODER_EVALUATION_FILE,
    AUTOENCODER_PREDICTIONS_FILE,
    FEATURES_FILE,
    LABEL_ANOMALY,
    LABEL_NORMAL,
    LABEL_TO_BINARY,
    RANDOM_STATE,
    TRAIN_TEST_SPLIT_RATIO,
)
from utils.exceptions import EvaluationError
from utils.logger import get_logger

logger = get_logger(__name__)

ID_COLUMN = "BlockId"


@dataclass
class AutoencoderEvaluationReport:
    """Classification metrics comparing Autoencoder predictions against ground truth."""

    generated_at: str
    predictions_file: str
    features_file: str
    total_predicted_blocks: int
    unmatched_prediction_blocks: int
    labeled_normal_blocks: int
    training_blocks_excluded: int
    holdout_normal_blocks: int
    anomaly_blocks: int
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


class AutoencoderEvaluator:
    """
    Computes evaluation metrics for Autoencoder predictions against the
    ground-truth HDFS block labels, restricted to blocks the model
    never saw during training.

    Parameters
    ----------
    predictions_file : Path
        CSV produced by ``AutoencoderDetector.run()``.
    features_file : Path
        Feature matrix produced by ``FeatureExtractor``. Used only to
        reproduce the exact row order/count needed to replicate the
        Autoencoder's train/test split -- no feature values are used
        in metric computation.
    output_file : Path
        Destination JSON for the evaluation report.
    train_split_ratio : float
        Must match the ratio used by ``AutoencoderDetector`` (and the
        Isolation Forest) so the reproduced split identifies the
        correct training blocks.
    """

    def __init__(
        self,
        predictions_file: Path = AUTOENCODER_PREDICTIONS_FILE,
        features_file: Path = FEATURES_FILE,
        output_file: Path = AUTOENCODER_EVALUATION_FILE,
        train_split_ratio: float = TRAIN_TEST_SPLIT_RATIO,
    ) -> None:
        self.predictions_file = Path(predictions_file)
        self.features_file = Path(features_file)
        self.output_file = Path(output_file)
        self.train_split_ratio = train_split_ratio

    def _load_predictions(self) -> pd.DataFrame:
        if not self.predictions_file.exists():
            raise EvaluationError(f"Predictions file not found: {self.predictions_file}")
        try:
            return pd.read_csv(self.predictions_file)
        except (OSError, pd.errors.ParserError) as exc:
            raise EvaluationError(f"Failed to read {self.predictions_file}: {exc}") from exc

    @staticmethod
    def _load_ground_truth() -> pd.DataFrame:
        collector = LogCollector()
        return collector.load_anomaly_labels()

    def _identify_training_blocks(self, labels_df: pd.DataFrame) -> Tuple[Set[str], int]:
        """
        Reproduce the exact BlockId set used to train the Autoencoder
        (models/autoencoder_model.py), which in turn uses the identical
        split logic as the Isolation Forest -- same features file row
        order, same left-join, same ``RANDOM_STATE``/``TRAIN_TEST_SPLIT_RATIO``.

        Returns
        -------
        Tuple[Set[str], int]
            The set of BlockIds used for Autoencoder training, and the
            total number of labeled Normal blocks found.
        """
        if not self.features_file.exists():
            raise EvaluationError(
                f"Features file not found: {self.features_file}. It is required "
                "to reproduce the Autoencoder train/test split for leakage-free "
                "evaluation."
            )

        try:
            block_ids = pd.read_csv(self.features_file, usecols=[ID_COLUMN])
        except (OSError, pd.errors.ParserError, ValueError) as exc:
            raise EvaluationError(f"Failed to read {self.features_file}: {exc}") from exc

        merged = block_ids.merge(labels_df, on=ID_COLUMN, how="left")
        normal_blocks = merged.loc[merged["Label"] == LABEL_NORMAL]

        if normal_blocks.empty:
            raise EvaluationError(
                "No blocks labeled 'Normal' were found while reproducing the "
                "Autoencoder train/test split; cannot determine holdout set."
            )

        if len(normal_blocks) < 2:
            train_blocks = normal_blocks
        else:
            train_blocks, _ = train_test_split(
                normal_blocks,
                train_size=self.train_split_ratio,
                random_state=RANDOM_STATE,
            )

        training_block_ids = set(train_blocks[ID_COLUMN].astype(str))
        return training_block_ids, len(normal_blocks)

    def evaluate(self) -> Dict:
        """
        Run the full Module 4, Part B evaluation on holdout data only.

        Returns
        -------
        dict
            Summary of computed metrics.

        Raises
        ------
        EvaluationError
            If predictions/labels/features cannot be loaded, joined, or
            the reproduced split does not yield a usable holdout set.
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

        training_block_ids, labeled_normal_count = self._identify_training_blocks(labels_df)

        is_training_block = merged[ID_COLUMN].astype(str).isin(training_block_ids)
        holdout = merged.loc[~is_training_block].copy()
        training_blocks_excluded = int(is_training_block.sum())

        if holdout.empty:
            raise EvaluationError(
                "No blocks remain for evaluation after excluding Autoencoder "
                "training blocks."
            )

        holdout_normal_count = int((holdout["Label"] == LABEL_NORMAL).sum())
        anomaly_count = int((holdout["Label"] == LABEL_ANOMALY).sum())

        logger.info(
            "Excluded %d block(s) used in Autoencoder training. Evaluating on "
            "%d holdout block(s) (%d Normal, %d Anomaly).",
            training_blocks_excluded,
            len(holdout),
            holdout_normal_count,
            anomaly_count,
        )

        y_true = holdout["Label"].map(LABEL_TO_BINARY).to_numpy()
        y_pred = holdout["autoencoder_prediction"].map(LABEL_TO_BINARY).to_numpy()
        y_score = holdout["autoencoder_anomaly_score"].to_numpy()

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

        report = AutoencoderEvaluationReport(
            generated_at=pd.Timestamp.now(tz="UTC").isoformat(),
            predictions_file=str(self.predictions_file),
            features_file=str(self.features_file),
            total_predicted_blocks=len(predictions_df),
            unmatched_prediction_blocks=unmatched,
            labeled_normal_blocks=labeled_normal_count,
            training_blocks_excluded=training_blocks_excluded,
            holdout_normal_blocks=holdout_normal_count,
            anomaly_blocks=anomaly_count,
            evaluated_blocks=len(holdout),
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
            "Autoencoder evaluation complete (holdout-only): accuracy=%.4f "
            "precision=%.4f recall=%.4f f1=%.4f roc_auc=%s pr_auc=%s",
            report.accuracy,
            report.precision,
            report.recall,
            report.f1_score,
            report.roc_auc,
            report.pr_auc,
        )

        return asdict(report)

    def _write_report(self, report: AutoencoderEvaluationReport) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.output_file, mode="w", encoding="utf-8") as handle:
                json.dump(asdict(report), handle, indent=2)
        except OSError as exc:
            raise EvaluationError(f"Failed to write evaluation report: {exc}") from exc
        logger.info("Autoencoder evaluation report written to %s", self.output_file)


def main() -> Dict:
    """Entry point for running evaluation as a standalone step."""
    evaluator = AutoencoderEvaluator()
    return evaluator.evaluate()


if __name__ == "__main__":
    result = main()
    logger.info("Autoencoder evaluation summary: %s", json.dumps(result, indent=2))
