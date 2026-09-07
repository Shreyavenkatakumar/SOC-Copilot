"""
Module 3, Part C: Evaluation.

Joins Isolation Forest predictions against ground-truth block labels
(``anomaly_label.csv``) and computes standard classification metrics.

Data-leakage fix
-----------------
``IsolationForestDetector`` (models/isolation_forest_model.py) trains on
80% of the ground-truth *Normal* blocks and then scores *every* block,
including the ones it was trained on. Evaluating against all 575,061
blocks therefore mixes in-sample and out-of-sample performance: the
446,578 Normal blocks the model was fit on are trivially easy for it to
score as Normal, which inflates accuracy/precision relative to the
model's true generalization performance.

This module never modifies the training/prediction artifacts. Instead it
independently *reproduces* the exact same train/test split — same
``FEATURES_FILE`` row order, same left-join against
``anomaly_label.csv``, same ``RANDOM_STATE``, and same
``TRAIN_TEST_SPLIT_RATIO`` used inside ``IsolationForestDetector``
-- purely to recover the set of BlockIds that were used for training, so
they can be excluded before computing metrics.

Threshold calibration
----------------------
``IsolationForestDetector.score()`` labels each block using
``sklearn.ensemble.IsolationForest.predict()``, whose Normal/Anomaly
cutoff is entirely determined by the ``contamination`` parameter set at
training time. If that value does not match the population's true
anomaly rate, the resulting predictions can be badly miscalibrated
(e.g. flagging ~25% of blocks as anomalous when only ~2.9% actually
are), which manifests as low precision/accuracy despite a good
underlying ROC-AUC.

This module now *additionally* computes an F1-optimal decision
threshold directly from the continuous ``iforest_anomaly_score``
column on the holdout set, using ``precision_recall_curve``. This does
not touch the trained model or the persisted ``iforest_prediction``
column — it is a purely evaluation-side recalibration, reported
alongside the original contamination-based metrics so both can be
compared.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from data.log_collector import LogCollector
from utils.config import (
    FEATURES_FILE,
    ISOLATION_FOREST_EVALUATION_FILE,
    ISOLATION_FOREST_PREDICTIONS_FILE,
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
class EvaluationReport:
    """Classification metrics comparing predictions against ground truth."""

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
    # --- contamination-based metrics (original decision from model.predict()) ---
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: List[List[int]]
    confusion_matrix_labels: List[str]
    # --- NEW: F1-optimal threshold, calibrated on holdout scores ---
    calibrated_threshold: float
    calibrated_accuracy: float
    calibrated_precision: float
    calibrated_recall: float
    calibrated_f1_score: float
    calibrated_confusion_matrix: List[List[int]]
    elapsed_seconds: float = 0.0
    warnings: List[str] = None


class IsolationForestEvaluator:
    """
    Computes evaluation metrics for Isolation Forest predictions against
    the ground-truth HDFS block labels, restricted to blocks the model
    never saw during training (holdout Normal blocks + all Anomaly
    blocks) to avoid data leakage.

    Parameters
    ----------
    predictions_file : Path
        CSV produced by ``IsolationForestDetector.run()``.
    features_file : Path
        Feature matrix produced by ``FeatureExtractor``. Used only to
        reproduce the exact row order/count needed to replicate the
        Isolation Forest's train/test split — no feature values are
        used in metric computation.
    output_file : Path
        Destination JSON for the evaluation report.
    train_split_ratio : float
        Must match the ratio used by ``IsolationForestDetector`` so the
        reproduced split identifies the correct training blocks.
    """

    def __init__(
        self,
        predictions_file: Path = ISOLATION_FOREST_PREDICTIONS_FILE,
        features_file: Path = FEATURES_FILE,
        output_file: Path = ISOLATION_FOREST_EVALUATION_FILE,
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
        Reproduce the exact BlockId set used to train the Isolation
        Forest, without touching ``models/isolation_forest_model.py`` or
        its persisted outputs.
        """
        if not self.features_file.exists():
            raise EvaluationError(
                f"Features file not found: {self.features_file}. It is required "
                "to reproduce the Isolation Forest train/test split for "
                "leakage-free evaluation."
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
                "Isolation Forest train/test split; cannot determine holdout set."
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

    @staticmethod
    def _calibrate_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Find the anomaly-score cutoff that maximizes F1 on the holdout
        set, independent of whatever ``contamination`` produced.

        ``y_score`` here is ``iforest_anomaly_score`` (already 0-1,
        higher = more anomalous), so no sign-flip is needed.

        Returns
        -------
        Tuple[float, numpy.ndarray]
            The chosen threshold, and the resulting binary predictions
            (1 = Anomaly, 0 = Normal) for the holdout set.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        f1_scores = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) != 0,
        )
        # precision_recall_curve returns one more point than thresholds
        best_idx = int(np.argmax(f1_scores[:-1])) if len(thresholds) > 0 else 0
        best_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5

        y_pred_calibrated = (y_score >= best_threshold).astype(int)
        return best_threshold, y_pred_calibrated

    def evaluate(self) -> Dict:
        """
        Run the full Module 3, Part C evaluation on holdout data only.

        Returns
        -------
        dict
            Summary of computed metrics.
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
                "No blocks remain for evaluation after excluding Isolation "
                "Forest training blocks."
            )

        holdout_normal_count = int((holdout["Label"] == LABEL_NORMAL).sum())
        anomaly_count = int((holdout["Label"] == LABEL_ANOMALY).sum())

        logger.info(
            "Excluded %d block(s) used in Isolation Forest training. "
            "Evaluating on %d holdout block(s) (%d Normal, %d Anomaly).",
            training_blocks_excluded,
            len(holdout),
            holdout_normal_count,
            anomaly_count,
        )

        y_true = holdout["Label"].map(LABEL_TO_BINARY).to_numpy()
        y_pred = holdout["iforest_prediction"].map(LABEL_TO_BINARY).to_numpy()
        y_score = holdout["iforest_anomaly_score"].to_numpy()

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

        # --- NEW: threshold calibration, independent of `contamination` ---
        try:
            calibrated_threshold, y_pred_calibrated = self._calibrate_threshold(y_true, y_score)
            calibrated_accuracy = accuracy_score(y_true, y_pred_calibrated)
            calibrated_precision = precision_score(y_true, y_pred_calibrated, zero_division=0)
            calibrated_recall = recall_score(y_true, y_pred_calibrated, zero_division=0)
            calibrated_f1 = f1_score(y_true, y_pred_calibrated, zero_division=0)
            calibrated_cm = confusion_matrix(y_true, y_pred_calibrated, labels=[0, 1])
            logger.info(
                "Calibrated threshold=%.6f -> accuracy=%.4f precision=%.4f "
                "recall=%.4f f1=%.4f (contamination-based f1=%.4f)",
                calibrated_threshold,
                calibrated_accuracy,
                calibrated_precision,
                calibrated_recall,
                calibrated_f1,
                f1,
            )
        except ValueError as exc:
            raise EvaluationError(f"Failed to compute calibrated metrics: {exc}") from exc

        elapsed = time.time() - start_time

        report = EvaluationReport(
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
            confusion_matrix=cm.tolist(),
            confusion_matrix_labels=["Normal", "Anomaly"],
            calibrated_threshold=round(calibrated_threshold, 6),
            calibrated_accuracy=round(float(calibrated_accuracy), 6),
            calibrated_precision=round(float(calibrated_precision), 6),
            calibrated_recall=round(float(calibrated_recall), 6),
            calibrated_f1_score=round(float(calibrated_f1), 6),
            calibrated_confusion_matrix=calibrated_cm.tolist(),
            elapsed_seconds=round(elapsed, 2),
            warnings=warnings,
        )
        self._write_report(report)

        logger.info(
            "Evaluation complete (holdout-only): accuracy=%.4f precision=%.4f "
            "recall=%.4f f1=%.4f roc_auc=%s | calibrated: accuracy=%.4f "
            "precision=%.4f recall=%.4f f1=%.4f",
            report.accuracy,
            report.precision,
            report.recall,
            report.f1_score,
            report.roc_auc,
            report.calibrated_accuracy,
            report.calibrated_precision,
            report.calibrated_recall,
            report.calibrated_f1_score,
        )

        return asdict(report)

    def _write_report(self, report: EvaluationReport) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.output_file, mode="w", encoding="utf-8") as handle:
                json.dump(asdict(report), handle, indent=2)
        except OSError as exc:
            raise EvaluationError(f"Failed to write evaluation report: {exc}") from exc
        logger.info("Evaluation report written to %s", self.output_file)


def main() -> Dict:
    """Entry point for running evaluation as a standalone step."""
    evaluator = IsolationForestEvaluator()
    return evaluator.evaluate()


if __name__ == "__main__":
    result = main()
    logger.info("Evaluation summary: %s", json.dumps(result, indent=2))