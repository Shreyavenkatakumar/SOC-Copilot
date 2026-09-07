"""
Module 3, Part B: Isolation Forest Anomaly Detection.

Trains a scikit-learn ``IsolationForest`` on ground-truth *Normal* blocks
only (semi-supervised), then scores every block in the feature matrix
produced by ``features.feature_extractor``.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from data.log_collector import LogCollector
from utils.config import (
    FEATURES_FILE,
    ISOLATION_FOREST_MODEL_FILE,
    ISOLATION_FOREST_PARAMS,
    ISOLATION_FOREST_PREDICTIONS_FILE,
    LABEL_ANOMALY,
    LABEL_NORMAL,
    RANDOM_STATE,
    TRAIN_TEST_SPLIT_RATIO,
)
from utils.exceptions import ModelInferenceError, ModelTrainingError
from utils.logger import get_logger

logger = get_logger(__name__)

ID_COLUMN = "BlockId"


@dataclass
class IsolationForestRunSummary:
    """Summary statistics produced after a full train + score run."""

    generated_at: str
    features_file: str
    model_file: str
    predictions_file: str
    total_blocks: int
    labeled_normal_blocks: int
    training_blocks: int
    holdout_normal_blocks: int
    predicted_anomaly_count: int
    predicted_normal_count: int
    elapsed_seconds: float = 0.0


class IsolationForestDetector:
    """
    Trains and applies an Isolation Forest anomaly detector over
    per-block HDFS features.

    Parameters
    ----------
    features_file : Path
        Feature matrix produced by ``FeatureExtractor``.
    model_file : Path
        Destination path for the persisted (joblib) trained model.
    predictions_file : Path
        Destination CSV for per-block anomaly scores and predictions.
    train_split_ratio : float
        Fraction of ground-truth Normal blocks used for training; the
        remainder is folded back into the scored/evaluated population.
    """

    def __init__(
        self,
        features_file: Path = FEATURES_FILE,
        model_file: Path = ISOLATION_FOREST_MODEL_FILE,
        predictions_file: Path = ISOLATION_FOREST_PREDICTIONS_FILE,
        train_split_ratio: float = TRAIN_TEST_SPLIT_RATIO,
    ) -> None:
        self.features_file = Path(features_file)
        self.model_file = Path(model_file)
        self.predictions_file = Path(predictions_file)
        self.train_split_ratio = train_split_ratio
        self.model: IsolationForest = None

    def _load_features(self) -> pd.DataFrame:
        """Load the per-block feature matrix produced by Module 3, Part A."""
        if not self.features_file.exists():
            raise ModelTrainingError(f"Features file not found: {self.features_file}")
        try:
            return pd.read_csv(self.features_file)
        except (OSError, pd.errors.ParserError) as exc:
            raise ModelTrainingError(f"Failed to read {self.features_file}: {exc}") from exc

    @staticmethod
    def _feature_columns(features_df: pd.DataFrame) -> list:
        """Return all feature columns (everything except the block identifier)."""
        return [c for c in features_df.columns if c != ID_COLUMN]

    def _select_training_blocks(
        self, features_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, int]:
        """
        Identify blocks labeled ``Normal`` in the ground truth and split
        off a training subset, leaving the remainder to be scored
        alongside all anomalous blocks.

        Returns
        -------
        Tuple[pandas.DataFrame, int]
            The training feature rows, and the total count of labeled
            Normal blocks found.
        """
        collector = LogCollector()
        labels_df = collector.load_anomaly_labels()

        merged = features_df.merge(labels_df, on=ID_COLUMN, how="left")
        normal_blocks = merged.loc[merged["Label"] == LABEL_NORMAL]

        if normal_blocks.empty:
            raise ModelTrainingError(
                "No blocks labeled 'Normal' were found after joining features "
                "with anomaly_label.csv; cannot train Isolation Forest."
            )

        if len(normal_blocks) < 2:
            train_blocks = normal_blocks
        else:
            train_blocks, _ = train_test_split(
                normal_blocks,
                train_size=self.train_split_ratio,
                random_state=RANDOM_STATE,
            )

        return train_blocks, len(normal_blocks)

    def train(self, features_df: pd.DataFrame) -> Dict:
        """
        Fit the Isolation Forest on the Normal-only training subset.

        Returns
        -------
        dict
            Training-set sizing information, used later for reporting.
        """
        train_blocks, labeled_normal_count = self._select_training_blocks(features_df)
        feature_cols = self._feature_columns(features_df)

        try:
            self.model = IsolationForest(**ISOLATION_FOREST_PARAMS)
            self.model.fit(train_blocks[feature_cols])
        except (ValueError, MemoryError) as exc:
            raise ModelTrainingError(f"Isolation Forest training failed: {exc}") from exc

        logger.info(
            "Isolation Forest trained on %d Normal block(s) (out of %d labeled Normal), "
            "%d feature(s)",
            len(train_blocks),
            labeled_normal_count,
            len(feature_cols),
        )

        self.model_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            dump({"model": self.model, "feature_columns": feature_cols}, self.model_file)
        except OSError as exc:
            raise ModelTrainingError(f"Failed to persist Isolation Forest model: {exc}") from exc
        logger.info("Isolation Forest model persisted to %s", self.model_file)

        return {
            "training_blocks": len(train_blocks),
            "labeled_normal_blocks": labeled_normal_count,
        }

    def score(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score every block in ``features_df`` with the trained model.

        Returns
        -------
        pandas.DataFrame
            Columns: ``BlockId``, ``iforest_raw_score``,
            ``iforest_anomaly_score`` (0-1, higher = more anomalous),
            ``iforest_prediction`` ("Normal"/"Anomaly").
        """
        if self.model is None:
            raise ModelInferenceError("Isolation Forest model has not been trained yet.")

        feature_cols = self._feature_columns(features_df)
        try:
            raw_decision = self.model.decision_function(features_df[feature_cols])
            raw_prediction = self.model.predict(features_df[feature_cols])
        except (ValueError, MemoryError) as exc:
            raise ModelInferenceError(f"Isolation Forest scoring failed: {exc}") from exc

        # sklearn: decision_function is higher for inliers (normal); flip the
        # sign so that higher == more anomalous, then min-max scale to [0, 1].
        inverted = -raw_decision
        min_val, max_val = inverted.min(), inverted.max()
        if max_val > min_val:
            normalized = (inverted - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(inverted)

        predicted_label = np.where(raw_prediction == -1, LABEL_ANOMALY, LABEL_NORMAL)

        return pd.DataFrame(
            {
                ID_COLUMN: features_df[ID_COLUMN].values,
                "iforest_raw_score": raw_decision,
                "iforest_anomaly_score": normalized,
                "iforest_prediction": predicted_label,
            }
        )

    def run(self) -> Dict:
        """
        Execute the full Module 3, Part B pipeline: load features, train
        on Normal-only data, score every block, and persist results.

        Returns
        -------
        dict
            Summary of the run.
        """
        start_time = time.time()
        features_df = self._load_features()

        train_info = self.train(features_df)
        predictions_df = self.score(features_df)

        self.predictions_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            predictions_df.to_csv(self.predictions_file, index=False)
        except OSError as exc:
            raise ModelInferenceError(f"Failed to write predictions: {exc}") from exc

        elapsed = time.time() - start_time
        anomaly_count = int((predictions_df["iforest_prediction"] == LABEL_ANOMALY).sum())
        normal_count = int((predictions_df["iforest_prediction"] == LABEL_NORMAL).sum())

        summary = IsolationForestRunSummary(
            generated_at=pd.Timestamp.now(tz='UTC').isoformat(),
            features_file=str(self.features_file),
            model_file=str(self.model_file),
            predictions_file=str(self.predictions_file),
            total_blocks=len(features_df),
            labeled_normal_blocks=train_info["labeled_normal_blocks"],
            training_blocks=train_info["training_blocks"],
            holdout_normal_blocks=train_info["labeled_normal_blocks"] - train_info["training_blocks"],
            predicted_anomaly_count=anomaly_count,
            predicted_normal_count=normal_count,
            elapsed_seconds=round(elapsed, 2),
        )

        logger.info(
            "Isolation Forest scoring complete: %d blocks scored (%d Anomaly, "
            "%d Normal), %.2fs elapsed",
            summary.total_blocks,
            anomaly_count,
            normal_count,
            elapsed,
        )

        return asdict(summary)


def main() -> Dict:
    """Entry point for running Isolation Forest training + scoring standalone."""
    detector = IsolationForestDetector()
    return detector.run()


if __name__ == "__main__":
    result = main()
    logger.info("Isolation Forest summary: %s", json.dumps(result, indent=2))
