"""
Module 4, Part A: Autoencoder Anomaly Detection.

Trains a dense (fully-connected) autoencoder on ground-truth *Normal*
blocks only, then scores every block by its reconstruction error --
blocks the network cannot reconstruct well are flagged as anomalous.

Consistency with the Isolation Forest (Module 3b)
---------------------------------------------------
This module never imports or modifies ``models/isolation_forest_model.py``.
Instead it independently reproduces the identical Normal-only train/test
split -- same ``FEATURES_FILE`` row order, same left-join against
``anomaly_label.csv``, same ``RANDOM_STATE``/``TRAIN_TEST_SPLIT_RATIO`` --
so the Autoencoder is trained and evaluated on exactly the same blocks
as the Isolation Forest. This keeps the two detectors' metrics directly
comparable and keeps both modules self-contained, matching the same
approach already used in ``evaluation/evaluate_isolation_forest.py`` to
avoid data leakage.

Threshold selection
--------------------
The Normal/Anomaly cutoff is the ``AUTOENCODER_CONTAMINATION`` percentile
of the reconstruction error computed on the TRAINING (Normal-only) data
only -- never on holdout or anomalous blocks -- so no evaluation leakage
is introduced by the threshold itself.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import dump
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from data.log_collector import LogCollector
from utils.config import (
    AUTOENCODER_CONTAMINATION,
    AUTOENCODER_MODEL_FILE,
    AUTOENCODER_PARAMS,
    AUTOENCODER_PREDICTIONS_FILE,
    AUTOENCODER_TRAINING_HISTORY_FILE,
    FEATURES_FILE,
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
class AutoencoderRunSummary:
    """Summary statistics produced after a full train + score run."""

    generated_at: str
    features_file: str
    model_file: str
    predictions_file: str
    total_blocks: int
    labeled_normal_blocks: int
    training_blocks: int
    holdout_normal_blocks: int
    input_dim: int
    encoding_dims: List[int]
    epochs_run: int
    final_train_loss: float
    final_val_loss: float
    reconstruction_error_threshold: float
    predicted_anomaly_count: int
    predicted_normal_count: int
    elapsed_seconds: float = 0.0


class AutoencoderDetector:
    """
    Trains and applies a dense autoencoder anomaly detector over
    per-block HDFS features.

    Parameters
    ----------
    features_file : Path
        Feature matrix produced by ``FeatureExtractor`` (Module 3a).
    model_file : Path
        Destination path for the persisted Keras model.
    predictions_file : Path
        Destination CSV for per-block reconstruction errors and
        predictions.
    history_file : Path
        Destination JSON for per-epoch training/validation loss.
    train_split_ratio : float
        Fraction of ground-truth Normal blocks used for training; must
        match the Isolation Forest's ratio for a directly comparable
        holdout split.
    contamination : float
        Assumed fraction of anomalous blocks, used to pick the
        reconstruction-error percentile that becomes the Anomaly cutoff.
    """

    def __init__(
        self,
        features_file: Path = FEATURES_FILE,
        model_file: Path = AUTOENCODER_MODEL_FILE,
        predictions_file: Path = AUTOENCODER_PREDICTIONS_FILE,
        history_file: Path = AUTOENCODER_TRAINING_HISTORY_FILE,
        train_split_ratio: float = TRAIN_TEST_SPLIT_RATIO,
        contamination: float = AUTOENCODER_CONTAMINATION,
    ) -> None:
        self.features_file = Path(features_file)
        self.model_file = Path(model_file)
        self.predictions_file = Path(predictions_file)
        self.history_file = Path(history_file)
        self.train_split_ratio = train_split_ratio
        self.contamination = contamination

        self.model: keras.Model = None
        self.feature_columns: List[str] = []
        self.reconstruction_error_threshold: float = 0.0

        tf.random.set_seed(RANDOM_STATE)

    def _load_features(self) -> pd.DataFrame:
        """Load the per-block feature matrix produced by Module 3a."""
        if not self.features_file.exists():
            raise ModelTrainingError(f"Features file not found: {self.features_file}")
        try:
            return pd.read_csv(self.features_file)
        except (OSError, pd.errors.ParserError) as exc:
            raise ModelTrainingError(f"Failed to read {self.features_file}: {exc}") from exc

    @staticmethod
    def _feature_columns(features_df: pd.DataFrame) -> List[str]:
        """Return all feature columns (everything except the block identifier)."""
        return [c for c in features_df.columns if c != ID_COLUMN]

    def _select_training_blocks(
        self, features_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, int]:
        """
        Identify blocks labeled ``Normal`` in the ground truth and split
        off a training subset, using the identical procedure as
        ``IsolationForestDetector._select_training_blocks``
        (models/isolation_forest_model.py) so both detectors share the
        same train/holdout split without this module importing that one.

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
                "with anomaly_label.csv; cannot train the Autoencoder."
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

    @staticmethod
    def _build_model(input_dim: int) -> keras.Model:
        """Construct a symmetric dense autoencoder from AUTOENCODER_PARAMS."""
        encoding_dims = AUTOENCODER_PARAMS["encoding_dims"]
        activation = AUTOENCODER_PARAMS["activation"]
        output_activation = AUTOENCODER_PARAMS["output_activation"]
        learning_rate = AUTOENCODER_PARAMS["learning_rate"]

        inputs = keras.Input(shape=(input_dim,), name="feature_input")

        x = inputs
        for units in encoding_dims:
            x = layers.Dense(units, activation=activation)(x)

        for units in reversed(encoding_dims[:-1]):
            x = layers.Dense(units, activation=activation)(x)

        outputs = layers.Dense(input_dim, activation=output_activation, name="reconstruction")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="hdfs_block_autoencoder")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
        )
        return model

    def train(self, features_df: pd.DataFrame) -> Dict:
        """
        Fit the autoencoder on the Normal-only training subset and
        derive the reconstruction-error threshold from training data.

        Returns
        -------
        dict
            Training-run sizing/loss information, used later for reporting.
        """
        train_blocks, labeled_normal_count = self._select_training_blocks(features_df)
        self.feature_columns = self._feature_columns(features_df)

        x_train = train_blocks[self.feature_columns].to_numpy(dtype=np.float32)

        self.model = self._build_model(input_dim=x_train.shape[1])

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=AUTOENCODER_PARAMS["early_stopping_patience"],
            restore_best_weights=True,
        )

        try:
            history = self.model.fit(
                x_train,
                x_train,
                batch_size=AUTOENCODER_PARAMS["batch_size"],
                epochs=AUTOENCODER_PARAMS["epochs"],
                validation_split=AUTOENCODER_PARAMS["validation_split"],
                shuffle=True,
                callbacks=[early_stopping],
                verbose=2,
            )
        except (ValueError, tf.errors.OpError) as exc:
            raise ModelTrainingError(f"Autoencoder training failed: {exc}") from exc

        epochs_run = len(history.history["loss"])
        logger.info(
            "Autoencoder trained on %d Normal block(s) (out of %d labeled Normal), "
            "%d feature(s), %d epoch(s) run (early stopping patience=%d)",
            len(train_blocks),
            labeled_normal_count,
            len(self.feature_columns),
            epochs_run,
            AUTOENCODER_PARAMS["early_stopping_patience"],
        )

        train_reconstructions = self.model.predict(x_train, verbose=0)
        train_errors = np.mean(np.square(x_train - train_reconstructions), axis=1)
        percentile = 100.0 * (1.0 - self.contamination)
        self.reconstruction_error_threshold = float(np.percentile(train_errors, percentile))

        logger.info(
            "Reconstruction-error threshold set to %.8f (%.1fth percentile of "
            "training error, contamination=%.4f)",
            self.reconstruction_error_threshold,
            percentile,
            self.contamination,
        )

        self._persist(history)

        return {
            "training_blocks": len(train_blocks),
            "labeled_normal_blocks": labeled_normal_count,
            "input_dim": x_train.shape[1],
            "epochs_run": epochs_run,
            "final_train_loss": float(history.history["loss"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]),
        }

    def _persist(self, history: keras.callbacks.History) -> None:
        """Persist the trained model, its metadata, and training history."""
        self.model_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.model.save(self.model_file)
        except OSError as exc:
            raise ModelTrainingError(f"Failed to persist Autoencoder model: {exc}") from exc
        logger.info("Autoencoder model persisted to %s", self.model_file)

        metadata_file = self.model_file.with_suffix(".meta.joblib")
        try:
            dump(
                {
                    "feature_columns": self.feature_columns,
                    "reconstruction_error_threshold": self.reconstruction_error_threshold,
                },
                metadata_file,
            )
        except OSError as exc:
            raise ModelTrainingError(f"Failed to persist Autoencoder metadata: {exc}") from exc
        logger.info("Autoencoder metadata persisted to %s", metadata_file)

        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.history_file, mode="w", encoding="utf-8") as handle:
                json.dump(history.history, handle, indent=2)
        except OSError as exc:
            raise ModelTrainingError(f"Failed to persist training history: {exc}") from exc
        logger.info("Training history written to %s", self.history_file)

    def score(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score every block in ``features_df`` with the trained autoencoder.

        Returns
        -------
        pandas.DataFrame
            Columns: ``BlockId``, ``autoencoder_reconstruction_error``
            (raw MSE), ``autoencoder_anomaly_score`` (0-1, higher = more
            anomalous), ``autoencoder_prediction`` ("Normal"/"Anomaly").
        """
        if self.model is None:
            raise ModelInferenceError("Autoencoder model has not been trained yet.")

        x = features_df[self.feature_columns].to_numpy(dtype=np.float32)

        try:
            reconstructions = self.model.predict(x, verbose=0)
        except (ValueError, tf.errors.OpError) as exc:
            raise ModelInferenceError(f"Autoencoder scoring failed: {exc}") from exc

        reconstruction_error = np.mean(np.square(x - reconstructions), axis=1)

        min_err, max_err = reconstruction_error.min(), reconstruction_error.max()
        if max_err > min_err:
            normalized_score = (reconstruction_error - min_err) / (max_err - min_err)
        else:
            normalized_score = np.zeros_like(reconstruction_error)

        predicted_label = np.where(
            reconstruction_error > self.reconstruction_error_threshold,
            LABEL_ANOMALY,
            LABEL_NORMAL,
        )

        return pd.DataFrame(
            {
                ID_COLUMN: features_df[ID_COLUMN].values,
                "autoencoder_reconstruction_error": reconstruction_error,
                "autoencoder_anomaly_score": normalized_score,
                "autoencoder_prediction": predicted_label,
            }
        )

    def run(self) -> Dict:
        """
        Execute the full Module 4, Part A pipeline: load features, train
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
        anomaly_count = int((predictions_df["autoencoder_prediction"] == LABEL_ANOMALY).sum())
        normal_count = int((predictions_df["autoencoder_prediction"] == LABEL_NORMAL).sum())

        summary = AutoencoderRunSummary(
            generated_at=pd.Timestamp.now(tz="UTC").isoformat(),
            features_file=str(self.features_file),
            model_file=str(self.model_file),
            predictions_file=str(self.predictions_file),
            total_blocks=len(features_df),
            labeled_normal_blocks=train_info["labeled_normal_blocks"],
            training_blocks=train_info["training_blocks"],
            holdout_normal_blocks=train_info["labeled_normal_blocks"] - train_info["training_blocks"],
            input_dim=train_info["input_dim"],
            encoding_dims=list(AUTOENCODER_PARAMS["encoding_dims"]),
            epochs_run=train_info["epochs_run"],
            final_train_loss=round(train_info["final_train_loss"], 8),
            final_val_loss=round(train_info["final_val_loss"], 8),
            reconstruction_error_threshold=round(self.reconstruction_error_threshold, 8),
            predicted_anomaly_count=anomaly_count,
            predicted_normal_count=normal_count,
            elapsed_seconds=round(elapsed, 2),
        )

        logger.info(
            "Autoencoder scoring complete: %d blocks scored (%d Anomaly, %d Normal), "
            "%.2fs elapsed",
            summary.total_blocks,
            anomaly_count,
            normal_count,
            elapsed,
        )

        return asdict(summary)


def main() -> Dict:
    """Entry point for running Autoencoder training + scoring standalone."""
    detector = AutoencoderDetector()
    return detector.run()


if __name__ == "__main__":
    result = main()
    logger.info("Autoencoder summary: %s", json.dumps(result, indent=2))
