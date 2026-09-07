"""
Module 3, Part A: Feature Extraction.

Aggregates the row-level output of Module 2 (``data/processed/parsed_logs.csv``,
one row per raw log line) into one feature vector per HDFS Block ID, suitable
for unsupervised anomaly detection.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import MinMaxScaler

from utils.config import (
    EVENT_DENSITY_EPSILON,
    FEATURE_EXTRACTION_CHUNK_SIZE,
    FEATURE_EXTRACTION_REPORT_FILE,
    FEATURE_SCALER_FILE,
    FEATURES_FILE,
    KNOWN_LOG_LEVELS,
    PARSED_LOGS_FILE,
    PARSED_LOGS_USECOLS,
)
from utils.exceptions import FeatureExtractionError
from utils.logger import get_logger

logger = get_logger(__name__)

# Non-feature identifier column kept alongside the numeric feature matrix.
ID_COLUMN = "BlockId"

# Fixed, non-template feature columns produced by this module (template
# frequency columns are appended dynamically based on the EventIds observed
# in the data).
BASE_FEATURE_COLUMNS = [
    "total_log_events",
    "unique_template_count",
    "event_sequence_length",
    "time_duration_seconds",
    "unique_component_count",
    "event_density",
] + [f"{level.lower()}_count" for level in KNOWN_LOG_LEVELS] + ["other_level_count"]


@dataclass
class FeatureExtractionReport:
    """Summary statistics produced after a full feature extraction run."""

    generated_at: str
    input_file: str
    output_file: str
    scaler_file: str
    total_rows_read: int
    total_blocks: int
    total_feature_columns: int
    template_feature_columns: int
    blocks_with_zero_duration: int
    missing_value_cells_filled: int
    elapsed_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)


class FeatureExtractor:
    """
    Builds a per-block numerical feature matrix from ``parsed_logs.csv``.

    Parameters
    ----------
    input_file : Path
        Parsed log CSV produced by Module 2.
    output_file : Path
        Destination CSV for the final (scaled) feature matrix.
    chunk_size : int
        Number of parsed-log rows read per chunk while streaming the input.
    """

    def __init__(
        self,
        input_file: Path = PARSED_LOGS_FILE,
        output_file: Path = FEATURES_FILE,
        chunk_size: int = FEATURE_EXTRACTION_CHUNK_SIZE,
    ) -> None:
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.chunk_size = chunk_size
        self.warnings: List[str] = []

    def _read_parsed_logs(self) -> pd.DataFrame:
        """
        Stream ``parsed_logs.csv`` in chunks, restricted to the columns
        needed for feature engineering, and concatenate into a single
        DataFrame with light dtypes.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        FeatureExtractionError
            If the input file is missing or malformed.
        """
        if not self.input_file.exists():
            raise FeatureExtractionError(f"Parsed logs file not found: {self.input_file}")

        dtype_map = {
            "BlockId": "string",
            "Timestamp": "string",
            "Level": "category",
            "Component": "category",
            "EventId": "category",
        }

        chunks: List[pd.DataFrame] = []
        try:
            reader = pd.read_csv(
                self.input_file,
                usecols=list(PARSED_LOGS_USECOLS),
                dtype=dtype_map,
                chunksize=self.chunk_size,
            )
            for chunk in reader:
                chunks.append(chunk)
        except (OSError, pd.errors.ParserError, ValueError) as exc:
            raise FeatureExtractionError(f"Failed to read {self.input_file}: {exc}") from exc

        if not chunks:
            raise FeatureExtractionError(f"No data found in {self.input_file}")

        logs_df = pd.concat(chunks, ignore_index=True)

        missing_block_id = logs_df["BlockId"].isna() | (logs_df["BlockId"] == "")
        if missing_block_id.any():
            dropped = int(missing_block_id.sum())
            self.warnings.append(
                f"Dropped {dropped} row(s) with no BlockId (could not be "
                "associated with any block)."
            )
            logs_df = logs_df.loc[~missing_block_id].copy()

        return logs_df

    @staticmethod
    def _parse_timestamps(logs_df: pd.DataFrame) -> pd.Series:
        """
        Convert the ``YYMMDDHHMMSS`` ``Timestamp`` string column into
        proper ``datetime64`` values, coercing unparsable values to NaT.
        """
        return pd.to_datetime(
            logs_df["Timestamp"], format="%y%m%d%H%M%S", errors="coerce"
        )

    def _compute_base_aggregates(self, logs_df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-block count/duration/diversity features."""
        logs_df = logs_df.copy()
        logs_df["_ts"] = self._parse_timestamps(logs_df)

        bad_timestamps = int(logs_df["_ts"].isna().sum())
        if bad_timestamps:
            self.warnings.append(
                f"{bad_timestamps} row(s) had an unparsable Timestamp and were "
                "excluded from duration calculations only."
            )

        grouped = logs_df.groupby(ID_COLUMN, observed=True)

        total_log_events = grouped.size().rename("total_log_events")
        unique_template_count = grouped["EventId"].nunique().rename("unique_template_count")
        unique_component_count = grouped["Component"].nunique().rename("unique_component_count")

        duration = (
            grouped["_ts"].max() - grouped["_ts"].min()
        ).dt.total_seconds().rename("time_duration_seconds")

        base = pd.concat(
            [total_log_events, unique_template_count, unique_component_count, duration],
            axis=1,
        )
        base["event_sequence_length"] = base["total_log_events"]
        base["time_duration_seconds"] = base["time_duration_seconds"].fillna(0.0).clip(lower=0.0)
        base["event_density"] = base["total_log_events"] / (
            base["time_duration_seconds"] + EVENT_DENSITY_EPSILON
        )

        return base

    @staticmethod
    def _compute_level_counts(logs_df: pd.DataFrame) -> pd.DataFrame:
        """Pivot log Level into fixed INFO/WARN/ERROR/OTHER count columns."""
        level_str = logs_df["Level"].astype(str).str.upper()
        normalized = level_str.where(level_str.isin(KNOWN_LOG_LEVELS), other="OTHER")

        pivot = pd.crosstab(logs_df[ID_COLUMN], normalized)
        for level in list(KNOWN_LOG_LEVELS) + ["OTHER"]:
            if level not in pivot.columns:
                pivot[level] = 0

        pivot = pivot[list(KNOWN_LOG_LEVELS) + ["OTHER"]]
        pivot.columns = [f"{c.lower()}_count" for c in pivot.columns]
        return pivot

    @staticmethod
    def _compute_template_frequencies(logs_df: pd.DataFrame) -> pd.DataFrame:
        """Pivot EventId into one occurrence-count column per template."""
        pivot = pd.crosstab(logs_df[ID_COLUMN], logs_df["EventId"])
        pivot.columns = [f"tmpl_{col}_count" for col in pivot.columns]
        return pivot

    def _assemble_feature_matrix(self, logs_df: pd.DataFrame) -> pd.DataFrame:
        """Join all per-block feature groups into a single wide matrix."""
        base_features = self._compute_base_aggregates(logs_df)
        level_features = self._compute_level_counts(logs_df)
        template_features = self._compute_template_frequencies(logs_df)

        features = base_features.join(level_features, how="left").join(
            template_features, how="left"
        )
        features.index.name = ID_COLUMN
        features = features.reset_index()

        return features

    @staticmethod
    def _handle_missing_values(features: pd.DataFrame) -> int:
        """
        Fill missing values in numeric feature columns with 0 in place.

        Returns
        -------
        int
            Number of cells that were filled.
        """
        numeric_cols = features.columns.drop(ID_COLUMN)
        missing_before = int(features[numeric_cols].isna().sum().sum())
        features[numeric_cols] = features[numeric_cols].fillna(0)
        return missing_before

    def _scale_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fit a MinMaxScaler on all numeric feature columns and persist it."""
        numeric_cols = [c for c in features.columns if c != ID_COLUMN]
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(features[numeric_cols].astype(np.float64))

        scaled = features[[ID_COLUMN]].copy()
        scaled[numeric_cols] = scaled_values

        FEATURE_SCALER_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            dump({"scaler": scaler, "columns": numeric_cols}, FEATURE_SCALER_FILE)
        except OSError as exc:
            raise FeatureExtractionError(f"Failed to persist feature scaler: {exc}") from exc

        logger.info("Feature scaler persisted to %s", FEATURE_SCALER_FILE)
        return scaled

    def extract(self) -> Dict:
        """
        Run the full Module 3, Part A pipeline: read, aggregate, handle
        missing values, scale, and persist the feature matrix.

        Returns
        -------
        dict
            Summary of the extraction run.
        """
        start_time = time.time()
        logs_df = self._read_parsed_logs()
        total_rows_read = len(logs_df)

        features = self._assemble_feature_matrix(logs_df)
        missing_filled = self._handle_missing_values(features)
        zero_duration_blocks = int((features["time_duration_seconds"] == 0).sum())

        scaled_features = self._scale_features(features)

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            scaled_features.to_csv(self.output_file, index=False)
        except OSError as exc:
            raise FeatureExtractionError(f"Failed to write {self.output_file}: {exc}") from exc

        elapsed = time.time() - start_time
        template_cols = [c for c in features.columns if c.startswith("tmpl_")]

        report = FeatureExtractionReport(
            generated_at=pd.Timestamp.now(tz='UTC').isoformat(),
            input_file=str(self.input_file),
            output_file=str(self.output_file),
            scaler_file=str(FEATURE_SCALER_FILE),
            total_rows_read=total_rows_read,
            total_blocks=len(features),
            total_feature_columns=len(features.columns) - 1,
            template_feature_columns=len(template_cols),
            blocks_with_zero_duration=zero_duration_blocks,
            missing_value_cells_filled=missing_filled,
            elapsed_seconds=round(elapsed, 2),
            warnings=self.warnings,
        )
        self._write_report(report)

        logger.info(
            "Feature extraction complete: %d blocks, %d feature columns "
            "(%d template columns), %.2fs elapsed",
            report.total_blocks,
            report.total_feature_columns,
            report.template_feature_columns,
            elapsed,
        )

        return {
            "output_file": str(self.output_file),
            "scaler_file": str(FEATURE_SCALER_FILE),
            "report_file": str(FEATURE_EXTRACTION_REPORT_FILE),
            "total_blocks": report.total_blocks,
            "total_feature_columns": report.total_feature_columns,
        }

    @staticmethod
    def _write_report(report: FeatureExtractionReport) -> None:
        """Persist the feature extraction report as JSON."""
        try:
            with open(FEATURE_EXTRACTION_REPORT_FILE, mode="w", encoding="utf-8") as handle:
                json.dump(asdict(report), handle, indent=2)
        except OSError as exc:
            raise FeatureExtractionError(f"Failed to write feature extraction report: {exc}") from exc
        logger.info("Feature extraction report written to %s", FEATURE_EXTRACTION_REPORT_FILE)


def main() -> Dict:
    """Entry point for running feature extraction as a standalone step."""
    extractor = FeatureExtractor()
    return extractor.extract()


if __name__ == "__main__":
    result = main()
    logger.info("Feature extraction summary: %s", json.dumps(result, indent=2))
