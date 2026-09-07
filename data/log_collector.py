"""
Module 1: Log Collection.

Responsible for discovering, validating, and streaming raw security log
files (HDFS.log and companion CSV files) from ``data/raw`` prior to
Drain3 parsing (Module 2).

This module never mutates raw source files. It only reads them, computes
quality statistics, and persists a JSON collection report under
``outputs/reports``.
"""

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd

from utils.config import (
    ALLOWED_LOG_EXTENSIONS,
    ANOMALY_LABEL_FILE,
    HDFS_LOG_FILE,
    LOG_ENCODING,
    LOG_ENCODING_FALLBACK,
    LOG_READ_CHUNK_SIZE,
    RAW_DATA_DIR,
    REPORTS_DIR,
    REQUIRED_RAW_FILES,
)
from utils.exceptions import LogCollectionError
from utils.logger import get_logger

logger = get_logger(__name__)

# Typical HDFS log line: "081109 203518 143 INFO dfs.DataNode$DataXceiver: ..."
HDFS_LINE_PATTERN = re.compile(
    r"^\d{6}\s+\d{6}\s+\d+\s+[A-Z]+\s+.+$"
)

BLOCK_ID_PATTERN = re.compile(r"(blk_-?\d+)")


@dataclass
class FileStats:
    """Quality and size statistics collected for a single raw log file."""

    file_name: str
    file_path: str
    extension: str
    size_bytes: int
    line_count: int = 0
    empty_line_count: int = 0
    malformed_line_count: Optional[int] = None
    sample_lines: List[str] = field(default_factory=list)


@dataclass
class CollectionReport:
    """Aggregate report produced after a full collection run."""

    generated_at: str
    raw_data_dir: str
    required_files_present: bool
    discovered_files: List[FileStats]
    total_lines_collected: int
    total_bytes_collected: int
    warnings: List[str] = field(default_factory=list)


class LogCollector:
    """
    Discovers and streams raw log files for the SOC Copilot pipeline.

    Parameters
    ----------
    raw_dir : Path
        Directory containing raw log/CSV input files. Defaults to
        ``data/raw`` as defined in ``utils.config``.
    chunk_size : int
        Number of lines read per chunk when streaming large files.
    sample_size : int
        Number of sample lines to retain per file for reporting/debugging.
    """

    def __init__(
        self,
        raw_dir: Path = RAW_DATA_DIR,
        chunk_size: int = LOG_READ_CHUNK_SIZE,
        sample_size: int = 5,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.chunk_size = chunk_size
        self.sample_size = sample_size

    def validate_environment(self) -> None:
        """
        Ensure the raw data directory and all required Phase I input
        files exist and are non-empty.

        Raises
        ------
        LogCollectionError
            If the raw directory or any required file is missing/empty.
        """
        if not self.raw_dir.exists():
            raise LogCollectionError(
                f"Raw data directory does not exist: {self.raw_dir}"
            )

        missing = [str(p) for p in REQUIRED_RAW_FILES if not p.exists()]
        if missing:
            raise LogCollectionError(
                "Missing required raw input file(s): " + ", ".join(missing)
            )

        empty = [str(p) for p in REQUIRED_RAW_FILES if p.stat().st_size == 0]
        if empty:
            raise LogCollectionError(
                "Required raw input file(s) are empty: " + ", ".join(empty)
            )

        logger.info("Environment validation passed. All required raw files present.")

    def discover_log_files(self) -> List[Path]:
        """
        Discover all candidate log input files in ``raw_dir`` matching
        ``ALLOWED_LOG_EXTENSIONS``.

        Returns
        -------
        List[Path]
            Sorted list of discovered file paths.
        """
        discovered = sorted(
            p
            for p in self.raw_dir.iterdir()
            if p.is_file() and p.suffix.lower() in ALLOWED_LOG_EXTENSIONS
        )
        logger.info("Discovered %d candidate log file(s) in %s", len(discovered), self.raw_dir)
        return discovered

    def _open_with_fallback(self, file_path: Path):
        """Open a text file trying the primary encoding, then a fallback."""
        try:
            handle = open(file_path, mode="r", encoding=LOG_ENCODING, errors="strict")
            handle.readline()
            handle.seek(0)
            return handle
        except (UnicodeDecodeError, UnicodeError):
            logger.warning(
                "Falling back to '%s' encoding for %s", LOG_ENCODING_FALLBACK, file_path
            )
            return open(file_path, mode="r", encoding=LOG_ENCODING_FALLBACK, errors="replace")

    def stream_log_lines(self, file_path: Path) -> Iterator[List[str]]:
        """
        Lazily stream a raw log file in fixed-size chunks of stripped
        lines. Designed to handle multi-gigabyte files (e.g. HDFS.log,
        11M+ lines) without loading the entire file into memory.

        Parameters
        ----------
        file_path : Path
            Path to the log file to stream.

        Yields
        ------
        List[str]
            A chunk of up to ``self.chunk_size`` raw (CRLF-stripped) lines.

        Raises
        ------
        LogCollectionError
            If the file cannot be opened or read.
        """
        if not file_path.exists():
            raise LogCollectionError(f"Log file not found: {file_path}")

        try:
            handle = self._open_with_fallback(file_path)
        except OSError as exc:
            raise LogCollectionError(f"Unable to open {file_path}: {exc}") from exc

        chunk: List[str] = []
        try:
            with handle:
                for raw_line in handle:
                    line = raw_line.rstrip("\r\n")
                    chunk.append(line)
                    if len(chunk) >= self.chunk_size:
                        yield chunk
                        chunk = []
                if chunk:
                    yield chunk
        except OSError as exc:
            raise LogCollectionError(f"Error while reading {file_path}: {exc}") from exc

    def compute_file_stats(
        self, file_path: Path, validate_hdfs_pattern: bool = False
    ) -> FileStats:
        """
        Compute size, line count, empty-line count, and optional format
        validation for a single log file.

        Parameters
        ----------
        file_path : Path
            File to analyze.
        validate_hdfs_pattern : bool
            If True, checks each line against the expected HDFS log line
            pattern and records how many lines do not conform.

        Returns
        -------
        FileStats
        """
        stats = FileStats(
            file_name=file_path.name,
            file_path=str(file_path),
            extension=file_path.suffix.lower(),
            size_bytes=file_path.stat().st_size,
        )

        malformed_count = 0
        start_time = time.time()

        for chunk in self.stream_log_lines(file_path):
            for line in chunk:
                stats.line_count += 1
                if not line.strip():
                    stats.empty_line_count += 1
                    continue
                if len(stats.sample_lines) < self.sample_size:
                    stats.sample_lines.append(line)
                if validate_hdfs_pattern and not HDFS_LINE_PATTERN.match(line):
                    malformed_count += 1

        if validate_hdfs_pattern:
            stats.malformed_line_count = malformed_count

        elapsed = time.time() - start_time
        logger.info(
            "Processed %s: %d lines, %d empty, %.2fs",
            file_path.name,
            stats.line_count,
            stats.empty_line_count,
            elapsed,
        )
        return stats

    def load_anomaly_labels(self) -> pd.DataFrame:
        """
        Load the ground-truth block-level anomaly labels.

        Returns
        -------
        pandas.DataFrame
            Columns: ``BlockId`` (str), ``Label`` (str, one of
            {"Normal", "Anomaly"}).

        Raises
        ------
        LogCollectionError
            If the file is missing or has an unexpected schema.
        """
        if not ANOMALY_LABEL_FILE.exists():
            raise LogCollectionError(f"Anomaly label file not found: {ANOMALY_LABEL_FILE}")

        try:
            labels_df = pd.read_csv(ANOMALY_LABEL_FILE, dtype=str)
        except (OSError, pd.errors.ParserError) as exc:
            raise LogCollectionError(f"Failed to read {ANOMALY_LABEL_FILE}: {exc}") from exc

        expected_columns = {"BlockId", "Label"}
        if not expected_columns.issubset(set(labels_df.columns)):
            raise LogCollectionError(
                f"Unexpected schema in {ANOMALY_LABEL_FILE}. "
                f"Expected columns {expected_columns}, found {set(labels_df.columns)}."
            )

        labels_df["Label"] = labels_df["Label"].str.strip()
        valid_labels = {"Normal", "Anomaly"}
        unexpected = set(labels_df["Label"].unique()) - valid_labels
        if unexpected:
            logger.warning("Unexpected label values found: %s", unexpected)

        logger.info(
            "Loaded %d anomaly labels (%d Normal, %d Anomaly)",
            len(labels_df),
            int((labels_df["Label"] == "Normal").sum()),
            int((labels_df["Label"] == "Anomaly").sum()),
        )
        return labels_df

    def collect(self) -> Dict:
        """
        Run the full log collection stage: validate the environment,
        discover input files, compute per-file statistics, load ground
        truth labels, and persist a JSON collection report.

        Returns
        -------
        dict
            Summary of the collection run, including the path to the
            persisted report.
        """
        self.validate_environment()
        discovered_files = self.discover_log_files()

        file_stats: List[FileStats] = []
        warnings: List[str] = []
        total_lines = 0
        total_bytes = 0

        for file_path in discovered_files:
            is_primary_log = file_path.resolve() == HDFS_LOG_FILE.resolve()
            try:
                stats = self.compute_file_stats(
                    file_path, validate_hdfs_pattern=is_primary_log
                )
            except LogCollectionError as exc:
                logger.error("Skipping %s due to error: %s", file_path, exc)
                warnings.append(f"Skipped {file_path.name}: {exc}")
                continue

            if is_primary_log and stats.malformed_line_count:
                ratio = stats.malformed_line_count / max(stats.line_count, 1)
                if ratio > 0.01:
                    warnings.append(
                        f"{file_path.name}: {ratio:.2%} of lines do not match the "
                        "expected HDFS log format."
                    )

            file_stats.append(stats)
            total_lines += stats.line_count
            total_bytes += stats.size_bytes

        try:
            self.load_anomaly_labels()
        except LogCollectionError as exc:
            warnings.append(str(exc))

        report = CollectionReport(
            generated_at=pd.Timestamp.now(tz='UTC').isoformat(),
            raw_data_dir=str(self.raw_dir),
            required_files_present=True,
            discovered_files=file_stats,
            total_lines_collected=total_lines,
            total_bytes_collected=total_bytes,
            warnings=warnings,
        )

        report_path = REPORTS_DIR / "log_collection_report.json"
        try:
            with open(report_path, mode="w", encoding="utf-8") as handle:
                json.dump(asdict(report), handle, indent=2)
        except OSError as exc:
            raise LogCollectionError(f"Failed to write collection report: {exc}") from exc

        logger.info("Collection report written to %s", report_path)

        return {
            "report_path": str(report_path),
            "total_files": len(file_stats),
            "total_lines_collected": total_lines,
            "total_bytes_collected": total_bytes,
            "warnings": warnings,
        }


def main() -> Dict:
    """Entry point for running log collection as a standalone step."""
    collector = LogCollector()
    return collector.collect()


if __name__ == "__main__":
    result = main()
    logger.info("Log collection summary: %s", json.dumps(result, indent=2))
