"""
Module 2: Drain3 Log Parsing.

Streams ``data/raw/HDFS.log`` through the ``LogCollector`` from Module 1,
extracts the structural header of every line, mines a stable log
template for the free-text content using Drain3, and persists:

  * ``data/processed/parsed_logs.csv``      — one row per raw log line
  * ``outputs/models/drain3_state.bin``     — the trained Drain3 tree
  * ``outputs/reports/parsing_report.json`` — parsing statistics
"""

import csv
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

from data.log_collector import LogCollector
from parsers.log_header_parser import extract_header
from utils.config import (
    DRAIN3_CONFIG_FILE,
    DRAIN3_STATE_FILE,
    HDFS_LOG_FILE,
    PARSED_LOGS_FILE,
    PARSING_REPORT_FILE,
    PARSING_WRITE_BATCH_SIZE,
    TOP_N_TEMPLATES_IN_REPORT,
)
from utils.exceptions import LogParsingError
from utils.logger import get_logger

logger = get_logger(__name__)

CSV_COLUMNS = [
    "LineId",
    "Date",
    "Time",
    "Timestamp",
    "Pid",
    "Level",
    "Component",
    "Content",
    "BlockId",
    "EventId",
    "EventTemplate",
    "ParameterList",
    "RawLine",
]


@dataclass
class TemplateStats:
    """Aggregate information about a single mined template/cluster."""

    event_id: str
    template: str
    occurrence_count: int


@dataclass
class ParsingReport:
    """Summary statistics produced after a full parsing run."""

    generated_at: str
    input_file: str
    output_file: str
    drain3_state_file: str
    total_lines_read: int
    total_lines_parsed: int
    total_lines_failed: int
    parsing_success_rate: float
    unique_template_count: int
    most_frequent_templates: List[TemplateStats] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)


class Drain3LogParser:
    """
    Parses raw HDFS log lines into structured log templates using Drain3.

    Parameters
    ----------
    input_file : Path
        Raw log file to parse. Defaults to ``data/raw/HDFS.log``.
    output_file : Path
        Destination CSV for parsed rows.
    config_file : Path
        Drain3 ``.ini`` configuration file.
    state_file : Path
        Path where the trained Drain3 tree snapshot is persisted.
    batch_size : int
        Number of parsed rows buffered in memory before a CSV flush.
    """

    def __init__(
        self,
        input_file: Path = HDFS_LOG_FILE,
        output_file: Path = PARSED_LOGS_FILE,
        config_file: Path = DRAIN3_CONFIG_FILE,
        state_file: Path = DRAIN3_STATE_FILE,
        batch_size: int = PARSING_WRITE_BATCH_SIZE,
    ) -> None:
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.config_file = Path(config_file)
        self.state_file = Path(state_file)
        self.batch_size = batch_size

        self.collector = LogCollector()
        self.template_miner = self._build_template_miner()

        self.template_occurrences: Counter = Counter()
        self.total_lines_read = 0
        self.total_lines_parsed = 0
        self.total_lines_failed = 0
        self.warnings: List[str] = []

    def _build_template_miner(self) -> TemplateMiner:
        """Construct a Drain3 ``TemplateMiner`` with file-based persistence."""
        if not self.config_file.exists():
            raise LogParsingError(f"Drain3 config file not found: {self.config_file}")

        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        config = TemplateMinerConfig()
        config.load(str(self.config_file))
        # drain3's TemplateMinerConfig.load() reads this value via
        # ConfigParser.get() instead of getint(), leaving it as a str.
        # Cast explicitly to avoid a TypeError inside cachetools' LRUCache.
        config.parameter_extraction_cache_capacity = int(
            config.parameter_extraction_cache_capacity
        )

        persistence_handler = FilePersistence(str(self.state_file))
        miner = TemplateMiner(persistence_handler=persistence_handler, config=config)
        logger.info(
            "Drain3 TemplateMiner initialized (config=%s, state=%s)",
            self.config_file,
            self.state_file,
        )
        return miner

    def _parse_line(self, raw_line: str, line_id: int) -> Optional[Dict]:
        """
        Parse a single raw log line into a structured row dict.

        Returns
        -------
        Optional[Dict]
            A row ready to be written to CSV, or ``None`` if the line
            could not be parsed (header mismatch).
        """
        try:
            header = extract_header(raw_line)
        except LogParsingError as exc:
            self.total_lines_failed += 1
            if self.total_lines_failed <= 20:
                logger.warning("Failed to parse line %d: %s", line_id, exc)
            return None

        result = self.template_miner.add_log_message(header.content)
        cluster_id = result["cluster_id"]
        event_id = f"E{cluster_id}"
        template = result["template_mined"]

        extracted_params = self.template_miner.extract_parameters(
            template, header.content, exact_matching=True
        )
        parameter_values = (
            [p.value for p in extracted_params] if extracted_params else []
        )

        self.template_occurrences[event_id] += 1
        if self.template_occurrences[event_id] == 1:
            self._register_template(event_id, template)

        return {
            "LineId": line_id,
            "Date": header.date,
            "Time": header.time,
            "Timestamp": header.timestamp,
            "Pid": header.pid,
            "Level": header.level,
            "Component": header.component,
            "Content": header.content,
            "BlockId": header.block_id or "",
            "EventId": event_id,
            "EventTemplate": template,
            "ParameterList": json.dumps(parameter_values),
            "RawLine": raw_line,
        }

    def _register_template(self, event_id: str, template: str) -> None:
        """Track the canonical template text for an EventId as it evolves."""
        self._event_templates = getattr(self, "_event_templates", {})
        self._event_templates[event_id] = template

    def parse(self) -> Dict:
        """
        Execute the full Module 2 parsing run.

        Returns
        -------
        dict
            Summary of the parsing run including output paths and key
            statistics.

        Raises
        ------
        LogParsingError
            If the input file is missing or the CSV cannot be written.
        """
        if not self.input_file.exists():
            raise LogParsingError(f"Input log file not found: {self.input_file}")

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self._event_templates: Dict[str, str] = {}

        start_time = time.time()
        line_id = 0

        try:
            with open(self.output_file, mode="w", newline="", encoding="utf-8") as out_handle:
                writer = csv.DictWriter(out_handle, fieldnames=CSV_COLUMNS)
                writer.writeheader()

                buffer: List[Dict] = []
                for chunk in self.collector.stream_log_lines(self.input_file):
                    for raw_line in chunk:
                        if not raw_line.strip():
                            continue
                        line_id += 1
                        self.total_lines_read += 1

                        row = self._parse_line(raw_line, line_id)
                        if row is not None:
                            self.total_lines_parsed += 1
                            buffer.append(row)

                        if len(buffer) >= self.batch_size:
                            writer.writerows(buffer)
                            buffer.clear()

                if buffer:
                    writer.writerows(buffer)
                    buffer.clear()
        except OSError as exc:
            raise LogParsingError(f"Failed to write parsed output: {exc}") from exc

        # Force a final snapshot so the learned tree is reusable without
        # re-training on the next run.
        self.template_miner.save_state("final")

        elapsed = time.time() - start_time
        report = self._build_report(elapsed)
        self._write_report(report)

        logger.info(
            "Parsing complete: %d/%d lines parsed (%.2f%% success), "
            "%d unique templates, %.2fs elapsed",
            self.total_lines_parsed,
            self.total_lines_read,
            report.parsing_success_rate * 100,
            report.unique_template_count,
            elapsed,
        )

        return {
            "output_file": str(self.output_file),
            "report_file": str(PARSING_REPORT_FILE),
            "state_file": str(self.state_file),
            "total_lines_read": self.total_lines_read,
            "total_lines_parsed": self.total_lines_parsed,
            "total_lines_failed": self.total_lines_failed,
            "parsing_success_rate": report.parsing_success_rate,
            "unique_template_count": report.unique_template_count,
        }

    def _build_report(self, elapsed_seconds: float) -> ParsingReport:
        """Assemble the ``ParsingReport`` from accumulated run statistics."""
        success_rate = (
            self.total_lines_parsed / self.total_lines_read
            if self.total_lines_read > 0
            else 0.0
        )

        most_frequent = [
            TemplateStats(
                event_id=event_id,
                template=self._event_templates.get(event_id, ""),
                occurrence_count=count,
            )
            for event_id, count in self.template_occurrences.most_common(
                TOP_N_TEMPLATES_IN_REPORT
            )
        ]

        if self.total_lines_failed:
            self.warnings.append(
                f"{self.total_lines_failed} line(s) failed header extraction "
                "and were excluded from parsed_logs.csv."
            )

        return ParsingReport(
            generated_at=pd.Timestamp.now(tz='UTC').isoformat(),
            input_file=str(self.input_file),
            output_file=str(self.output_file),
            drain3_state_file=str(self.state_file),
            total_lines_read=self.total_lines_read,
            total_lines_parsed=self.total_lines_parsed,
            total_lines_failed=self.total_lines_failed,
            parsing_success_rate=round(success_rate, 6),
            unique_template_count=len(self.template_occurrences),
            most_frequent_templates=most_frequent,
            elapsed_seconds=round(elapsed_seconds, 2),
            warnings=self.warnings,
        )

    @staticmethod
    def _write_report(report: ParsingReport) -> None:
        """Persist the parsing report as JSON."""
        try:
            with open(PARSING_REPORT_FILE, mode="w", encoding="utf-8") as handle:
                json.dump(asdict(report), handle, indent=2)
        except OSError as exc:
            raise LogParsingError(f"Failed to write parsing report: {exc}") from exc
        logger.info("Parsing report written to %s", PARSING_REPORT_FILE)


def main() -> Dict:
    """Entry point for running Drain3 log parsing as a standalone step."""
    parser = Drain3LogParser()
    return parser.parse()


if __name__ == "__main__":
    result = main()
    logger.info("Drain3 parsing summary: %s", json.dumps(result, indent=2))
