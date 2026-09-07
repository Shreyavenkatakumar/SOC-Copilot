"""
Module 3 orchestrator: Feature Extraction + Isolation Forest Detection +
Evaluation.

Runs Parts A, B, and C of Module 3 end to end:

  1. ``features.feature_extractor``            — parsed_logs.csv -> features.csv
  2. ``models.isolation_forest_model``         — train + score Isolation Forest
  3. ``evaluation.evaluate_isolation_forest``  — metrics vs. ground truth
"""

import json
from typing import Dict

from evaluation.evaluate_isolation_forest import IsolationForestEvaluator
from features.feature_extractor import FeatureExtractor
from models.isolation_forest_model import IsolationForestDetector
from utils.exceptions import SOCCopilotError
from utils.logger import get_logger

logger = get_logger(__name__)


def run_feature_extraction() -> Dict:
    """Execute Module 3, Part A: Feature Extraction."""
    logger.info("--- Module 3a: Feature Extraction — START ---")
    summary = FeatureExtractor().extract()
    logger.info("--- Module 3a: Feature Extraction — COMPLETE ---")
    return summary


def run_isolation_forest() -> Dict:
    """Execute Module 3, Part B: Isolation Forest training + scoring."""
    logger.info("--- Module 3b: Isolation Forest — START ---")
    summary = IsolationForestDetector().run()
    logger.info("--- Module 3b: Isolation Forest — COMPLETE ---")
    return summary


def run_evaluation() -> Dict:
    """Execute Module 3, Part C: Evaluation against ground truth."""
    logger.info("--- Module 3c: Evaluation — START ---")
    summary = IsolationForestEvaluator().evaluate()
    logger.info("--- Module 3c: Evaluation — COMPLETE ---")
    return summary


def main() -> Dict:
    """Run all three parts of Module 3 in sequence and return a combined summary."""
    feature_summary = run_feature_extraction()
    iforest_summary = run_isolation_forest()
    evaluation_summary = run_evaluation()

    return {
        "feature_extraction": feature_summary,
        "isolation_forest": iforest_summary,
        "evaluation": evaluation_summary,
    }


if __name__ == "__main__":
    try:
        result = main()
    except SOCCopilotError as exc:
        logger.error("Module 3 pipeline halted: %s", exc)
        raise SystemExit(1) from exc

    logger.info("Module 3 summary: %s", json.dumps(result, indent=2))
