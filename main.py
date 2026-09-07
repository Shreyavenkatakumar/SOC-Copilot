"""
SOC Copilot Phase I — main orchestrator (PRODUCTION, approved architecture).

Pipeline:
  Module 1  Log Collection
  Module 2  Drain3 Log Parsing
  Module 3a Feature Extraction (58 features)
  Module 3b Isolation Forest v2 (19 features; the ONLY Isolation Forest
             that executes -- the original Isolation Forest,
             models/isolation_forest_model.py, is retained on disk for
             reference/rollback but is not imported or run here)
  Module 4  Autoencoder (58 features; architecture/training/threshold
             unchanged)
  Module 5  Fusion Engine -- fixed, approved configuration:
             0.10 * calibrated IF-v2 + 0.90 * calibrated AE,
             threshold = 0.323459 (validation-selected in a prior,
             separate experiment; not re-searched here)
  Module 5b Fusion Evaluation (leakage-free, untouched test slice)
"""

import json
import sys

from data.log_collector import LogCollector
from evaluation.evaluate_autoencoder import AutoencoderEvaluator
from evaluation.evaluate_fusion import FusionEvaluator
from features.feature_extractor import FeatureExtractor
from models.autoencoder_model import AutoencoderDetector
from models.fusion_engine import FusionEngine
from models.isolation_forest_model_v2 import IsolationForestV2Validator
from parsers.parser import Drain3LogParser
from utils.exceptions import SOCCopilotError
from utils.logger import get_logger

logger = get_logger(__name__)


def run_log_collection() -> dict:
    """Execute Module 1: Log Collection and return its summary dict."""
    logger.info("=== Module 1: Log Collection — START ===")
    collector = LogCollector()
    summary = collector.collect()
    logger.info("=== Module 1: Log Collection — COMPLETE ===")
    return summary


def run_log_parsing() -> dict:
    """Execute Module 2: Drain3 Log Parsing and return its summary dict."""
    logger.info("=== Module 2: Drain3 Log Parsing — START ===")
    parser = Drain3LogParser()
    summary = parser.parse()
    logger.info("=== Module 2: Drain3 Log Parsing — COMPLETE ===")
    return summary


def run_feature_extraction() -> dict:
    """Execute Module 3a: Feature Extraction (58 features) and return its summary dict."""
    logger.info("=== Module 3a: Feature Extraction — START ===")
    summary = FeatureExtractor().extract()
    logger.info("=== Module 3a: Feature Extraction — COMPLETE ===")
    return summary


def run_isolation_forest_v2() -> dict:
    """
    Execute Module 3b: Isolation Forest v2 -- the sole production
    Isolation Forest (19 validated features, 446,578 training-Normal
    blocks, training-only scaler, validation-only threshold/calibration).
    ``include_original_if_comparison=False`` guarantees the original
    Isolation Forest is never fit or executed here.
    """
    logger.info("=== Module 3b: Isolation Forest v2 (production) — START ===")
    summary = IsolationForestV2Validator().run(include_original_if_comparison=False)
    logger.info("=== Module 3b: Isolation Forest v2 (production) — COMPLETE ===")
    return summary


def run_autoencoder() -> dict:
    """Execute Module 4a: Autoencoder training + scoring (unchanged)."""
    logger.info("=== Module 4a: Autoencoder — START ===")
    summary = AutoencoderDetector().run()
    logger.info("=== Module 4a: Autoencoder — COMPLETE ===")
    return summary


def run_autoencoder_evaluation() -> dict:
    """Execute Module 4b: Autoencoder evaluation against ground truth (unchanged)."""
    logger.info("=== Module 4b: Autoencoder Evaluation — START ===")
    summary = AutoencoderEvaluator().evaluate()
    logger.info("=== Module 4b: Autoencoder Evaluation — COMPLETE ===")
    return summary


def run_fusion() -> dict:
    """
    Execute Module 5a: Fusion Engine -- applies the fixed, approved
    configuration (IF-v2 weight=0.10, AE weight=0.90, threshold=0.323459)
    after calibrating both scores on validation only. No grid search.
    """
    logger.info("=== Module 5a: Fusion Engine (production) — START ===")
    summary = FusionEngine().run()
    logger.info("=== Module 5a: Fusion Engine (production) — COMPLETE ===")
    return summary


def run_fusion_evaluation() -> dict:
    """Execute Module 5b: Fusion evaluation on the untouched test slice."""
    logger.info("=== Module 5b: Fusion Evaluation — START ===")
    summary = FusionEvaluator().evaluate()
    logger.info("=== Module 5b: Fusion Evaluation — COMPLETE ===")
    return summary


def main() -> int:
    """Run the complete approved production pipeline."""
    try:
        collection_summary = run_log_collection()
        parsing_summary = run_log_parsing()
        feature_summary = run_feature_extraction()
        ifv2_summary = run_isolation_forest_v2()
        autoencoder_summary = run_autoencoder()
        autoencoder_evaluation_summary = run_autoencoder_evaluation()
        fusion_summary = run_fusion()
        fusion_evaluation_summary = run_fusion_evaluation()
    except SOCCopilotError as exc:
        logger.error("Pipeline halted: %s", exc)
        return 1

    logger.info("Collection summary:\n%s", json.dumps(collection_summary, indent=2))
    logger.info("Parsing summary:\n%s", json.dumps(parsing_summary, indent=2))
    logger.info("Feature extraction summary:\n%s", json.dumps(feature_summary, indent=2))
    logger.info("Isolation Forest v2 summary:\n%s", json.dumps(ifv2_summary, indent=2))
    logger.info("Autoencoder summary:\n%s", json.dumps(autoencoder_summary, indent=2))
    logger.info(
        "Autoencoder evaluation summary:\n%s",
        json.dumps(autoencoder_evaluation_summary, indent=2),
    )
    logger.info("Fusion summary:\n%s", json.dumps(fusion_summary, indent=2))
    logger.info(
        "Fusion evaluation summary:\n%s", json.dumps(fusion_evaluation_summary, indent=2)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
