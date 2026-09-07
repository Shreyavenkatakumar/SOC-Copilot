"""
Module 4 orchestrator: Autoencoder Training + Scoring + Evaluation.

Runs Parts A and B of Module 4 end to end:

  1. ``models.autoencoder_model``        -- train + score the Autoencoder
  2. ``evaluation.evaluate_autoencoder`` -- leakage-free metrics vs. ground truth
"""

import json
from typing import Dict

from evaluation.evaluate_autoencoder import AutoencoderEvaluator
from models.autoencoder_model import AutoencoderDetector
from utils.exceptions import SOCCopilotError
from utils.logger import get_logger

logger = get_logger(__name__)


def run_autoencoder() -> Dict:
    """Execute Module 4, Part A: Autoencoder training + scoring."""
    logger.info("--- Module 4a: Autoencoder Training + Scoring — START ---")
    summary = AutoencoderDetector().run()
    logger.info("--- Module 4a: Autoencoder Training + Scoring — COMPLETE ---")
    return summary


def run_evaluation() -> Dict:
    """Execute Module 4, Part B: Evaluation against ground truth."""
    logger.info("--- Module 4b: Autoencoder Evaluation — START ---")
    summary = AutoencoderEvaluator().evaluate()
    logger.info("--- Module 4b: Autoencoder Evaluation — COMPLETE ---")
    return summary


def main() -> Dict:
    """Run both parts of Module 4 in sequence and return a combined summary."""
    autoencoder_summary = run_autoencoder()
    evaluation_summary = run_evaluation()

    return {
        "autoencoder": autoencoder_summary,
        "evaluation": evaluation_summary,
    }


if __name__ == "__main__":
    try:
        result = main()
    except SOCCopilotError as exc:
        logger.error("Module 4 pipeline halted: %s", exc)
        raise SystemExit(1) from exc

    logger.info("Module 4 summary: %s", json.dumps(result, indent=2))
