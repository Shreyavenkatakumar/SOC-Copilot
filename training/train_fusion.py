"""
Module 5 orchestrator: Fusion Engine + Evaluation.

Runs Parts A and B of Module 5 end to end:

  1. ``models.fusion_engine``       -- grid-search fusion weight/threshold,
                                        score every block
  2. ``evaluation.evaluate_fusion`` -- leakage-free metrics on the
                                        untouched test slice
"""

import json
from typing import Dict

from evaluation.evaluate_fusion import FusionEvaluator
from models.fusion_engine import FusionEngine
from utils.exceptions import SOCCopilotError
from utils.logger import get_logger

logger = get_logger(__name__)


def run_fusion() -> Dict:
    """Execute Module 5, Part A: Fusion Engine."""
    logger.info("--- Module 5a: Fusion Engine — START ---")
    summary = FusionEngine().run()
    logger.info("--- Module 5a: Fusion Engine — COMPLETE ---")
    return summary


def run_evaluation() -> Dict:
    """Execute Module 5, Part B: Fusion Evaluation."""
    logger.info("--- Module 5b: Fusion Evaluation — START ---")
    summary = FusionEvaluator().evaluate()
    logger.info("--- Module 5b: Fusion Evaluation — COMPLETE ---")
    return summary


def main() -> Dict:
    """Run both parts of Module 5 in sequence and return a combined summary."""
    fusion_summary = run_fusion()
    evaluation_summary = run_evaluation()

    return {
        "fusion": fusion_summary,
        "evaluation": evaluation_summary,
    }


if __name__ == "__main__":
    try:
        result = main()
    except SOCCopilotError as exc:
        logger.error("Module 5 pipeline halted: %s", exc)
        raise SystemExit(1) from exc

    logger.info("Module 5 summary: %s", json.dumps(result, indent=2))
