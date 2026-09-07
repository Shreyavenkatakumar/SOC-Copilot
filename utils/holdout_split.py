"""
Shared holdout-split reproduction utility.

Isolation Forest training, Autoencoder training, and their evaluators
all derive their Normal-only training set the same way: an
``TRAIN_TEST_SPLIT_RATIO`` (80/20) split of ground-truth Normal blocks,
via ``sklearn.model_selection.train_test_split`` seeded with
``RANDOM_STATE``, applied to the block order found in ``FEATURES_FILE``.

Because that split is a deterministic function of row order, row count,
and the random seed, any caller can reproduce it exactly without needing
the original training code. This module centralizes that reproduction
so the Fusion Engine (Module 5) can identify the same training blocks
without duplicating the logic a fourth time or importing
``models/isolation_forest_model.py`` / ``models/autoencoder_model.py``
directly.

This is a new, additive file -- it does not modify or get imported by
any existing Module 1-4 file.
"""

from pathlib import Path
from typing import Set, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from data.log_collector import LogCollector
from utils.config import FEATURES_FILE, LABEL_NORMAL, RANDOM_STATE, TRAIN_TEST_SPLIT_RATIO
from utils.exceptions import SOCCopilotError
from utils.logger import get_logger

logger = get_logger(__name__)

ID_COLUMN = "BlockId"


def identify_training_normal_blocks(
    features_file: Path = FEATURES_FILE,
    train_split_ratio: float = TRAIN_TEST_SPLIT_RATIO,
    random_state: int = RANDOM_STATE,
) -> Tuple[Set[str], int]:
    """
    Reproduce the set of BlockIds used to train the Isolation Forest and
    Autoencoder (both use this identical Normal-only 80/20 split).

    Parameters
    ----------
    features_file : Path
        Feature matrix whose row order/count anchors the split.
    train_split_ratio : float
        Fraction of labeled Normal blocks used for training.
    random_state : int
        Seed used by ``train_test_split``; must match the value used
        when the detectors were actually trained.

    Returns
    -------
    Tuple[Set[str], int]
        The set of BlockIds used for training, and the total number of
        labeled Normal blocks found.

    Raises
    ------
    SOCCopilotError
        If the features file is missing, unreadable, or no Normal
        blocks are found.
    """
    features_file = Path(features_file)
    if not features_file.exists():
        raise SOCCopilotError(
            f"Features file not found: {features_file}. It is required to "
            "reproduce the anomaly detectors' train/test split."
        )

    try:
        block_ids = pd.read_csv(features_file, usecols=[ID_COLUMN])
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        raise SOCCopilotError(f"Failed to read {features_file}: {exc}") from exc

    collector = LogCollector()
    labels_df = collector.load_anomaly_labels()

    merged = block_ids.merge(labels_df, on=ID_COLUMN, how="left")
    normal_blocks = merged.loc[merged["Label"] == LABEL_NORMAL]

    if normal_blocks.empty:
        raise SOCCopilotError(
            "No blocks labeled 'Normal' were found while reproducing the "
            "anomaly detectors' train/test split."
        )

    if len(normal_blocks) < 2:
        train_blocks = normal_blocks
    else:
        train_blocks, _ = train_test_split(
            normal_blocks,
            train_size=train_split_ratio,
            random_state=random_state,
        )

    training_block_ids = set(train_blocks[ID_COLUMN].astype(str))
    logger.debug(
        "Reproduced training split: %d of %d labeled Normal block(s)",
        len(training_block_ids),
        len(normal_blocks),
    )
    return training_block_ids, len(normal_blocks)
