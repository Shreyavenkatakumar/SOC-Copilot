"""
Central configuration module for SOC Copilot Phase I.

All paths, constants, and tunable parameters used across the pipeline
are defined here so that every module shares a single source of truth.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base directories
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

OUTPUTS_DIR: Path = BASE_DIR / "outputs"
LOGS_DIR: Path = OUTPUTS_DIR / "logs"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
MODELS_DIR: Path = OUTPUTS_DIR / "models"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"

for _directory in (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    LOGS_DIR,
    REPORTS_DIR,
    MODELS_DIR,
    FIGURES_DIR,
):
    _directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Raw dataset files (LogHub HDFS dataset)
# ---------------------------------------------------------------------------
HDFS_LOG_FILE: Path = RAW_DATA_DIR / "HDFS.log"
ANOMALY_LABEL_FILE: Path = RAW_DATA_DIR / "anomaly_label.csv"
LOG_TEMPLATES_FILE: Path = RAW_DATA_DIR / "HDFS.log_templates.csv"
EVENT_TRACES_FILE: Path = RAW_DATA_DIR / "Event_traces.csv"
EVENT_OCCURRENCE_MATRIX_FILE: Path = RAW_DATA_DIR / "Event_occurrence_matrix.csv"
HDFS_NPZ_FILE: Path = RAW_DATA_DIR / "HDFS.npz"

# Files strictly required for the Phase I pipeline to run end to end.
REQUIRED_RAW_FILES = (HDFS_LOG_FILE, ANOMALY_LABEL_FILE)

# Files that exist purely for validation / cross-checking, never used to
# replace any pipeline stage.
VALIDATION_ONLY_FILES = (
    LOG_TEMPLATES_FILE,
    EVENT_TRACES_FILE,
    EVENT_OCCURRENCE_MATRIX_FILE,
    HDFS_NPZ_FILE,
)

# Accepted raw log input extensions for the log collector.
ALLOWED_LOG_EXTENSIONS = (".log", ".txt", ".csv")

# ---------------------------------------------------------------------------
# Log collection parameters
# ---------------------------------------------------------------------------
LOG_READ_CHUNK_SIZE = 200_000          # lines per chunk when streaming HDFS.log
LOG_ENCODING = "utf-8"
LOG_ENCODING_FALLBACK = "latin-1"

# HDFS log lines contain a block id token of the form "blk_<signed_int>".
BLOCK_ID_REGEX = r"(blk_-?\d+)"

# ---------------------------------------------------------------------------
# Drain3 parser configuration (Module 2)
# ---------------------------------------------------------------------------
DRAIN3_CONFIG_FILE: Path = BASE_DIR / "parsers" / "drain3_config.ini"
DRAIN3_STATE_FILE: Path = OUTPUTS_DIR / "models" / "drain3_state.bin"

PARSED_LOGS_FILE: Path = PROCESSED_DATA_DIR / "parsed_logs.csv"
PARSING_REPORT_FILE: Path = REPORTS_DIR / "parsing_report.json"

# Number of parsed rows buffered in memory before being flushed to CSV.
PARSING_WRITE_BATCH_SIZE = 100_000

# Number of highest-frequency templates to retain in the parsing report.
TOP_N_TEMPLATES_IN_REPORT = 20

# ---------------------------------------------------------------------------
# Feature extraction parameters (Module 3)
# ---------------------------------------------------------------------------
FEATURES_FILE: Path = PROCESSED_DATA_DIR / "features.csv"
FEATURE_EXTRACTION_REPORT_FILE: Path = REPORTS_DIR / "feature_extraction_report.json"

# Columns actually read from parsed_logs.csv for feature extraction. Keeping
# this narrow avoids loading RawLine/ParameterList (unused here) for an
# 11M+ row file.
PARSED_LOGS_USECOLS = ("BlockId", "Timestamp", "Level", "Component", "EventId")

# Log levels normalized into fixed count columns; anything else is bucketed
# under "OTHER".
KNOWN_LOG_LEVELS = ("INFO", "WARN", "ERROR")

# Chunk size used when streaming parsed_logs.csv for aggregation.
FEATURE_EXTRACTION_CHUNK_SIZE = 500_000

# Small constant to avoid division-by-zero when computing event density for
# blocks whose entire lifetime falls inside a single second.
EVENT_DENSITY_EPSILON = 1e-6

FEATURE_SCALER_FILE: Path = MODELS_DIR / "feature_scaler.joblib"

# ---------------------------------------------------------------------------
# Model hyper-parameters (Module 4 / 5)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

ISOLATION_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_samples": "auto",
    "contamination": "auto",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

ISOLATION_FOREST_MODEL_FILE: Path = MODELS_DIR / "isolation_forest_model.joblib"
ISOLATION_FOREST_PREDICTIONS_FILE: Path = REPORTS_DIR / "isolation_forest_predictions.csv"
ISOLATION_FOREST_EVALUATION_FILE: Path = REPORTS_DIR / "isolation_forest_evaluation.json"

# ---------------------------------------------------------------------------
# Isolation Forest v2 -- ISOLATED validation experiment only.
# Separate artifacts from the production model above; the original
# isolation_forest_model.joblib is never read or written by V2.
# Not imported by main.py, fusion_engine.py, or any severity/dashboard code.
# ---------------------------------------------------------------------------
ISOLATION_FOREST_V2_PARAMS = {
    "n_estimators": 200,
    "max_samples": 0.5,
    "max_features": 1.0,
    "bootstrap": False,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
ISOLATION_FOREST_V2_MODEL_FILE: Path = MODELS_DIR / "isolation_forest_model_v2.joblib"
ISOLATION_FOREST_V2_PREDICTIONS_FILE: Path = REPORTS_DIR / "isolation_forest_v2_predictions.csv"
ISOLATION_FOREST_V2_REPORT_FILE: Path = REPORTS_DIR / "isolation_forest_v2_validation_report.json"

# Isolation Forest is trained on a TRAIN_TEST_SPLIT_RATIO fraction of
# ground-truth Normal blocks only (semi-supervised); it is then scored
# against every block (remaining Normal + all Anomaly) for evaluation.

# Anomaly label encoding used across evaluation modules.
LABEL_NORMAL = "Normal"
LABEL_ANOMALY = "Anomaly"
LABEL_TO_BINARY = {LABEL_NORMAL: 0, LABEL_ANOMALY: 1}

AUTOENCODER_PARAMS = {
    "encoding_dims": [32, 16, 8],
    "activation": "relu",
    "output_activation": "sigmoid",
    "learning_rate": 1e-3,
    "batch_size": 256,
    "epochs": 50,
    "validation_split": 0.1,
    "early_stopping_patience": 5,
}

# Assumed anomaly rate used to pick a reconstruction-error cutoff from the
# TRAINING (Normal-only) error distribution -- e.g. 0.03 -> the 97th
# percentile of training reconstruction error becomes the Anomaly cutoff.
# Matches the tuned Isolation Forest contamination for a fair comparison.
AUTOENCODER_CONTAMINATION = 0.03

AUTOENCODER_MODEL_FILE: Path = MODELS_DIR / "autoencoder_model.keras"
AUTOENCODER_PREDICTIONS_FILE: Path = REPORTS_DIR / "autoencoder_predictions.csv"
AUTOENCODER_EVALUATION_FILE: Path = REPORTS_DIR / "autoencoder_evaluation.json"
AUTOENCODER_TRAINING_HISTORY_FILE: Path = REPORTS_DIR / "autoencoder_training_history.json"

# ---------------------------------------------------------------------------
# Fusion engine parameters (Module 5)
# ---------------------------------------------------------------------------
FUSION_WEIGHT_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
DEFAULT_ISOLATION_FOREST_WEIGHT = 0.5

# Full weight grid used by the multi-strategy fusion search (Module 5,
# revised): 0.0-1.0 in steps of 0.1, plus a fine AE-dominant sweep since
# the score-scale analysis showed the Autoencoder's normalized score is
# heavily right-skewed (a few extreme reconstruction errors dominate its
# min-max range) -- small IF weights are worth checking explicitly rather
# than only at the 0.1 grid spacing.
FUSION_WEIGHT_GRID_FULL = tuple(round(i * 0.1, 2) for i in range(0, 11))
FUSION_WEIGHT_GRID_AE_DOMINANT = (0.05, 0.10, 0.15, 0.20)

# Fusion strategies evaluated during grid search (Module 5, revised):
#   "weighted_raw"        -- w*iforest_anomaly_score + (1-w)*autoencoder_anomaly_score,
#                             using each detector's own already-persisted
#                             0-1 score (min-max fit over ALL scored blocks).
#   "weighted_normalized" -- same weighted average, but both scores are
#                             first re-normalized using min-max bounds
#                             fit ONLY on the train+validation population
#                             (avoids the test-set leaking into the scale).
#   "rank"                -- both scores converted to percentile rank
#                             against the train+validation distribution,
#                             then weighted-averaged. Robust to the
#                             Autoencoder's skewed raw distribution.
#   "max"                 -- element-wise max of the two re-normalized
#                             scores; either detector alone can trigger
#                             a high fused score.
FUSION_STRATEGIES = ("weighted_raw", "weighted_normalized", "rank", "max")

# Minimum acceptable recall when selecting a "precision at recall floor"
# threshold. SOC alerting: missing a real intrusion is costlier than an
# extra alert to triage, so recall is protected first.
FUSION_RECALL_FLOOR = 0.80

# F-beta used to pick the winning (weight, threshold) pair during grid
# search. beta=1.0 -> standard F1 (precision/recall weighted equally).
# beta=2.0 -> F2, weights recall twice as heavily as precision -- often
# a better fit for SOC alerting, where missing a real intrusion (false
# negative) is costlier than an extra alert to triage (false positive).
FUSION_OPTIMIZATION_BETA = 1.0

# The Normal-holdout + all-Anomaly population (blocks never used to train
# Isolation Forest or the Autoencoder) is further split into a validation
# set (used to grid-search the fusion weight and decision threshold) and a
# test set (untouched during weight/threshold selection, used only for the
# final reported metrics). Stratified by ground-truth label.
FUSION_VALIDATION_SPLIT_RATIO = 0.5

# ---------------------------------------------------------------------------
# APPROVED PRODUCTION fusion configuration (validated via
# models/complementarity_analysis.py; see
# outputs/reports/ae_ifv2_complementarity_report.json step6). Fixed, not
# re-searched at runtime -- production fusion applies this configuration
# directly rather than performing a new grid search.
# ---------------------------------------------------------------------------
FUSION_PRODUCTION_IF_WEIGHT = 0.10
FUSION_PRODUCTION_AE_WEIGHT = 0.90
FUSION_PRODUCTION_THRESHOLD = 0.323459

FUSION_CONFIG_FILE: Path = MODELS_DIR / "fusion_config.joblib"
FUSION_PREDICTIONS_FILE: Path = REPORTS_DIR / "fusion_predictions.csv"
FUSION_EVALUATION_FILE: Path = REPORTS_DIR / "fusion_evaluation.json"
FUSION_GRID_SEARCH_REPORT_FILE: Path = REPORTS_DIR / "fusion_grid_search_report.json"
SCORE_SCALE_ANALYSIS_FILE: Path = REPORTS_DIR / "score_scale_analysis.json"

# Comparison plots (Module 5, revised)
PLOT_CONFUSION_MATRICES_FILE: Path = FIGURES_DIR / "confusion_matrices.png"
PLOT_PR_CURVES_FILE: Path = FIGURES_DIR / "pr_curves.png"
PLOT_ROC_CURVES_FILE: Path = FIGURES_DIR / "roc_curves.png"
PLOT_WEIGHT_VS_F1_FILE: Path = FIGURES_DIR / "fusion_weight_vs_f1.png"
PLOT_WEIGHT_VS_RECALL_FILE: Path = FIGURES_DIR / "fusion_weight_vs_recall.png"
PLOT_THRESHOLD_VS_PR_FILE: Path = FIGURES_DIR / "fusion_threshold_vs_precision_recall.png"
MODEL_COMPARISON_TABLE_FILE: Path = REPORTS_DIR / "model_comparison_table.json"

# ---------------------------------------------------------------------------
# Severity scoring thresholds (used starting Module 7)
# ---------------------------------------------------------------------------
SEVERITY_BANDS = (
    (0, 25, "Low"),
    (26, 50, "Medium"),
    (51, 75, "High"),
    (76, 100, "Critical"),
)

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TRAIN_TEST_SPLIT_RATIO = 0.8

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("SOC_COPILOT_LOG_LEVEL", "INFO")
LOG_FILE: Path = LOGS_DIR / "soc_copilot.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
