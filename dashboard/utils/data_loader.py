"""
Data loading layer for the SOC Copilot Analyst Dashboard.

Strictly read-only: this module never trains, retrains, fits, or
recalculates any model. It only reads the already-generated output
files from the finalized ML pipeline (python main.py) and adapts to
their actual schemas.

Every loader gracefully returns None (or an empty DataFrame) when its
source file is missing, so the dashboard can degrade gracefully instead
of crashing with a traceback.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st
from joblib import load as joblib_load

# ---------------------------------------------------------------------------
# Project paths (dashboard/ -> project root is one level up)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS_REPORTS = PROJECT_ROOT / "outputs" / "reports"
OUTPUTS_MODELS = PROJECT_ROOT / "outputs" / "models"

FUSION_PREDICTIONS_FILE = OUTPUTS_REPORTS / "fusion_predictions.csv"
AUTOENCODER_PREDICTIONS_FILE = OUTPUTS_REPORTS / "autoencoder_predictions.csv"
ISOLATION_FOREST_V2_PREDICTIONS_FILE = OUTPUTS_REPORTS / "isolation_forest_v2_predictions.csv"
FEATURES_FILE = DATA_PROCESSED / "features.csv"
PARSED_LOGS_FILE = DATA_PROCESSED / "parsed_logs.csv"
LABELS_FILE = PROJECT_ROOT / "data" / "raw" / "anomaly_label.csv"

FUSION_EVALUATION_FILE = OUTPUTS_REPORTS / "fusion_evaluation.json"
AUTOENCODER_EVALUATION_FILE = OUTPUTS_REPORTS / "autoencoder_evaluation.json"
ISOLATION_FOREST_V2_REPORT_FILE = OUTPUTS_REPORTS / "isolation_forest_v2_validation_report.json"
LOG_COLLECTION_REPORT_FILE = OUTPUTS_REPORTS / "log_collection_report.json"
PARSING_REPORT_FILE = OUTPUTS_REPORTS / "parsing_report.json"
FEATURE_EXTRACTION_REPORT_FILE = OUTPUTS_REPORTS / "feature_extraction_report.json"
FUSION_CONFIG_FILE = OUTPUTS_MODELS / "fusion_config.joblib"
FEATURE_SCALER_FILE = OUTPUTS_MODELS / "feature_scaler.joblib"

BLOCK_ID_COLUMN = "BlockId"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def file_status() -> Dict[str, bool]:
    """Return which expected output files currently exist on disk."""
    return {
        "fusion_predictions": FUSION_PREDICTIONS_FILE.exists(),
        "autoencoder_predictions": AUTOENCODER_PREDICTIONS_FILE.exists(),
        "isolation_forest_v2_predictions": ISOLATION_FOREST_V2_PREDICTIONS_FILE.exists(),
        "features": FEATURES_FILE.exists(),
        "parsed_logs": PARSED_LOGS_FILE.exists(),
        "labels": LABELS_FILE.exists(),
        "fusion_evaluation": FUSION_EVALUATION_FILE.exists(),
        "autoencoder_evaluation": AUTOENCODER_EVALUATION_FILE.exists(),
        "isolation_forest_v2_report": ISOLATION_FOREST_V2_REPORT_FILE.exists(),
        "log_collection_report": LOG_COLLECTION_REPORT_FILE.exists(),
        "parsing_report": PARSING_REPORT_FILE.exists(),
        "feature_extraction_report": FEATURE_EXTRACTION_REPORT_FILE.exists(),
        "fusion_config": FUSION_CONFIG_FILE.exists(),
        "feature_scaler": FEATURE_SCALER_FILE.exists(),
    }


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Cached loaders -- prediction / feature data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading fusion predictions...")
def load_fusion_predictions() -> Optional[pd.DataFrame]:
    if not FUSION_PREDICTIONS_FILE.exists():
        return None
    return pd.read_csv(FUSION_PREDICTIONS_FILE)


@st.cache_data(show_spinner="Loading Autoencoder predictions...")
def load_autoencoder_predictions() -> Optional[pd.DataFrame]:
    if not AUTOENCODER_PREDICTIONS_FILE.exists():
        return None
    return pd.read_csv(AUTOENCODER_PREDICTIONS_FILE)


@st.cache_data(show_spinner="Loading IF-v2 predictions...")
def load_ifv2_predictions() -> Optional[pd.DataFrame]:
    if not ISOLATION_FOREST_V2_PREDICTIONS_FILE.exists():
        return None
    return pd.read_csv(ISOLATION_FOREST_V2_PREDICTIONS_FILE)


@st.cache_data(show_spinner="Loading ground-truth labels...")
def load_labels() -> Optional[pd.DataFrame]:
    if not LABELS_FILE.exists():
        return None
    return pd.read_csv(LABELS_FILE, dtype=str)


@st.cache_data(show_spinner="Loading behavioural features...")
def load_features_raw() -> Optional[pd.DataFrame]:
    """
    Load features.csv and, if the fitted global MinMaxScaler is
    available, invert the 0-1 scaling back to human-readable raw values
    (event counts, seconds, etc.) for display purposes only. This never
    touches the model's own scaled inputs -- it is purely a dashboard
    presentation convenience.
    """
    if not FEATURES_FILE.exists():
        return None
    scaled_df = pd.read_csv(FEATURES_FILE)

    if not FEATURE_SCALER_FILE.exists():
        return scaled_df

    try:
        bundle = joblib_load(FEATURE_SCALER_FILE)
        scaler = bundle["scaler"]
        columns = bundle["columns"]
        raw_values = scaler.inverse_transform(scaled_df[columns].to_numpy())
        raw_df = pd.DataFrame(raw_values, columns=columns)
        raw_df.insert(0, BLOCK_ID_COLUMN, scaled_df[BLOCK_ID_COLUMN].values)
        return raw_df
    except (KeyError, ValueError, OSError):
        # Fall back to scaled values rather than crash the dashboard.
        return scaled_df


@st.cache_data(show_spinner="Building the master alert table...")
def build_master_table() -> Optional[pd.DataFrame]:
    """
    Join fusion + AE + IF-v2 predictions, ground-truth labels, and
    human-readable behavioural features into one analyst-facing table.
    Returns None only if the core fusion predictions are missing --
    everything else degrades gracefully via left-joins.
    """
    fusion_df = load_fusion_predictions()
    if fusion_df is None:
        return None

    table = fusion_df.copy()

    labels_df = load_labels()
    if labels_df is not None and "BlockId" in labels_df.columns and "Label" in labels_df.columns:
        table = table.merge(labels_df[[BLOCK_ID_COLUMN, "Label"]], on=BLOCK_ID_COLUMN, how="left")
    else:
        table["Label"] = None

    features_df = load_features_raw()
    if features_df is not None:
        feature_cols = [c for c in features_df.columns if c != BLOCK_ID_COLUMN]
        table = table.merge(features_df, on=BLOCK_ID_COLUMN, how="left", suffixes=("", "_feat"))
    else:
        feature_cols = []

    return table


# ---------------------------------------------------------------------------
# Cached loaders -- JSON evaluation / configuration reports
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_fusion_evaluation() -> Optional[Dict]:
    return _load_json(FUSION_EVALUATION_FILE)


@st.cache_data(show_spinner=False)
def load_autoencoder_evaluation() -> Optional[Dict]:
    return _load_json(AUTOENCODER_EVALUATION_FILE)


@st.cache_data(show_spinner=False)
def load_ifv2_report() -> Optional[Dict]:
    return _load_json(ISOLATION_FOREST_V2_REPORT_FILE)


@st.cache_data(show_spinner=False)
def load_log_collection_report() -> Optional[Dict]:
    return _load_json(LOG_COLLECTION_REPORT_FILE)


@st.cache_data(show_spinner=False)
def load_parsing_report() -> Optional[Dict]:
    return _load_json(PARSING_REPORT_FILE)


@st.cache_data(show_spinner=False)
def load_feature_extraction_report() -> Optional[Dict]:
    return _load_json(FEATURE_EXTRACTION_REPORT_FILE)


@st.cache_resource(show_spinner=False)
def load_fusion_config() -> Optional[Dict]:
    if not FUSION_CONFIG_FILE.exists():
        return None
    try:
        bundle = joblib_load(FUSION_CONFIG_FILE)
        return bundle.get("config")
    except (OSError, KeyError):
        return None


@st.cache_resource(show_spinner=False)
def load_ifv2_model_metadata() -> Optional[Dict]:
    """Selected feature list + threshold, from the persisted IF-v2 bundle."""
    ifv2_model_file = OUTPUTS_MODELS / "isolation_forest_model_v2.joblib"
    if not ifv2_model_file.exists():
        return None
    try:
        bundle = joblib_load(ifv2_model_file)
        return {
            "selected_features": bundle.get("selected_features"),
            "threshold": bundle.get("threshold"),
            "random_state": bundle.get("random_state"),
        }
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Per-block raw log lookup (parsed_logs.csv is large -- never loaded whole)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-block raw log lookup
# ---------------------------------------------------------------------------
# parsed_logs.csv holds ~11.2 million rows (~1.5 GB) -- a linear scan per
# analyst click is far too slow for an interactive UI (~80s measured).
# Instead, a lightweight SQLite index (BlockId -> matching rows) is built
# ONCE, lazily, on first use, and cached inside dashboard/.cache/ (never
# touching any file outside dashboard/). After the one-time build, every
# per-block lookup is an indexed query taking a few milliseconds. The
# index is built from parsed_logs.csv only -- it never reads HDFS.log.

import sqlite3

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
LOG_INDEX_DB = CACHE_DIR / "parsed_logs_index.db"
_INDEX_COLUMNS = ["BlockId", "Timestamp", "Component", "EventId", "EventTemplate", "Content"]


def _build_log_index(progress_callback=None) -> bool:
    """
    One-time ETL: stream parsed_logs.csv in bounded chunks into a local
    SQLite database with an index on BlockId. Returns True on success.
    """
    if not PARSED_LOGS_FILE.exists():
        return False

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = LOG_INDEX_DB.with_suffix(".building.db")
    if tmp_path.exists():
        tmp_path.unlink()

    conn = sqlite3.connect(str(tmp_path))
    conn.execute(
        "CREATE TABLE logs (BlockId TEXT, Timestamp TEXT, Component TEXT, "
        "EventId TEXT, EventTemplate TEXT, Content TEXT)"
    )

    chunk_size = 500_000
    rows_written = 0
    try:
        reader = pd.read_csv(PARSED_LOGS_FILE, usecols=_INDEX_COLUMNS, chunksize=chunk_size, dtype=str)
        for i, chunk in enumerate(reader):
            chunk = chunk.where(pd.notna(chunk), None)
            conn.executemany(
                "INSERT INTO logs VALUES (?, ?, ?, ?, ?, ?)",
                chunk[_INDEX_COLUMNS].itertuples(index=False, name=None),
            )
            rows_written += len(chunk)
            if progress_callback is not None:
                progress_callback(rows_written)
        conn.execute("CREATE INDEX idx_block_id ON logs (BlockId)")
        conn.commit()
    except (OSError, pd.errors.ParserError, ValueError, sqlite3.Error):
        conn.close()
        if tmp_path.exists():
            tmp_path.unlink()
        return False
    finally:
        conn.close()

    tmp_path.replace(LOG_INDEX_DB)
    return True


def ensure_log_index() -> bool:
    """Build the SQLite log index if it doesn't already exist. Returns True if usable."""
    if LOG_INDEX_DB.exists():
        return True
    if not PARSED_LOGS_FILE.exists():
        return False

    progress_bar = st.progress(0.0, text="Building one-time log search index (first use only)...")
    total_estimate = 11_200_000  # rough guide for the progress bar only

    def _progress(rows_written: int) -> None:
        progress_bar.progress(min(rows_written / total_estimate, 1.0))

    success = _build_log_index(progress_callback=_progress)
    progress_bar.empty()
    return success


@st.cache_data(show_spinner="Searching parsed logs for this block...")
def get_block_raw_logs(block_id: str) -> Optional[pd.DataFrame]:
    """
    Fast, indexed lookup of every parsed log line belonging to
    ``block_id``, via the one-time SQLite index (built lazily on first
    call). Cached per BlockId so repeat lookups are instant.
    """
    if not ensure_log_index():
        return None
    try:
        conn = sqlite3.connect(str(LOG_INDEX_DB))
        query = f"SELECT {', '.join(_INDEX_COLUMNS)} FROM logs WHERE BlockId = ?"
        df = pd.read_sql_query(query, conn, params=(block_id,))
        conn.close()
        return df
    except sqlite3.Error:
        return None


@st.cache_data(show_spinner=False)
def load_event_template_map() -> Dict[str, str]:
    """
    EventId -> EventTemplate lookup (e.g. "E12" -> "Deleting block <BlockId> file <*>").

    Built from the same one-time SQLite index used for per-block log
    lookups (``ensure_log_index`` / ``get_block_raw_logs``) -- this never
    re-scans the 11M-row ``parsed_logs.csv`` on its own; it issues one
    cheap ``SELECT DISTINCT`` against the already-built index (~48 rows).
    """
    if not ensure_log_index():
        return {}
    try:
        conn = sqlite3.connect(str(LOG_INDEX_DB))
        df = pd.read_sql_query("SELECT DISTINCT EventId, EventTemplate FROM logs", conn)
        conn.close()
        return dict(zip(df["EventId"], df["EventTemplate"]))
    except sqlite3.Error:
        return {}


@st.cache_data(show_spinner="Computing anomaly trend (one-time index scan)...")
def get_block_first_seen_all() -> Optional[pd.DataFrame]:
    """
    Every block's first-seen timestamp, in one aggregate pass over the
    SQLite index (``GROUP BY BlockId``) -- used for the SOC Overview
    anomaly-trend chart and the Alerts page's Timestamp column / date
    filter. This is a heavier, whole-index aggregate (unlike a single
    per-block lookup), but it is computed at most ONCE per running
    dashboard process thanks to ``st.cache_data`` -- never repeated per
    interaction -- and it reads only the pre-built index, never
    ``parsed_logs.csv`` itself.
    """
    if not ensure_log_index():
        return None
    try:
        conn = sqlite3.connect(str(LOG_INDEX_DB))
        df = pd.read_sql_query(
            "SELECT BlockId, MIN(Timestamp) as first_seen_raw FROM logs GROUP BY BlockId", conn
        )
        conn.close()
    except sqlite3.Error:
        return None
    df["first_seen"] = pd.to_datetime(df["first_seen_raw"], format="%y%m%d%H%M%S", errors="coerce")
    return df[["BlockId", "first_seen"]]


def attach_timeline_columns(block_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Pure transform (no disk I/O) over an already-fetched per-block log
    dataframe: parses the ``YYMMDDHHMMSS`` Timestamp string into a real
    datetime, sorts chronologically, and adds a sequential step number
    -- everything the block activity timeline chart needs.
    """
    df = block_logs.copy()
    df["_ts"] = pd.to_datetime(df["Timestamp"], format="%y%m%d%H%M%S", errors="coerce")
    df = df.sort_values("_ts", kind="stable").reset_index(drop=True)
    df["_step"] = range(1, len(df) + 1)
    return df


@st.cache_data(show_spinner=False)
def get_dataset_summary() -> Dict:
    """Aggregate a few headline counts, preferring real report values over hard-coded ones."""
    collection = load_log_collection_report() or {}
    parsing = load_parsing_report() or {}
    features = load_feature_extraction_report() or {}
    labels_df = load_labels()

    normal_count = anomaly_count = total_blocks = None
    if labels_df is not None and "Label" in labels_df.columns:
        total_blocks = len(labels_df)
        normal_count = int((labels_df["Label"] == "Normal").sum())
        anomaly_count = int((labels_df["Label"] == "Anomaly").sum())

    return {
        "total_log_lines": parsing.get("total_lines_read") or collection.get("total_lines_collected"),
        "unique_templates": parsing.get("unique_template_count"),
        "total_blocks": total_blocks or features.get("total_blocks"),
        "normal_blocks": normal_count,
        "anomaly_blocks": anomaly_count,
        "total_feature_columns": features.get("total_feature_columns"),
    }
