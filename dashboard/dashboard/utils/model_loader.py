"""
Read-only loaders for the already-trained, frozen production models
(Autoencoder + IF-v2) and their persisted scalers/calibrators.

Used only to compute SHAP explanations for a single selected block in
the Alert Investigation page. This module NEVER trains, retrains,
fits, or refits anything -- models are only ``.predict()``-ed /
``.decision_function()``-ed, and scalers are only ``.transform()``-ed,
exactly as persisted by the finalized pipeline (``python main.py``).

Every loader returns ``None`` if its source artifact is missing, so the
dashboard degrades gracefully instead of crashing.
"""

from typing import Dict, Optional

import pandas as pd
import streamlit as st
from joblib import load as joblib_load

from utils.data_loader import FEATURE_SCALER_FILE, FEATURES_FILE, OUTPUTS_MODELS

AUTOENCODER_MODEL_FILE = OUTPUTS_MODELS / "autoencoder_model.keras"
AUTOENCODER_META_FILE = OUTPUTS_MODELS / "autoencoder_model.meta.joblib"
ISOLATION_FOREST_V2_MODEL_FILE = OUTPUTS_MODELS / "isolation_forest_model_v2.joblib"


@st.cache_resource(show_spinner="Loading Autoencoder for explainability...")
def load_ae_model():
    """Load the already-trained Keras Autoencoder. Read-only -- never refit."""
    if not AUTOENCODER_MODEL_FILE.exists():
        return None
    from tensorflow import keras  # local import: heavy, only needed on this page

    try:
        return keras.models.load_model(AUTOENCODER_MODEL_FILE)
    except (OSError, ValueError):
        return None


@st.cache_resource(show_spinner=False)
def load_ae_metadata() -> Optional[Dict]:
    """Load {'feature_columns': [...58], 'reconstruction_error_threshold': ...}."""
    if not AUTOENCODER_META_FILE.exists():
        return None
    try:
        return joblib_load(AUTOENCODER_META_FILE)
    except OSError:
        return None


@st.cache_resource(show_spinner=False)
def load_ifv2_bundle() -> Optional[Dict]:
    """Load {'model', 'local_scaler', 'calibrator', 'selected_features', 'threshold', ...}."""
    if not ISOLATION_FOREST_V2_MODEL_FILE.exists():
        return None
    try:
        return joblib_load(ISOLATION_FOREST_V2_MODEL_FILE)
    except OSError:
        return None


@st.cache_resource(show_spinner=False)
def load_global_feature_scaler() -> Optional[Dict]:
    """Load {'scaler': MinMaxScaler, 'columns': [...58]}, fit over all 575,061 blocks."""
    if not FEATURE_SCALER_FILE.exists():
        return None
    try:
        return joblib_load(FEATURE_SCALER_FILE)
    except OSError:
        return None


@st.cache_data(show_spinner="Loading feature matrix for model inference...")
def load_features_scaled() -> Optional[pd.DataFrame]:
    """
    Load ``features.csv`` exactly as the models see it at inference time
    (already 0-1 MinMax-scaled by the production ``feature_scaler.joblib``).

    Distinct from ``utils.data_loader.load_features_raw()``, which
    inverts that scaling back to human-readable values for dashboard
    display only -- models must be fed the scaled values they were
    actually trained on.
    """
    if not FEATURES_FILE.exists():
        return None
    return pd.read_csv(FEATURES_FILE)
