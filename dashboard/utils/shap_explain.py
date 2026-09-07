"""
SHAP explainability for the FROZEN Autoencoder and IF-v2 models.

Method chosen and why
----------------------
Both explainers wrap the *exact* scoring function each production
module uses (see ``models/autoencoder_model.py::score`` and
``models/isolation_forest_model_v2.py::_if_anomaly_score``), and use
SHAP's model-agnostic Permutation explainer (``shap.Explainer`` with a
callable + a small background sample) rather than an
architecture-specific explainer (e.g. DeepExplainer/GradientExplainer
for the Autoencoder, or a tree-specific explainer for IF-v2). This is a
deliberate choice, not a shortcut:

  - Autoencoder: the quantity to explain (mean squared reconstruction
    error) is not a raw model output neuron -- it's a derived scalar.
    Wrapping it in a plain Python function and explaining that function
    directly guarantees SHAP is attributing the actual anomaly evidence
    described in the brief, with no risk of accidentally explaining an
    arbitrary output neuron. It also sidesteps Keras-version-specific
    gradient-explainer compatibility issues, since a black-box explainer
    only ever needs ``model.predict``.
  - IF-v2: ``sklearn.ensemble.IsolationForest`` + a locally-fit
    ``MinMaxScaler`` is a two-stage pipeline, and different SHAP
    tree-explainer configurations for IsolationForest have historically
    disagreed on sign convention (whether a positive SHAP value pushes
    the score toward "more normal" or "more anomalous"). Wrapping the
    literal ``-model.decision_function(scaler.transform(x))`` function
    that production already uses removes that ambiguity by construction:
    a positive SHAP value here is *guaranteed* to mean "pushed the exact
    same score used in production toward Anomaly."

Both explained quantities are monotonically related (via a fixed
min-max rescaling) to the 0-1 scores shown elsewhere in the dashboard,
so their relative feature attributions carry over directly; this is
stated explicitly in the UI rather than claimed to be pixel-identical
to the displayed 0-1 score.

Never retrains, refits, or modifies any model/scaler/calibrator.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import load_features_raw, load_features_scaled, load_fusion_predictions
from utils.model_loader import (
    autoencoder_reconstruction_error,
    ifv2_raw_anomaly_score,
    load_autoencoder,
    load_ifv2_bundle,
)

BACKGROUND_SIZE = 50
BACKGROUND_RANDOM_STATE = 42


@st.cache_data(show_spinner=False)
def _background_block_ids() -> List[str]:
    """
    A small, deterministic sample of BlockIds predicted Normal by the
    production fusion decision, used as the SHAP background/baseline
    population for both explainers. Fixed random_state so results are
    reproducible across analyst sessions.
    """
    fusion_df = load_fusion_predictions()
    if fusion_df is None or fusion_df.empty:
        return []
    normal_ids = fusion_df.loc[fusion_df["fusion_prediction"] == "Normal", "BlockId"]
    if normal_ids.empty:
        return []
    n = min(BACKGROUND_SIZE, len(normal_ids))
    return normal_ids.sample(n=n, random_state=BACKGROUND_RANDOM_STATE).tolist()


def _shap_explainer_available() -> bool:
    try:
        import shap  # noqa: F401

        return True
    except ImportError:
        return False


@st.cache_data(show_spinner="Computing Autoencoder SHAP explanation...")
def explain_autoencoder_block(block_id: str) -> Optional[Dict]:
    """
    SHAP explanation of the Autoencoder's reconstruction-error anomaly
    evidence for a single BlockId, over all 58 features.

    Returns
    -------
    Optional[Dict]
        ``{"feature_names", "feature_values", "shap_values", "base_value",
        "f_x", "method"}``, or ``None`` if SHAP / the model / the
        feature row is unavailable (caller must show a clear message,
        never crash).
    """
    if not _shap_explainer_available():
        return None

    bundle = load_autoencoder()
    if bundle is None or bundle.get("model") is None:
        return None

    feature_columns = bundle["feature_columns"]
    features_scaled = load_features_scaled()
    if features_scaled is None or not set(feature_columns).issubset(features_scaled.columns):
        return None

    row = features_scaled.loc[features_scaled["BlockId"] == block_id]
    if row.empty:
        return None

    background_ids = _background_block_ids()
    background_df = features_scaled.loc[features_scaled["BlockId"].isin(background_ids)]
    if background_df.empty:
        background_df = features_scaled.head(BACKGROUND_SIZE)

    x_row = row[feature_columns].to_numpy(dtype="float32")
    x_background = background_df[feature_columns].to_numpy(dtype="float32")

    model = bundle["model"]

    def f(x: np.ndarray) -> np.ndarray:
        return autoencoder_reconstruction_error(model, x)

    import shap

    try:
        explainer = shap.Explainer(f, x_background, algorithm="permutation")
        explanation = explainer(x_row, max_evals=2 * len(feature_columns) + 1)
    except Exception:  # noqa: BLE001 -- any SHAP/backend failure must degrade, not crash
        return None

    shap_values = np.asarray(explanation.values[0], dtype="float64")
    base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
    f_x = float(f(x_row)[0])

    return {
        "feature_names": feature_columns,
        "feature_values": x_row[0].tolist(),
        "shap_values": shap_values.tolist(),
        "base_value": base_value,
        "f_x": f_x,
        "method": "SHAP Permutation explainer over the exact reconstruction-error function "
        "(mean squared error between input and Autoencoder output), 58 features.",
    }


@st.cache_data(show_spinner="Computing IF-v2 SHAP explanation...")
def explain_ifv2_block(block_id: str) -> Optional[Dict]:
    """
    SHAP explanation of IF-v2's statistical-isolation anomaly evidence
    for a single BlockId, over exactly the persisted 19 selected
    features. Positive SHAP = pushed the score toward Anomaly.

    Returns
    -------
    Optional[Dict]
        Same shape as ``explain_autoencoder_block``, or ``None``.
    """
    if not _shap_explainer_available():
        return None

    bundle = load_ifv2_bundle()
    if bundle is None or bundle.get("model") is None or bundle.get("local_scaler") is None:
        return None

    selected_features = bundle["selected_features"]
    features_raw = load_features_raw()
    if features_raw is None or not set(selected_features).issubset(features_raw.columns):
        return None

    row = features_raw.loc[features_raw["BlockId"] == block_id]
    if row.empty:
        return None

    background_ids = _background_block_ids()
    background_df = features_raw.loc[features_raw["BlockId"].isin(background_ids)]
    if background_df.empty:
        background_df = features_raw.head(BACKGROUND_SIZE)

    x_row = row[selected_features].to_numpy(dtype="float64")
    x_background = background_df[selected_features].to_numpy(dtype="float64")

    model = bundle["model"]
    local_scaler = bundle["local_scaler"]

    def f(x: np.ndarray) -> np.ndarray:
        return ifv2_raw_anomaly_score(model, local_scaler, x)

    import shap

    try:
        explainer = shap.Explainer(f, x_background, algorithm="permutation")
        explanation = explainer(x_row, max_evals=2 * len(selected_features) + 1)
    except Exception:  # noqa: BLE001
        return None

    shap_values = np.asarray(explanation.values[0], dtype="float64")
    base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
    f_x = float(f(x_row)[0])

    return {
        "feature_names": selected_features,
        "feature_values": x_row[0].tolist(),
        "shap_values": shap_values.tolist(),
        "base_value": base_value,
        "f_x": f_x,
        "method": "SHAP Permutation explainer over the exact IF-v2 scoring function "
        "(-decision_function after the persisted local scaler), 19 features. "
        "Positive contribution = evidence toward Anomaly.",
    }


def top_contributors(explanation: Dict, n: int = 5) -> Dict[str, List[Dict]]:
    """
    Split an explanation dict's per-feature SHAP contributions into the
    top-``n`` positive (toward anomaly) and top-``n`` negative (away
    from anomaly) contributors, each with the feature's raw value.
    """
    names = explanation["feature_names"]
    values = explanation["feature_values"]
    shap_values = explanation["shap_values"]

    rows = [
        {"feature": name, "value": value, "shap": shap}
        for name, value, shap in zip(names, values, shap_values)
    ]
    positive = sorted([r for r in rows if r["shap"] > 0], key=lambda r: -r["shap"])[:n]
    negative = sorted([r for r in rows if r["shap"] < 0], key=lambda r: r["shap"])[:n]
    return {"positive": positive, "negative": negative}
