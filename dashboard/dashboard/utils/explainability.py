"""
Read-only SHAP explainability for the Alert Investigation page.

Computes SHAP feature-contribution values for a single selected block
against the already-trained, frozen production models:
  - Autoencoder (58 input features)
  - IF-v2       (its exact persisted 19 selected features)

This module NEVER trains, fits, or refits anything. Every scaler is
only ``.transform()``-ed and every model is only scored, exactly as
persisted by the finalized pipeline. Ground-truth labels are never
used to build or select the explanation for the block being
investigated -- the SHAP background reference is a plain random
sample of the scored population (see ``_background_sample``).

SHAP values here explain which input features push each model's own
anomaly score up or down for this specific block. This is feature
contribution to model evidence, NOT a proof of root cause -- the
dashboard surfaces that caveat alongside every SHAP result.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from utils import model_loader as ml

ID_COLUMN = "BlockId"

# A small, deterministic random sample of the full scored population,
# used only as the SHAP baseline ("what does a typical block look
# like"). Not label-filtered: with a ~97% Normal prevalence in this
# dataset a plain random sample is already overwhelmingly
# representative of normal behaviour, and this keeps the explanation
# fully independent of any ground-truth label.
BACKGROUND_SAMPLE_SIZE = 60
BACKGROUND_RANDOM_STATE = 42

# Bounds KernelExplainer runtime to a few seconds even for the
# Autoencoder's 58-feature input space.
AE_SHAP_NSAMPLES = 120
IFV2_SHAP_NSAMPLES = "auto"


def _background_sample(features_scaled: pd.DataFrame, columns: List[str]) -> np.ndarray:
    n = min(BACKGROUND_SAMPLE_SIZE, len(features_scaled))
    sample = features_scaled.sample(n=n, random_state=BACKGROUND_RANDOM_STATE)
    return sample[columns].to_numpy(dtype=np.float64)


def _as_1d(shap_values) -> np.ndarray:
    """Normalize shap's various return shapes down to a flat 1-D array."""
    arr = np.asarray(shap_values)
    return arr.reshape(-1)


@st.cache_data(show_spinner="Computing Autoencoder SHAP explanation...")
def explain_autoencoder(block_id: str) -> Optional[Dict]:
    """
    SHAP explanation of the Autoencoder's reconstruction-error anomaly
    score for a single block, over its 58 input features.
    """
    model = ml.load_ae_model()
    meta = ml.load_ae_metadata()
    features_scaled = ml.load_features_scaled()
    if model is None or meta is None or features_scaled is None:
        return None

    columns = meta.get("feature_columns")
    if not columns:
        return None

    row = features_scaled.loc[features_scaled[ID_COLUMN] == block_id]
    if row.empty:
        return None

    x_instance = row[columns].to_numpy(dtype=np.float64)
    background = _background_sample(features_scaled, columns)

    def reconstruction_error(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        recon = model.predict(x, verbose=0)
        return np.mean(np.square(x - recon), axis=1)

    import shap  # local import: heavy, only needed on this page

    explainer = shap.KernelExplainer(reconstruction_error, background)
    raw_shap = explainer.shap_values(x_instance, nsamples=AE_SHAP_NSAMPLES)
    shap_values = _as_1d(raw_shap)

    base_value = explainer.expected_value
    base_value = float(base_value if np.isscalar(base_value) else np.asarray(base_value).reshape(-1)[0])

    return {
        "model": "autoencoder",
        "feature_names": columns,
        "shap_values": shap_values,
        "feature_values": x_instance.reshape(-1),
        "base_value": base_value,
        "predicted_value": float(reconstruction_error(x_instance)[0]),
    }


@st.cache_data(show_spinner="Computing IF-v2 SHAP explanation...")
def explain_isolation_forest_v2(block_id: str) -> Optional[Dict]:
    """
    SHAP explanation of IF-v2's isolation anomaly score for a single
    block, over its exact persisted 19 selected features.

    Sign convention matches ``models/isolation_forest_model_v2.py``
    (``-model.decision_function(x)``, higher = more anomalous) so SHAP
    contributions here are directly consistent with the persisted
    ``if_v2_anomaly_score``.
    """
    bundle = ml.load_ifv2_bundle()
    scaler_bundle = ml.load_global_feature_scaler()
    features_scaled = ml.load_features_scaled()
    if bundle is None or scaler_bundle is None or features_scaled is None:
        return None

    selected_features = bundle.get("selected_features")
    model = bundle.get("model")
    local_scaler = bundle.get("local_scaler")
    if not selected_features or model is None or local_scaler is None:
        return None

    global_scaler = scaler_bundle.get("scaler")
    global_columns = scaler_bundle.get("columns")
    if global_scaler is None or not global_columns:
        return None

    row = features_scaled.loc[features_scaled[ID_COLUMN] == block_id]
    if row.empty:
        return None

    # Invert the global 0-1 scaling to recover raw feature values, then
    # apply IF-v2's own persisted local scaler -- exactly the two-step
    # transform models/isolation_forest_model_v2.py uses at inference
    # time. Both are .transform()/.inverse_transform() calls only.
    raw_row = global_scaler.inverse_transform(row[global_columns].to_numpy(dtype=np.float64))
    raw_row_df = pd.DataFrame(raw_row, columns=global_columns)
    x_instance = local_scaler.transform(raw_row_df[selected_features].to_numpy(dtype=np.float64))

    bg_rows = features_scaled.sample(
        n=min(BACKGROUND_SAMPLE_SIZE, len(features_scaled)), random_state=BACKGROUND_RANDOM_STATE
    )
    bg_raw = global_scaler.inverse_transform(bg_rows[global_columns].to_numpy(dtype=np.float64))
    bg_raw_df = pd.DataFrame(bg_raw, columns=global_columns)
    background = local_scaler.transform(bg_raw_df[selected_features].to_numpy(dtype=np.float64))

    def anomaly_score(x: np.ndarray) -> np.ndarray:
        return -model.decision_function(x)

    import shap

    explainer = shap.KernelExplainer(anomaly_score, background)
    raw_shap = explainer.shap_values(x_instance, nsamples=IFV2_SHAP_NSAMPLES)
    shap_values = _as_1d(raw_shap)

    base_value = explainer.expected_value
    base_value = float(base_value if np.isscalar(base_value) else np.asarray(base_value).reshape(-1)[0])

    return {
        "model": "if_v2",
        "feature_names": selected_features,
        "shap_values": shap_values,
        "feature_values": x_instance.reshape(-1),
        "base_value": base_value,
        "predicted_value": float(anomaly_score(x_instance)[0]),
    }


def drain3_mapping_table(
    ae_explanation: Optional[Dict], ifv2_explanation: Optional[Dict], template_map: Dict[str, str], top_n: int = 12
) -> pd.DataFrame:
    """
    Feature -> Event ID -> full EventTemplate text, for the template
    (``tmpl_E<n>_count``) features that actually appear in this block's
    SHAP results, ranked by their largest |SHAP value| across both
    models. Gives the analyst the untruncated template text behind the
    short labels shown on the SHAP charts.
    """
    rows: Dict[str, Dict] = {}
    for explanation in (ae_explanation, ifv2_explanation):
        if not explanation:
            continue
        for name, value in zip(explanation["feature_names"], explanation["shap_values"]):
            if not (name.startswith("tmpl_") and name.endswith("_count")):
                continue
            event_id = name[len("tmpl_") : -len("_count")]
            entry = rows.setdefault(
                name,
                {
                    "Feature": name,
                    "Event ID": event_id,
                    "EventTemplate": template_map.get(event_id, "(template text unavailable)"),
                    "_max_abs_shap": 0.0,
                },
            )
            entry["_max_abs_shap"] = max(entry["_max_abs_shap"], abs(float(value)))

    if not rows:
        return pd.DataFrame(columns=["Feature", "Event ID", "EventTemplate"])

    df = pd.DataFrame(rows.values()).sort_values("_max_abs_shap", ascending=False).head(top_n)
    return df[["Feature", "Event ID", "EventTemplate"]].reset_index(drop=True)


def top_positive_contributor_label(explanation: Optional[Dict], template_map: Dict[str, str]) -> Optional[str]:
    """The single feature pushing this block's anomaly score up the most, or None."""
    if not explanation:
        return None
    from utils.feature_labels import friendly_feature_label

    values = explanation["shap_values"]
    idx = int(np.argmax(values))
    if values[idx] <= 0:
        return None
    return friendly_feature_label(explanation["feature_names"][idx], template_map)
