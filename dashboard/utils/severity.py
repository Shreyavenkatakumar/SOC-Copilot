"""
SOC Copilot Dashboard -- presentation-layer severity classification.

IMPORTANT: severity is a DISPLAY concept only. It never changes,
overrides, or feeds back into the ML anomaly decision. The ML decision
(Normal / Anomaly) always comes directly from ``fusion_prediction`` in
outputs/reports/fusion_predictions.csv, produced by the finalized
production pipeline (models/fusion_engine.py). This module only maps
the already-computed ``fusion_score`` onto a human-readable severity
label for analyst triage.

============================================================
SEVERITY BOUNDARIES (the one place they are defined)
============================================================
fusion_score is a weighted blend (0.10 IF-v2 + 0.90 AE) of two
validation-calibrated, isotonic-regression scores, each roughly a
pseudo-probability in [0, 1]. The production ML decision threshold is
0.323459 (anything at or above this is predicted "Anomaly").

    fusion_score >= 0.75              -> CRITICAL
    0.323459 <= fusion_score < 0.75   -> HIGH        (ML: Anomaly)
    0.15 <= fusion_score < 0.323459   -> MEDIUM       (ML: Normal, but
                                                        worth a second look)
    fusion_score < 0.15               -> LOW          (ML: Normal)

CRITICAL and HIGH both correspond to ML Decision = Anomaly (fusion_score
>= 0.323459); the CRITICAL/HIGH split exists purely to help an analyst
prioritize among Anomaly alerts. MEDIUM and LOW both correspond to ML
Decision = Normal; MEDIUM exists to flag blocks with meaningful anomaly
evidence that nonetheless fell just short of the production threshold.
"""

from dataclasses import dataclass
from typing import Tuple

import pandas as pd

CRITICAL_THRESHOLD = 0.75
HIGH_THRESHOLD = 0.323459  # == production ML decision threshold
MEDIUM_THRESHOLD = 0.15

SEVERITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

SEVERITY_COLORS = {
    "CRITICAL": "#ff3b5c",
    "HIGH": "#ff9f1c",
    "MEDIUM": "#ffd23f",
    "LOW": "#4dd4ac",
}

SEVERITY_DESCRIPTIONS = {
    "CRITICAL": "Very high fusion score with strong combined anomaly evidence from both models.",
    "HIGH": "Fusion score at or above the production decision threshold -- classified as an anomaly.",
    "MEDIUM": "Below the production decision threshold, but with moderate anomaly evidence worth a second look.",
    "LOW": "Weak anomaly evidence; consistent with normal HDFS block behaviour.",
}


@dataclass
class SeverityConfig:
    critical_threshold: float = CRITICAL_THRESHOLD
    high_threshold: float = HIGH_THRESHOLD
    medium_threshold: float = MEDIUM_THRESHOLD


def classify_severity(fusion_score: float, config: SeverityConfig = SeverityConfig()) -> str:
    """Map a single fusion_score to a severity label. Display only."""
    if fusion_score is None or pd.isna(fusion_score):
        return "LOW"
    if fusion_score >= config.critical_threshold:
        return "CRITICAL"
    if fusion_score >= config.high_threshold:
        return "HIGH"
    if fusion_score >= config.medium_threshold:
        return "MEDIUM"
    return "LOW"


def add_severity_column(df: pd.DataFrame, score_column: str = "fusion_score") -> pd.DataFrame:
    """Return a copy of ``df`` with a ``severity`` column added. Does not mutate the input."""
    if score_column not in df.columns:
        return df
    out = df.copy()
    out["severity"] = out[score_column].apply(classify_severity)
    return out


def severity_badge_html(severity: str) -> str:
    color = SEVERITY_COLORS.get(severity, "#8a94a6")
    return (
        f'<span style="background-color:{color}22;color:{color};'
        f'border:1px solid {color};padding:2px 10px;border-radius:12px;'
        f'font-weight:600;font-size:0.78rem;letter-spacing:0.03em;">{severity}</span>'
    )


def decision_badge_html(decision: str) -> str:
    is_anomaly = str(decision).strip().lower() == "anomaly"
    color = "#ff3b5c" if is_anomaly else "#4dd4ac"
    label = "ANOMALY" if is_anomaly else "NORMAL"
    return (
        f'<span style="background-color:{color}22;color:{color};'
        f'border:1px solid {color};padding:2px 10px;border-radius:12px;'
        f'font-weight:700;font-size:0.78rem;letter-spacing:0.05em;">{label}</span>'
    )


# Lower number = higher priority; used to sort the Alerts queue
# (Critical -> High -> Medium -> Low, highest fusion score first within
# each severity band). Display-layer only, same as severity itself.
PRIORITY_RANK = {"CRITICAL": 1, "HIGH": 2, "MEDIUM": 3, "LOW": 4}

TRIAGE_STATUS = {
    "CRITICAL": "Needs Investigation",
    "HIGH": "Needs Investigation",
    "MEDIUM": "Monitor",
    "LOW": "No Action Needed",
}


SEVERITY_EMOJI = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}


def severity_emoji_label(severity: str) -> str:
    """Compact text indicator for use inside plain st.dataframe cells (e.g. the Alerts table)."""
    return f"{SEVERITY_EMOJI.get(severity, '⚪')} {severity}"


def priority_rank(severity: str) -> int:
    return PRIORITY_RANK.get(severity, 99)


def triage_status(severity: str) -> str:
    """
    A derived triage suggestion based purely on the already-computed
    SOC severity -- NOT a persisted ticketing/workflow state (this
    pipeline has no ticketing system to read from).
    """
    return TRIAGE_STATUS.get(severity, "Unknown")
