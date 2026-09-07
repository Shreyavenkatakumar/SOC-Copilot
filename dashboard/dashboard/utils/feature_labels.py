"""
Human-readable labels for feature columns, used in SHAP explanations
and behavioural-feature displays. Presentation only -- never affects
any model input, feature order, or scaling.
"""

from typing import Dict

FRIENDLY_LABELS: Dict[str, str] = {
    "total_log_events": "Total log events",
    "unique_template_count": "Unique template count",
    "event_sequence_length": "Event sequence length",
    "time_duration_seconds": "Block duration (s)",
    "unique_component_count": "Unique component count",
    "event_density": "Event density",
    "info_count": "INFO-level count",
    "warn_count": "WARN-level count",
    "error_count": "ERROR-level count",
    "other_level_count": "Other-level count",
}

TEMPLATE_LABEL_MAX_LEN = 70


def friendly_feature_label(column: str, template_map: Dict[str, str]) -> str:
    """
    Map a raw feature column name to an analyst-friendly label.

    Template frequency columns (``tmpl_E<n>_count``) are resolved to
    their actual Drain3 ``EventTemplate`` text via ``template_map``
    (EventId -> EventTemplate), falling back to the raw EventId if the
    mapping is unavailable.
    """
    if column in FRIENDLY_LABELS:
        return FRIENDLY_LABELS[column]

    if column.startswith("tmpl_") and column.endswith("_count"):
        event_id = column[len("tmpl_") : -len("_count")]
        template = template_map.get(event_id)
        if template:
            short = (
                template
                if len(template) <= TEMPLATE_LABEL_MAX_LEN
                else template[: TEMPLATE_LABEL_MAX_LEN - 3] + "..."
            )
            return f"{event_id}: {short}"
        return f"{event_id} (template text unavailable)"

    return column
