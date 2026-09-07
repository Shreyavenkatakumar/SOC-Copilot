"""
Read-only EventId -> EventTemplate text lookup.

Maps the ``tmpl_E<n>_count`` feature columns (and any ``EventId`` shown
in per-block log evidence) back to the actual human-readable Drain3
template text -- e.g. ``tmpl_E1_count`` -> ``E1`` ->
"Receiving block <BlockId> src: /<IP_PORT> dest: /<IP_PORT>".

Never re-mines or re-trains Drain3. Two read-only sources are tried, in
order of completeness:

  1. The persisted Drain3 tree snapshot (``outputs/models/drain3_state.bin``),
     loaded via the ``drain3`` library's own file-persistence loader --
     this is the authoritative source and covers all 48 templates.
  2. ``outputs/reports/parsing_report.json``'s ``most_frequent_templates``
     list, which only covers the top-N templates by frequency but
     requires no extra dependency -- used only if (1) is unavailable.

If neither source is available, an empty mapping is returned and
callers must show the EventId with a clear "template text unavailable"
note (never invent template text).
"""

from typing import Dict

import streamlit as st

from utils.data_loader import DRAIN3_CONFIG_FILE, DRAIN3_STATE_FILE, load_parsing_report


@st.cache_resource(show_spinner=False)
def _load_template_map_from_drain3_state() -> Dict[str, str]:
    """
    Load the full EventId -> EventTemplate mapping directly from the
    trained Drain3 tree snapshot, via the same ``drain3`` library the
    production parser (``parsers/parser.py``) uses. Read-only: only
    ``TemplateMiner``'s persistence loader is used, nothing is mined or
    added to the tree.
    """
    if not DRAIN3_STATE_FILE.exists():
        return {}
    try:
        from drain3 import TemplateMiner
        from drain3.file_persistence import FilePersistence
        from drain3.template_miner_config import TemplateMinerConfig
    except ImportError:
        return {}

    try:
        config = TemplateMinerConfig()
        if DRAIN3_CONFIG_FILE.exists():
            config.load(str(DRAIN3_CONFIG_FILE))
            config.parameter_extraction_cache_capacity = int(
                config.parameter_extraction_cache_capacity
            )
        persistence_handler = FilePersistence(str(DRAIN3_STATE_FILE))
        miner = TemplateMiner(persistence_handler=persistence_handler, config=config)

        mapping: Dict[str, str] = {}
        for cluster in miner.drain.clusters:
            event_id = f"E{cluster.cluster_id}"
            mapping[event_id] = cluster.get_template()
        return mapping
    except (OSError, ValueError, KeyError, AttributeError):
        return {}


@st.cache_data(show_spinner=False)
def _load_template_map_from_parsing_report() -> Dict[str, str]:
    """Fallback: the (partial, top-N) mapping already saved in the parsing report."""
    report = load_parsing_report()
    if not report or not report.get("most_frequent_templates"):
        return {}
    return {
        row["event_id"]: row["template"]
        for row in report["most_frequent_templates"]
        if row.get("event_id") and row.get("template")
    }


def get_event_template_map() -> Dict[str, str]:
    """
    Best-available EventId -> EventTemplate mapping, preferring the
    complete Drain3-state-derived map and falling back to the partial
    parsing-report map. Never invents template text for an EventId that
    isn't found in either source.
    """
    mapping = _load_template_map_from_drain3_state()
    if mapping:
        return mapping
    return _load_template_map_from_parsing_report()


def template_feature_to_event_id(feature_name: str) -> str:
    """``"tmpl_E12_count"`` -> ``"E12"``. Returns the input unchanged if it doesn't match."""
    if feature_name.startswith("tmpl_") and feature_name.endswith("_count"):
        return feature_name[len("tmpl_"):-len("_count")]
    return feature_name


def describe_feature(feature_name: str, template_map: Dict[str, str]) -> str:
    """
    Human-readable label for a feature name in SHAP/behavioural tables:
    template features get their actual EventTemplate text appended;
    everything else is shown as-is.
    """
    if feature_name.startswith("tmpl_") and feature_name.endswith("_count"):
        event_id = template_feature_to_event_id(feature_name)
        template_text = template_map.get(event_id)
        if template_text:
            return f"{feature_name} ({event_id}: {template_text})"
        return f"{feature_name} ({event_id}: template text unavailable)"
    return feature_name
