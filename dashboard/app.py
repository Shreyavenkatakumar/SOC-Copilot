"""
SOC Copilot -- SOC Analyst Dashboard (Phase 1).

Strictly a visualization / investigation layer over the finalized,
already-trained production pipeline. This file NEVER trains, retrains,
fits, or recalculates any model, scaler, or calibrator, and never
imports the retired original Isolation Forest
(models/isolation_forest_model.py). It only reads the output files
already produced by ``python main.py``.

Run with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from components import charts
from utils import data_loader as dl
from utils import explainability as expl
from utils.feature_labels import friendly_feature_label
from utils.severity import (
    add_severity_column,
    decision_badge_html,
    priority_rank,
    severity_badge_html,
    severity_emoji_label,
    triage_status,
)

st.set_page_config(
    page_title="SOC Copilot",
    page_icon=":shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    css_path = Path(__file__).resolve().parent / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


load_css()

PRODUCTION_THRESHOLD = 0.323459

PAGES = ["SOC Overview", "Alerts", "Alert Investigation", "Analytics", "Model & System Evidence"]
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = PAGES[0]
if "selected_block" not in st.session_state:
    st.session_state["selected_block"] = None


def missing_file_banner(message: str) -> None:
    st.markdown(f'<div class="warning-banner">{message}</div>', unsafe_allow_html=True)


def get_selection_rows(dataframe_event) -> list:
    """Defensive access to st.dataframe's selection state across Streamlit versions."""
    if dataframe_event is None:
        return []
    selection = getattr(dataframe_event, "selection", None)
    if selection is None and isinstance(dataframe_event, dict):
        selection = dataframe_event.get("selection")
    if selection is None:
        return []
    if isinstance(selection, dict):
        return selection.get("rows", [])
    return getattr(selection, "rows", [])


# ---------------------------------------------------------------------------
# Sidebar -- navigation + filters
# ---------------------------------------------------------------------------

st.sidebar.markdown('<div class="soc-title">SOC COPILOT</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="soc-subtitle">HDFS Anomaly Detection &mdash; Phase 1</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.radio("Navigation", PAGES, key="nav_page", label_visibility="collapsed")
page = st.session_state["nav_page"]

st.sidebar.markdown("---")
st.sidebar.markdown("**Filters**")

status = dl.file_status()
master_df = dl.build_master_table()

if master_df is not None:
    master_df = add_severity_column(master_df, "fusion_score")

    severity_filter = st.sidebar.multiselect(
        "Severity", ["CRITICAL", "HIGH", "MEDIUM", "LOW"], default=["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    )
    decision_filter = st.sidebar.multiselect(
        "ML Decision", ["Anomaly", "Normal"], default=["Anomaly", "Normal"]
    )
    score_range = st.sidebar.slider("Fusion score range", 0.0, 1.0, (0.0, 1.0), step=0.01)
    block_search = st.sidebar.text_input("Block ID search", placeholder="blk_-...")
    max_alerts = st.sidebar.slider("Max alerts displayed", 10, 2000, 200, step=10)

    filtered_df = master_df[
        master_df["severity"].isin(severity_filter)
        & master_df["fusion_prediction"].isin(decision_filter)
        & master_df["fusion_score"].between(score_range[0], score_range[1])
    ]
    if block_search:
        filtered_df = filtered_df[filtered_df["BlockId"].str.contains(block_search, na=False)]
else:
    filtered_df = None
    severity_filter = decision_filter = []
    score_range = (0.0, 1.0)
    block_search = ""
    max_alerts = 200

st.sidebar.markdown("---")


# ---------------------------------------------------------------------------
# PAGE 1 -- SOC Overview
# ---------------------------------------------------------------------------

def page_soc_overview() -> None:
    st.markdown('<div class="section-header">SOC Overview</div>', unsafe_allow_html=True)

    if master_df is None:
        missing_file_banner(
            "Required prediction output not found "
            f"(<code>{dl.FUSION_PREDICTIONS_FILE.name}</code>). Run <code>python main.py</code> first."
        )
        return

    # SOC Overview KPIs reflect the trained model's actual scored output
    # (fusion_predictions.csv via master_df), NOT ground-truth labels.
    # This keeps Normal/Anomalous counts consistent with the severity bar,
    # pie chart, and Alerts queue, which all read from fusion_prediction.
    total = len(master_df)
    normal = int((master_df["fusion_prediction"] == "Normal").sum())
    anomaly = int((master_df["fusion_prediction"] == "Anomaly").sum())
    anomaly_rate = (anomaly / total * 100) if total else None

    critical_count = int((master_df["severity"] == "CRITICAL").sum())
    high_count = int((master_df["severity"] == "HIGH").sum())
    needs_investigation = critical_count + high_count

    st.markdown(
        '<div class="evidence-box"><b>SYSTEM STATUS</b> &nbsp;&nbsp; '
        '<span style="color:#4dd4ac;font-size:1.1rem;">&#9679;</span> Operational &mdash; '
        f"pipeline outputs loaded, {total:,} blocks scored.</div>",
        unsafe_allow_html=True,
    )

    kpi_cols = st.columns(6)
    kpi_defs = [
        ("Total Blocks", f"{total:,}" if total else "N/A", ""),
        ("Normal Blocks", f"{normal:,}" if normal is not None else "N/A", "good"),
        ("Anomalous Blocks", f"{anomaly:,}" if anomaly is not None else "N/A", "critical"),
        ("Critical Alerts", f"{critical_count:,}", "critical"),
        ("High Alerts", f"{high_count:,}", "warn"),
        ("Needs Investigation", f"{needs_investigation:,}", "critical"),
    ]
    for col, (label, value, cls) in zip(kpi_cols, kpi_defs):
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value {cls}">{value}</div></div>',
            unsafe_allow_html=True,
        )
    if anomaly_rate is not None:
        st.caption(f"Anomaly rate: **{anomaly_rate:.2f}%** of all scored blocks.")

    st.markdown('<div class="section-header">Severity &amp; Trend</div>', unsafe_allow_html=True)
    trend_cols = st.columns(2)
    trend_cols[0].plotly_chart(charts.severity_distribution_bar(master_df), use_container_width=True)

    first_seen_df = dl.get_block_first_seen_all()
    if first_seen_df is not None:
        merged = (
            master_df[["BlockId", "fusion_prediction"]]
            .merge(first_seen_df, on="BlockId", how="inner")
            .dropna(subset=["first_seen"])
        )
        if not merged.empty:
            merged["bucket"] = merged["first_seen"].dt.floor("h")
            trend = merged.groupby(["bucket", "fusion_prediction"]).size().unstack(fill_value=0).reset_index()
            trend_cols[1].plotly_chart(charts.anomaly_trend_line(trend), use_container_width=True)
        else:
            trend_cols[1].info("No parseable timestamps were found to build a trend chart.")
    else:
        trend_cols[1].info(
            "Anomaly trend requires the parsed-log index (built automatically the first time "
            "Alerts or Alert Investigation is opened)."
        )

    st.markdown('<div class="section-header">Alert Distribution</div>', unsafe_allow_html=True)
    dist_cols = st.columns(2)
    dist_cols[0].plotly_chart(charts.normal_vs_anomaly_pie(master_df), use_container_width=True)
    dist_cols[1].plotly_chart(
        charts.score_histogram(master_df, "fusion_score", "Fusion Score Distribution", "#4dd4ac"),
        use_container_width=True,
    )

    score_cols = st.columns(2)
    if "autoencoder_anomaly_score" in master_df.columns:
        score_cols[0].plotly_chart(
            charts.score_histogram(master_df, "autoencoder_anomaly_score", "AE Score Distribution", "#5aa9ff"),
            use_container_width=True,
        )
    if "if_v2_anomaly_score" in master_df.columns:
        score_cols[1].plotly_chart(
            charts.score_histogram(master_df, "if_v2_anomaly_score", "IF-v2 Score Distribution", "#ff9f1c"),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# PAGE 2 -- Alerts
# ---------------------------------------------------------------------------

def page_alerts() -> None:
    st.markdown('<div class="section-header">Alerts</div>', unsafe_allow_html=True)

    if filtered_df is None or filtered_df.empty:
        missing_file_banner("No alerts available with the current filters, or predictions are missing.")
        return

    working_df = filtered_df

    with st.expander("Advanced filters (Event ID, first-seen time range)"):
        template_map = dl.load_event_template_map()
        if template_map:
            event_options = sorted(template_map.keys(), key=lambda e: int(e[1:]) if e[1:].isdigit() else 0)
            event_filter = st.multiselect("Block contains Event ID", event_options)
            if event_filter:
                mask = pd.Series(False, index=working_df.index)
                for eid in event_filter:
                    col_name = f"tmpl_{eid}_count"
                    if col_name in working_df.columns:
                        mask = mask | (working_df[col_name] > 0)
                working_df = working_df[mask]
        else:
            st.caption("Event ID filter requires the parsed-log index (built automatically on first use).")

        first_seen_df = dl.get_block_first_seen_all()
        if first_seen_df is not None:
            working_df = working_df.merge(first_seen_df, on="BlockId", how="left")
            valid_dates = working_df["first_seen"].dropna()
            if not valid_dates.empty:
                min_d, max_d = valid_dates.min().to_pydatetime(), valid_dates.max().to_pydatetime()
                if min_d < max_d:
                    date_range = st.slider(
                        "First-seen time range", min_value=min_d, max_value=max_d, value=(min_d, max_d)
                    )
                    working_df = working_df[
                        working_df["first_seen"].isna()
                        | working_df["first_seen"].between(date_range[0], date_range[1])
                    ]
        else:
            st.caption("Date/time filter requires the parsed-log index (built automatically on first use).")

    working_df = working_df.copy()
    working_df["Priority"] = working_df["severity"].map(priority_rank)
    working_df["Status"] = working_df["severity"].map(triage_status)
    working_df["SeverityDisplay"] = working_df["severity"].map(severity_emoji_label)
    working_df = working_df.sort_values(["Priority", "fusion_score"], ascending=[True, False])

    rename_map = {
        "BlockId": "Block ID",
        "SeverityDisplay": "Severity",
        "fusion_prediction": "Decision",
        "fusion_score": "Fusion Score",
        "first_seen": "Timestamp",
    }
    base_cols = [c for c in ["Priority", "BlockId", "SeverityDisplay", "fusion_prediction", "fusion_score", "first_seen", "Status"] if c in working_df.columns]
    table = working_df.head(max_alerts)[base_cols].rename(columns=rename_map)

    st.caption(
        f"Showing top {len(table):,} of {len(working_df):,} matching alerts, "
        "sorted by priority (Critical -> High -> Medium -> Low), then highest fusion score."
    )

    column_config = {}
    if "Fusion Score" in table.columns:
        column_config["Fusion Score"] = st.column_config.NumberColumn(format="%.6f")
    if "Timestamp" in table.columns:
        column_config["Timestamp"] = st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss")

    event = st.dataframe(
        table,
        use_container_width=True,
        height=420,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config=column_config,
    )

    selected_rows = get_selection_rows(event)
    if selected_rows:
        picked_block = table.iloc[selected_rows[0]]["Block ID"]
        st.success(f"Selected alert: `{picked_block}`")
        if st.button(f"Investigate {picked_block} \u2192", type="primary"):
            st.session_state["selected_block"] = picked_block
            st.session_state["nav_page"] = "Alert Investigation"
            st.rerun()
    else:
        st.caption("Select a row in the table above, then click Investigate to open Alert Investigation.")


# ---------------------------------------------------------------------------
# PAGE 3 -- Alert Investigation
# ---------------------------------------------------------------------------

def page_alert_investigation() -> None:
    st.markdown('<div class="section-header">Alert Investigation</div>', unsafe_allow_html=True)

    if master_df is None:
        missing_file_banner("Predictions not found. Run <code>python main.py</code> first.")
        return

    selected_block = st.session_state.get("selected_block")
    with st.expander("Change Block ID", expanded=not selected_block):
        manual = st.text_input("Block ID", value=selected_block or "", placeholder="blk_-...")
        if manual:
            selected_block = manual
            st.session_state["selected_block"] = manual

    if not selected_block:
        st.info("Select an alert on the Alerts page, or enter a Block ID above, to begin an investigation.")
        return

    row_match = master_df[master_df["BlockId"] == selected_block]
    if row_match.empty:
        missing_file_banner("Selected Block ID was not found in the prediction output.")
        return
    row = row_match.iloc[0]

    ae_score = row.get("autoencoder_anomaly_score", 0)
    if_score = row.get("if_v2_anomaly_score", 0)
    fusion_score = row.get("fusion_score", 0)
    fusion_prediction = row.get("fusion_prediction")
    severity = row.get("severity")
    template_map = dl.load_event_template_map()

    block_logs = dl.get_block_raw_logs(selected_block) if status["parsed_logs"] else None
    first_seen_display = None
    if block_logs is not None and not block_logs.empty:
        parsed_ts = pd.to_datetime(block_logs["Timestamp"], format="%y%m%d%H%M%S", errors="coerce")
        if parsed_ts.notna().any():
            first_seen_display = parsed_ts.min()

    # -- A. Alert Summary ---------------------------------------------------
    st.markdown('<div class="section-header">A. Alert Summary</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(f"**Block ID**  \n`{selected_block}`")
    s1.markdown(f"**ML Decision:** {decision_badge_html(fusion_prediction)}", unsafe_allow_html=True)
    s2.markdown(f"**SOC Severity:** {severity_badge_html(severity)}", unsafe_allow_html=True)
    if first_seen_display is not None:
        s2.markdown(f"**Timestamp (first seen):**  \n{first_seen_display}")
    s3.metric("Fusion Score", f"{fusion_score:.6f}")
    s4.metric("Production Threshold", f"{PRODUCTION_THRESHOLD}")

    # -- B. Why Was This Block Flagged? --------------------------------------
    ae_explanation = expl.explain_autoencoder(selected_block)
    ifv2_explanation = expl.explain_isolation_forest_v2(selected_block)
    top_ae_feature = expl.top_positive_contributor_label(ae_explanation, template_map)
    top_ifv2_feature = expl.top_positive_contributor_label(ifv2_explanation, template_map)

    st.markdown('<div class="section-header">B. Why Was This Block Flagged?</div>', unsafe_allow_html=True)
    relation = "at or above" if fusion_score >= PRODUCTION_THRESHOLD else "below"
    narrative = [
        f"Block <b>{selected_block}</b> was classified as <b>{fusion_prediction}</b> "
        f"(SOC severity <b>{severity}</b>). Its fusion score of <b>{fusion_score:.4f}</b> is "
        f"{relation} the production threshold of <b>{PRODUCTION_THRESHOLD}</b>.",
        f"The Autoencoder identified deviation from learned normal behaviour (score {ae_score:.4f}), "
        f"while IF-v2 identified statistical isolation in its selected feature space "
        f"(score {if_score:.4f}).",
    ]
    if top_ae_feature:
        narrative.append(f"The Autoencoder's strongest evidence came from <b>{top_ae_feature}</b>.")
    if top_ifv2_feature:
        narrative.append(f"IF-v2's strongest evidence came from <b>{top_ifv2_feature}</b>.")
    st.markdown(f'<div class="evidence-box">{" ".join(narrative)}</div>', unsafe_allow_html=True)
    st.caption(
        "Model explanations describe feature contribution to anomaly evidence; they do not prove root cause."
    )

    # -- C. Model Evidence ---------------------------------------------------
    st.markdown('<div class="section-header">C. Model Evidence</div>', unsafe_allow_html=True)
    ev_cols = st.columns(3)
    ev_cols[0].metric("Autoencoder Score", f"{ae_score:.6g}")
    ev_cols[1].metric("IF-v2 Score", f"{if_score:.6f}")
    ev_cols[2].metric("Fusion Score", f"{fusion_score:.6f}")

    # -- D. SHAP Explainability -----------------------------------------------
    st.markdown('<div class="section-header">D. SHAP Explainability</div>', unsafe_allow_html=True)
    st.caption(
        "SHAP shows how much each input feature pushed this block's anomaly score up (red, stronger "
        "anomaly evidence) or down (green, weaker anomaly evidence) relative to a typical block."
    )

    def _shap_feature_table(explanation) -> pd.DataFrame:
        labels = [friendly_feature_label(f, template_map) for f in explanation["feature_names"]]
        df = pd.DataFrame(
            {
                "Feature": labels,
                "Feature Value": explanation["feature_values"],
                "SHAP Contribution": explanation["shap_values"],
            }
        )
        return df.reindex(df["SHAP Contribution"].abs().sort_values(ascending=False).index).head(10).reset_index(drop=True)

    shap_tabs = st.tabs(["Autoencoder", "IF-v2"])
    with shap_tabs[0]:
        if ae_explanation is not None:
            ae_labels = [friendly_feature_label(f, template_map) for f in ae_explanation["feature_names"]]
            c1, c2 = st.columns(2)
            c1.plotly_chart(
                charts.shap_contribution_bar(ae_labels, ae_explanation["shap_values"], "Autoencoder -- Top Contributors"),
                use_container_width=True,
            )
            c2.plotly_chart(
                charts.shap_waterfall(
                    ae_labels, ae_explanation["shap_values"], ae_explanation["base_value"],
                    ae_explanation["predicted_value"], "Autoencoder -- Waterfall",
                ),
                use_container_width=True,
            )
            st.dataframe(_shap_feature_table(ae_explanation), use_container_width=True, hide_index=True)
        else:
            st.info("Autoencoder SHAP explanation unavailable (model, metadata, or feature file not found).")

    with shap_tabs[1]:
        if ifv2_explanation is not None:
            ifv2_labels = [friendly_feature_label(f, template_map) for f in ifv2_explanation["feature_names"]]
            c1, c2 = st.columns(2)
            c1.plotly_chart(
                charts.shap_contribution_bar(ifv2_labels, ifv2_explanation["shap_values"], "IF-v2 -- Top Contributors"),
                use_container_width=True,
            )
            c2.plotly_chart(
                charts.shap_waterfall(
                    ifv2_labels, ifv2_explanation["shap_values"], ifv2_explanation["base_value"],
                    ifv2_explanation["predicted_value"], "IF-v2 -- Waterfall",
                ),
                use_container_width=True,
            )
            st.dataframe(_shap_feature_table(ifv2_explanation), use_container_width=True, hide_index=True)
        else:
            st.info("IF-v2 SHAP explanation unavailable (model bundle or feature scaler not found).")

    # -- E. Drain3 Feature Mapping --------------------------------------------
    st.markdown('<div class="section-header">E. Drain3 Feature Mapping</div>', unsafe_allow_html=True)
    mapping_df = expl.drain3_mapping_table(ae_explanation, ifv2_explanation, template_map)
    if mapping_df.empty:
        st.caption("No template-frequency features appear among this block's top SHAP contributors.")
    else:
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)

    # -- F. Behavioural Features -----------------------------------------------
    st.markdown('<div class="section-header">F. Behavioural Features</div>', unsafe_allow_html=True)
    NON_FEATURE_COLS = {
        "BlockId", "Label", "severity", "fusion_score", "fusion_prediction",
        "autoencoder_anomaly_score", "autoencoder_prediction", "autoencoder_reconstruction_error",
        "autoencoder_calibrated_score", "if_v2_anomaly_score", "if_v2_prediction",
        "if_v2_calibrated_score",
    }
    if ae_explanation is not None:
        feature_cols = [c for c in ae_explanation["feature_names"] if c in row.index]
    else:
        feature_cols = [c for c in row.index if c not in NON_FEATURE_COLS]

    if feature_cols:
        feat_cols = st.columns(4)
        for i, col_name in enumerate(feature_cols):
            value = row[col_name]
            display = f"{value:,.2f}" if isinstance(value, float) else str(value)
            feat_cols[i % 4].markdown(
                f'<div class="kpi-card"><div class="kpi-label">{friendly_feature_label(col_name, template_map)}</div>'
                f'<div class="kpi-value">{display}</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No behavioural feature values are available for this block.")

    with st.expander("View all features as a table"):
        all_df = pd.DataFrame(
            {
                "Feature": [friendly_feature_label(c, template_map) for c in feature_cols],
                "Value": [float(row[c]) if c in row and pd.notna(row[c]) else None for c in feature_cols],
            }
        )
        st.dataframe(all_df, use_container_width=True, hide_index=True, height=320)

    # -- G. Block Event-Template Analysis ---------------------------------------
    st.markdown('<div class="section-header">G. Block Event-Template Analysis</div>', unsafe_allow_html=True)
    if block_logs is None or block_logs.empty:
        missing_file_banner("No parsed log records could be mapped to this Block ID.")
    else:
        freq = block_logs["EventId"].value_counts().reset_index()
        freq.columns = ["Event ID", "Occurrence Count"]
        total_events = freq["Occurrence Count"].sum()
        freq["EventTemplate"] = freq["Event ID"].map(template_map).fillna("(template text unavailable)")
        freq["Percentage of Block Events"] = (freq["Occurrence Count"] / total_events * 100).round(2)
        freq = freq.sort_values("Occurrence Count", ascending=False)

        chart_freq = freq.rename(columns={"Occurrence Count": "count"}).copy()
        chart_freq["label"] = chart_freq.apply(lambda r: f"{r['Event ID']}: {r['EventTemplate'][:60]}", axis=1)
        st.plotly_chart(charts.block_template_frequency_bar(chart_freq), use_container_width=True)
        st.dataframe(
            freq[["Event ID", "EventTemplate", "Occurrence Count", "Percentage of Block Events"]],
            use_container_width=True,
            hide_index=True,
        )

    # -- H. Block Activity Timeline ------------------------------------------
    st.markdown('<div class="section-header">H. Block Activity Timeline</div>', unsafe_allow_html=True)
    if block_logs is None or block_logs.empty:
        missing_file_banner("No parsed log records available to build a timeline for this Block ID.")
    else:
        timeline_df = dl.attach_timeline_columns(block_logs)
        st.plotly_chart(charts.block_activity_timeline(timeline_df), use_container_width=True)

    # -- I. Supporting Log Evidence -------------------------------------------
    st.markdown('<div class="section-header">I. Supporting Log Evidence</div>', unsafe_allow_html=True)
    if not status["parsed_logs"]:
        missing_file_banner(f"Parsed logs file not found (<code>{dl.PARSED_LOGS_FILE.name}</code>).")
    elif block_logs is None or block_logs.empty:
        st.info("No parsed log records could be mapped to this Block ID.")
    else:
        with st.expander(f"Show all {len(block_logs):,} parsed log events for this block", expanded=False):
            st.dataframe(
                block_logs[["Timestamp", "Component", "EventId", "EventTemplate", "Content"]],
                use_container_width=True,
                height=320,
            )


# ---------------------------------------------------------------------------
# PAGE 4 -- Analytics
# ---------------------------------------------------------------------------

def page_analytics() -> None:
    st.markdown('<div class="section-header">Analytics</div>', unsafe_allow_html=True)

    if master_df is None:
        missing_file_banner("Predictions not found. Run <code>python main.py</code> first.")
        return

    view = st.radio("Filter", ["All", "Normal", "Anomaly"], horizontal=True)
    view_df = master_df if view == "All" else master_df[master_df["fusion_prediction"] == view]

    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.top_suspicious_bar(master_df, n=15), use_container_width=True)
    if {"autoencoder_anomaly_score", "if_v2_anomaly_score"}.issubset(view_df.columns):
        c2.plotly_chart(charts.scatter_if_vs_ae(view_df.sample(min(len(view_df), 20000))), use_container_width=True)

    st.markdown('<div class="section-header">Behavioural Analysis</div>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    if "warn_count" in view_df.columns:
        b1.plotly_chart(
            charts.score_histogram(view_df, "warn_count", "Warning-Event Count", "#ff9f1c"),
            use_container_width=True,
        )
    if "event_density" in view_df.columns:
        b2.plotly_chart(
            charts.score_histogram(view_df, "event_density", "Event Density", "#5aa9ff"),
            use_container_width=True,
        )
    if "time_duration_seconds" in view_df.columns:
        b3.plotly_chart(
            charts.score_histogram(view_df, "time_duration_seconds", "Block Duration (s)", "#4dd4ac"),
            use_container_width=True,
        )

    template_cols = [c for c in view_df.columns if c.startswith("tmpl_")]
    if template_cols:
        st.markdown('<div class="section-header">Event-Template Analytics (selected view)</div>', unsafe_allow_html=True)
        totals = view_df[template_cols].sum().sort_values(ascending=False).head(15)
        st.bar_chart(totals)

    st.caption(
        "A single dataset-wide timeline isn't meaningful here since every block has its own "
        "independent time range; open a block in Alert Investigation for its exact "
        "chronological Block Activity Timeline."
    )
    st.caption(
        "Top Suspicious Users / IPs are not shown: the HDFS block-level feature set produced by "
        "this pipeline contains no user or IP identity fields, and this dashboard never invents data."
    )


# ---------------------------------------------------------------------------
# PAGE 5 -- Model & System Evidence
# ---------------------------------------------------------------------------

def page_model_system_evidence() -> None:
    st.markdown('<div class="section-header">Model &amp; System Evidence</div>', unsafe_allow_html=True)
    st.caption("Architecture, validated performance, and configuration -- for mentor/viva/research review.")

    fusion_config = dl.load_fusion_config()
    ifv2_meta = dl.load_ifv2_model_metadata()

    st.markdown('<div class="section-header">Architecture</div>', unsafe_allow_html=True)
    pipeline_cols = st.columns(4)
    steps = [
        "IF + AE", "Fusion", "Severity", "SHAP",
    ]
    for col, step in zip(pipeline_cols, steps):
        col.markdown(f'<div class="pipeline-step">{step}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Performance (test set, ablation view)</div>', unsafe_allow_html=True)
    ae_eval = dl.load_autoencoder_evaluation()
    ifv2_report = dl.load_ifv2_report()
    fusion_eval = dl.load_fusion_evaluation()

    def metrics_block(title, eval_dict, metrics_source):
        st.markdown(f"**{title}**")
        if eval_dict is None:
            missing_file_banner("Evaluation report not found for this model.")
            return
        m_cols = st.columns(6)
        keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
        for col, key in zip(m_cols, keys):
            value = eval_dict.get(key) if metrics_source == "flat" else eval_dict.get("if_v2_test_metrics_uncalibrated", {}).get(key)
            col.metric(key.replace("_", " ").upper(), f"{value:.4f}" if isinstance(value, (int, float)) else "N/A")

    metrics_block("Autoencoder only", ae_eval, "flat")
    metrics_block("IF-v2 only", ifv2_report, "nested")
    metrics_block("Fusion (final production decision)", fusion_eval, "flat")

    if fusion_eval:
        cm = fusion_eval.get("confusion_matrix", [[0, 0], [0, 0]])
        tn, fp = cm[0]
        fn, tp = cm[1]
        fpr = fp / (fp + tn) if (fp + tn) else None
        fp_cols = st.columns(5)
        for col, (label, value) in zip(
            fp_cols,
            [
                ("False Positives", fp), ("False Negatives", fn),
                ("True Positives", tp), ("True Negatives", tn),
                ("False Positive Rate", f"{fpr:.4%}" if fpr is not None else "N/A"),
            ],
        ):
            col.markdown(
                f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div></div>',
                unsafe_allow_html=True,
            )
        st.caption(
            f"Evaluated on the untouched test set only ({fusion_eval.get('evaluated_blocks', 'N/A'):,} blocks) "
            "-- never on training or validation blocks."
        )

    st.markdown('<div class="section-header">Confusion Matrices (test set)</div>', unsafe_allow_html=True)
    cm_cols = st.columns(3)
    if ae_eval and ae_eval.get("confusion_matrix"):
        cm_cols[0].plotly_chart(charts.confusion_matrix_heatmap(ae_eval["confusion_matrix"], title="Autoencoder"), use_container_width=True)
    if ifv2_report and ifv2_report.get("if_v2_test_metrics_uncalibrated", {}).get("confusion_matrix"):
        cm_cols[1].plotly_chart(
            charts.confusion_matrix_heatmap(ifv2_report["if_v2_test_metrics_uncalibrated"]["confusion_matrix"], title="IF-v2"),
            use_container_width=True,
        )
    if fusion_eval and fusion_eval.get("confusion_matrix"):
        cm_cols[2].plotly_chart(charts.confusion_matrix_heatmap(fusion_eval["confusion_matrix"], title="Fusion"), use_container_width=True)

    st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)
    checklist = [
        ("Log Collection", status.get("labels", False)),
        ("Drain3 Parsing", status.get("parsed_logs", False)),
        ("Feature Extraction", status.get("features", False)),
        ("IF-v2", status.get("isolation_forest_v2_predictions", False)),
        ("Autoencoder", status.get("autoencoder_predictions", False)),
        ("Fusion", status.get("fusion_predictions", False)),
        ("Severity (display layer)", status.get("fusion_predictions", False)),
        ("SHAP (on-demand)", status.get("feature_scaler", False)),
        ("Dashboard", True),
    ]
    check_cols = st.columns(3)
    for i, (label, ok) in enumerate(checklist):
        mark = "\u2705" if ok else "\u274c"
        check_cols[i % 3].markdown(f"{mark} {label}")

    summary = dl.get_dataset_summary()
    stat_cols = st.columns(4)
    stat_defs = [
        ("HDFS Log Lines", summary.get("total_log_lines")),
        ("Drain3 Templates", summary.get("unique_templates")),
        ("Feature Count", summary.get("total_feature_columns")),
        (
            "Train / Val / Test",
            f"{fusion_config['training_blocks']:,} / {fusion_config['validation_blocks']:,} / {fusion_config['test_blocks']:,}"
            if fusion_config else "446,578 / 64,241 / 64,242",
        ),
    ]
    for col, (label, value) in zip(stat_cols, stat_defs):
        display = f"{value:,}" if isinstance(value, (int, float)) else (value or "N/A")
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{display}</div></div>',
            unsafe_allow_html=True,
        )

    with st.expander("Technical Details"):
        st.markdown("**IF-v2 Parameters**")
        st.code(
            "n_estimators=200, max_samples=0.5, max_features=1.0,\n"
            "bootstrap=False, random_state=42, n_jobs=-1",
            language="python",
        )
        st.markdown("**Autoencoder Architecture**")
        st.code(
            "Input(58) -> 32 -> 16 -> 8 (bottleneck) -> 16 -> 32 -> Output(58), "
            "ReLU hidden, sigmoid output, MSE loss",
            language="text",
        )
        if ifv2_meta and ifv2_meta.get("selected_features"):
            st.markdown("**IF-v2 Selected Features**")
            st.write(", ".join(ifv2_meta["selected_features"]))
        st.markdown("**Raw Pipeline Reports**")
        st.write("Log Collection:", dl.load_log_collection_report())
        st.write("Parsing:", dl.load_parsing_report())
        st.write("Feature Extraction:", dl.load_feature_extraction_report())


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

PAGE_FUNCTIONS = {
    "SOC Overview": page_soc_overview,
    "Alerts": page_alerts,
    "Alert Investigation": page_alert_investigation,
    "Analytics": page_analytics,
    "Model & System Evidence": page_model_system_evidence,
}

PAGE_FUNCTIONS[page]()
