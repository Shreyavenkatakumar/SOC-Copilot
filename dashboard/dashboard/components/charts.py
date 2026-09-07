"""Reusable Plotly chart builders for the SOC Copilot dashboard."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

DARK_LAYOUT = dict(
    paper_bgcolor="#0b0f19",
    plot_bgcolor="#0b0f19",
    font=dict(color="#c3ccd9", family="Segoe UI, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

SEVERITY_COLOR_MAP = {
    "CRITICAL": "#ff3b5c",
    "HIGH": "#ff9f1c",
    "MEDIUM": "#ffd23f",
    "LOW": "#4dd4ac",
}


def normal_vs_anomaly_pie(df: pd.DataFrame, column: str = "fusion_prediction") -> go.Figure:
    counts = df[column].value_counts()
    fig = go.Figure(
        data=[
            go.Pie(
                labels=counts.index,
                values=counts.values,
                hole=0.55,
                marker=dict(colors=["#4dd4ac", "#ff3b5c"]),
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(title="Normal vs. Anomaly (ML Decision)", **DARK_LAYOUT)
    return fig


def severity_distribution_bar(df: pd.DataFrame) -> go.Figure:
    order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    counts = df["severity"].value_counts().reindex(order).fillna(0)
    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index,
                y=counts.values,
                marker_color=[SEVERITY_COLOR_MAP[s] for s in counts.index],
                text=counts.values,
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Severity Distribution",
        xaxis_title="SOC Severity",
        yaxis_title="Block Count",
        **DARK_LAYOUT,
    )
    return fig


def score_histogram(df: pd.DataFrame, column: str, title: str, color: str = "#4dd4ac") -> go.Figure:
    fig = px.histogram(df, x=column, nbins=60, color_discrete_sequence=[color])
    fig.update_layout(
        title=title,
        xaxis_title="Score",
        yaxis_title="Block Count",
        bargap=0.02,
        **DARK_LAYOUT,
    )
    return fig


def scatter_if_vs_ae(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="autoencoder_anomaly_score",
        y="if_v2_anomaly_score",
        color="fusion_prediction",
        color_discrete_map={"Normal": "#4dd4ac", "Anomaly": "#ff3b5c"},
        opacity=0.5,
        hover_data=["BlockId"] if "BlockId" in df.columns else None,
    )
    fig.update_layout(
        title="Autoencoder Score vs. IF-v2 Score",
        xaxis_title="Autoencoder anomaly score",
        yaxis_title="IF-v2 anomaly score",
        **DARK_LAYOUT,
    )
    return fig


def top_suspicious_bar(df: pd.DataFrame, n: int = 15) -> go.Figure:
    top = df.nlargest(n, "fusion_score")[["BlockId", "fusion_score"]]
    fig = go.Figure(
        data=[
            go.Bar(
                x=top["fusion_score"],
                y=top["BlockId"],
                orientation="h",
                marker_color="#ff3b5c",
            )
        ]
    )
    fig.update_layout(
        title=f"Top {n} Most Suspicious Blocks (by fusion score)",
        xaxis_title="Fusion Score",
        yaxis=dict(autorange="reversed"),
        **DARK_LAYOUT,
    )
    return fig


def confusion_matrix_heatmap(cm, labels=("Normal", "Anomaly"), title: str = "Confusion Matrix") -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"Predicted {l}" for l in labels],
            y=[f"Actual {l}" for l in labels],
            colorscale=[[0, "#0e1420"], [1, "#4dd4ac"]],
            text=cm,
            texttemplate="%{text}",
            showscale=False,
        )
    )
    fig.update_layout(title=title, **DARK_LAYOUT)
    return fig


def shap_contribution_bar(feature_labels, shap_values, title: str, top_n: int = 10) -> go.Figure:
    """
    Horizontal bar chart of the top +/- SHAP contributions for a single
    block. Red = pushes the anomaly score up; green = pushes it down.
    """
    shap_values = list(shap_values)
    order = sorted(range(len(shap_values)), key=lambda i: abs(shap_values[i]), reverse=True)[:top_n]
    order = order[::-1]  # largest contribution at the top of the chart
    labels = [feature_labels[i] for i in order]
    values = [shap_values[i] for i in order]
    colors = ["#ff3b5c" if v > 0 else "#4dd4ac" for v in values]

    fig = go.Figure(
        data=[go.Bar(x=values, y=labels, orientation="h", marker_color=colors)]
    )
    fig.update_layout(
        title=title,
        xaxis_title="SHAP value (impact on anomaly score)",
        **DARK_LAYOUT,
    )
    return fig


def block_template_frequency_bar(freq_df: pd.DataFrame, title: str = "Block Event Template Frequency") -> go.Figure:
    """Horizontal bar chart of event-template occurrence counts within a single block."""
    fig = go.Figure(
        data=[
            go.Bar(
                x=freq_df["count"],
                y=freq_df["label"],
                orientation="h",
                marker_color="#5aa9ff",
                text=freq_df["count"],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Occurrences in this block",
        yaxis=dict(autorange="reversed"),
        **DARK_LAYOUT,
    )
    return fig


def block_activity_timeline(timeline_df: pd.DataFrame) -> go.Figure:
    """
    Chronological scatter of every parsed log event belonging to one
    block: x = actual timestamp, y = sequence position, colored by
    EventId, with Component/EventTemplate on hover.
    """
    fig = px.scatter(
        timeline_df,
        x="_ts",
        y="_step",
        color="EventId",
        hover_data=["Component", "EventTemplate"],
    )
    fig.add_trace(
        go.Scatter(
            x=timeline_df["_ts"],
            y=timeline_df["_step"],
            mode="lines",
            line=dict(color="#3a4560", width=1),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title="Block Activity Timeline (chronological event sequence)",
        xaxis_title="Timestamp",
        yaxis_title="Event sequence #",
        **DARK_LAYOUT,
    )
    return fig


def anomaly_trend_line(trend_df: pd.DataFrame) -> go.Figure:
    """Stacked-line trend of Normal vs. Anomaly block counts over time buckets."""
    fig = go.Figure()
    if "Normal" in trend_df.columns:
        fig.add_trace(
            go.Scatter(
                x=trend_df["bucket"], y=trend_df["Normal"], mode="lines",
                name="Normal", line=dict(color="#4dd4ac", width=2),
            )
        )
    if "Anomaly" in trend_df.columns:
        fig.add_trace(
            go.Scatter(
                x=trend_df["bucket"], y=trend_df["Anomaly"], mode="lines",
                name="Anomaly", line=dict(color="#ff3b5c", width=2),
            )
        )
    fig.update_layout(
        title="Anomaly Trend Over Time (blocks by first-seen hour)",
        xaxis_title="Time",
        yaxis_title="Block Count",
        **DARK_LAYOUT,
    )
    return fig


def shap_waterfall(feature_labels, shap_values, base_value: float, predicted_value: float, title: str, top_n: int = 8) -> go.Figure:
    """
    Waterfall view of a single SHAP explanation: starts at the
    background baseline, shows each top feature's push up (red) or
    down (green), collapses the rest into one "other features" bar,
    and ends at the model's actual output for this block.
    """
    shap_values = list(shap_values)
    order = sorted(range(len(shap_values)), key=lambda i: abs(shap_values[i]), reverse=True)
    top_idx = order[:top_n]
    other_idx = order[top_n:]

    labels = [feature_labels[i] for i in top_idx]
    values = [shap_values[i] for i in top_idx]
    if other_idx:
        labels.append(f"Other {len(other_idx)} feature(s)")
        values.append(sum(shap_values[i] for i in other_idx))

    x_labels = ["Baseline"] + labels + ["Model output"]
    y_values = [base_value] + values + [predicted_value]
    measures = ["absolute"] + ["relative"] * len(values) + ["total"]

    fig = go.Figure(
        go.Waterfall(
            x=x_labels,
            y=y_values,
            measure=measures,
            increasing=dict(marker=dict(color="#ff3b5c")),
            decreasing=dict(marker=dict(color="#4dd4ac")),
            totals=dict(marker=dict(color="#5aa9ff")),
            connector=dict(line=dict(color="#2a3448")),
        )
    )
    fig.update_layout(title=title, showlegend=False, **DARK_LAYOUT)
    return fig


def fusion_composition_bar(if_score: float, ae_score: float, if_weight: float, ae_weight: float, fusion_score: float) -> go.Figure:
    """Small stacked bar visually showing IF-v2 + Autoencoder -> final fusion score."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=["Fusion"], x=[if_weight * if_score], orientation="h",
            name=f"IF-v2 ({if_weight:.0%})", marker_color="#ff9f1c",
        )
    )
    fig.add_trace(
        go.Bar(
            y=["Fusion"], x=[ae_weight * ae_score], orientation="h",
            name=f"Autoencoder ({ae_weight:.0%})", marker_color="#5aa9ff",
        )
    )
    fig.update_layout(
        barmode="stack",
        title=f"Fusion Composition -- final score {fusion_score:.4f}",
        xaxis_title="Weighted contribution",
        height=200,
        **DARK_LAYOUT,
    )
    return fig


def metric_gauge(value: float, title: str, max_value: float = 1.0) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"valueformat": ".4f"},
            gauge={
                "axis": {"range": [0, max_value], "tickcolor": "#c3ccd9"},
                "bar": {"color": "#4dd4ac"},
                "bgcolor": "#10192a",
                "borderwidth": 0,
            },
            title={"text": title, "font": {"size": 14}},
        )
    )
    fig.update_layout(height=220, **DARK_LAYOUT)
    return fig
