"""
Chart building with Plotly.

Two entry points:
  - suggest_chart_type: a small heuristic that picks a sensible chart
    type from the shape of the data, used as a fallback / sanity check
    on whatever the LLM proposes.
  - build_chart: turns (DataFrame, chart_type, x, y, ...) into a
    plotly.graph_objects.Figure that Streamlit renders directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SUPPORTED_TYPES = {"bar", "line", "pie", "scatter"}


@dataclass
class ChartSpec:
    chart_type: str
    x: str
    y: str
    color: str | None = None
    title: str = ""


def suggest_chart_type(df: pd.DataFrame, x: str, y: str) -> str:
    """Heuristic chart-type suggestion based on column dtypes/cardinality."""
    x_is_datetime = pd.api.types.is_datetime64_any_dtype(df[x]) or "date" in x.lower()
    x_is_numeric = pd.api.types.is_numeric_dtype(df[x])
    y_is_numeric = pd.api.types.is_numeric_dtype(df[y])

    if x_is_datetime and y_is_numeric:
        return "line"
    if x_is_numeric and y_is_numeric:
        return "scatter"
    if not x_is_numeric and y_is_numeric:
        n_unique = df[x].nunique()
        if n_unique <= 8:
            return "pie" if n_unique <= 6 else "bar"
        return "bar"
    return "bar"


def build_chart(df: pd.DataFrame, spec: ChartSpec) -> go.Figure:
    chart_type = spec.chart_type if spec.chart_type in SUPPORTED_TYPES else "bar"

    common = dict(title=spec.title or f"{spec.y} by {spec.x}")

    if chart_type == "bar":
        fig = px.bar(df, x=spec.x, y=spec.y, color=spec.color, **common)
    elif chart_type == "line":
        fig = px.line(df, x=spec.x, y=spec.y, color=spec.color, markers=True, **common)
    elif chart_type == "pie":
        fig = px.pie(df, names=spec.x, values=spec.y, **common)
    elif chart_type == "scatter":
        fig = px.scatter(df, x=spec.x, y=spec.y, color=spec.color, **common)
    else:  # pragma: no cover - guarded above
        raise ValueError(f"Unsupported chart type: {chart_type}")

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
        legend_title_text=spec.color or "",
    )
    return fig
