"""
Statistical anomaly detection.

Deliberately NOT LLM-based: anomalies are found with real statistics
(z-score and IQR), per numeric column and optionally within groups
(e.g. flag outliers within each region rather than globally, which
catches anomalies that a global check would wash out). The LLM's role
is downstream - explaining *why* something flagged, in plain language.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Anomaly:
    row_index: int
    column: str
    value: float
    method: str
    score: float
    group: str | None = None
    context: dict | None = None


def _zscore_flags(series: pd.Series, threshold: float) -> pd.Series:
    clean = series.dropna()
    if clean.std(ddof=0) == 0 or len(clean) < 5:
        return pd.Series(False, index=series.index)
    z = (series - clean.mean()) / clean.std(ddof=0)
    return z.abs() > threshold


def _iqr_flags(series: pd.Series, k: float = 1.5) -> pd.Series:
    clean = series.dropna()
    if len(clean) < 5:
        return pd.Series(False, index=series.index)
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=series.index)
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (series < lower) | (series > upper)


def detect_anomalies(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    group_col: str | None = None,
    method: str = "zscore",
    threshold: float = 3.0,
    max_results: int = 25,
    id_col: str | None = None,
) -> list[Anomaly]:
    """
    Detect outliers in numeric columns, optionally within groups
    (e.g. detect revenue outliers *per region* rather than globally).
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    anomalies: list[Anomaly] = []

    groups = df.groupby(group_col) if group_col and group_col in df.columns else [(None, df)]

    for group_val, gdf in groups:
        for col in numeric_cols:
            if col not in gdf.columns:
                continue
            series = gdf[col]
            flags = _zscore_flags(series, threshold) if method == "zscore" else _iqr_flags(series)
            flagged = gdf[flags]
            for idx, row in flagged.iterrows():
                clean = series.dropna()
                if method == "zscore" and clean.std(ddof=0) > 0:
                    score = float((row[col] - clean.mean()) / clean.std(ddof=0))
                else:
                    score = float(row[col])
                anomalies.append(Anomaly(
                    row_index=int(idx),
                    column=col,
                    value=float(row[col]) if pd.notna(row[col]) else float("nan"),
                    method=method,
                    score=round(score, 2),
                    group=str(group_val) if group_val is not None else None,
                    context={id_col: row[id_col]} if id_col and id_col in row else None,
                ))

    # most extreme first
    anomalies.sort(key=lambda a: abs(a.score), reverse=True)
    return anomalies[:max_results]


def anomalies_to_text(anomalies: list[Anomaly]) -> str:
    if not anomalies:
        return "No statistically significant anomalies detected."
    lines = []
    for a in anomalies:
        group_str = f", group={a.group}" if a.group else ""
        ctx_str = f", {a.context}" if a.context else ""
        lines.append(
            f"- row {a.row_index}: column '{a.column}' = {a.value} "
            f"({a.method} score={a.score}{group_str}{ctx_str})"
        )
    return "\n".join(lines)
