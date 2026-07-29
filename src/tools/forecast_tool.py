"""
Lightweight forecasting.

This is intentionally a simple, dependency-free baseline (linear trend
via numpy.polyfit, blended with a seasonal-naive component when there's
enough history) rather than ARIMA/Prophet - good for demonstrating the
end-to-end feature without pulling in heavy forecasting libraries.
Swap in statsmodels/Prophet here if you need production-grade accuracy;
the graph node that calls this doesn't care how the numbers are made,
only that it gets a DataFrame back.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    history: pd.DataFrame       # columns: period, value
    forecast: pd.DataFrame      # columns: period, value, lower, upper
    method: str
    period_freq: str


def forecast_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    periods: int = 3,
    freq: str = "MS",
) -> ForecastResult:
    """
    Aggregate `value_col` by calendar period (default: month start) and
    project `periods` steps forward with a linear-trend baseline.
    """
    work = df[[date_col, value_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col])
    work = work.set_index(date_col).resample(freq)[value_col].sum().reset_index()
    work.columns = ["period", "value"]

    if len(work) < 3:
        raise ValueError("Not enough time-series history to forecast (need >= 3 periods).")

    x = np.arange(len(work))
    y = work["value"].values.astype(float)

    # linear trend
    slope, intercept = np.polyfit(x, y, 1)
    trend_fit = slope * x + intercept
    residuals = y - trend_fit
    resid_std = float(np.std(residuals)) if len(residuals) > 1 else 0.0

    # simple seasonal component: average deviation by calendar month,
    # only applied if we have >= 2 full years of monthly data
    seasonal = np.zeros(periods)
    if freq == "MS" and len(work) >= 24:
        month_idx = work["period"].dt.month.values
        detrended = y - trend_fit
        month_avg = {m: detrended[month_idx == m].mean() for m in range(1, 13)}
        future_months = [
            (work["period"].iloc[-1] + pd.DateOffset(months=i)).month
            for i in range(1, periods + 1)
        ]
        seasonal = np.array([month_avg.get(m, 0.0) for m in future_months])

    future_x = np.arange(len(work), len(work) + periods)
    future_trend = slope * future_x + intercept
    future_values = future_trend + seasonal
    future_periods = pd.date_range(
        work["period"].iloc[-1] + pd.DateOffset(months=1) if freq == "MS" else work["period"].iloc[-1],
        periods=periods,
        freq=freq,
    )

    forecast_df = pd.DataFrame({
        "period": future_periods,
        "value": np.maximum(future_values, 0),  # revenue/qty can't go negative
        "lower": np.maximum(future_values - 1.96 * resid_std, 0),
        "upper": future_values + 1.96 * resid_std,
    })

    return ForecastResult(
        history=work,
        forecast=forecast_df,
        method="linear-trend + seasonal-naive" if seasonal.any() else "linear-trend",
        period_freq=freq,
    )


def forecast_to_text(result: ForecastResult) -> str:
    lines = [f"Method: {result.method}", "Forecast:"]
    for _, row in result.forecast.iterrows():
        lines.append(
            f"  - {row['period'].date()}: {row['value']:.2f} "
            f"(range {row['lower']:.2f} - {row['upper']:.2f})"
        )
    return "\n".join(lines)
