import pandas as pd
import pytest

from src.tools.forecast_tool import forecast_series


def _monthly_df(n=12, start_value=100, growth=5):
    dates = pd.date_range("2024-01-01", periods=n, freq="MS")
    values = [start_value + growth * i for i in range(n)]
    return pd.DataFrame({"order_date": dates, "revenue": values})


def test_forecast_projects_upward_trend():
    df = _monthly_df()
    result = forecast_series(df, "order_date", "revenue", periods=3)
    assert len(result.forecast) == 3
    # trend is increasing, so forecast values should exceed last history value
    last_history_value = result.history["value"].iloc[-1]
    assert result.forecast["value"].iloc[0] > last_history_value * 0.9


def test_forecast_bounds_are_ordered():
    df = _monthly_df()
    result = forecast_series(df, "order_date", "revenue", periods=3)
    assert (result.forecast["lower"] <= result.forecast["value"]).all()
    assert (result.forecast["value"] <= result.forecast["upper"]).all()


def test_forecast_no_negative_values():
    dates = pd.date_range("2024-01-01", periods=6, freq="MS")
    values = [10, 8, 6, 4, 2, 1]  # sharp downward trend
    df = pd.DataFrame({"order_date": dates, "revenue": values})
    result = forecast_series(df, "order_date", "revenue", periods=5)
    assert (result.forecast["value"] >= 0).all()


def test_forecast_insufficient_history_raises():
    df = _monthly_df(n=2)
    with pytest.raises(ValueError):
        forecast_series(df, "order_date", "revenue", periods=3)
