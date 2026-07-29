import pandas as pd

from src.tools.chart_tool import suggest_chart_type


def test_datetime_x_numeric_y_suggests_line():
    df = pd.DataFrame({
        "order_date": pd.date_range("2024-01-01", periods=5, freq="MS"),
        "revenue": [100, 120, 90, 150, 130],
    })
    assert suggest_chart_type(df, "order_date", "revenue") == "line"


def test_low_cardinality_category_suggests_pie():
    df = pd.DataFrame({"region": ["N", "S", "E", "W"], "revenue": [100, 90, 80, 70]})
    assert suggest_chart_type(df, "region", "revenue") == "pie"


def test_high_cardinality_category_suggests_bar():
    df = pd.DataFrame({"customer": [f"C{i}" for i in range(20)], "revenue": range(20)})
    assert suggest_chart_type(df, "customer", "revenue") == "bar"


def test_two_numeric_columns_suggest_scatter():
    df = pd.DataFrame({"quantity": [1, 2, 3, 4], "revenue": [10, 20, 30, 40]})
    assert suggest_chart_type(df, "quantity", "revenue") == "scatter"
