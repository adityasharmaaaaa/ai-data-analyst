import numpy as np
import pandas as pd

from src.tools.quality_tool import assess_quality


def test_clean_data_scores_high():
    df = pd.DataFrame({"a": range(50), "b": [f"x{i}" for i in range(50)]})
    report = assess_quality(df, "t")
    assert report.overall_score > 90
    assert report.duplicate_rows == 0


def test_missing_values_detected():
    df = pd.DataFrame({"a": [1, 2, None, None, 5] * 10})
    report = assess_quality(df, "t")
    col = report.columns[0]
    assert col.missing_count == 20
    assert any("missing" in i for i in report.issues)


def test_duplicate_rows_detected():
    df = pd.DataFrame({"a": [1, 1, 2, 3]})
    report = assess_quality(df, "t")
    assert report.duplicate_rows == 1


def test_bad_dates_detected():
    df = pd.DataFrame({"order_date": ["2024-01-01", "2024-02-01", "not-a-date"] * 10})
    report = assess_quality(df, "t")
    assert any("date" in i.lower() for i in report.issues)


def test_all_null_column():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [np.nan, np.nan, np.nan]})
    report = assess_quality(df, "t")
    assert any("100%" in i for i in report.issues)
