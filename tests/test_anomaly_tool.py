import numpy as np
import pandas as pd

from src.tools.anomaly_tool import detect_anomalies


def test_detects_obvious_outlier():
    rng = np.random.default_rng(0)
    values = rng.normal(100, 5, size=200).tolist()
    values[10] = 1000.0  # obvious outlier
    df = pd.DataFrame({"revenue": values})
    anomalies = detect_anomalies(df, numeric_cols=["revenue"], method="zscore", threshold=3.0)
    assert any(a.row_index == 10 for a in anomalies)


def test_no_anomalies_in_uniform_data():
    df = pd.DataFrame({"revenue": [100.0] * 50})
    anomalies = detect_anomalies(df, numeric_cols=["revenue"])
    assert anomalies == []


def test_group_aware_detection():
    # An order of 300 is normal for region A but an outlier for region B
    rng = np.random.default_rng(1)
    region_a = rng.normal(300, 10, size=100)
    region_b = rng.normal(50, 5, size=100)
    region_b[0] = 300  # outlier only relative to region B
    df = pd.DataFrame({
        "region": ["A"] * 100 + ["B"] * 100,
        "revenue": np.concatenate([region_a, region_b]),
    })
    anomalies = detect_anomalies(df, numeric_cols=["revenue"], group_col="region", threshold=3.0)
    flagged_in_b = [a for a in anomalies if a.group == "B"]
    assert len(flagged_in_b) >= 1


def test_max_results_respected():
    rng = np.random.default_rng(2)
    values = rng.normal(0, 1, size=500)
    # make many extreme outliers
    values[:100] = 999
    df = pd.DataFrame({"x": values})
    anomalies = detect_anomalies(df, numeric_cols=["x"], max_results=10)
    assert len(anomalies) <= 10
