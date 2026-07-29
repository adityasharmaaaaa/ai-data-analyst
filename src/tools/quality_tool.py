"""
Data quality checks.

Pure pandas/numpy - deterministic and testable without an LLM. The LLM's
job (in the quality graph node) is only to turn this structured report
into a readable narrative with recommendations; it never invents the
numbers itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ColumnQuality:
    column: str
    dtype: str
    missing_count: int
    missing_pct: float
    n_unique: int
    notes: list[str] = field(default_factory=list)


@dataclass
class QualityReport:
    table_name: str
    n_rows: int
    n_cols: int
    duplicate_rows: int
    columns: list[ColumnQuality]
    overall_score: float  # 0-100, heuristic
    issues: list[str]


def _score(missing_pct_avg: float, dup_pct: float, n_issue_cols: int, n_cols: int) -> float:
    score = 100.0
    score -= min(40.0, missing_pct_avg * 100 * 2)  # heavier weight on missingness
    score -= min(30.0, dup_pct * 100 * 3)
    if n_cols:
        score -= min(20.0, (n_issue_cols / n_cols) * 40)
    return round(max(0.0, score), 1)


def assess_quality(df: pd.DataFrame, table_name: str) -> QualityReport:
    n_rows, n_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())
    dup_pct = duplicate_rows / n_rows if n_rows else 0.0

    columns: list[ColumnQuality] = []
    issues: list[str] = []
    missing_pcts: list[float] = []
    n_issue_cols = 0

    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        missing_pct = missing / n_rows if n_rows else 0.0
        missing_pcts.append(missing_pct)
        notes: list[str] = []

        if missing_pct == 1.0:
            notes.append("column is entirely empty")
            issues.append(f"'{col}' is 100% missing.")
            n_issue_cols += 1
        elif missing_pct > 0.2:
            notes.append(f"{missing_pct:.0%} missing")
            issues.append(f"'{col}' is {missing_pct:.0%} missing (severe).")
            n_issue_cols += 1
        elif missing_pct > 0.05:
            notes.append(f"{missing_pct:.0%} missing")
            issues.append(f"'{col}' is {missing_pct:.0%} missing (moderate).")
            n_issue_cols += 1
        elif missing_pct > 0:
            notes.append(f"{missing_pct:.0%} missing")
            issues.append(f"'{col}' has {missing} missing value(s) ({missing_pct:.1%}, minor).")

        n_unique = int(series.nunique(dropna=True))
        if n_unique == 1 and n_rows > 1:
            notes.append("constant value across all rows")

        # numeric-specific checks
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if not non_null.empty:
                if (non_null < 0).any() and col not in ("lat", "latitude"):
                    neg_pct = (non_null < 0).mean()
                    if neg_pct > 0:
                        notes.append(f"{neg_pct:.0%} negative values")
                        if neg_pct > 0.01:
                            issues.append(
                                f"'{col}' has negative values ({neg_pct:.0%} of rows) - "
                                "verify this is expected for this field."
                            )
                            n_issue_cols += 1

        # attempt date parsing on columns that look like dates by name.
        # Note: pandas >= 2.x may back text columns with either numpy
        # 'object' dtype or the newer pandas StringDtype ('str') depending
        # on version/config, so check for "is text" rather than a single
        # exact dtype.
        if "date" in col.lower() and (
            pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
        ) and not pd.api.types.is_numeric_dtype(series):
            parsed = pd.to_datetime(series, errors="coerce")
            bad = int(parsed.isna().sum() - series.isna().sum())
            if bad > 0:
                notes.append(f"{bad} unparseable date value(s)")
                issues.append(f"'{col}' has {bad} value(s) that don't parse as dates.")
                n_issue_cols += 1

        columns.append(ColumnQuality(
            column=col,
            dtype=str(series.dtype),
            missing_count=missing,
            missing_pct=round(missing_pct, 4),
            n_unique=n_unique,
            notes=notes,
        ))

    if duplicate_rows > 0:
        issues.append(f"{duplicate_rows} duplicate row(s) found ({dup_pct:.1%} of the table).")

    avg_missing = float(np.mean(missing_pcts)) if missing_pcts else 0.0
    score = _score(avg_missing, dup_pct, n_issue_cols, n_cols)

    if not issues:
        issues.append("No significant data quality issues detected.")

    return QualityReport(
        table_name=table_name,
        n_rows=n_rows,
        n_cols=n_cols,
        duplicate_rows=duplicate_rows,
        columns=columns,
        overall_score=score,
        issues=issues,
    )


def report_to_text(report: QualityReport) -> str:
    lines = [
        f"Table '{report.table_name}': {report.n_rows} rows x {report.n_cols} cols, "
        f"quality score {report.overall_score}/100, {report.duplicate_rows} duplicate rows.",
        "Issues:",
    ]
    lines += [f"  - {i}" for i in report.issues]
    lines.append("Per-column detail:")
    for c in report.columns:
        note_str = f" ({'; '.join(c.notes)})" if c.notes else ""
        lines.append(
            f"  - {c.column} [{c.dtype}]: {c.missing_count} missing "
            f"({c.missing_pct:.1%}), {c.n_unique} unique{note_str}"
        )
    return "\n".join(lines)
