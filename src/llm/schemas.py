"""
Structured-output schemas for LLM calls.

Using `.with_structured_output(Model)` instead of parsing free text out
of the LLM's response is what makes the router and SQL-generation steps
reliable enough to drive a graph: we get typed, validated objects back
instead of hoping the model's prose contains a parseable code block.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Intent = Literal[
    "sql_qa",       # factual/aggregation questions answerable via SQL
    "chart",        # explicit request for a visualization
    "insight",      # "summarize", "give me insights", open-ended analysis
    "anomaly",      # anomaly / outlier detection requests
    "quality",      # data quality / data cleanliness questions
    "forecast",     # trend projection / "what will next month look like"
    "dashboard",    # "give me a dashboard / overview of everything"
    "general",      # greetings, clarifications, questions unrelated to the data
]


class RouterDecision(BaseModel):
    intent: Intent = Field(description="The single best-matching category for the user's request.")
    reasoning: str = Field(description="One short sentence explaining the classification.")


class SQLGeneration(BaseModel):
    sql: str = Field(description="A single read-only SQLite SELECT/WITH statement that answers the question.")
    explanation: str = Field(description="Plain-language explanation of what the query does and why.")


class ChartPlan(BaseModel):
    chart_type: Literal["bar", "line", "pie", "scatter"]
    x: str = Field(description="Column name to use for the x-axis / categories.")
    y: str = Field(description="Column name to use for the y-axis / values.")
    color: Optional[str] = Field(default=None, description="Optional column to color/group by.")
    title: str = Field(description="Short, descriptive chart title.")
    sql: str = Field(description="A single read-only SQLite SELECT/WITH statement producing the data to chart.")


class ForecastPlan(BaseModel):
    date_column: str = Field(description="Column to use as the time axis.")
    value_column: str = Field(description="Numeric column to forecast.")
    table: str = Field(description="Which table to pull date_column/value_column from.")
    periods: int = Field(default=3, description="How many future periods to project.")


class InsightPlan(BaseModel):
    queries: list[SQLGeneration] = Field(
        description="2-4 SQL queries whose results together would reveal useful "
        "business insights (top/bottom performers, trends, concentration, etc.)."
    )


class AnomalyPlan(BaseModel):
    table: str = Field(description="Table to analyze for anomalies.")
    numeric_columns: list[str] = Field(description="Numeric columns to check for outliers.")
    group_column: Optional[str] = Field(
        default=None,
        description="Optional column to check outliers within-group rather than globally "
        "(e.g. detect outliers per region instead of across all rows).",
    )
    id_column: Optional[str] = Field(
        default=None, description="Optional identifier column to include for context (e.g. order_id)."
    )


class QualityPlan(BaseModel):
    tables: list[str] = Field(description="Which table(s) to run data quality checks on.")


class DashboardPlan(BaseModel):
    charts: list[ChartPlan] = Field(description="3-4 charts giving a rounded overview of the dataset.")
