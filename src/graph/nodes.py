"""
Specialized graph nodes, one per intent.

Every node is a *factory* (`make_x_node(store)`) that closes over the
current session's SQLStore and returns the actual LangGraph node
function. This keeps the SQLite connection / DataFrames out of the
(potentially checkpointed) graph state while still giving each node
real data access - see state.py for why.

Shared pattern across the data-driven nodes:
  1. Ask Gemini (structured output) to translate the question into a
     concrete plan (SQL query, chart spec, which table/columns, etc.)
  2. Execute that plan with a deterministic Python tool (sql_tool,
     anomaly_tool, quality_tool, forecast_tool, chart_tool) - the LLM
     never sees or touches raw data beyond what these tools return.
  3. If step 2 fails (e.g. bad SQL), feed the error back to the LLM
     once for a self-correction retry before giving up gracefully.
  4. Ask Gemini to turn the structured result into a plain-language,
     reasoned answer.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.graph_objects as go
from langchain_core.messages import AIMessage, SystemMessage

from src.data.sql_store import SQLStore
from src.graph.state import AgentState
from src.llm.gemini import extract_text, get_llm
from src.llm.schemas import (
    AnomalyPlan,
    ChartPlan,
    DashboardPlan,
    ForecastPlan,
    InsightPlan,
    QualityPlan,
    SQLGeneration,
)
from src.logging_config import get_logger
from src.tools.anomaly_tool import anomalies_to_text, detect_anomalies
from src.tools.chart_tool import ChartSpec, build_chart
from src.tools.forecast_tool import forecast_to_text, forecast_series
from src.tools.quality_tool import assess_quality, report_to_text
from src.tools.sql_tool import SQLResult, run_sql

log = get_logger(__name__)

# Fields every node should reset to None at the start of a fresh turn so
# stale artifacts from a previous intent never leak into a new answer.
# Consumed by the Streamlit app when invoking the graph - see app.py.
TRANSIENT_STATE_KEYS = [
    "sql_query", "sql_preview_md", "sql_row_count",
    "chart_json", "chart_title", "dashboard_charts",
    "anomaly_summary", "quality_summary", "forecast_summary", "error",
]


def _df_preview_md(df, max_rows: int = 15) -> str:
    if df is None or df.empty:
        return "(no rows returned)"
    return df.head(max_rows).to_markdown(index=False)


def _recent_history_text(state: AgentState, max_turns: int = 3) -> str:
    """Format the last few conversation turns as plain text context, so
    follow-up questions ('now break that down by product') resolve
    correctly. Excludes the current turn's own question, which is
    always the last message in state["messages"] (see finalize_node /
    the LangGraph input-merge note in general_node)."""
    history = state.get("messages", [])[:-1]  # drop current turn's question
    if not history:
        return ""
    recent = history[-(max_turns * 2):]
    lines = []
    for m in recent:
        role = "User" if m.__class__.__name__ == "HumanMessage" else "Assistant"
        content = extract_text(m.content)
        lines.append(f"{role}: {content[:400]}")
    return "Recent conversation:\n" + "\n".join(lines) + "\n\n"


def _generate_and_run_sql(
    store: SQLStore, schema_context: str, question: str, history_text: str = ""
) -> tuple[SQLGeneration, SQLResult]:
    """Generate SQL for `question`, run it, and self-correct once on failure."""
    llm = get_llm()
    structured = llm.with_structured_output(SQLGeneration)

    prompt = (
        "You write SQLite SELECT queries for a data-analyst app.\n\n"
        f"{history_text}"
        f"Schema:\n{schema_context}\n\n"
        f"Question: {question}\n\n"
        "Rules: SQLite dialect only. Read-only (SELECT/WITH) only. "
        "Reference only the tables/columns shown above. If the question "
        "refers back to something from the recent conversation (e.g. 'now "
        "break that down by product'), use that context."
    )
    plan: SQLGeneration = structured.invoke(prompt)
    result = run_sql(store, plan.sql)

    if not result.ok:
        log.info("SQL failed, retrying once with error feedback: %s", result.error)
        retry_prompt = (
            f"{prompt}\n\nYour previous query failed:\n{plan.sql}\n\n"
            f"Error: {result.error}\n\nWrite a corrected query."
        )
        plan = structured.invoke(retry_prompt)
        result = run_sql(store, plan.sql)

    return plan, result


# --------------------------------------------------------------- sql_qa
def make_sql_qa_node(store: SQLStore) -> Callable[[AgentState], dict]:
    def node(state: AgentState) -> dict:
        schema_context = state.get("schema_context", "")
        question = state["question"]

        plan, result = _generate_and_run_sql(store, schema_context, question, _recent_history_text(state))

        if not result.ok:
            return {
                "sql_query": plan.sql,
                "error": result.error,
                "final_answer": (
                    f"I wasn't able to run a query for that.\n\n"
                    f"SQL attempted:\n```sql\n{plan.sql}\n```\n\nError: {result.error}"
                ),
            }

        preview_md = _df_preview_md(result.df)
        llm = get_llm()
        answer_prompt = (
            f"Question: {question}\n\n"
            f"SQL used:\n{plan.sql}\n\n"
            f"Query explanation: {plan.explanation}\n\n"
            f"Result ({result.row_count} row(s)"
            f"{', truncated to first ' + str(len(result.df)) if result.truncated else ''}):\n"
            f"{preview_md}\n\n"
            "Write a direct, concise answer to the question for a business user. "
            "Reference specific numbers from the result. Then briefly explain your "
            "reasoning (why this query answers the question)."
        )
        answer = extract_text(llm.invoke(answer_prompt).content)

        return {
            "sql_query": plan.sql,
            "sql_preview_md": preview_md,
            "sql_row_count": result.row_count,
            "final_answer": answer,
        }

    return node


# --------------------------------------------------------------- chart
def make_chart_node(store: SQLStore) -> Callable[[AgentState], dict]:
    def node(state: AgentState) -> dict:
        schema_context = state.get("schema_context", "")
        question = state["question"]

        llm = get_llm()
        structured = llm.with_structured_output(ChartPlan)
        prompt = (
            "Plan a chart answering the user's request.\n\n"
            f"{_recent_history_text(state)}"
            f"Schema:\n{schema_context}\n\n"
            f"Request: {question}\n\n"
            "Choose the most appropriate chart_type (bar/line/pie/scatter), pick x/y "
            "columns (or aliases from your SQL), and write a single read-only SQLite "
            "query (SELECT/WITH only) that produces the exact rows to plot - "
            "pre-aggregated if needed (e.g. GROUP BY for a bar chart). If the request "
            "refers back to the recent conversation, use that context."
        )
        plan: ChartPlan = structured.invoke(prompt)
        result = run_sql(store, plan.sql)

        if not result.ok:
            retry_prompt = f"{prompt}\n\nPrevious query failed:\n{plan.sql}\nError: {result.error}\nFix it."
            plan = structured.invoke(retry_prompt)
            result = run_sql(store, plan.sql)

        if not result.ok:
            return {
                "sql_query": plan.sql,
                "error": result.error,
                "final_answer": f"I couldn't build that chart. SQL error: {result.error}",
            }

        try:
            fig = build_chart(result.df, ChartSpec(
                chart_type=plan.chart_type, x=plan.x, y=plan.y, color=plan.color, title=plan.title,
            ))
        except Exception as exc:  # noqa: BLE001
            log.exception("Chart build failed")
            return {
                "sql_query": plan.sql,
                "error": str(exc),
                "final_answer": f"I ran the query but couldn't render the chart: {exc}",
            }

        answer = (
            f"Here's your {plan.chart_type} chart: **{plan.title}**.\n\n"
            f"Built from {result.row_count} row(s) using:\n```sql\n{plan.sql}\n```"
        )

        return {
            "sql_query": plan.sql,
            "sql_preview_md": _df_preview_md(result.df),
            "chart_json": fig.to_json(),
            "chart_title": plan.title,
            "final_answer": answer,
        }

    return node


# --------------------------------------------------------------- insight
def make_insight_node(store: SQLStore) -> Callable[[AgentState], dict]:
    def node(state: AgentState) -> dict:
        schema_context = state.get("schema_context", "")
        question = state["question"]

        llm = get_llm()
        structured = llm.with_structured_output(InsightPlan)
        prompt = (
            "Plan 2-4 SQL queries (SQLite, read-only) whose results would together "
            "reveal genuinely useful business insights for this dataset - think top/"
            "bottom performers, trends over time, concentration (e.g. revenue share), "
            "and anything notable.\n\n"
            f"{_recent_history_text(state)}"
            f"Schema:\n{schema_context}\n\nUser request: {question}"
        )
        plan: InsightPlan = structured.invoke(prompt)

        sections = []
        for q in plan.queries:
            result = run_sql(store, q.sql)
            if result.ok:
                sections.append(f"Query: {q.explanation}\nSQL: {q.sql}\nResult:\n{_df_preview_md(result.df, 10)}")
            else:
                sections.append(f"Query: {q.explanation}\nSQL: {q.sql}\nFailed: {result.error}")

        combined = "\n\n".join(sections) if sections else "(no queries produced results)"
        answer_prompt = (
            f"User asked: {question}\n\n"
            f"Here is data pulled to answer this:\n\n{combined}\n\n"
            "Write a concise business-insights summary (bullet points are fine). "
            "Call out specific numbers, name top/bottom performers, and note any "
            "trend. Ground every claim in the data above - don't invent figures."
        )
        answer = extract_text(llm.invoke(answer_prompt).content)

        return {"final_answer": answer}

    return node


# --------------------------------------------------------------- anomaly
def make_anomaly_node(store: SQLStore) -> Callable[[AgentState], dict]:
    def node(state: AgentState) -> dict:
        schema_context = state.get("schema_context", "")
        question = state["question"]

        llm = get_llm()
        structured = llm.with_structured_output(AnomalyPlan)
        prompt = (
            "Decide which table/columns to check for anomalies based on the request.\n\n"
            f"Schema:\n{schema_context}\n\nRequest: {question}"
        )
        plan: AnomalyPlan = structured.invoke(prompt)

        if plan.table not in store.tables:
            return {"final_answer": f"I couldn't find a table called '{plan.table}' to check."}

        df = store.tables[plan.table]
        anomalies = detect_anomalies(
            df,
            numeric_cols=[c for c in plan.numeric_columns if c in df.columns] or None,
            group_col=plan.group_column if plan.group_column in df.columns else None,
            id_col=plan.id_column if plan.id_column in df.columns else None,
        )
        summary_text = anomalies_to_text(anomalies)

        answer_prompt = (
            f"Request: {question}\n\n"
            f"Statistically flagged anomalies (z-score/IQR based) in table "
            f"'{plan.table}':\n{summary_text}\n\n"
            "Explain, in plain language, why these specific records were flagged "
            "(unusually high/low relative to the rest of the data / their group), "
            "and note anything they might have in common. If none were found, say so plainly."
        )
        answer = extract_text(llm.invoke(answer_prompt).content)

        return {"anomaly_summary": summary_text, "final_answer": answer}

    return node


# --------------------------------------------------------------- quality
def make_quality_node(store: SQLStore) -> Callable[[AgentState], dict]:
    def node(state: AgentState) -> dict:
        schema_context = state.get("schema_context", "")
        question = state["question"]

        if len(store.tables) == 1:
            tables_to_check = list(store.tables.keys())
        else:
            llm = get_llm()
            structured = llm.with_structured_output(QualityPlan)
            prompt = (
                "Which table(s) should be checked for data quality issues, based on "
                f"the request? If unclear, include all tables.\n\nSchema:\n{schema_context}"
                f"\n\nRequest: {question}"
            )
            plan: QualityPlan = structured.invoke(prompt)
            tables_to_check = [t for t in plan.tables if t in store.tables] or list(store.tables.keys())

        reports_text = []
        for t in tables_to_check:
            report = assess_quality(store.tables[t], t)
            reports_text.append(report_to_text(report))
        combined = "\n\n".join(reports_text)

        llm = get_llm()
        answer_prompt = (
            f"Request: {question}\n\nData quality report(s):\n{combined}\n\n"
            "Summarize the data quality findings in plain language for a business "
            "user, prioritized by severity, and give 2-3 concrete recommendations."
        )
        answer = extract_text(llm.invoke(answer_prompt).content)

        return {"quality_summary": combined, "final_answer": answer}

    return node


# --------------------------------------------------------------- forecast
def make_forecast_node(store: SQLStore) -> Callable[[AgentState], dict]:
    def node(state: AgentState) -> dict:
        schema_context = state.get("schema_context", "")
        question = state["question"]

        llm = get_llm()
        structured = llm.with_structured_output(ForecastPlan)
        prompt = (
            "Decide which table/date column/numeric column to forecast, and how "
            f"many future periods to project, based on the request.\n\nSchema:\n"
            f"{schema_context}\n\nRequest: {question}"
        )
        plan: ForecastPlan = structured.invoke(prompt)

        if plan.table not in store.tables:
            return {"final_answer": f"I couldn't find a table called '{plan.table}' to forecast."}

        df = store.tables[plan.table]
        try:
            fc = forecast_series(df, plan.date_column, plan.value_column, periods=max(1, min(plan.periods, 12)))
        except (ValueError, KeyError) as exc:
            return {"final_answer": f"I couldn't build a forecast: {exc}"}

        summary_text = forecast_to_text(fc)

        # bonus: chart the history + forecast together
        fig = go.Figure()
        fig.add_scatter(x=fc.history["period"], y=fc.history["value"], mode="lines+markers", name="History")
        fig.add_scatter(x=fc.forecast["period"], y=fc.forecast["value"], mode="lines+markers", name="Forecast",
                         line=dict(dash="dash"))
        fig.add_scatter(
            x=pd.concat([fc.forecast["period"], fc.forecast["period"][::-1]]),
            y=pd.concat([fc.forecast["upper"], fc.forecast["lower"][::-1]]),
            fill="toself", fillcolor="rgba(99,110,250,0.15)", line=dict(width=0),
            name="Confidence range", showlegend=True,
        )
        fig.update_layout(template="plotly_white", title=f"{plan.value_column} forecast", margin=dict(l=40, r=20, t=60, b=40))

        answer_prompt = (
            f"Request: {question}\n\nForecast results ({fc.method}):\n{summary_text}\n\n"
            "Explain the projection in plain language, mention the method briefly "
            "(it's a lightweight trend-based baseline, not a heavyweight statistical "
            "model - be upfront about that), and note the uncertainty range."
        )
        answer = extract_text(llm.invoke(answer_prompt).content)

        return {
            "forecast_summary": summary_text,
            "chart_json": fig.to_json(),
            "chart_title": f"{plan.value_column} forecast",
            "final_answer": answer,
        }

    return node


# -------------------------------------------------------------- dashboard
def make_dashboard_node(store: SQLStore) -> Callable[[AgentState], dict]:
    def node(state: AgentState) -> dict:
        schema_context = state.get("schema_context", "")
        question = state["question"]

        llm = get_llm()
        structured = llm.with_structured_output(DashboardPlan)
        prompt = (
            "Plan 3-4 charts (with SQLite SELECT/WITH queries) that together give a "
            "well-rounded overview dashboard of this dataset - e.g. a breakdown by "
            "category, a trend over time, and a top-N ranking.\n\n"
            f"Schema:\n{schema_context}\n\nRequest: {question}"
        )
        plan: DashboardPlan = structured.invoke(prompt)

        charts_json = []
        built_titles = []
        for chart_plan in plan.charts:
            result = run_sql(store, chart_plan.sql)
            if not result.ok:
                continue
            try:
                fig = build_chart(result.df, ChartSpec(
                    chart_type=chart_plan.chart_type, x=chart_plan.x, y=chart_plan.y,
                    color=chart_plan.color, title=chart_plan.title,
                ))
            except Exception:  # noqa: BLE001
                continue
            charts_json.append({"title": chart_plan.title, "figure": fig.to_json()})
            built_titles.append(chart_plan.title)

        answer = (
            f"Here's a dashboard with {len(charts_json)} chart(s): {', '.join(built_titles)}."
            if charts_json else
            "I wasn't able to build any dashboard charts from that request."
        )

        return {"dashboard_charts": charts_json, "final_answer": answer}

    return node


# --------------------------------------------------------------- general
def general_node(state: AgentState) -> dict:
    """Greetings, capability questions, or anything not about the data itself.
    Uses prior conversation turns so follow-ups stay coherent. Note:
    state["messages"] already ends with this turn's HumanMessage - LangGraph
    merges the invoke() input into state before any node runs - so it is
    NOT re-appended here."""
    llm = get_llm()
    system_note = SystemMessage(content=(
        "You are a helpful AI data analyst assistant. If the user asks what you "
        "can do, mention: answering questions about their uploaded CSV data, "
        "generating SQL, charts, business insights, anomaly detection, data "
        "quality checks, and forecasts. Keep replies brief."
    ))
    history = state.get("messages", [])
    answer = extract_text(llm.invoke([system_note, *history]).content)
    return {"final_answer": answer}


# --------------------------------------------------------------- finalize
def finalize_node(state: AgentState) -> dict:
    """Appends the turn's answer to conversation history for memory."""
    return {"messages": [AIMessage(content=state.get("final_answer", ""))]}
