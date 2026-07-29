"""
Shared state passed between LangGraph nodes.

Design note: the actual DataFrames / SQLite connection are deliberately
*not* stored in this state. LangGraph state gets checkpointed (for
conversation memory) and can be logged/serialized, and it's also what
gets passed to the LLM in places - none of that should carry raw
session data. Instead, each compiled graph is bound (via closures, see
build_graph.py) to one session's SQLStore, and this state only carries
the LLM-facing text, the routing decision, and small result artifacts
(a markdown preview table, a chart's JSON spec, etc.) needed to render
the final answer.
"""
from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # conversation memory - accumulates across turns via the checkpointer
    messages: Annotated[list, add_messages]

    # per-turn input
    question: str
    schema_context: str

    # routing
    intent: str
    route_reasoning: str

    # working artifacts produced by whichever node handled this turn
    sql_query: Optional[str]
    sql_preview_md: Optional[str]
    sql_row_count: Optional[int]
    chart_json: Optional[str]
    chart_title: Optional[str]
    dashboard_charts: Optional[list[dict[str, Any]]]
    anomaly_summary: Optional[str]
    quality_summary: Optional[str]
    forecast_summary: Optional[str]

    # final output for this turn
    final_answer: str
    error: Optional[str]
