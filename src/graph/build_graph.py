"""
Builds the compiled LangGraph graph for one session.

    START -> router -> (conditional edge on state["intent"]) -> {
        sql_qa, chart, insight, anomaly, quality, forecast, dashboard, general
    } -> finalize -> END

`router` is the only node reached from START; every specialized node is
reached only via the conditional edge keyed on the router's classified
intent, and every specialized node flows into `finalize`, which appends
the answer to conversation history for the checkpointer to persist.
Conversation memory across turns is handled by LangGraph's checkpointer
(MemorySaver, in-process) keyed on a per-session `thread_id` - see
app.py for how the Streamlit session id is used as that thread_id.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.data.sql_store import SQLStore
from src.graph.nodes import (
    finalize_node,
    general_node,
    make_anomaly_node,
    make_chart_node,
    make_dashboard_node,
    make_forecast_node,
    make_insight_node,
    make_quality_node,
    make_sql_qa_node,
)
from src.graph.router import route_condition, router_node
from src.graph.state import AgentState

_INTENT_NODE_NAMES = [
    "sql_qa", "chart", "insight", "anomaly", "quality", "forecast", "dashboard", "general",
]


def build_graph(store: SQLStore):
    """Compile a fresh graph bound to this session's SQLStore, with an
    in-memory checkpointer for conversation memory across turns."""
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("sql_qa", make_sql_qa_node(store))
    graph.add_node("chart", make_chart_node(store))
    graph.add_node("insight", make_insight_node(store))
    graph.add_node("anomaly", make_anomaly_node(store))
    graph.add_node("quality", make_quality_node(store))
    graph.add_node("forecast", make_forecast_node(store))
    graph.add_node("dashboard", make_dashboard_node(store))
    graph.add_node("general", general_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_condition,
        {name: name for name in _INTENT_NODE_NAMES},
    )
    for name in _INTENT_NODE_NAMES:
        graph.add_edge(name, "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=MemorySaver())
