"""
Router node: classifies the user's question into one of the intents in
src.llm.schemas.Intent, using Gemini structured output. Every other node
in the graph is reached only through this classification, which is what
makes this an explicit multi-node graph rather than a single tool-calling
agent - routing is a visible, inspectable step of its own.
"""
from __future__ import annotations

from src.graph.state import AgentState
from src.llm.gemini import get_llm
from src.llm.schemas import RouterDecision
from src.logging_config import get_logger

log = get_logger(__name__)

_ROUTER_PROMPT = """You are the routing layer of an AI data analyst application.
Classify the user's message into exactly one category based on what
action is needed.

Categories:
- sql_qa: A factual or aggregation question answerable by querying the
  data (e.g. "which region had the highest revenue", "top 5 customers",
  "generate SQL for this").
- chart: The user explicitly wants a visualization/chart/plot/graph.
- insight: Open-ended requests for a summary, business insights, or
  "tell me about this data".
- anomaly: Requests to find outliers, anomalies, or unusual/suspicious
  records.
- quality: Questions about missing data, duplicates, data cleanliness,
  or data quality in general.
- forecast: Requests to project a trend or predict future values.
- dashboard: Requests for an overview/dashboard covering multiple
  aspects of the data at once.
- general: Greetings, meta-questions, or anything not about analyzing
  the uploaded data.

Dataset schema currently loaded:
{schema_context}

User message: {question}
"""


def router_node(state: AgentState) -> dict:
    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(RouterDecision)

    prompt = _ROUTER_PROMPT.format(
        schema_context=state.get("schema_context", "No data loaded."),
        question=state["question"],
    )

    try:
        decision: RouterDecision = structured_llm.invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        log.exception("Router classification failed, defaulting to general")
        return {"intent": "general", "route_reasoning": f"routing failed: {exc}"}

    log.info("Routed question %r -> intent=%s (%s)", state["question"], decision.intent, decision.reasoning)
    return {"intent": decision.intent, "route_reasoning": decision.reasoning}


def route_condition(state: AgentState) -> str:
    """Used as the conditional edge function - just reads the classified intent."""
    return state.get("intent", "general")
