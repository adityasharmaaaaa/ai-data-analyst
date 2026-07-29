"""
AI-powered Data Analyst - Streamlit entry point.

Upload one or more CSVs, then ask questions in natural language. Under
the hood: a LangGraph multi-node agent (router -> specialized node)
backed by Gemini generates SQL/analysis plans, executes them
deterministically against an in-memory SQLite store, and explains the
results. See src/graph/build_graph.py for the agent architecture.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from langchain_core.messages import HumanMessage

from src.config import SETTINGS
from src.data.loader import CSVValidationError, validate_and_load_csv
from src.data.sql_store import SQLStore
from src.graph.build_graph import build_graph
from src.graph.nodes import TRANSIENT_STATE_KEYS
from src.llm.gemini import MissingAPIKeyError
from src.logging_config import get_logger
from src.tools.quality_tool import assess_quality

log = get_logger(__name__)

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

SAMPLE_DIR = Path(__file__).parent / "sample_data"


# --------------------------------------------------------------- session
def init_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "store" not in st.session_state:
        st.session_state.store = SQLStore()
    if "graph" not in st.session_state:
        st.session_state.graph = build_graph(st.session_state.store)
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = [] 
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()


init_session()
store: SQLStore = st.session_state.store


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("📊 AI Data Analyst")
    st.caption("Upload CSVs, then ask questions in plain English.")

    if not SETTINGS.google_api_key:
        st.error(
            "No Gemini API key configured. Set `GOOGLE_API_KEY` in your `.env` "
            "file (see `.env.example`), then restart the app."
        )

    st.subheader("1. Data")
    uploaded_files = st.file_uploader(
        "Upload CSV file(s)", type=["csv"], accept_multiple_files=True,
        help="Multiple files can be joined in SQL if they share key columns.",
    )
    if uploaded_files:
        for f in uploaded_files:
            if f.file_id not in st.session_state.processed_files:
                existing = set(store.table_names())
                try:
                    loaded = validate_and_load_csv(f.name, f.getvalue(), existing)
                    if loaded.table_name not in store.table_names():
                        store.add_table(loaded)
                        st.session_state.processed_files.add(f.file_id)
                        
                        st.success(f"Loaded **{f.name}** -> table `{loaded.table_name}` "
                                   f"({len(loaded.df)} rows)")
                except CSVValidationError as e:
                    st.error(f"{f.name}: {e}")

    if not store.has_data() and SAMPLE_DIR.exists():
        if st.button("Or load sample dataset (sales/customers/products)"):
            for csv_path in sorted(SAMPLE_DIR.glob("*.csv")):
                existing = set(store.table_names())
                loaded = validate_and_load_csv(csv_path.name, csv_path.read_bytes(), existing)
                store.add_table(loaded)
            st.rerun()

    if store.has_data():
        st.subheader("Loaded tables")
        for name, df in store.tables.items():
            with st.expander(f"`{name}` — {len(df)} rows × {len(df.columns)} cols"):
                st.dataframe(df.head(5), use_container_width=True)
                q = assess_quality(df, name)
                st.caption(f"Data quality score: {q.overall_score}/100")
                if st.button("Remove", key=f"remove_{name}"):
                    store.remove_table(name)
                    st.rerun()

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.chat_log = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    with st.expander("Example questions"):
        st.markdown(
            "- Which region generated the highest revenue?\n"
            "- Show monthly sales trends.\n"
            "- Which products are underperforming?\n"
            "- What are the top five customers?\n"
            "- Generate SQL for total revenue by product category.\n"
            "- Detect anomalies in the sales data.\n"
            "- Check the sales data for quality issues.\n"
            "- Forecast next 3 months of revenue.\n"
            "- Give me a dashboard overview of this data."
        )


# ------------------------------------------------------------------- main
st.header("Chat with your data")

if not store.has_data():
    st.info("👋 Upload one or more CSV files in the sidebar (or load the sample "
            "dataset) to get started.")


def render_message(msg: dict):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sql_query"):
            with st.expander("SQL used"):
                st.code(msg["sql_query"], language="sql")
        if msg.get("sql_preview_md"):
            with st.expander("Result data"):
                st.markdown(msg["sql_preview_md"])
        if msg.get("chart_json"):
            fig = go.Figure(json.loads(msg["chart_json"]))
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{msg['id']}")
        if msg.get("dashboard_charts"):
            cols = st.columns(2)
            for i, chart in enumerate(msg["dashboard_charts"]):
                with cols[i % 2]:
                    st.plotly_chart(
                        go.Figure(json.loads(chart["figure"])),
                        use_container_width=True,
                        key=f"dash_{msg['id']}_{i}",
                    )


for msg in st.session_state.chat_log:
    render_message(msg)

question = st.chat_input(
    "Ask a question about your data..." if store.has_data() else "Upload data first, then ask a question...",
    disabled=not store.has_data() or not SETTINGS.google_api_key,
)

if question:
    user_msg = {"id": str(uuid.uuid4()), "role": "user", "content": question}
    st.session_state.chat_log.append(user_msg)
    render_message(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reset = {k: None for k in TRANSIENT_STATE_KEYS}
            graph_input = {
                **reset,
                "question": question,
                "schema_context": store.schema_context(),
                "messages": [HumanMessage(content=question)],
            }
            config = {"configurable": {"thread_id": st.session_state.session_id}}

            try:
                t0 = time.time()
                result = st.session_state.graph.invoke(graph_input, config=config)
                log.info("Turn handled by intent=%s in %.1fs", result.get("intent"), time.time() - t0)
            except MissingAPIKeyError as e:
                result = {"final_answer": str(e)}
            except Exception as e:  # noqa: BLE001
                log.exception("Graph invocation failed")
                result = {"final_answer": f"Something went wrong processing that request: {e}"}

        assistant_msg = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": result.get("final_answer", "(no answer produced)"),
            "sql_query": result.get("sql_query"),
            "sql_preview_md": result.get("sql_preview_md"),
            "chart_json": result.get("chart_json"),
            "dashboard_charts": result.get("dashboard_charts"),
        }
        st.session_state.chat_log.append(assistant_msg)
        st.markdown(assistant_msg["content"])
        if assistant_msg.get("sql_query"):
            with st.expander("SQL used"):
                st.code(assistant_msg["sql_query"], language="sql")
        if assistant_msg.get("sql_preview_md"):
            with st.expander("Result data"):
                st.markdown(assistant_msg["sql_preview_md"])
        if assistant_msg.get("chart_json"):
            st.plotly_chart(go.Figure(json.loads(assistant_msg["chart_json"])),
                             use_container_width=True, key=f"chart_{assistant_msg['id']}")
        if assistant_msg.get("dashboard_charts"):
            cols = st.columns(2)
            for i, chart in enumerate(assistant_msg["dashboard_charts"]):
                with cols[i % 2]:
                    st.plotly_chart(go.Figure(json.loads(chart["figure"])), use_container_width=True,
                                     key=f"dash_{assistant_msg['id']}_{i}")
