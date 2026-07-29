"""
Safe SQL execution against the session's SQLite store.

Only SELECT / WITH (CTE) statements are allowed - this is a read-only
analyst tool, not a database admin console, and the LLM's generated SQL
should never be trusted to mutate data. Results are capped to
MAX_SQL_ROWS to keep prompts small and the UI responsive.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

import pandas as pd

from src.config import SETTINGS
from src.data.sql_store import SQLStore
from src.logging_config import get_logger

log = get_logger(__name__)

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM)\b",
    re.IGNORECASE,
)


@dataclass
class SQLResult:
    query: str
    ok: bool
    df: pd.DataFrame | None = None
    error: str | None = None
    truncated: bool = False
    row_count: int = 0


def is_read_only(query: str) -> bool:
    stripped = query.strip().rstrip(";")
    if not stripped:
        return False
    if _FORBIDDEN.search(stripped):
        return False
    return stripped[:1].upper() != "" and (
        stripped.upper().startswith("SELECT") or stripped.upper().startswith("WITH")
    )


def run_sql(store: SQLStore, query: str) -> SQLResult:
    """Execute a read-only SQL query against the session's tables."""
    query = query.strip().rstrip(";")

    if not is_read_only(query):
        msg = (
            "Only read-only SELECT/WITH queries are allowed. Rejected query: "
            f"{query[:200]}"
        )
        log.warning(msg)
        return SQLResult(query=query, ok=False, error=msg)

    try:
        df = pd.read_sql_query(query, store.conn)
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        log.warning("SQL error for query %r: %s", query, exc)
        return SQLResult(query=query, ok=False, error=str(exc))

    row_count = len(df)
    truncated = row_count > SETTINGS.max_sql_rows
    if truncated:
        df = df.head(SETTINGS.max_sql_rows)

    return SQLResult(query=query, ok=True, df=df, truncated=truncated, row_count=row_count)
