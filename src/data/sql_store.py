"""
SQLite-backed data store for the current session.

Every uploaded CSV becomes a real SQLite table (in-memory), so the app
can execute genuine SQL - including JOINs across multiple uploaded
files - rather than faking it with pandas. This class also builds the
schema/context string that gets fed to Gemini so it can write correct
SQL against whatever the user uploaded.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import pandas as pd

from src.data.loader import LoadedTable
from src.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class SQLStore:
    """Owns one in-memory SQLite connection for a single user session."""

    conn: sqlite3.Connection = field(default_factory=lambda: sqlite3.connect(":memory:", check_same_thread=False))
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    original_filenames: dict[str, str] = field(default_factory=dict)

    def add_table(self, loaded: LoadedTable) -> None:
        loaded.df.to_sql(loaded.table_name, self.conn, if_exists="replace", index=False)
        self.tables[loaded.table_name] = loaded.df
        self.original_filenames[loaded.table_name] = loaded.original_filename
        log.info("Loaded table '%s' into SQLite (%d rows)", loaded.table_name, len(loaded.df))

    def remove_table(self, table_name: str) -> None:
        if table_name in self.tables:
            self.conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            del self.tables[table_name]
            self.original_filenames.pop(table_name, None)

    def has_data(self) -> bool:
        return bool(self.tables)

    def schema_context(self, sample_rows: int = 3) -> str:
        """Human/LLM-readable description of every table: columns, dtypes, sample rows."""
        if not self.tables:
            return "No tables loaded yet."

        blocks = []
        for name, df in self.tables.items():
            cols = ", ".join(f"{c} ({str(dt)})" for c, dt in df.dtypes.items())
            sample = df.head(sample_rows).to_csv(index=False)
            blocks.append(
                f"Table: {name}  (source file: {self.original_filenames.get(name, '?')}, "
                f"{len(df)} rows)\n"
                f"Columns: {cols}\n"
                f"Sample rows:\n{sample}"
            )
        return "\n\n".join(blocks)

    def table_names(self) -> list[str]:
        return list(self.tables.keys())
