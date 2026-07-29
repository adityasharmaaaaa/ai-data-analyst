"""
CSV validation and loading.

Turns raw uploaded file bytes into clean pandas DataFrames + a sanitized
table name, with real validation (not just "does it parse"): empty files,
duplicate headers, zero-row files, and oversized uploads are all caught
here and surfaced as human-readable errors instead of crashing the app.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass

import pandas as pd

from src.config import SETTINGS
from src.logging_config import get_logger

log = get_logger(__name__)


class CSVValidationError(Exception):
    """Raised when an uploaded file fails validation."""


@dataclass
class LoadedTable:
    table_name: str
    original_filename: str
    df: pd.DataFrame


_NAME_RE = re.compile(r"[^a-zA-Z0-9_]")


def sanitize_table_name(filename: str, existing: set[str]) -> str:
    """Turn 'Sales Q1 (final).csv' into a safe, unique SQL table name."""
    stem = filename.rsplit(".", 1)[0]
    name = _NAME_RE.sub("_", stem).strip("_").lower() or "table"
    if name[0].isdigit():
        name = f"t_{name}"

    candidate = name
    i = 2
    while candidate in existing:
        candidate = f"{name}_{i}"
        i += 1
    return candidate


def validate_and_load_csv(filename: str, raw_bytes: bytes, existing_tables: set[str]) -> LoadedTable:
    """
    Validate an uploaded CSV's bytes and parse it into a DataFrame.

    Raises CSVValidationError with a user-facing message on any problem.
    """
    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > SETTINGS.max_upload_mb:
        raise CSVValidationError(
            f"'{filename}' is {size_mb:.1f} MB, which exceeds the "
            f"{SETTINGS.max_upload_mb} MB limit."
        )

    if not raw_bytes.strip():
        raise CSVValidationError(f"'{filename}' is empty.")

    # Check for duplicate headers on the *raw* first line. pandas silently
    # renames duplicate columns (e.g. 'a' -> 'a', 'a.1') rather than
    # erroring, so this must happen before pd.read_csv or it's unreachable.
    first_line = raw_bytes.splitlines()[0].decode("utf-8", errors="replace")
    raw_headers = next(csv.reader([first_line]))
    dupe_headers = [h for h in raw_headers if raw_headers.count(h) > 1]
    if dupe_headers:
        raise CSVValidationError(
            f"'{filename}' has duplicate column headers: {sorted(set(dupe_headers))}."
        )

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except pd.errors.EmptyDataError:
        raise CSVValidationError(f"'{filename}' has no parsable columns.")
    except pd.errors.ParserError as exc:
        raise CSVValidationError(f"'{filename}' is not valid CSV: {exc}")
    except UnicodeDecodeError:
        # retry with a more permissive encoding before giving up
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding="latin1")
        except Exception as exc:  # noqa: BLE001
            raise CSVValidationError(f"'{filename}' has an unsupported text encoding: {exc}")

    if df.shape[1] == 0:
        raise CSVValidationError(f"'{filename}' has no columns.")

    if df.shape[0] == 0:
        raise CSVValidationError(f"'{filename}' has headers but zero data rows.")

    # normalize column names to be SQL/identifier friendly, but keep a
    # mapping so we can show the user their original names if needed
    df.columns = [
        _NAME_RE.sub("_", str(c)).strip("_").lower() or f"col_{i}"
        for i, c in enumerate(df.columns)
    ]

    table_name = sanitize_table_name(filename, existing_tables)
    log.info("Validated '%s' -> table '%s' (%d rows, %d cols)",
              filename, table_name, df.shape[0], df.shape[1])

    return LoadedTable(table_name=table_name, original_filename=filename, df=df)
