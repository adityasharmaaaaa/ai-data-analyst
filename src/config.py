"""
Central application configuration.

All runtime settings are read from environment variables (loaded from a
local .env file via python-dotenv). Nothing here should be hardcoded
secrets — see .env.example for the variables this app expects.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env once, at import time, before anything else reads os.environ.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    google_api_key: str | None
    gemini_model: str
    gemini_temperature: float
    max_sql_rows: int
    log_level: str
    max_upload_mb: int


def get_settings() -> Settings:
    return Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        # Default picked for broad availability/stability; override freely.
        # Google renames/retires Gemini model ids fairly often — if this
        # default 404s on your key, check https://ai.google.dev/gemini-api/docs/models
        # and set GEMINI_MODEL in your .env instead of editing code.
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        gemini_temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.2")),
        max_sql_rows=int(os.getenv("MAX_SQL_ROWS", "500")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "50")),
    )


SETTINGS = get_settings()
