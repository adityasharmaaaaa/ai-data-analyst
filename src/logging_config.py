"""
Shared logging setup.

Every module gets a logger via `get_logger(__name__)`. Configure the
verbosity with LOG_LEVEL in .env. Logs go to stdout (picked up by
Docker/whatever process manager runs the app) and to a rotating file
under logs/ for local debugging.
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config import SETTINGS

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(SETTINGS.log_level)

    formatter = logging.Formatter(_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = RotatingFileHandler(
        _LOG_DIR / "app.log", maxBytes=2_000_000, backupCount=3
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)
