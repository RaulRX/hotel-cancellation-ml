"""Structured logging configuration for the API."""
from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(handler)
    root.setLevel(level.upper())
