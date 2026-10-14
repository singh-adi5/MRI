"""Structured JSON logging with a per-request id.

One line per event, one JSON object per line. The request id is carried in a
ContextVar so any log call inside a request picks it up without being passed it,
and it is echoed back on the `X-Request-ID` response header.
"""
from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Final

request_id_var: ContextVar[str] = ContextVar("mri_request_id", default="-")

REQUEST_ID_HEADER: Final[str] = "X-Request-ID"

# Attributes LogRecord always carries; anything else an adapter attached is
# treated as structured context and merged into the line.
_STANDARD: Final[frozenset[str]] = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            # The formatted traceback goes to the log, never to the client.
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    # Replace handlers rather than add: uvicorn installs its own and we want one
    # machine-readable stream, not two formats interleaved.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(getattr(logging, level, logging.INFO))

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = [handler]
        logger.propagate = False
