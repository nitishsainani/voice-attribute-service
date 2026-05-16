"""
Structured JSON logging for voice-attribute-service.

Provides a pre-configured logger that outputs structured JSON lines,
making it easy to ingest logs into observability platforms (ELK, Datadog,
CloudWatch, etc.).

Usage:
    from app.logger import logger
    logger.info("Processing audio", extra={"duration_s": 3.2})
"""

import logging
import json
import sys
from datetime import datetime, timezone

from app.config import APP_NAME


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge any extra fields passed via `extra={...}`
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in (
                    "name", "msg", "args", "created", "relativeCreated",
                    "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                    "filename", "module", "pathname", "thread", "threadName",
                    "process", "processName", "levelname", "levelno",
                    "message", "msecs", "taskName",
                ):
                    log_entry[key] = value
        # Include exception traceback if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


def _create_logger() -> logging.Logger:
    """Create and configure the application logger."""
    _logger = logging.getLogger(APP_NAME)
    _logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if logger is re-created
    if not _logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        _logger.addHandler(handler)

    return _logger


logger = _create_logger()
