"""Centralised logging configuration for the call summariser package.

Call ``setup_logging()`` once at application startup. All modules then obtain
their logger via ``logging.getLogger(__name__)``.

Log file location : logs/call_summarizer.log
Max file size     : 5 MB
Backup count      : 3  (logs rotated as .log.1, .log.2, .log.3)
"""

import logging
import logging.handlers
from pathlib import Path

_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "call_summarizer.log"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_BACKUP_COUNT = 3
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s.%(funcName)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a rotating file handler and a console handler.

    Args:
        level: Logging level applied to both handlers (default: INFO).

    Creates:
        ``logs/`` directory if it does not already exist.
        ``logs/call_summarizer.log`` as the primary log file.
    """
    _LOG_DIR.mkdir(exist_ok=True)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        _LOG_FILE,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.getLogger(__name__).info(
        "Logging initialised — file: %s, max size: %d MB, backups: %d",
        _LOG_FILE,
        _MAX_BYTES // (1024 * 1024),
        _BACKUP_COUNT,
    )
