"""Centralised logging configuration for the call summariser package.

Call :func:`setup_logging` once at application startup (in the FastAPI
lifespan, in ``run.py``, or at the top of the CLI entry point).  Every module
in the package then obtains its own logger via::

    import logging
    logger = logging.getLogger(__name__)

Log configuration
-----------------
Log file   : ``logs/call_summarizer.log``
Max size   : 5 MB per file
Backups    : 3 rotated copies (.log.1, .log.2, .log.3)
Format     : ``timestamp | LEVEL    | module.function | message``
"""

import logging
import logging.handlers
from pathlib import Path

_LOG_DIR: Path = Path("logs")
_LOG_FILE: Path = _LOG_DIR / "call_summarizer.log"
_MAX_FILE_BYTES: int = 5 * 1024 * 1024  # 5 MB
_BACKUP_COUNT: int = 3
_LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s.%(funcName)s | %(message)s"
_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a rotating file handler and a console handler.

    This function is idempotent: calling it multiple times does not add
    duplicate handlers (they are cleared and rebuilt each call).

    Handlers created
    ----------------
    1. :class:`~logging.handlers.RotatingFileHandler` — writes to
       ``logs/call_summarizer.log``, rotates at 5 MB, keeps 3 backups.
    2. :class:`~logging.StreamHandler` — writes to ``stderr`` at the same level.

    Args:
        level: Root logging level applied to all handlers.
            Defaults to :data:`logging.INFO`.
            Pass :data:`logging.DEBUG` for verbose output during development.

    Side effects:
        Creates ``logs/`` directory if it does not already exist.
        Replaces any existing handlers on the root logger.
    """
    _LOG_DIR.mkdir(exist_ok=True)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        _LOG_FILE,
        maxBytes=_MAX_FILE_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # Clear existing handlers to prevent duplication on repeated calls.
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.getLogger(__name__).info(
        "Logging initialised — file: %s, max size: %d MB, backups: %d",
        _LOG_FILE,
        _MAX_FILE_BYTES // (1024 * 1024),
        _BACKUP_COUNT,
    )
