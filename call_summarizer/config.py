"""Application configuration loaded from environment variables.

Usage::

    from call_summarizer.config import load_config
    config = load_config()
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_GROQ_MODEL_DEFAULT = "llama-3.1-8b-instant"
_RATE_LIMIT_DELAY_SECONDS = 3.0


@dataclass(frozen=True)
class Config:
    """Immutable application configuration.

    Attributes:
        groq_api_key: Groq API key used to authenticate LLM requests.
        groq_model: Groq model identifier string.
        input_dir: Directory from which transcript .txt files are read.
        output_dir: Directory where generated summary files are written.
        rate_limit_delay_seconds: Pause inserted between consecutive LLM calls
            to stay within Groq's 6K tokens-per-minute limit.
    """

    groq_api_key: str
    groq_model: str
    input_dir: Path
    output_dir: Path
    rate_limit_delay_seconds: float


def _log_langsmith_status() -> None:
    """Log whether LangSmith tracing is active so operators can confirm observability.

    LangSmith tracing is activated automatically by LangChain/LangGraph when the
    LANGSMITH_TRACING environment variable is set to ``true``.  This function
    reads the same variables and writes a clear INFO or WARNING log line so the
    startup output is unambiguous about whether traces are being sent.

    Called automatically at the end of :func:`load_config`.
    """
    tracing = os.getenv("LANGSMITH_TRACING", "false").lower()
    if tracing in ("true", "1", "yes"):
        project = os.getenv("LANGSMITH_PROJECT", "<default>")
        endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        api_key_set = bool(os.getenv("LANGSMITH_API_KEY"))
        logger.info(
            "LangSmith tracing ENABLED — project: %r, endpoint: %s, api_key: %s",
            project,
            endpoint,
            "set" if api_key_set else "MISSING — traces will not be sent",
        )
        if not api_key_set:
            logger.warning(
                "LANGSMITH_API_KEY is not set. "
                "Add it to .env or set LANGSMITH_TRACING=false to suppress this warning."
            )
    else:
        logger.info("LangSmith tracing DISABLED (set LANGSMITH_TRACING=true to enable)")


def load_config() -> Config:
    """Load and validate application configuration from environment variables.

    Reads a ``.env`` file if present via python-dotenv, then reads individual
    environment variables. Overrides can be set via env vars:
    ``GROQ_API_KEY``, ``GROQ_MODEL``, ``INPUT_DIR``, ``OUTPUT_DIR``.

    Returns:
        A fully validated :class:`Config` instance.

    Raises:
        EnvironmentError: If ``GROQ_API_KEY`` is absent or still the placeholder
            value, indicating the developer has not yet configured credentials.
    """
    load_dotenv()
    logger.debug("Loaded .env file")

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key_here":
        logger.error("GROQ_API_KEY is missing or not set in .env")
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file before running."
        )

    config = Config(
        groq_api_key=api_key,
        groq_model=os.getenv("GROQ_MODEL", _GROQ_MODEL_DEFAULT),
        input_dir=Path(os.getenv("INPUT_DIR", "Input_data")),
        output_dir=Path(os.getenv("OUTPUT_DIR", "Output_data")),
        rate_limit_delay_seconds=_RATE_LIMIT_DELAY_SECONDS,
    )

    logger.info(
        "Config loaded — model: %s, input: %s, output: %s",
        config.groq_model,
        config.input_dir,
        config.output_dir,
    )
    _log_langsmith_status()
    return config
