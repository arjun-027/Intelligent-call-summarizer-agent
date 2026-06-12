"""CLI entry point for the call summarisation agent.

Usage::

    uv run main.py

Reads all ``.txt`` transcripts from ``Input_data/``, generates a structured
summary for each using Groq's LLM, and writes results to ``Output_data/``.
Configuration is driven by ``.env`` — see README for setup instructions.
"""

import logging

from call_summarizer.config import load_config
from call_summarizer.observability.logging import setup_logging
from call_summarizer.service import process_directory

logger = logging.getLogger(__name__)


def main() -> None:
    """Parse configuration, run the batch pipeline, and report results."""
    setup_logging()

    try:
        config = load_config()
    except EnvironmentError as exc:
        logger.error("Configuration error: %s", exc)
        print(f"Configuration error: {exc}")
        return

    results = process_directory(config)

    if not results:
        return

    print("\n── Summary ──────────────────────────────")
    for result in results:
        status = "OK " if result.success else "ERR"
        name = result.transcript_path.name
        if result.success:
            print(f"  [{status}] {name} → {result.output_path.name}")
        else:
            print(f"  [{status}] {name} — {result.error}")
        if result.issues:
            for issue in result.issues:
                print(f"         ⚠ {issue}")
    print("─────────────────────────────────────────")


if __name__ == "__main__":
    main()
