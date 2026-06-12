"""Backward-compatibility shim — the canonical module is now call_summarizer.observability.logging."""
from call_summarizer.observability.logging import setup_logging  # noqa: F401

__all__ = ["setup_logging"]
