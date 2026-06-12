"""Backward-compatibility shim — the canonical module is now call_summarizer.utils.storage."""
from call_summarizer.utils.storage import derive_output_path, save_summary  # noqa: F401

__all__ = ["derive_output_path", "save_summary"]
