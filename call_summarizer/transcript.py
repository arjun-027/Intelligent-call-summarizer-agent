"""Backward-compatibility shim — the canonical module is now call_summarizer.utils.transcript."""
from call_summarizer.utils.transcript import find_transcripts, load_transcript  # noqa: F401

__all__ = ["find_transcripts", "load_transcript"]
