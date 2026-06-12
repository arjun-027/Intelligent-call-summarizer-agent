"""Backward-compatibility shim — the canonical module is now call_summarizer.utils.validator."""
from call_summarizer.utils.validator import (  # noqa: F401
    ValidationResult,
    validate_input_file,
    validate_summary,
)

__all__ = ["ValidationResult", "validate_input_file", "validate_summary"]
