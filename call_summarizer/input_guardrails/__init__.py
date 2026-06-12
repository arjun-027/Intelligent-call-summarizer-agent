"""Input guardrail engine for uploaded call transcript files.

Three-tier validation is applied to every transcript *before* the LLM call so
that bad or hostile content is caught early, before API quota is spent.

Package structure
-----------------
constants.py          Token budget math, injection patterns, PII regexes.
models.py             InputFinding, InputValidationResult data classes.
tier1_token_budget.py Tier 1 — reject transcripts exceeding the token budget.
tier2_injection.py    Tier 2 — block prompt injection attempts (OWASP LLM01).
tier3_pii.py          Tier 3 — audit PII categories (GDPR Article 30, non-blocking).
runner.py             Orchestrates all tiers; exposes the public API.

Public API
----------
    from call_summarizer.input_guardrails import (
        validate_transcript_input,
        InputFinding,
        InputValidationResult,
    )

    result = validate_transcript_input(transcript_text, filename)
    if not result.allowed:
        # reject request — do not call the LLM
        ...
"""

from .constants import _MAX_TRANSCRIPT_CHARS, _MAX_TRANSCRIPT_TOKENS
from .models import InputFinding, InputValidationResult
from .runner import validate_transcript_input
from .tier1_token_budget import _check_token_budget
from .tier2_injection import _check_injection
from .tier3_pii import _audit_pii

__all__ = [
    # Primary public API
    "validate_transcript_input",
    "InputFinding",
    "InputValidationResult",
    # Constants and internal functions re-exported for tests
    "_MAX_TRANSCRIPT_CHARS",
    "_MAX_TRANSCRIPT_TOKENS",
    "_check_token_budget",
    "_check_injection",
    "_audit_pii",
]
