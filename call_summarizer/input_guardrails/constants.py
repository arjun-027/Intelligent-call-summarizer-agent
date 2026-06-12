"""Constants and compiled regex patterns for the input guardrail engine.

Token-budget constants are computed at import time from the live SYSTEM_PROMPT
so they stay in sync automatically if the prompt ever changes.
"""

import re

from ..summarizer import SYSTEM_PROMPT

# ── Tier 1 — Token budget ──────────────────────────────────────────────────────

_GROQ_TPM_LIMIT: int = 6_000        # Groq llama-3.1-8b-instant tokens/minute
_MAX_OUTPUT_TOKENS: int = 600       # matches max_tokens=600 in build_llm()
_CHARS_PER_TOKEN: int = 4           # standard BPE approximation

_HUMAN_PREFIX: str = "Summarise this call transcript:\n\n"

_SYSTEM_PROMPT_TOKENS: int = max(1, len(SYSTEM_PROMPT) // _CHARS_PER_TOKEN)
_HUMAN_PREFIX_TOKENS: int = max(1, len(_HUMAN_PREFIX) // _CHARS_PER_TOKEN)
_FIXED_OVERHEAD_TOKENS: int = _SYSTEM_PROMPT_TOKENS + _HUMAN_PREFIX_TOKENS

_MAX_TRANSCRIPT_TOKENS: int = _GROQ_TPM_LIMIT - _FIXED_OVERHEAD_TOKENS - _MAX_OUTPUT_TOKENS
_MAX_TRANSCRIPT_CHARS: int = _MAX_TRANSCRIPT_TOKENS * _CHARS_PER_TOKEN

# ── Tier 2 — Prompt injection patterns (OWASP LLM01) ─────────────────────────
#
# 14 case-insensitive patterns tuned to avoid false positives on legitimate
# insurance call language (e.g. "act as a witness" does NOT match because the
# pattern requires an AI-related noun after "act as").

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(previous|all|prior)\s+instructions?",
        r"disregard\s+(the\s+)?(above|previous|all|prior)",
        r"forget\s+(everything|all|your\s+instructions?)",
        r"\bpretend\s+(you\s+are|to\s+be)\b",
        r"\bnew\s+instruction\s*:",
        r"(?m)^system\s*:",                          # line-start only
        r"\[system\]",
        r"reveal\s+(your\s+)?(system\s+prompt|instructions?|prompt)",
        r"print\s+(your\s+)?(instructions?|system\s+prompt)",
        r"do\s+not\s+summ(ari[sz]|ary)",
        r"instead\s+of\s+summ(ari[sz]|ary)",
        r"output\s+the\s+following\b",
        r"you\s+are\s+now\s+(?:an?\s+)?(?:AI|LLM|language\s+model|assistant)\b",
        r"\bact\s+as\s+(?:an?\s+)?(?:AI|LLM|language\s+model|assistant)\b",
    ]
]

# ── Tier 3 — PII detection patterns (GDPR Article 30) ────────────────────────

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b")
_PHONE_RE = re.compile(
    r"(?<!\w)"
    r"(?:\+\d{1,3}[\s\-.]?)?"
    r"\(?\d{3,4}\)?"
    r"[\s\-.]?\d{3,4}"
    r"[\s\-.]?\d{3,5}"
    r"(?!\w)",
)
_UK_IE_POSTCODE_RE = re.compile(
    r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b"
    r"|\b[ACD-FHKNPRTV-Y]\d{2}\s*[AC-FHKNPRTV-Y]\d[AC-FHKNPRTV-Y]\d\b",
    re.IGNORECASE,
)
_DOB_CONTEXT_RE = re.compile(
    r"\b(?:date\s+of\s+birth|dob|born\s+on|born\s+the)\b",
    re.IGNORECASE,
)

_PII_CHECKS: list[tuple[str, re.Pattern[str]]] = [
    ("email_address",        _EMAIL_RE),
    ("IBAN",                 _IBAN_RE),
    ("phone_number",         _PHONE_RE),
    ("UK/IE_postcode",       _UK_IE_POSTCODE_RE),
    ("date_of_birth_context", _DOB_CONTEXT_RE),
]
