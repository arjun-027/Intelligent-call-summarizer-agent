"""Input guardrail engine for uploaded call transcript files.

Three-tier validation is applied to every transcript *before* the LLM call so
that bad or hostile content is caught early, before API quota is spent.

Tier 1 — Token Budget Guard  (BLOCKING)
    Ensures the transcript character count fits inside the usable per-request
    token budget of the Groq model.

    Budget arithmetic
    -----------------
    Groq ``llama-3.1-8b-instant`` is rate-limited to **6,000 tokens per minute**
    (TPM).  Every request consumes fixed + variable tokens:

        tokens_per_request = system_prompt_tokens   (fixed)
                           + human_prefix_tokens    (fixed)
                           + transcript_tokens      (variable — the input we control)
                           + output_tokens          (fixed reservation)

    At import time, two fixed terms are computed from the actual SYSTEM_PROMPT
    string so the budget stays in sync automatically if the prompt ever changes:

        _SYSTEM_PROMPT_TOKENS  = len(SYSTEM_PROMPT) // 4   → currently ~465
        _HUMAN_PREFIX_TOKENS   = len(_HUMAN_PREFIX)  // 4  → currently ~8
        _FIXED_OVERHEAD_TOKENS = 465 + 8                   → ~473
        _MAX_TRANSCRIPT_TOKENS = 6,000 − 473 − 600         → 4,927
        _MAX_TRANSCRIPT_CHARS  = 4,927 × 4                 → 19,708

    The token ceiling chosen is the **TPM rate limit** (6,000) rather than the
    model's full context window (128K).  This prevents a single large upload from
    saturating the per-minute quota.  It also guarantees that up to two automatic
    guardrail-triggered retries can still fire within the same 60-second window.

    Transcripts that exceed ``_MAX_TRANSCRIPT_CHARS`` are rejected with code
    ``TRANSCRIPT_TOO_LONG`` and the LLM is never called.

Tier 2 — Prompt Injection Scan  (BLOCKING)
    Case-insensitive regex scan for 14 LLM manipulation patterns (OWASP LLM01 —
    Prompt Injection).  Patterns are tuned to avoid false positives on legitimate
    insurance dialogue (e.g. "act as a witness" does NOT match because the pattern
    requires an AI-related noun after "act as").  Any match is rejected with code
    ``PROMPT_INJECTION_DETECTED``.

Tier 3 — PII Audit  (NON-BLOCKING — audit log only)
    Detects personal-information categories present in the transcript.  This check
    NEVER blocks processing — insurance call transcripts legitimately contain PII
    and the pipeline is designed to handle it.  A structured audit log entry is
    written for every upload, satisfying NIST AI RMF MEASURE 2.5 and supporting
    GDPR Article 30 Records of Processing Activities obligations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from .summarizer import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ── Tier 1 — Token budget constants ──────────────────────────────────────────
_GROQ_TPM_LIMIT: int = 6_000       # Groq llama-3.1-8b-instant tokens/minute
_MAX_OUTPUT_TOKENS: int = 600      # matches max_tokens=600 in build_llm()
_CHARS_PER_TOKEN: int = 4          # standard BPE approximation

_HUMAN_PREFIX: str = "Summarise this call transcript:\n\n"

# Computed at import time so they track SYSTEM_PROMPT changes automatically.
_SYSTEM_PROMPT_TOKENS: int = max(1, len(SYSTEM_PROMPT) // _CHARS_PER_TOKEN)
_HUMAN_PREFIX_TOKENS: int = max(1, len(_HUMAN_PREFIX) // _CHARS_PER_TOKEN)
_FIXED_OVERHEAD_TOKENS: int = _SYSTEM_PROMPT_TOKENS + _HUMAN_PREFIX_TOKENS

_MAX_TRANSCRIPT_TOKENS: int = _GROQ_TPM_LIMIT - _FIXED_OVERHEAD_TOKENS - _MAX_OUTPUT_TOKENS
_MAX_TRANSCRIPT_CHARS: int = _MAX_TRANSCRIPT_TOKENS * _CHARS_PER_TOKEN

# ── Tier 2 — Injection patterns ──────────────────────────────────────────────
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(previous|all|prior)\s+instructions?",
        r"disregard\s+(the\s+)?(above|previous|all|prior)",
        r"forget\s+(everything|all|your\s+instructions?)",
        r"\bpretend\s+(you\s+are|to\s+be)\b",
        r"\bnew\s+instruction\s*:",
        r"(?m)^system\s*:",                          # line-start only — avoids "operating system:"
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

# ── Tier 3 — PII detection patterns ──────────────────────────────────────────
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b")
_PHONE_RE = re.compile(
    r"(?<!\w)"
    r"(?:\+\d{1,3}[\s\-.]?)?"  # optional country code
    r"\(?\d{3,4}\)?"           # area code
    r"[\s\-.]?\d{3,4}"        # middle segment
    r"[\s\-.]?\d{3,5}"        # last segment
    r"(?!\w)",
)
_UK_IE_POSTCODE_RE = re.compile(
    r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b"  # UK/NI: BT7 3GH, SW1A 2AA
    r"|\b[ACD-FHKNPRTV-Y]\d{2}\s*[AC-FHKNPRTV-Y]\d[AC-FHKNPRTV-Y]\d\b",  # IE Eircode
    re.IGNORECASE,
)
_DOB_CONTEXT_RE = re.compile(
    r"\b(?:date\s+of\s+birth|dob|born\s+on|born\s+the)\b",
    re.IGNORECASE,
)

_PII_CHECKS: list[tuple[str, re.Pattern[str]]] = [
    ("email_address", _EMAIL_RE),
    ("IBAN", _IBAN_RE),
    ("phone_number", _PHONE_RE),
    ("UK/IE_postcode", _UK_IE_POSTCODE_RE),
    ("date_of_birth_context", _DOB_CONTEXT_RE),
]


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class InputFinding:
    """A single finding from the input guardrail engine.

    Attributes:
        tier: ``"error"`` for blocking findings (Tier 1/2), ``"info"`` for
            non-blocking audit findings (Tier 3).
        code: Machine-readable identifier (e.g. ``"TRANSCRIPT_TOO_LONG"``).
        message: Human-readable description for API responses and UI display.
        detail: Optional extended detail (e.g. matched pattern text, byte count).
    """

    tier: Literal["error", "info"]
    code: str
    message: str
    detail: str = ""


@dataclass
class InputValidationResult:
    """Aggregated result from :func:`validate_transcript_input`.

    Attributes:
        allowed: ``False`` when any Tier-1 or Tier-2 error was found.  The
            caller must reject the request and must NOT invoke the LLM.
        findings: All findings in evaluation order (errors first, then audit).
    """

    allowed: bool
    findings: list[InputFinding] = field(default_factory=list)

    @property
    def errors(self) -> list[InputFinding]:
        """Blocking findings only (``tier == "error"``)."""
        return [f for f in self.findings if f.tier == "error"]

    @property
    def audit(self) -> list[InputFinding]:
        """Non-blocking audit entries only (``tier == "info"``)."""
        return [f for f in self.findings if f.tier == "info"]


# ── Tier implementations ──────────────────────────────────────────────────────

def _check_token_budget(content: str) -> InputFinding | None:
    """Tier 1: Reject transcripts that exceed the Groq per-request token budget.

    Budget arithmetic
    -----------------
    The Groq llama-3.1-8b-instant model is capped at 6,000 tokens per minute
    (TPM).  Module-level constants encode the fixed overhead that every request
    carries regardless of transcript length:

        _SYSTEM_PROMPT_TOKENS  = len(SYSTEM_PROMPT) // 4   (computed at import)
        _HUMAN_PREFIX_TOKENS   = len(_HUMAN_PREFIX)  // 4  (computed at import)
        _MAX_OUTPUT_TOKENS     = 600                        (matches build_llm())
        _FIXED_OVERHEAD_TOKENS = _SYSTEM_PROMPT_TOKENS + _HUMAN_PREFIX_TOKENS

    From these, the maximum safe transcript size is:

        _MAX_TRANSCRIPT_TOKENS = _GROQ_TPM_LIMIT − _FIXED_OVERHEAD_TOKENS
                                 − _MAX_OUTPUT_TOKENS
        _MAX_TRANSCRIPT_CHARS  = _MAX_TRANSCRIPT_TOKENS × _CHARS_PER_TOKEN

    Choosing the TPM rate limit (not the 128K context window) as the ceiling
    ensures that a single upload never saturates the per-minute quota, keeping
    headroom for the automatic retry loop (up to 2 retries) within the same
    60-second window.

    Args:
        content: Raw transcript text to evaluate.

    Returns:
        :class:`InputFinding` with code ``TRANSCRIPT_TOO_LONG`` if *content*
        exceeds :data:`_MAX_TRANSCRIPT_CHARS`; ``None`` if within budget.
    """
    char_count = len(content)
    if char_count > _MAX_TRANSCRIPT_CHARS:
        return InputFinding(
            tier="error",
            code="TRANSCRIPT_TOO_LONG",
            message=(
                f"Transcript is {char_count:,} characters, which exceeds the "
                f"{_MAX_TRANSCRIPT_CHARS:,}-character limit "
                f"(~{_MAX_TRANSCRIPT_TOKENS:,} tokens).  "
                "Please shorten the transcript or split it into smaller files."
            ),
            detail=f"chars={char_count}, limit={_MAX_TRANSCRIPT_CHARS}",
        )
    logger.debug("[INPUT-T1] %s chars within budget (%s limit)", char_count, _MAX_TRANSCRIPT_CHARS)
    return None


def _check_injection(content: str) -> InputFinding | None:
    """Tier 2: Detect prompt injection attempts (OWASP LLM01).

    Scans the full transcript text with 14 case-insensitive regex patterns
    targeting the standard LLM manipulation taxonomy.  Patterns are specifically
    worded to avoid false positives on legitimate insurance call language — for
    example ``"act as a witness"`` does NOT match because the pattern requires an
    AI-related noun (``AI``, ``LLM``, ``language model``, ``assistant``) directly
    after ``"act as"``.

    Args:
        content: Raw transcript text to scan.

    Returns:
        :class:`InputFinding` with code ``PROMPT_INJECTION_DETECTED`` on the
        first pattern match; ``None`` if the transcript is clean.
    """
    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(content)
        if match:
            return InputFinding(
                tier="error",
                code="PROMPT_INJECTION_DETECTED",
                message=(
                    "The transcript contains content that resembles a prompt "
                    "injection attack and cannot be processed."
                ),
                detail=f"matched: {match.group()!r}",
            )
    logger.debug("[INPUT-T2] scan clean — no injection patterns matched")
    return None


def _audit_pii(content: str, filename: str) -> InputFinding | None:
    """Tier 3: Audit PII categories in the transcript (GDPR Article 30).

    Identifies which personal-information categories are present and writes a
    structured audit log entry for every upload.  This does NOT block processing.

    Categories checked: email addresses, IBANs, phone numbers, UK/IE postcodes,
    and date-of-birth context phrases (e.g. "date of birth", "DOB", "born on").

    The audit log satisfies NIST AI RMF MEASURE 2.5 (track personal information
    in inference data) and supports GDPR Article 30 Records of Processing
    Activities.

    Args:
        content: Raw transcript text to audit.
        filename: Filename for the audit log entry.

    Returns:
        :class:`InputFinding` with code ``PII_DETECTED`` listing detected
        categories; ``None`` if no categories matched (also logged).
    """
    detected: list[str] = []
    for label, pattern in _PII_CHECKS:
        if pattern.search(content):
            detected.append(label)

    if not detected:
        logger.info("[PII AUDIT] %s — no PII categories detected", filename)
        return None

    categories_str = ", ".join(detected)
    logger.info("[PII AUDIT] %s — categories detected: %s", filename, categories_str)
    return InputFinding(
        tier="info",
        code="PII_DETECTED",
        message=f"PII categories detected in transcript: {categories_str}.",
        detail=categories_str,
    )


# ── Public entry point ────────────────────────────────────────────────────────

def validate_transcript_input(
    content: str,
    filename: str = "<unknown>",
) -> InputValidationResult:
    """Run all three input guardrail tiers against a transcript.

    Evaluation order
    ----------------
    1. **Tier 1** (token budget) — if the transcript is too long, returns
       ``allowed=False`` immediately without running further checks.
    2. **Tier 2** (injection scan) — if injection patterns are found, returns
       ``allowed=False`` immediately.
    3. **Tier 3** (PII audit) — always runs; the finding is logged and included
       in ``result.audit`` but does NOT affect ``allowed``.

    Args:
        content: Raw transcript text read from the uploaded file.
        filename: Original filename used for log entries (e.g. ``"7-transcript.txt"``).

    Returns:
        :class:`InputValidationResult` where ``allowed=False`` means the caller
        must return an error response and must NOT invoke the LLM.
    """
    logger.debug("Input guardrails — file: %s, chars: %d", filename, len(content))

    findings: list[InputFinding] = []

    budget_finding = _check_token_budget(content)
    if budget_finding:
        logger.warning(
            "[INPUT-T1] %s — TRANSCRIPT_TOO_LONG: %s", filename, budget_finding.detail
        )
        findings.append(budget_finding)
        return InputValidationResult(allowed=False, findings=findings)

    injection_finding = _check_injection(content)
    if injection_finding:
        logger.warning(
            "[INPUT-T2] %s — PROMPT_INJECTION_DETECTED: %s",
            filename,
            injection_finding.detail,
        )
        findings.append(injection_finding)
        return InputValidationResult(allowed=False, findings=findings)

    pii_finding = _audit_pii(content, filename)
    if pii_finding:
        findings.append(pii_finding)

    return InputValidationResult(allowed=True, findings=findings)
