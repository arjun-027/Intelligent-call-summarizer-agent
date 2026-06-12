"""Tier 2 — Prompt injection scan (BLOCKING).

Case-insensitive regex scan for 14 LLM manipulation patterns (OWASP LLM01 —
Prompt Injection).  Patterns are tuned to avoid false positives on legitimate
insurance call language.

Rejection code: ``PROMPT_INJECTION_DETECTED``
"""

import logging
from typing import Optional

from .constants import _INJECTION_PATTERNS
from .models import InputFinding

logger = logging.getLogger(__name__)


def _check_injection(content: str) -> Optional[InputFinding]:
    """Tier 2: Detect prompt injection attempts (OWASP LLM01).

    Scans the full transcript text with 14 case-insensitive regex patterns.
    Returns on the *first* match so the caller receives the matched evidence.

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
