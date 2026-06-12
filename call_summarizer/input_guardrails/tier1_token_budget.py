"""Tier 1 — Token budget guard (BLOCKING).

Rejects transcripts that exceed the per-request token budget derived from
Groq's 6,000 TPM rate limit.  A single oversized upload would saturate the
per-minute quota and prevent the auto-retry loop from firing within the same
60-second window.

Rejection code: ``TRANSCRIPT_TOO_LONG``
"""

import logging
from typing import Optional

from .constants import _MAX_TRANSCRIPT_CHARS, _MAX_TRANSCRIPT_TOKENS
from .models import InputFinding

logger = logging.getLogger(__name__)


def _check_token_budget(content: str) -> Optional[InputFinding]:
    """Tier 1: Reject transcripts that exceed the Groq per-request token budget.

    Args:
        content: Raw transcript text to evaluate.

    Returns:
        :class:`InputFinding` with code ``TRANSCRIPT_TOO_LONG`` if *content*
        exceeds :data:`~.constants._MAX_TRANSCRIPT_CHARS`; ``None`` if within budget.
    """
    char_count = len(content)
    if char_count > _MAX_TRANSCRIPT_CHARS:
        return InputFinding(
            tier="error",
            code="TRANSCRIPT_TOO_LONG",
            message=(
                f"Transcript is {char_count:,} characters, which exceeds the "
                f"{_MAX_TRANSCRIPT_CHARS:,}-character limit "
                f"(~{_MAX_TRANSCRIPT_TOKENS:,} tokens). "
                "Please shorten the transcript or split it into smaller files."
            ),
            detail=f"chars={char_count}, limit={_MAX_TRANSCRIPT_CHARS}",
        )
    logger.debug("[INPUT-T1] %s chars within budget (%s limit)", char_count, _MAX_TRANSCRIPT_CHARS)
    return None
