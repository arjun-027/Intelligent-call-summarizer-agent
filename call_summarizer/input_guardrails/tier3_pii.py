"""Tier 3 — PII audit (NON-BLOCKING — audit log only).

Detects personal-information categories present in the transcript and writes a
structured audit log entry.  This check NEVER blocks processing — insurance
call transcripts legitimately contain PII and the pipeline is designed to
handle it.

Categories checked
------------------
- Email addresses
- IBANs (bank account numbers)
- Phone numbers
- UK / Irish postcodes
- Date-of-birth context phrases (DOB, "date of birth", "born on", …)

Compliance notes
----------------
- Satisfies NIST AI RMF MEASURE 2.5 (track personal information in inference data).
- Supports GDPR Article 30 Records of Processing Activities obligations.
"""

import logging
from typing import Optional

from .constants import _PII_CHECKS
from .models import InputFinding

logger = logging.getLogger(__name__)


def _audit_pii(content: str, filename: str) -> Optional[InputFinding]:
    """Tier 3: Audit PII categories in the transcript (GDPR Article 30).

    Writes a structured audit log entry regardless of whether PII is present.
    Does NOT affect the ``allowed`` flag of the result.

    Args:
        content: Raw transcript text to audit.
        filename: Original filename used for the audit log entry.

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
