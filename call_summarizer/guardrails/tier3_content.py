"""Tier 3 — Content integrity checks for generated call summaries.

All functions here compare the summary against the original transcript and
return ``Finding(tier='warning', ...)`` findings.  Tier 3 is skipped when no
transcript is supplied.

Checks in this tier
-------------------
- AMOUNT_NOT_IN_TRANSCRIPT
- REFERENCE_NOT_IN_TRANSCRIPT
- IBAN_NOT_IN_TRANSCRIPT
- EMAIL_NOT_IN_TRANSCRIPT
- UNVERIFIED_CONFIRMATION
- CONDITIONAL_SECTION_UNJUSTIFIED
"""

import logging
import re

from ..models import Finding
from .constants import _CONDITIONAL_DOMAIN_TERMS, _CONFIRMATION_PHRASE_CHECKS
from .helpers import (
    _extract_amounts,
    _extract_emails,
    _extract_ibans,
    _extract_references,
)

logger = logging.getLogger(__name__)


# ── Individual checks ──────────────────────────────────────────────────────────


def _check_amounts_in_transcript(summary: str, transcript: str) -> list[Finding]:
    """Warn for monetary amounts in summary that cannot be matched in the transcript."""
    summary_amounts = _extract_amounts(summary)
    transcript_amounts = _extract_amounts(transcript)
    findings: list[Finding] = []
    for amount in summary_amounts:
        if amount and amount not in transcript_amounts:
            if amount not in transcript.replace(",", "").replace(" ", ""):
                findings.append(Finding(
                    tier="warning",
                    code="AMOUNT_NOT_IN_TRANSCRIPT",
                    message=f"Amount '{amount}' in summary could not be verified in the transcript — possible hallucination.",
                    detail=amount,
                ))
    return findings


def _check_references_in_transcript(summary: str, transcript: str) -> list[Finding]:
    """Warn for reference numbers in summary not found in the transcript."""
    findings: list[Finding] = []
    for ref in _extract_references(summary):
        if ref not in transcript.upper():
            findings.append(Finding(
                tier="warning",
                code="REFERENCE_NOT_IN_TRANSCRIPT",
                message=f"Reference '{ref}' in summary not found in transcript — possible error.",
                detail=ref,
            ))
    return findings


def _check_ibans_in_transcript(summary: str, transcript: str) -> list[Finding]:
    """Warn for IBANs in summary that do not match those in the transcript."""
    transcript_ibans = _extract_ibans(transcript)
    findings: list[Finding] = []
    for iban in _extract_ibans(summary):
        if iban not in transcript_ibans and iban not in transcript.upper().replace(" ", ""):
            findings.append(Finding(
                tier="warning",
                code="IBAN_NOT_IN_TRANSCRIPT",
                message=f"IBAN '{iban}' in summary could not be matched to the transcript — verify carefully.",
                detail=iban,
            ))
    return findings


def _check_emails_in_transcript(summary: str, transcript: str) -> list[Finding]:
    """Warn for email addresses in summary not found in the transcript."""
    findings: list[Finding] = []
    for email in _extract_emails(summary):
        if email not in transcript.lower():
            findings.append(Finding(
                tier="warning",
                code="EMAIL_NOT_IN_TRANSCRIPT",
                message=f"Email '{email}' in summary not found in transcript — verify for typos.",
                detail=email,
            ))
    return findings


def _check_unverified_confirmations(summary: str, transcript: str) -> list[Finding]:
    """Warn for confirmation phrases in summary not supported by transcript evidence.

    Examples: "confirmed bank details", "waived the 10-day consideration period",
    "accepted the offer", "confirmed the settlement".
    """
    findings: list[Finding] = []
    transcript_lower = transcript.lower()
    summary_lower = summary.lower()

    for phrase_pattern, evidence_terms in _CONFIRMATION_PHRASE_CHECKS:
        if re.search(phrase_pattern, summary_lower):
            evidence_hits = sum(1 for t in evidence_terms if t in transcript_lower)
            if evidence_hits < 2:
                readable = re.sub(r"\\[a-z]|\?|\\s\+", " ", phrase_pattern).strip()
                findings.append(Finding(
                    tier="warning",
                    code="UNVERIFIED_CONFIRMATION",
                    message=(
                        f"Summary contains a confirmation ('{readable}') that "
                        "could not be verified in the transcript."
                    ),
                    detail=phrase_pattern,
                ))
    return findings


def _check_conditional_sections_justified(summary: str, transcript: str) -> list[Finding]:
    """Warn for conditional sections whose domain terms are absent from the transcript."""
    findings: list[Finding] = []
    transcript_lower = transcript.lower()

    for section, domain_terms in _CONDITIONAL_DOMAIN_TERMS.items():
        if not re.search(rf"^{re.escape(section)}:", summary, re.MULTILINE):
            continue
        if not any(term in transcript_lower for term in domain_terms):
            findings.append(Finding(
                tier="warning",
                code="CONDITIONAL_SECTION_UNJUSTIFIED",
                message=(
                    f"'{section}' section is included but no related terms were found "
                    "in the transcript. This section may not be relevant to this call."
                ),
                detail=section,
            ))
    return findings


# ── Collector ──────────────────────────────────────────────────────────────────


def run_content_integrity_checks(summary: str, transcript: str) -> list[Finding]:
    """Run all Tier-3 content integrity checks against the source transcript."""
    findings: list[Finding] = []
    findings.extend(_check_amounts_in_transcript(summary, transcript))
    findings.extend(_check_references_in_transcript(summary, transcript))
    findings.extend(_check_ibans_in_transcript(summary, transcript))
    findings.extend(_check_emails_in_transcript(summary, transcript))
    findings.extend(_check_unverified_confirmations(summary, transcript))
    findings.extend(_check_conditional_sections_justified(summary, transcript))
    logger.debug("Tier-3 content integrity checks: %d warning(s) found", len(findings))
    return findings
