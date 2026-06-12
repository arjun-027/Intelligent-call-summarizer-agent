"""Tier 1 — Structural error checks for generated call summaries.

All functions here return ``Finding(tier='error', ...)`` findings.  A summary
with any Tier-1 error cannot be saved; the auto-retry loop in ``service.py``
will attempt to correct these before presenting the result to the reviewer.

Checks in this tier
-------------------
- EMPTY_SUMMARY
- CHAR_LIMIT_EXCEEDED
- MISSING_SUBJECT
- MISSING_EXECUTIVE_SUMMARY
- MISSING_NEXT_STEPS
- SUBJECT_MULTILINE
- EXECUTIVE_SUMMARY_NO_BULLETS
- NEXT_STEPS_INCOMPLETE        (company action + Other line)
- PHANTOM_CONDITIONAL_SECTION  (section set to 'None')
- UNKNOWN_SECTION_HEADER
- CONDITIONAL_SECTION_EMPTY_BODY
"""

import logging
import re
from typing import Optional

from ..models import Finding
from ..summarizer import CHAR_LIMIT
from .constants import _CONDITIONAL_SECTION_NAMES, _KNOWN_SECTION_NAMES, _KNOWN_SUB_FIELD_LABELS
from .helpers import (
    _extract_next_steps_body,
    _extract_section_body,
    _get_all_section_header_names,
)

logger = logging.getLogger(__name__)


# ── Individual checks ──────────────────────────────────────────────────────────


def _check_empty(summary: str) -> Optional[Finding]:
    if not summary.strip():
        return Finding(tier="error", code="EMPTY_SUMMARY",
                       message="Summary is empty. The LLM produced no output.")
    return None


def _check_char_limit(summary: str) -> Optional[Finding]:
    count = len(summary)
    if count > CHAR_LIMIT:
        return Finding(
            tier="error",
            code="CHAR_LIMIT_EXCEEDED",
            message=(
                f"Summary is {count:,} characters — exceeds the {CHAR_LIMIT:,}-character "
                f"limit by {count - CHAR_LIMIT:,}."
            ),
            detail=str(count),
        )
    return None


def _check_missing_subject(summary: str) -> Optional[Finding]:
    if "Subject:" not in summary:
        return Finding(tier="error", code="MISSING_SUBJECT",
                       message="Required 'Subject:' section is missing.")
    return None


def _check_missing_executive_summary(summary: str) -> Optional[Finding]:
    if "Executive Summary:" not in summary:
        return Finding(tier="error", code="MISSING_EXECUTIVE_SUMMARY",
                       message="Required 'Executive Summary:' section is missing.")
    return None


def _check_missing_next_steps(summary: str) -> Optional[Finding]:
    if not re.search(r"Next Steps:?", summary):
        return Finding(tier="error", code="MISSING_NEXT_STEPS",
                       message="Required 'Next Steps' section is missing.")
    return None


def _check_subject_multiline(summary: str) -> Optional[Finding]:
    body = _extract_section_body(summary, "Subject")
    if body is None:
        return None
    non_empty = [ln for ln in body.split("\n") if ln.strip()]
    if len(non_empty) > 1:
        return Finding(
            tier="error",
            code="SUBJECT_MULTILINE",
            message=f"Subject spans {len(non_empty)} lines — it must be a single concise line.",
            detail=body[:120],
        )
    return None


def _check_executive_summary_bullets(summary: str) -> Optional[Finding]:
    """Tier-1 error: Executive Summary must contain '- ' bullet lines."""
    body = _extract_section_body(summary, "Executive Summary")
    if body is None:
        return None
    if not re.search(r"^\s*[-•*]", body, re.MULTILINE):
        return Finding(
            tier="error",
            code="EXECUTIVE_SUMMARY_NO_BULLETS",
            message=(
                "Executive Summary has no bullet points. "
                "The schema requires key facts listed as '- bullet' lines after the paragraph."
            ),
        )
    return None


def _check_next_steps_completeness(summary: str) -> list[Finding]:
    """Return errors when the Next Steps section is incomplete.

    Requires: (1) a company action line and (2) an ``Other:`` line.
    """
    body = _extract_next_steps_body(summary)
    if not body:
        return []

    findings: list[Finding] = []

    company_lines = [
        ln.strip()
        for ln in body.split("\n")
        if ":" in ln
        and not re.match(r"^\s*-?\s*Other\b", ln, re.IGNORECASE)
        and ln.strip()
    ]
    if not company_lines:
        findings.append(Finding(
            tier="error",
            code="NEXT_STEPS_INCOMPLETE",
            message="Next Steps is missing a company action line (e.g. 'Pemberton Insurance: ...').",
        ))

    if not re.search(r"\bOther\b.*:", body, re.IGNORECASE):
        findings.append(Finding(
            tier="error",
            code="NEXT_STEPS_INCOMPLETE",
            message="Next Steps is missing an 'Other:' line for third-party actions.",
        ))

    return findings


def _check_phantom_conditional_sections(summary: str) -> list[Finding]:
    """Return errors for conditional sections whose value is literally 'None'."""
    findings: list[Finding] = []
    for section in _CONDITIONAL_SECTION_NAMES:
        if re.search(rf"^{re.escape(section)}:\s*None\s*$", summary,
                     re.MULTILINE | re.IGNORECASE):
            findings.append(Finding(
                tier="error",
                code="PHANTOM_CONDITIONAL_SECTION",
                message=(
                    f"'{section}' is present but set to 'None'. "
                    "Omit the section entirely when that topic was not discussed."
                ),
                detail=section,
            ))
    return findings


def _check_unknown_section_headers(summary: str) -> list[Finding]:
    """Return errors for section headers outside the defined schema."""
    findings: list[Finding] = []
    for header in _get_all_section_header_names(summary):
        if header not in _KNOWN_SECTION_NAMES and header not in _KNOWN_SUB_FIELD_LABELS:
            findings.append(Finding(
                tier="error",
                code="UNKNOWN_SECTION_HEADER",
                message=(
                    f"Section '{header}:' is not part of the required schema. "
                    "Remove it or use a recognised section name."
                ),
                detail=header,
            ))
    return findings


def _check_conditional_section_empty_body(summary: str) -> list[Finding]:
    """Return errors for conditional section headers that are present but empty."""
    findings: list[Finding] = []
    for section in _CONDITIONAL_SECTION_NAMES:
        if re.search(rf"^{re.escape(section)}:\s*$", summary, re.MULTILINE):
            body = _extract_section_body(summary, section)
            if body is not None and not body.strip():
                findings.append(Finding(
                    tier="error",
                    code="CONDITIONAL_SECTION_EMPTY_BODY",
                    message=(
                        f"'{section}' section header is present but has no content. "
                        "Either add content or remove the section entirely."
                    ),
                    detail=section,
                ))
    return findings


# ── Collector ──────────────────────────────────────────────────────────────────


def run_structural_checks(summary: str) -> list[Finding]:
    """Run all Tier-1 structural checks and return combined error findings."""
    findings: list[Finding] = []

    for check in (
        _check_empty,
        _check_char_limit,
        _check_missing_subject,
        _check_missing_executive_summary,
        _check_missing_next_steps,
        _check_subject_multiline,
        _check_executive_summary_bullets,
    ):
        result = check(summary)
        if result:
            findings.append(result)

    findings.extend(_check_next_steps_completeness(summary))
    findings.extend(_check_phantom_conditional_sections(summary))
    findings.extend(_check_unknown_section_headers(summary))
    findings.extend(_check_conditional_section_empty_body(summary))

    logger.debug("Tier-1 structural checks: %d error(s) found", len(findings))
    return findings
