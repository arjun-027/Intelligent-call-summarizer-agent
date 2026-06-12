"""Tier 2 — Format quality warning checks for generated call summaries.

All functions here return ``Finding(tier='warning', ...)`` findings.  These do
not block saving but should be reviewed before the summary is filed.

Checks in this tier
-------------------
- MISSING_CALLER_LINE
- CALLER_DIRECTION_MISSING       (inbound / outbound)
- CALLER_RELATIONSHIP_UNRECOGNIZED
- NEXT_STEPS_BOTH_NONE
- CHAR_COUNT_HIGH                (approaching 1,500-char limit)
- VEHICLE_DAMAGE_TOWAGE_MISSING
- DUPLICATE_BULLET_CONTENT
"""

import logging
import re
from typing import Optional

from ..models import Finding
from ..summarizer import CHAR_LIMIT
from .constants import _CHAR_HIGH_WATERMARK, _KNOWN_CALLER_RELATIONSHIPS
from .helpers import (
    _extract_next_steps_body,
    _extract_section_body,
    _get_caller_line,
)

logger = logging.getLogger(__name__)


# ── Individual checks ──────────────────────────────────────────────────────────


def _check_caller_line_present(summary: str) -> Optional[Finding]:
    if not re.search(r"^Caller:", summary, re.MULTILINE):
        return Finding(
            tier="warning",
            code="MISSING_CALLER_LINE",
            message="'Caller:' line is missing. The schema requires name, relationship, and direction.",
        )
    return None


def _check_caller_direction(summary: str) -> Optional[Finding]:
    caller = _get_caller_line(summary)
    if caller is None:
        return None
    if not re.search(r"\b(?:inbound|outbound)\b", caller, re.IGNORECASE):
        return Finding(
            tier="warning",
            code="CALLER_DIRECTION_MISSING",
            message="Call direction ('inbound' or 'outbound') not found in the Caller line.",
            detail=caller,
        )
    return None


def _check_caller_relationship(summary: str) -> Optional[Finding]:
    caller = _get_caller_line(summary)
    if caller is None:
        return None
    if not any(rel in caller.lower() for rel in _KNOWN_CALLER_RELATIONSHIPS):
        return Finding(
            tier="warning",
            code="CALLER_RELATIONSHIP_UNRECOGNIZED",
            message=(
                "Caller relationship not recognised. Expected one of: policyholder, "
                "third party representative, solicitor, insurance company representative, "
                "family member."
            ),
            detail=caller,
        )
    return None


def _check_next_steps_both_none(summary: str) -> Optional[Finding]:
    body = _extract_next_steps_body(summary)
    if not body:
        return None
    none_count = len(re.findall(r":\s*None\b", body, re.IGNORECASE))
    if none_count >= 2:
        return Finding(
            tier="warning",
            code="NEXT_STEPS_BOTH_NONE",
            message=(
                "Both actions in Next Steps are 'None'. "
                "On any real call, at least one party should have a follow-up action."
            ),
        )
    return None


def _check_char_count_high(summary: str) -> Optional[Finding]:
    count = len(summary)
    if _CHAR_HIGH_WATERMARK < count <= CHAR_LIMIT:
        return Finding(
            tier="warning",
            code="CHAR_COUNT_HIGH",
            message=(
                f"Summary is {count:,} characters — within limit but close to the "
                f"{CHAR_LIMIT:,}-character cap. Check for redundant content."
            ),
            detail=str(count),
        )
    return None


def _check_duplicate_bullet_content(summary: str) -> list[Finding]:
    """Warn when the same numeric value appears in more than one Executive Summary bullet."""
    body = _extract_section_body(summary, "Executive Summary")
    if not body:
        return []

    bullets = [
        ln.strip().lstrip("-•*").strip()
        for ln in body.split("\n")
        if re.match(r"^\s*[-•*]", ln)
    ]
    if len(bullets) < 2:
        return []

    token_bullet_count: dict[str, int] = {}
    for bullet in bullets:
        seen: set[str] = set()
        for token in re.findall(r"[\d,]+(?:\.\d+)?", bullet):
            if len(token) > 1 and token not in seen:
                token_bullet_count[token] = token_bullet_count.get(token, 0) + 1
                seen.add(token)

    duplicates = [t for t, count in token_bullet_count.items() if count > 1]
    if duplicates:
        return [Finding(
            tier="warning",
            code="DUPLICATE_BULLET_CONTENT",
            message=(
                f"Numeric value(s) {', '.join(duplicates[:3])} appear in more than one bullet. "
                "The same fact may be stated twice."
            ),
            detail=", ".join(duplicates),
        )]
    return []


def _check_vehicle_damage_subfields(summary: str) -> Optional[Finding]:
    if not re.search(r"^Vehicle Damage:\s*$", summary, re.MULTILINE):
        return None
    body = _extract_section_body(summary, "Vehicle Damage")
    if body and "Towage:" not in body:
        return Finding(
            tier="warning",
            code="VEHICLE_DAMAGE_TOWAGE_MISSING",
            message=(
                "Vehicle Damage section is present but the 'Towage:' sub-field is missing. "
                "The schema requires Vehicle Status, Towage, and Car hire."
            ),
        )
    return None


# ── Collector ──────────────────────────────────────────────────────────────────


def run_format_checks(summary: str) -> list[Finding]:
    """Run all Tier-2 format quality checks and return combined warning findings."""
    findings: list[Finding] = []

    for check in (
        _check_caller_line_present,
        _check_caller_direction,
        _check_caller_relationship,
        _check_next_steps_both_none,
        _check_char_count_high,
        _check_vehicle_damage_subfields,
    ):
        result = check(summary)
        if result:
            findings.append(result)

    findings.extend(_check_duplicate_bullet_content(summary))

    logger.debug("Tier-2 format checks: %d warning(s) found", len(findings))
    return findings
