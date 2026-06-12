"""Section-extraction and entity-extraction helpers used by all three guardrail tiers."""

import re
from typing import Optional

from .constants import (
    _AMOUNT_RE,
    _EMAIL_RE,
    _IBAN_RE,
    _REFERENCE_RE,
    _SECTION_HEADER_LINE_RE,
)


# ── Section extraction ─────────────────────────────────────────────────────────


def _get_caller_line(summary: str) -> Optional[str]:
    """Return the content of the Caller line (after ``'Caller:'``), or ``None``."""
    match = re.search(r"^Caller:\s*(.+)$", summary, re.MULTILINE)
    return match.group(1).strip() if match else None


def _extract_section_body(summary: str, header: str) -> Optional[str]:
    """Extract the body text that follows *header*, stopping at the next section.

    Args:
        summary: Full summary text.
        header: Section name without trailing colon (e.g. ``"Subject"``).

    Returns:
        Stripped body text, or ``None`` when the header is absent.
    """
    pattern = re.compile(
        r"^" + re.escape(header) + r":\s*\n(.*?)(?=\n[A-Z][A-Za-z]+(?: [A-Za-z]+)*:\s*$|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(summary)
    return match.group(1).strip() if match else None


def _extract_next_steps_body(summary: str) -> str:
    """Extract the body text of the Next Steps section (with or without colon)."""
    match = re.search(r"^Next Steps:?\s*$", summary, re.MULTILINE)
    if not match:
        return ""
    start = match.end()
    next_header = re.search(r"\n[A-Z][A-Za-z]+(?: [A-Za-z]+)*:\s*$", summary[start:], re.MULTILINE)
    end = start + next_header.start() if next_header else len(summary)
    return summary[start:end].strip()


def _get_all_section_header_names(summary: str) -> list[str]:
    """Return all top-level section header names found (lines that are header+colon only)."""
    return _SECTION_HEADER_LINE_RE.findall(summary)


# ── Entity extraction ──────────────────────────────────────────────────────────


def _normalize_numeric(value: str) -> str:
    """Strip currency symbols, spaces, and commas for numeric comparison."""
    return re.sub(r"[€$£,\s]", "", value)


def _extract_amounts(text: str) -> set[str]:
    """Extract all currency amounts from *text*, normalized."""
    return {_normalize_numeric(m) for m in _AMOUNT_RE.findall(text)}


def _extract_ibans(text: str) -> set[str]:
    """Extract all IBAN strings from *text* (uppercased, no spaces)."""
    return {m.upper().replace(" ", "") for m in _IBAN_RE.findall(text)}


def _extract_emails(text: str) -> set[str]:
    """Extract all email addresses from *text* (lowercased)."""
    return {m.lower() for m in _EMAIL_RE.findall(text)}


def _extract_references(text: str) -> set[str]:
    """Extract claim/policy reference numbers from *text* (uppercased)."""
    return {m.upper() for m in _REFERENCE_RE.findall(text)}
