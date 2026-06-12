"""Three-tier output guardrail engine for generated call summaries.

Tier 1 — Structural Errors   : schema violations that make the summary unusable;
                                blocks saving until resolved.
Tier 2 — Format Warnings     : quality issues that a reviewer should see but
                                that do not prevent saving.
Tier 3 — Content Integrity   : cross-referenced against the source transcript;
                                requires ``transcript_content`` to be provided.

Public API::

    from call_summarizer.guardrails import run_guardrails, build_retry_prompt_addendum

    result = run_guardrails(summary, transcript_content)
    if not result.passed:
        addendum = build_retry_prompt_addendum(result)
"""

import logging
import re
from typing import Optional

from .models import Finding, GuardrailResult
from .summarizer import CHAR_LIMIT

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

_CHAR_HIGH_WATERMARK = 1200  # warn when approaching limit

_KNOWN_SECTION_NAMES: set[str] = {
    "Caller",
    "Subject",
    "Executive Summary",
    "Next Steps",
    "Liability Summary",
    "Negotiation Summary",
    "Vehicle Damage",
    "Injury",
    "Property",
}

# Sub-field labels inside conditional sections — NOT top-level section headers
_KNOWN_SUB_FIELD_LABELS: set[str] = {
    "Vehicle Status",
    "Towage",
    "Treatment",
}

_CONDITIONAL_SECTION_NAMES: list[str] = [
    "Liability Summary",
    "Negotiation Summary",
    "Vehicle Damage",
    "Injury",
    "Property",
]

# Domain terms that justify including each conditional section
_CONDITIONAL_DOMAIN_TERMS: dict[str, list[str]] = {
    "Liability Summary": ["liabilit", "liable", "fault", "responsibil", "at fault"],
    "Negotiation Summary": ["negotiat", "offer", "counter", "settlement discussion"],
    "Vehicle Damage": ["vehicle", " car ", "garage", "repair", "tow", "hire car", "car hire"],
    "Injury": ["injur", "pain", "treatment", "hospital", "medical", "whiplash", "physio"],
    "Property": ["property", "building", "house", "home", "roof", "contents", "flood"],
}

_KNOWN_CALLER_RELATIONSHIPS: list[str] = [
    "policyholder",
    "third party representative",
    "third-party representative",
    "third party",
    "solicitor",
    "insurance company representative",
    "insurance representative",
    "family member",
    "representative",
]

# Confirmation phrases that must be verifiable from the transcript
_CONFIRMATION_PHRASE_CHECKS: list[tuple[str, list[str]]] = [
    (
        r"confirmed\s+(?:their\s+)?bank\s+details",
        ["bank", "detail", "account", "confirm"],
    ),
    (
        r"waived?\s+(?:the\s+)?(?:10.day\s+)?consideration\s+period",
        ["waiv", "consideration", "10", "period"],
    ),
    (
        r"accepted\s+(?:the\s+)?(?:settlement\s+)?offer",
        ["accept", "offer", "settlement"],
    ),
    (
        r"confirmed\s+(?:the\s+)?settlement",
        ["confirm", "settlement", "agree"],
    ),
]

# Regex patterns for extracting critical facts
_AMOUNT_RE = re.compile(r"[€$£]\s*[\d,]+(?:\.\d{1,2})?")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")
_REFERENCE_RE = re.compile(r"\b[A-Z]{1,5}-\d{3,4}-\d{4,8}\b")

# A line that looks like a top-level section header (e.g. "Subject:", "Vehicle Damage:")
# Requires: starts at line beginning, one or more capitalized words, ends with colon + optional space
_SECTION_HEADER_LINE_RE = re.compile(
    r"^([A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*)*):\s*$", re.MULTILINE
)


# ── Section extraction helpers ─────────────────────────────────────────────


def _get_caller_line(summary: str) -> Optional[str]:
    """Return the content of the Caller line (after 'Caller:'), or None.

    Args:
        summary: Full summary text.

    Returns:
        Everything after 'Caller:' on the same line, stripped; or None if not found.
    """
    match = re.search(r"^Caller:\s*(.+)$", summary, re.MULTILINE)
    return match.group(1).strip() if match else None


def _extract_section_body(summary: str, header: str) -> Optional[str]:
    """Extract the body text that follows a named section header line.

    Stops at the next top-level section header or end of string.

    Args:
        summary: Full summary text.
        header: The exact section name without the trailing colon (e.g. ``"Subject"``).

    Returns:
        Stripped body text of the section, or None if the header is not present.
    """
    pattern = re.compile(
        r"^" + re.escape(header) + r":\s*\n(.*?)(?=\n[A-Z][A-Za-z]+(?: [A-Za-z]+)*:\s*$|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(summary)
    return match.group(1).strip() if match else None


def _extract_next_steps_body(summary: str) -> str:
    """Extract the body text of the Next Steps section.

    Handles both 'Next Steps:' (with colon) and 'Next Steps' (without).

    Args:
        summary: Full summary text.

    Returns:
        Stripped body text, or empty string if the section is absent.
    """
    match = re.search(r"^Next Steps:?\s*$", summary, re.MULTILINE)
    if not match:
        return ""
    start = match.end()
    next_header = re.search(r"\n[A-Z][A-Za-z]+(?: [A-Za-z]+)*:\s*$", summary[start:], re.MULTILINE)
    end = start + next_header.start() if next_header else len(summary)
    return summary[start:end].strip()


def _get_all_section_header_names(summary: str) -> list[str]:
    """Return all top-level section header names found in the summary.

    Only matches lines that consist solely of a header and colon
    (i.e. no inline content), to exclude sub-field labels like
    'Vehicle Status: repairable'.

    Args:
        summary: Full summary text.

    Returns:
        List of header name strings (without the colon).
    """
    return _SECTION_HEADER_LINE_RE.findall(summary)


# ── Tier 1: Structural error checks ───────────────────────────────────────


def _check_empty(summary: str) -> Optional[Finding]:
    """Return an error finding if the summary is blank or whitespace only."""
    if not summary.strip():
        return Finding(
            tier="error",
            code="EMPTY_SUMMARY",
            message="Summary is empty. The LLM produced no output.",
        )
    return None


def _check_char_limit(summary: str) -> Optional[Finding]:
    """Return an error finding if the summary exceeds 1,500 characters."""
    count = len(summary)
    if count > CHAR_LIMIT:
        return Finding(
            tier="error",
            code="CHAR_LIMIT_EXCEEDED",
            message=f"Summary is {count:,} characters — exceeds the {CHAR_LIMIT:,}-character limit by {count - CHAR_LIMIT:,}.",
            detail=str(count),
        )
    return None


def _check_missing_subject(summary: str) -> Optional[Finding]:
    """Return an error finding if the 'Subject:' section is absent."""
    if "Subject:" not in summary:
        return Finding(
            tier="error",
            code="MISSING_SUBJECT",
            message="Required 'Subject:' section is missing.",
        )
    return None


def _check_missing_executive_summary(summary: str) -> Optional[Finding]:
    """Return an error finding if the 'Executive Summary:' section is absent."""
    if "Executive Summary:" not in summary:
        return Finding(
            tier="error",
            code="MISSING_EXECUTIVE_SUMMARY",
            message="Required 'Executive Summary:' section is missing.",
        )
    return None


def _check_missing_next_steps(summary: str) -> Optional[Finding]:
    """Return an error finding if 'Next Steps' is absent (with or without colon)."""
    if not re.search(r"Next Steps:?", summary):
        return Finding(
            tier="error",
            code="MISSING_NEXT_STEPS",
            message="Required 'Next Steps' section is missing.",
        )
    return None


def _check_next_steps_completeness(summary: str) -> list[Finding]:
    """Return error findings for an incomplete Next Steps section.

    A complete Next Steps section must contain:
    1. A company action line (``CompanyName: action``).
    2. An ``Other:`` line (for third-party actions).

    Args:
        summary: Full summary text.

    Returns:
        List of zero, one, or two :class:`~call_summarizer.models.Finding` instances.
    """
    body = _extract_next_steps_body(summary)
    if not body:
        return []  # Caught by MISSING_NEXT_STEPS

    findings: list[Finding] = []

    # Company action line: any non-empty line containing ': ' that is NOT the Other line
    company_lines = [
        ln.strip()
        for ln in body.split("\n")
        if ":" in ln
        and not re.match(r"^\s*-?\s*Other\b", ln, re.IGNORECASE)
        and ln.strip()
    ]
    if not company_lines:
        findings.append(
            Finding(
                tier="error",
                code="NEXT_STEPS_INCOMPLETE",
                message="Next Steps is missing a company action line (e.g. 'Pemberton Insurance: ...').",
            )
        )

    # Other line
    if not re.search(r"\bOther\b.*:", body, re.IGNORECASE):
        findings.append(
            Finding(
                tier="error",
                code="NEXT_STEPS_INCOMPLETE",
                message="Next Steps is missing an 'Other:' line for third-party actions.",
            )
        )

    return findings


def _check_phantom_conditional_sections(summary: str) -> list[Finding]:
    """Return error findings for conditional sections set to 'None'.

    The schema requires that conditional sections (Liability Summary, etc.)
    are omitted entirely when the topic was not discussed — not included with
    a 'None' value.

    Args:
        summary: Full summary text.

    Returns:
        One finding per phantom section detected.
    """
    findings: list[Finding] = []
    for section in _CONDITIONAL_SECTION_NAMES:
        # Pattern: "Liability Summary: None" on same line
        if re.search(rf"^{re.escape(section)}:\s*None\s*$", summary, re.MULTILINE | re.IGNORECASE):
            findings.append(
                Finding(
                    tier="error",
                    code="PHANTOM_CONDITIONAL_SECTION",
                    message=(
                        f"'{section}' is present but set to 'None'. "
                        "Omit the section entirely when that topic was not discussed."
                    ),
                    detail=section,
                )
            )
    return findings


def _check_unknown_section_headers(summary: str) -> list[Finding]:
    """Return error findings for section headers outside the defined schema.

    Args:
        summary: Full summary text.

    Returns:
        One finding per unrecognised section header found.
    """
    findings: list[Finding] = []
    all_headers = _get_all_section_header_names(summary)
    for header in all_headers:
        if header not in _KNOWN_SECTION_NAMES and header not in _KNOWN_SUB_FIELD_LABELS:
            findings.append(
                Finding(
                    tier="error",
                    code="UNKNOWN_SECTION_HEADER",
                    message=(
                        f"Section '{header}:' is not part of the required schema. "
                        "Remove it or use a recognised section name."
                    ),
                    detail=header,
                )
            )
    return findings


def _check_subject_multiline(summary: str) -> Optional[Finding]:
    """Return an error finding if the Subject spans more than one line.

    The schema requires Subject to be a single concise line.

    Args:
        summary: Full summary text.

    Returns:
        A finding if the subject body contains multiple non-empty lines.
    """
    body = _extract_section_body(summary, "Subject")
    if body is None:
        return None  # Caught by MISSING_SUBJECT

    non_empty_lines = [ln for ln in body.split("\n") if ln.strip()]
    if len(non_empty_lines) > 1:
        return Finding(
            tier="error",
            code="SUBJECT_MULTILINE",
            message=(
                f"Subject spans {len(non_empty_lines)} lines — it must be a single concise line."
            ),
            detail=body[:120],
        )
    return None


def _check_conditional_section_empty_body(summary: str) -> list[Finding]:
    """Return error findings for conditional section headers with no body text.

    Args:
        summary: Full summary text.

    Returns:
        One finding per conditional section that is present but empty.
    """
    findings: list[Finding] = []
    for section in _CONDITIONAL_SECTION_NAMES:
        if re.search(rf"^{re.escape(section)}:\s*$", summary, re.MULTILINE):
            body = _extract_section_body(summary, section)
            if body is not None and not body.strip():
                findings.append(
                    Finding(
                        tier="error",
                        code="CONDITIONAL_SECTION_EMPTY_BODY",
                        message=(
                            f"'{section}' section header is present but has no content. "
                            "Either add content or remove the section entirely."
                        ),
                        detail=section,
                    )
                )
    return findings


def _run_structural_checks(summary: str) -> list[Finding]:
    """Run all Tier-1 structural checks and return combined findings.

    Args:
        summary: Full summary text to validate.

    Returns:
        List of :class:`~call_summarizer.models.Finding` with ``tier='error'``.
    """
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

    logger.debug("Tier-1 checks: %d error(s) found", len(findings))
    return findings


# ── Tier 2: Format quality warning checks ─────────────────────────────────


def _check_caller_line_present(summary: str) -> Optional[Finding]:
    """Return a warning if the 'Caller:' line is missing."""
    if not re.search(r"^Caller:", summary, re.MULTILINE):
        return Finding(
            tier="warning",
            code="MISSING_CALLER_LINE",
            message="'Caller:' line is missing. The schema requires name, relationship, and direction.",
        )
    return None


def _check_caller_direction(summary: str) -> Optional[Finding]:
    """Return a warning if neither 'inbound' nor 'outbound' appears in the Caller line."""
    caller = _get_caller_line(summary)
    if caller is None:
        return None  # Covered by MISSING_CALLER_LINE
    if not re.search(r"\b(?:inbound|outbound)\b", caller, re.IGNORECASE):
        return Finding(
            tier="warning",
            code="CALLER_DIRECTION_MISSING",
            message="Call direction ('inbound' or 'outbound') not found in the Caller line.",
            detail=caller,
        )
    return None


def _check_caller_relationship(summary: str) -> Optional[Finding]:
    """Return a warning if the caller's relationship is not a recognised value."""
    caller = _get_caller_line(summary)
    if caller is None:
        return None
    caller_lower = caller.lower()
    if not any(rel in caller_lower for rel in _KNOWN_CALLER_RELATIONSHIPS):
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


def _check_executive_summary_bullets(summary: str) -> Optional[Finding]:
    """Return an error if the Executive Summary section contains no bullet points.

    The schema explicitly requires key facts to be listed as ``-`` bullets after
    the narrative paragraph.  Absence of bullets is treated as a Tier-1 structural
    error so the auto-retry loop can correct it before the summary is presented to
    the reviewer.
    """
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


def _check_next_steps_both_none(summary: str) -> Optional[Finding]:
    """Return a warning if both the company action and Other line are 'None'."""
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
    """Return a warning when approaching but not yet exceeding the character limit."""
    count = len(summary)
    if _CHAR_HIGH_WATERMARK < count <= CHAR_LIMIT:
        return Finding(
            tier="warning",
            code="CHAR_COUNT_HIGH",
            message=(
                f"Summary is {count:,} characters — within limit but close to the {CHAR_LIMIT:,}-character cap. "
                "Check for redundant content."
            ),
            detail=str(count),
        )
    return None


def _check_duplicate_bullet_content(summary: str) -> list[Finding]:
    """Return a warning if the same numeric value appears in more than one bullet.

    This catches the pattern seen in okay-4 where a fact (e.g. '10-day waiver')
    was restated in two separate bullets.

    Args:
        summary: Full summary text.

    Returns:
        At most one finding listing the duplicate values found.
    """
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

    # Count how many bullets each numeric token appears in
    token_bullet_count: dict[str, int] = {}
    for bullet in bullets:
        seen_in_this_bullet: set[str] = set()
        for token in re.findall(r"[\d,]+(?:\.\d+)?", bullet):
            if len(token) > 1 and token not in seen_in_this_bullet:
                token_bullet_count[token] = token_bullet_count.get(token, 0) + 1
                seen_in_this_bullet.add(token)

    duplicates = [t for t, count in token_bullet_count.items() if count > 1]
    if duplicates:
        return [
            Finding(
                tier="warning",
                code="DUPLICATE_BULLET_CONTENT",
                message=(
                    f"Numeric value(s) {', '.join(duplicates[:3])} appear in more than one bullet. "
                    "The same fact may be stated twice."
                ),
                detail=", ".join(duplicates),
            )
        ]
    return []


def _check_vehicle_damage_subfields(summary: str) -> Optional[Finding]:
    """Return a warning if Vehicle Damage is present but the Towage sub-field is missing."""
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


def _run_format_checks(summary: str) -> list[Finding]:
    """Run all Tier-2 format quality checks and return combined findings.

    Args:
        summary: Full summary text to validate.

    Returns:
        List of :class:`~call_summarizer.models.Finding` with ``tier='warning'``.
    """
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

    logger.debug("Tier-2 checks: %d warning(s) found", len(findings))
    return findings


# ── Tier 3: Content integrity checks ──────────────────────────────────────


def _normalize_numeric(value: str) -> str:
    """Strip currency symbols, spaces, and commas for comparison.

    Args:
        value: A raw extracted string such as ``'€ 3,150.00'``.

    Returns:
        Normalized string e.g. ``'3150.00'``.
    """
    return re.sub(r"[€$£,\s]", "", value)


def _extract_amounts(text: str) -> set[str]:
    """Extract all currency amounts from text, normalized for comparison.

    Args:
        text: Any text to search.

    Returns:
        Set of normalized amount strings.
    """
    return {_normalize_numeric(m) for m in _AMOUNT_RE.findall(text)}


def _extract_ibans(text: str) -> set[str]:
    """Extract all IBAN strings from text.

    Args:
        text: Any text to search.

    Returns:
        Set of uppercase IBAN strings.
    """
    return {m.upper().replace(" ", "") for m in _IBAN_RE.findall(text)}


def _extract_emails(text: str) -> set[str]:
    """Extract all email addresses from text (case-insensitive).

    Args:
        text: Any text to search.

    Returns:
        Set of lowercased email strings.
    """
    return {m.lower() for m in _EMAIL_RE.findall(text)}


def _extract_references(text: str) -> set[str]:
    """Extract claim/policy reference numbers from text.

    Args:
        text: Any text to search.

    Returns:
        Set of uppercase reference strings.
    """
    return {m.upper() for m in _REFERENCE_RE.findall(text)}


def _check_amounts_in_transcript(summary: str, transcript: str) -> list[Finding]:
    """Return warnings for monetary amounts in summary not found in transcript.

    Args:
        summary: Generated summary text.
        transcript: Original transcript text.

    Returns:
        One warning per amount present in summary but absent from transcript.
    """
    summary_amounts = _extract_amounts(summary)
    transcript_amounts = _extract_amounts(transcript)
    findings: list[Finding] = []
    for amount in summary_amounts:
        if amount and amount not in transcript_amounts:
            # Also do a loose substring check in case of formatting differences
            if amount not in transcript.replace(",", "").replace(" ", ""):
                findings.append(
                    Finding(
                        tier="warning",
                        code="AMOUNT_NOT_IN_TRANSCRIPT",
                        message=f"Amount '{amount}' in summary could not be verified in the transcript — possible hallucination.",
                        detail=amount,
                    )
                )
    return findings


def _check_references_in_transcript(summary: str, transcript: str) -> list[Finding]:
    """Return warnings for reference numbers in summary not found in transcript.

    Args:
        summary: Generated summary text.
        transcript: Original transcript text.

    Returns:
        One warning per reference absent from transcript.
    """
    summary_refs = _extract_references(summary)
    findings: list[Finding] = []
    for ref in summary_refs:
        if ref not in transcript.upper():
            findings.append(
                Finding(
                    tier="warning",
                    code="REFERENCE_NOT_IN_TRANSCRIPT",
                    message=f"Reference '{ref}' in summary not found in transcript — possible error.",
                    detail=ref,
                )
            )
    return findings


def _check_ibans_in_transcript(summary: str, transcript: str) -> list[Finding]:
    """Return warnings for IBANs in summary not matching those in transcript.

    Args:
        summary: Generated summary text.
        transcript: Original transcript text.

    Returns:
        One warning per IBAN mismatch detected.
    """
    summary_ibans = _extract_ibans(summary)
    transcript_ibans = _extract_ibans(transcript)
    findings: list[Finding] = []
    for iban in summary_ibans:
        if iban not in transcript_ibans and iban not in transcript.upper().replace(" ", ""):
            findings.append(
                Finding(
                    tier="warning",
                    code="IBAN_NOT_IN_TRANSCRIPT",
                    message=f"IBAN '{iban}' in summary could not be matched to the transcript — verify carefully.",
                    detail=iban,
                )
            )
    return findings


def _check_emails_in_transcript(summary: str, transcript: str) -> list[Finding]:
    """Return warnings for email addresses in summary not found in transcript.

    Args:
        summary: Generated summary text.
        transcript: Original transcript text.

    Returns:
        One warning per email absent from transcript.
    """
    summary_emails = _extract_emails(summary)
    findings: list[Finding] = []
    for email in summary_emails:
        if email not in transcript.lower():
            findings.append(
                Finding(
                    tier="warning",
                    code="EMAIL_NOT_IN_TRANSCRIPT",
                    message=f"Email '{email}' in summary not found in transcript — verify for typos.",
                    detail=email,
                )
            )
    return findings


def _check_unverified_confirmations(summary: str, transcript: str) -> list[Finding]:
    """Return warnings for confirmation phrases not supported by the transcript.

    Examples from real failures: "caller confirmed bank details" (they didn't),
    "waived the 10-day consideration period" (not mentioned).

    Args:
        summary: Generated summary text.
        transcript: Original transcript text.

    Returns:
        One warning per unverifiable confirmation phrase.
    """
    findings: list[Finding] = []
    transcript_lower = transcript.lower()
    summary_lower = summary.lower()

    for phrase_pattern, evidence_terms in _CONFIRMATION_PHRASE_CHECKS:
        if re.search(phrase_pattern, summary_lower):
            evidence_hits = sum(1 for t in evidence_terms if t in transcript_lower)
            if evidence_hits < 2:
                readable = re.sub(r"\\[a-z]|\?|\\s\+", " ", phrase_pattern).strip()
                findings.append(
                    Finding(
                        tier="warning",
                        code="UNVERIFIED_CONFIRMATION",
                        message=(
                            f"Summary contains a confirmation ('{readable}') that "
                            "could not be verified in the transcript."
                        ),
                        detail=phrase_pattern,
                    )
                )
    return findings


def _check_conditional_sections_justified(summary: str, transcript: str) -> list[Finding]:
    """Return warnings for conditional sections not supported by the transcript content.

    For example, including a 'Vehicle Damage' section when the transcript
    contains no vehicle-related terms.

    Args:
        summary: Generated summary text.
        transcript: Original transcript text.

    Returns:
        One warning per unjustified conditional section.
    """
    findings: list[Finding] = []
    transcript_lower = transcript.lower()

    for section, domain_terms in _CONDITIONAL_DOMAIN_TERMS.items():
        section_present = re.search(rf"^{re.escape(section)}:", summary, re.MULTILINE)
        if not section_present:
            continue
        if not any(term in transcript_lower for term in domain_terms):
            findings.append(
                Finding(
                    tier="warning",
                    code="CONDITIONAL_SECTION_UNJUSTIFIED",
                    message=(
                        f"'{section}' section is included but no related terms were found "
                        "in the transcript. This section may not be relevant to this call."
                    ),
                    detail=section,
                )
            )
    return findings


def _run_content_integrity_checks(summary: str, transcript: str) -> list[Finding]:
    """Run all Tier-3 content integrity checks against the source transcript.

    Args:
        summary: Generated summary text.
        transcript: Original transcript text (required for these checks).

    Returns:
        List of :class:`~call_summarizer.models.Finding` with ``tier='warning'``.
    """
    findings: list[Finding] = []
    findings.extend(_check_amounts_in_transcript(summary, transcript))
    findings.extend(_check_references_in_transcript(summary, transcript))
    findings.extend(_check_ibans_in_transcript(summary, transcript))
    findings.extend(_check_emails_in_transcript(summary, transcript))
    findings.extend(_check_unverified_confirmations(summary, transcript))
    findings.extend(_check_conditional_sections_justified(summary, transcript))
    logger.debug("Tier-3 checks: %d warning(s) found", len(findings))
    return findings


# ── Public API ─────────────────────────────────────────────────────────────


def run_guardrails(
    summary: str,
    transcript_content: str = "",
) -> GuardrailResult:
    """Run all three guardrail tiers against a generated summary.

    Tier 3 (content integrity) is skipped when ``transcript_content`` is empty,
    which is appropriate when the summary has already been saved and the original
    transcript is no longer available.

    Args:
        summary: The LLM-generated (or user-edited) summary text.
        transcript_content: The original transcript text.  Pass an empty string
            to skip Tier-3 content integrity checks.

    Returns:
        A :class:`~call_summarizer.models.GuardrailResult` describing all
        findings.  ``passed`` is True only when no Tier-1 errors were found.
    """
    logger.debug("Running guardrails (transcript supplied: %s)", bool(transcript_content))

    findings: list[Finding] = []
    findings.extend(_run_structural_checks(summary))
    findings.extend(_run_format_checks(summary))

    if transcript_content:
        findings.extend(_run_content_integrity_checks(summary, transcript_content))

    errors = [f for f in findings if f.tier == "error"]
    char_count = len(summary)

    result = GuardrailResult(
        passed=len(errors) == 0,
        findings=findings,
        char_count=char_count,
        char_within_limit=char_count <= CHAR_LIMIT,
    )

    logger.info(
        "Guardrails complete — passed: %s, errors: %d, warnings: %d",
        result.passed,
        len(result.errors),
        len(result.warnings),
    )
    return result


def build_retry_prompt_addendum(result: GuardrailResult) -> str:
    """Build targeted additional prompt instructions based on Tier-1 guardrail errors.

    Used by the retry loop in the service layer to give the LLM specific
    corrective guidance on the second (or third) generation attempt.

    Args:
        result: The failed :class:`~call_summarizer.models.GuardrailResult` from
            the previous attempt.

    Returns:
        A string of additional instructions to append to the system prompt.
        Returns an empty string if there are no errors to address.
    """
    error_codes = {f.code for f in result.errors}
    addenda: list[str] = []

    if "PHANTOM_CONDITIONAL_SECTION" in error_codes or "CONDITIONAL_SECTION_EMPTY_BODY" in error_codes:
        addenda.append(
            "CRITICAL: Do NOT include Liability Summary, Negotiation Summary, "
            "Vehicle Damage, Injury, or Property sections unless explicitly discussed. "
            "Never write 'Section Name: None' — omit the section entirely."
        )

    if "CHAR_LIMIT_EXCEEDED" in error_codes:
        addenda.append(
            f"CRITICAL: Your previous response exceeded {CHAR_LIMIT} characters. "
            "Be significantly more concise. Prioritise facts over narrative."
        )

    if "MISSING_NEXT_STEPS" in error_codes or "NEXT_STEPS_INCOMPLETE" in error_codes:
        addenda.append(
            "CRITICAL: Always include a 'Next Steps:' section with "
            "a company action line and an 'Other:' line."
        )

    if "UNKNOWN_SECTION_HEADER" in error_codes:
        addenda.append(
            "CRITICAL: Only use these section headers: Caller, Subject, "
            "Executive Summary, Next Steps, and the five conditional sections "
            "(Liability Summary, Negotiation Summary, Vehicle Damage, Injury, Property)."
        )

    if "SUBJECT_MULTILINE" in error_codes:
        addenda.append(
            "CRITICAL: The Subject must be a single line — one concise sentence only."
        )

    if "EXECUTIVE_SUMMARY_NO_BULLETS" in error_codes:
        addenda.append(
            "CRITICAL: The Executive Summary MUST include '- ' bullet points listing key facts. "
            "After the narrative paragraph, add at least two bullet lines in the format:\n"
            "- [Key fact extracted from the call]\n"
            "- [Another key fact]\n"
            "Do NOT omit the bullet points."
        )

    return "\n".join(addenda)
