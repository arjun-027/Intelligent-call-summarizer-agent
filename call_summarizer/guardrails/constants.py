"""Shared constants, lookup tables, and compiled regex patterns for the output guardrail engine."""

import re

# ── Character limits ───────────────────────────────────────────────────────────

_CHAR_HIGH_WATERMARK: int = 1200  # advisory watermark (< CHAR_LIMIT from summarizer)

# ── Schema-defined section names ───────────────────────────────────────────────

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

# Sub-field labels that appear *inside* conditional sections — not top-level headers.
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

# Domain keywords that justify including each optional section.
_CONDITIONAL_DOMAIN_TERMS: dict[str, list[str]] = {
    "Liability Summary":    ["liabilit", "liable", "fault", "responsibil", "at fault"],
    "Negotiation Summary":  ["negotiat", "offer", "counter", "settlement discussion"],
    "Vehicle Damage":       ["vehicle", " car ", "garage", "repair", "tow", "hire car", "car hire"],
    "Injury":               ["injur", "pain", "treatment", "hospital", "medical", "whiplash", "physio"],
    "Property":             ["property", "building", "house", "home", "roof", "contents", "flood"],
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

# Confirmation phrases that must be verifiable from the transcript.
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

# ── Compiled regex patterns ────────────────────────────────────────────────────

_AMOUNT_RE = re.compile(r"[€$£]\s*[\d,]+(?:\.\d{1,2})?")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")
_REFERENCE_RE = re.compile(r"\b[A-Z]{1,5}-\d{3,4}-\d{4,8}\b")

# Matches lines that are *solely* a header + colon (e.g. "Subject:" alone on a line).
_SECTION_HEADER_LINE_RE = re.compile(
    r"^([A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*)*):\s*$", re.MULTILINE
)
