"""Transcript pre-processing: normalises raw ASR/transcription artefacts before LLM ingestion.

Handles five categories of real-world transcript quality issues, applied in a fixed,
deterministic order so the output is predictable and testable:

1. Encoding artefacts -- UTF-8 bytes mis-decoded as Windows-1252/Latin-1 (mojibake).
   Example: 'ae...' appears instead of the ellipsis character when a UTF-8 file is
   decoded as Latin-1.

2. Inaudible / garbled audio markers -- normalised to the canonical token ``[unclear]``.
   Preserving the marker (rather than deleting it) tells the LLM that a gap exists so
   it does not hallucinate content to fill the silence.

3. Speaker label variants -- normalised to canonical ``Caller:`` / ``Agent:`` so the LLM
   receives a consistent vocabulary regardless of the transcription tool that produced
   the file.

4. Filler words and false starts -- ``um``, ``uh``, ``er``, ``ah``, ``hmm`` and stutter
   patterns like ``"I- I"`` are removed to reduce noise and lower token usage.

5. Whitespace normalisation -- collapses runs of spaces/tabs and excess blank lines
   without altering line structure or factual content.

Facts (amounts, IBANs, reference numbers, names, dates) are never altered.
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── 1. Encoding artefact map ──────────────────────────────────────────────────
# These mojibake patterns arise when a UTF-8 encoded file is decoded as Windows-1252
# or Latin-1.  The key is the incorrect multi-byte Latin-1 string; the value is the
# correct Unicode character (expressed as escape sequences to avoid save-time corruption).
# Ordered longest-first so overlapping patterns match correctly.
_MOJIBAKE: list[tuple[str, str]] = [
    ("ae…",     "…"),  # HORIZONTAL ELLIPSIS (U+2026)
    ("ae™",     "’"),  # RIGHT SINGLE QUOTATION MARK  (U+2019)
    ("ae˜",     "‘"),  # LEFT SINGLE QUOTATION MARK   (U+2018)
    ("ae\x9d",       "”"),  # RIGHT DOUBLE QUOTATION MARK  (U+201d)
    ("aeœ",     "“"),  # LEFT DOUBLE QUOTATION MARK   (U+201c)
    ("ae“",     "—"),  # EM DASH  (U+2014)
    ("ae–",     "–"),  # EN DASH  (U+2013)
    ("\xc2\xa3",     "\xa3"),    # POUND SIGN  (U+00a3)
    ("\xc2\x80",     "€"),  # EURO SIGN   (U+20ac)
    ("\xc2\xb0",     "\xb0"),    # DEGREE SIGN (U+00b0)
    ("\xc3\xa9",     "\xe9"),    # e with acute (U+00e9)
    ("\xc3\xa8",     "\xe8"),    # e with grave (U+00e8)
    ("\xc3\xa0",     "\xe0"),    # a with grave (U+00e0)
    ("\xc3\xa2",     "\xe2"),    # a with circumflex (U+00e2)
]

# The most common real-world mojibake pattern in insurance transcripts: the ellipsis
# used to mark speech pauses becomes the 3-byte sequence C3 A2 C2 80 C2 A6 when
# UTF-8 is decoded as Latin-1.  Match it directly as a literal string.
_ELLIPSIS_MOJIBAKE = "â€¦"

# ── 2. Inaudible / garbled audio markers ─────────────────────────────────────
_INAUDIBLE_RE = re.compile(
    r"""
    \[(?:inaudible|unclear|crosstalk|noise|unintelligible|indistinct)\]
    | \((?:inaudible|unclear|crosstalk|noise|indistinct)\)
    | \*{2,3}               # ** or *** used by some ASR platforms
    | \[[\?\!]{2,}\]        # [??] or [!!]
    | <[Uu][Uu]>            # <uu> used by some ASR systems
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ── 3. Speaker label normalisation ────────────────────────────────────────────
_SPEAKER_NORMALISATIONS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"^(customer|caller|client|policyholder|claimant"
            r"|insured|third[\s_-]?party|complainant)\s*:",
            re.IGNORECASE,
        ),
        "Caller:",
    ),
    (
        re.compile(
            r"^(agent|rep(?:resentative)?|advisor|handler|staff"
            r"|operator|associate|claims[\s_-]?handler|adjuster)\s*:",
            re.IGNORECASE,
        ),
        "Agent:",
    ),
    # Speaker 1 / S1 -> Caller (typically the inbound party)
    (re.compile(r"^(?:speaker\s*1|s1)\s*:", re.IGNORECASE), "Caller:"),
    # Speaker 2 / S2 -> Agent (typically the company representative)
    (re.compile(r"^(?:speaker\s*2|s2)\s*:", re.IGNORECASE), "Agent:"),
]

# ── 4. Filler words ───────────────────────────────────────────────────────────
# Word-boundary anchors prevent matching inside real words (e.g. "umbrella", "Ahmed").
_FILLER_RE = re.compile(
    r"\b(u+m+|u+h+|e+r+|a+h+|h+m+m*|m+m+h?)\b",
    re.IGNORECASE,
)

# False starts: "I- I was" -> "I was",  "we- we can" -> "we can"
_FALSE_START_RE = re.compile(r"\b(\w+)-\s+\1\b", re.IGNORECASE)

# Orphaned punctuation left when a filler word is stripped from mid-sentence
_ORPHANED_PUNCTUATION_RE = re.compile(r"\s+([,;])")


@dataclass
class PreprocessResult:
    """Outcome of :func:`preprocess_transcript`.

    Attributes:
        cleaned: Normalised transcript text ready for guardrail checks and LLM ingestion.
        notes: Human-readable list of changes applied (empty when nothing changed).
    """

    cleaned: str
    notes: list[str] = field(default_factory=list)


def preprocess_transcript(text: str) -> PreprocessResult:
    """Normalise a raw transcript for LLM ingestion.

    Applies five fix categories in deterministic order.  Facts (amounts, IBANs,
    reference numbers, names, dates) are never altered.  ``[unclear]`` markers are
    preserved so the LLM knows gaps exist rather than hallucinating content.

    Args:
        text: Raw transcript string read from the uploaded file.

    Returns:
        :class:`PreprocessResult` with the cleaned text and a list of change notes.
        Notes are empty when the transcript required no changes.
    """
    notes: list[str] = []

    # ── Step 1: Fix encoding artefacts ───────────────────────────────────────
    # Handle the most common ellipsis mojibake pattern first (3-byte sequence)
    if _ELLIPSIS_MOJIBAKE in text:
        count = text.count(_ELLIPSIS_MOJIBAKE)
        text = text.replace(_ELLIPSIS_MOJIBAKE, "...")
        notes.append(f"encoding artefact: {count}x ellipsis mojibake fixed")

    for artefact, replacement in _MOJIBAKE:
        if artefact in text:
            count = text.count(artefact)
            text = text.replace(artefact, replacement)
            notes.append(f"encoding artefact: {count}x mojibake corrected")

    # ── Step 2: Normalise inaudible/garbled markers ───────────────────────────
    text, inaudible_count = _INAUDIBLE_RE.subn("[unclear]", text)
    if inaudible_count:
        notes.append(f"{inaudible_count} inaudible marker(s) normalised to [unclear]")

    # ── Step 3: Normalise speaker labels (per-line, never mid-sentence) ───────
    lines = text.splitlines()
    speaker_counts: dict[str, int] = {}
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        for pattern, canonical in _SPEAKER_NORMALISATIONS:
            m = pattern.match(stripped)
            if m:
                indent = line[: len(line) - len(stripped)]
                rest = stripped[m.end():]
                lines[i] = f"{indent}{canonical}{rest}"
                original = m.group(0).rstrip(": \t")
                speaker_counts[original] = speaker_counts.get(original, 0) + 1
                break
    text = "\n".join(lines)
    for label, count in speaker_counts.items():
        notes.append(f"speaker label '{label}:' normalised to canonical form ({count} turn(s))")

    # ── Step 4a: Resolve false starts before filler removal ───────────────────
    text, fs_count = _FALSE_START_RE.subn(r"\1", text)
    if fs_count:
        notes.append(f"{fs_count} false start(s) resolved")

    # ── Step 4b: Remove filler words ─────────────────────────────────────────
    text, filler_count = _FILLER_RE.subn("", text)
    if filler_count:
        # Clean up punctuation artefacts left when a filler sat between two commas:
        # "Hello, um, I" -> "Hello,  , I" -> "Hello, I"
        text = re.sub(r",(\s*,)+", ",", text)   # multiple commas -> single comma
        text = _ORPHANED_PUNCTUATION_RE.sub(r"\1", text)
        notes.append(f"{filler_count} filler word(s) removed")

    # ── Step 5: Whitespace normalisation ─────────────────────────────────────
    text = re.sub(r"[ \t]{2,}", " ", text)           # multiple spaces/tabs -> one space
    text = re.sub(r"\n{3,}", "\n\n", text)            # 3+ blank lines -> one blank line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = text.strip()

    if notes:
        logger.info("Transcript pre-processing: %s", "; ".join(notes))
    else:
        logger.debug("Transcript pre-processing: no changes needed")

    return PreprocessResult(cleaned=text, notes=notes)
