"""Lightweight input-file and output-summary validation.

Validation is intentionally kept separate from business logic so the same
checks can be reused by the CLI, REST API, and Streamlit UI without
duplicating rules.  For deep structural validation of *summaries* use the
three-tier guardrail engine in :mod:`call_summarizer.guardrails`; the checks
here are a fast pre-flight only.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..summarizer import CHAR_LIMIT

logger = logging.getLogger(__name__)

# Minimum number of meaningful characters a transcript must contain.
_MIN_TRANSCRIPT_CHARS: int = 50

# Sections that every valid summary must include.
_REQUIRED_SUMMARY_SECTIONS: list[str] = [
    "Caller:",
    "Subject:",
    "Executive Summary:",
    "Next Steps:",
]


@dataclass
class ValidationResult:
    """Outcome of a lightweight pre-flight validation check.

    Attributes:
        is_valid: ``True`` when no issues were found; ``False`` otherwise.
        issues: Human-readable description of each problem detected.
            Empty when ``is_valid`` is ``True``.
    """

    is_valid: bool
    issues: list[str] = field(default_factory=list)


def validate_input_file(file_path: Path) -> ValidationResult:
    """Validate that *file_path* is a readable, non-trivial ``.txt`` transcript.

    Checks performed (in order)
    ---------------------------
    1. File exists at the given path.
    2. File has a ``.txt`` extension.
    3. File content is at least :data:`_MIN_TRANSCRIPT_CHARS` characters long.

    Args:
        file_path: Path to the transcript file to validate.

    Returns:
        :class:`ValidationResult` with ``is_valid=False`` and a populated
        ``issues`` list when any check fails; ``is_valid=True`` otherwise.
    """
    logger.debug("Validating input file: %s", file_path)
    issues: list[str] = []

    if not file_path.exists():
        issues.append(f"File does not exist: {file_path}")
        logger.warning("Input validation failed — file not found: %s", file_path)
        return ValidationResult(is_valid=False, issues=issues)

    if file_path.suffix.lower() != ".txt":
        issues.append(f"Expected a .txt file, got: '{file_path.suffix}'")

    try:
        raw_content = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        issues.append(f"Cannot read file: {exc}")
        logger.error("Input validation failed — read error for %s: %s", file_path.name, exc)
        return ValidationResult(is_valid=False, issues=issues)

    stripped_length = len(raw_content.strip())
    if stripped_length < _MIN_TRANSCRIPT_CHARS:
        issues.append(
            f"File appears to be empty or too short "
            f"({stripped_length} chars, minimum {_MIN_TRANSCRIPT_CHARS})"
        )

    if issues:
        logger.warning("Input validation issues for %s: %s", file_path.name, issues)
    else:
        logger.info("Input file validated OK: %s", file_path.name)

    return ValidationResult(is_valid=len(issues) == 0, issues=issues)


def validate_summary(summary_text: str) -> ValidationResult:
    """Validate that *summary_text* meets the required output format.

    This is a lightweight pre-flight check used by
    :func:`~call_summarizer.service.process_transcript_file`.  For the full
    structural and content-integrity analysis, use
    :func:`~call_summarizer.guardrails.run_guardrails`.

    Checks performed
    ----------------
    - All four required sections are present (Caller, Subject, Executive
      Summary, Next Steps).
    - Total character count does not exceed :data:`~call_summarizer.summarizer.CHAR_LIMIT`.

    Args:
        summary_text: The generated summary text to validate.

    Returns:
        :class:`ValidationResult` describing any issues found.
    """
    logger.debug("Validating summary (%d chars)", len(summary_text))
    issues: list[str] = []

    for required_section in _REQUIRED_SUMMARY_SECTIONS:
        if required_section not in summary_text:
            issues.append(f"Required section missing: '{required_section}'")

    if len(summary_text) > CHAR_LIMIT:
        issues.append(
            f"Summary exceeds character limit: {len(summary_text)} chars "
            f"(limit {CHAR_LIMIT})"
        )

    if issues:
        logger.warning("Summary validation issues (%d): %s", len(issues), issues)
    else:
        logger.info("Summary validated OK (%d chars)", len(summary_text))

    return ValidationResult(is_valid=len(issues) == 0, issues=issues)
