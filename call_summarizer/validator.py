"""Input and output validation for the call summariser pipeline.

Validation is intentionally separated from business logic so that the same
checks can be reused by the CLI, a future REST API, and a Streamlit UI without
duplicating rules.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .summarizer import CHAR_LIMIT

logger = logging.getLogger(__name__)

_REQUIRED_SUMMARY_SECTIONS = [
    "Caller:",
    "Subject:",
    "Executive Summary:",
    "Next Steps:",
]

_MIN_TRANSCRIPT_CHARS = 50


@dataclass
class ValidationResult:
    """Outcome of a validation check.

    Attributes:
        is_valid: True when no issues were found.
        issues: Human-readable descriptions of each problem detected.
    """

    is_valid: bool
    issues: list[str] = field(default_factory=list)


def validate_input_file(path: Path) -> ValidationResult:
    """Validate that *path* is a readable, non-empty transcript file.

    Checks performed:
    - File exists.
    - File has a ``.txt`` extension.
    - File is not empty (content exceeds minimum threshold).

    Args:
        path: Path to the transcript file to validate.

    Returns:
        A :class:`ValidationResult` describing any issues found.
    """
    logger.debug("Validating input file: %s", path)
    issues: list[str] = []

    if not path.exists():
        issues.append(f"File does not exist: {path}")
        logger.warning("Input validation failed — file not found: %s", path)
        return ValidationResult(is_valid=False, issues=issues)

    if path.suffix.lower() != ".txt":
        issues.append(f"Expected a .txt file, got: {path.suffix}")

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        issues.append(f"Cannot read file: {exc}")
        logger.error("Input validation failed — read error: %s", exc)
        return ValidationResult(is_valid=False, issues=issues)

    if len(content.strip()) < _MIN_TRANSCRIPT_CHARS:
        issues.append(
            f"File appears to be empty or too short "
            f"({len(content.strip())} chars, minimum {_MIN_TRANSCRIPT_CHARS})"
        )

    if issues:
        logger.warning("Input validation issues for %s: %s", path.name, issues)
    else:
        logger.info("Input file validated OK: %s", path.name)

    return ValidationResult(is_valid=len(issues) == 0, issues=issues)


def validate_summary(summary: str) -> ValidationResult:
    """Validate that *summary* meets the required output format and constraints.

    Checks performed:
    - All required sections are present (Caller, Subject, Executive Summary, Next Steps).
    - Total character count is within :data:`~call_summarizer.summarizer.CHAR_LIMIT`.

    Args:
        summary: The generated summary text to validate.

    Returns:
        A :class:`ValidationResult` describing any issues found.
    """
    logger.debug("Validating summary (%d chars)", len(summary))
    issues: list[str] = []

    for section in _REQUIRED_SUMMARY_SECTIONS:
        if section not in summary:
            issues.append(f"Required section missing: '{section}'")

    if len(summary) > CHAR_LIMIT:
        issues.append(
            f"Summary exceeds character limit: {len(summary)} chars (limit {CHAR_LIMIT})"
        )

    if issues:
        logger.warning("Summary validation issues: %s", issues)
    else:
        logger.info("Summary validated OK (%d chars)", len(summary))

    return ValidationResult(is_valid=len(issues) == 0, issues=issues)
