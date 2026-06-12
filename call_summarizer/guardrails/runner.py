"""Public orchestrator: runs all three guardrail tiers and builds retry prompts."""

import logging

from ..models import Finding, GuardrailResult
from ..observability.tracing import traceable
from ..summarizer import CHAR_LIMIT
from .tier1_structural import run_structural_checks
from .tier2_format import run_format_checks
from .tier3_content import run_content_integrity_checks

logger = logging.getLogger(__name__)


@traceable(name="run_output_guardrails", run_type="chain")
def run_guardrails(
    summary: str,
    transcript_content: str = "",
) -> GuardrailResult:
    """Run all three guardrail tiers against a generated summary.

    Tier 3 (content integrity) is skipped when *transcript_content* is empty.

    Args:
        summary: The LLM-generated (or user-edited) summary text.
        transcript_content: The original transcript text.  Pass an empty string
            to skip Tier-3 content integrity checks.

    Returns:
        :class:`~call_summarizer.models.GuardrailResult` describing all findings.
        ``passed`` is ``True`` only when no Tier-1 errors were found.
    """
    logger.debug("Running output guardrails (transcript supplied: %s)", bool(transcript_content))

    findings: list[Finding] = []
    findings.extend(run_structural_checks(summary))
    findings.extend(run_format_checks(summary))

    if transcript_content:
        findings.extend(run_content_integrity_checks(summary, transcript_content))

    errors = [f for f in findings if f.tier == "error"]
    char_count = len(summary)

    result = GuardrailResult(
        passed=len(errors) == 0,
        findings=findings,
        char_count=char_count,
        char_within_limit=char_count <= CHAR_LIMIT,
    )

    logger.info(
        "Output guardrails complete — passed: %s, errors: %d, warnings: %d",
        result.passed,
        len(result.errors),
        len(result.warnings),
    )
    return result


def build_retry_prompt_addendum(result: GuardrailResult) -> str:
    """Build targeted corrective instructions from Tier-1 guardrail errors.

    Appended to the system prompt on the next LLM generation attempt so the
    model receives specific, actionable guidance rather than generic retry.

    Args:
        result: The failed :class:`~call_summarizer.models.GuardrailResult`.

    Returns:
        A multi-line correction string, or an empty string when there are no
        errors to address.
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
