"""Public orchestrator: runs all three input guardrail tiers in order."""

import logging

from ..observability.tracing import traceable
from .models import InputFinding, InputValidationResult
from .tier1_token_budget import _check_token_budget
from .tier2_injection import _check_injection
from .tier3_pii import _audit_pii

logger = logging.getLogger(__name__)


@traceable(name="validate_input", run_type="chain")
def validate_transcript_input(
    content: str,
    filename: str = "<unknown>",
) -> InputValidationResult:
    """Run all three input guardrail tiers against a transcript.

    Evaluation order
    ----------------
    1. **Tier 1** (token budget) — if the transcript is too long, returns
       ``allowed=False`` immediately without running further checks.
    2. **Tier 2** (injection scan) — if injection patterns are found, returns
       ``allowed=False`` immediately.
    3. **Tier 3** (PII audit) — always runs; the finding is logged and included
       in ``result.audit`` but does NOT affect ``allowed``.

    Args:
        content: Raw transcript text read from the uploaded file.
        filename: Original filename used for log entries.

    Returns:
        :class:`InputValidationResult` where ``allowed=False`` means the caller
        must return an error response and must NOT invoke the LLM.
    """
    logger.debug("Input guardrails — file: %s, chars: %d", filename, len(content))

    findings: list[InputFinding] = []

    # Tier 1: token budget
    budget_finding = _check_token_budget(content)
    if budget_finding:
        logger.warning("[INPUT-T1] %s — TRANSCRIPT_TOO_LONG: %s", filename, budget_finding.detail)
        findings.append(budget_finding)
        return InputValidationResult(allowed=False, findings=findings)

    # Tier 2: prompt injection scan
    injection_finding = _check_injection(content)
    if injection_finding:
        logger.warning(
            "[INPUT-T2] %s — PROMPT_INJECTION_DETECTED: %s",
            filename,
            injection_finding.detail,
        )
        findings.append(injection_finding)
        return InputValidationResult(allowed=False, findings=findings)

    # Tier 3: PII audit (non-blocking)
    pii_finding = _audit_pii(content, filename)
    if pii_finding:
        findings.append(pii_finding)

    logger.debug("Input guardrails passed — file: %s", filename)
    return InputValidationResult(allowed=True, findings=findings)
