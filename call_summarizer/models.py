"""Data models shared across the call summariser package."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, TypedDict

if TYPE_CHECKING:
    from .evaluator import EvaluationReport


class SummaryState(TypedDict):
    """LangGraph pipeline state passed between nodes.

    Attributes:
        transcript_path: Absolute or relative path to the source transcript file.
        transcript_content: Raw text content of the loaded transcript.
        summary: Generated summary text produced by the LLM.
        output_path: Destination path where the summary will be saved.
        error: Human-readable error message if any node fails; None on success.
    """

    transcript_path: str
    transcript_content: str
    summary: str
    output_path: str
    error: Optional[str]


@dataclass
class Finding:
    """A single issue detected by the guardrails engine.

    Attributes:
        tier: ``"error"`` blocks saving; ``"warning"`` is advisory only.
        code: Machine-readable identifier (e.g. ``"PHANTOM_CONDITIONAL_SECTION"``).
        message: Human-readable description shown in the UI.
        detail: Optional extracted text that triggered the check.
    """

    tier: Literal["error", "warning"]
    code: str
    message: str
    detail: str = ""


@dataclass
class GuardrailResult:
    """Aggregated outcome of running all guardrail tiers against a summary.

    Attributes:
        passed: True only when no Tier-1 (blocking) errors were found.
        findings: All findings from all tiers combined.
        char_count: Total character count of the evaluated summary.
        char_within_limit: True when ``char_count`` ≤ 1,500.
    """

    passed: bool
    findings: list[Finding]
    char_count: int
    char_within_limit: bool

    @property
    def errors(self) -> list[Finding]:
        """Return only Tier-1 blocking error findings."""
        return [f for f in self.findings if f.tier == "error"]

    @property
    def warnings(self) -> list[Finding]:
        """Return only Tier-2 / Tier-3 advisory warning findings."""
        return [f for f in self.findings if f.tier == "warning"]


@dataclass
class ProcessingResult:
    """Outcome of processing a single transcript.

    Returned by the service layer so that callers (CLI, API, Streamlit) all
    receive a uniform result regardless of how they invoked the pipeline.

    Attributes:
        transcript_path: Source transcript file that was processed.
        output_path: Destination file where the summary was written (None on error).
        success: True if the transcript was summarised and saved without errors.
        summary: The generated summary text (empty string on error).
        error: Failure description; None when success is True.
        issues: Deprecated flat list kept for backward compatibility — prefer
            ``guardrail_result`` for structured findings.
        guardrail_result: Structured guardrail findings, or None if guardrails
            were not run (e.g. when the LLM call itself failed).
    """

    transcript_path: Path
    output_path: Optional[Path]
    success: bool
    summary: str = ""
    error: Optional[str] = None
    issues: list[str] = field(default_factory=list)
    guardrail_result: Optional[GuardrailResult] = None
    # Populated by the agentic eval-feedback loop in service.py when a
    # transcript is available.  None when generation failed before evaluation.
    evaluation_report: "Optional[EvaluationReport]" = None
