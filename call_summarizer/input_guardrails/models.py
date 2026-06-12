"""Data models for the input guardrail engine."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class InputFinding:
    """A single finding from the input guardrail engine.

    Attributes:
        tier: ``"error"`` for blocking findings (Tier 1/2); ``"info"`` for
            non-blocking audit findings (Tier 3).
        code: Machine-readable identifier (e.g. ``"TRANSCRIPT_TOO_LONG"``).
        message: Human-readable description for API responses and UI display.
        detail: Optional extended detail (matched pattern, byte count, etc.).
    """

    tier: Literal["error", "info"]
    code: str
    message: str
    detail: str = ""


@dataclass
class InputValidationResult:
    """Aggregated result from :func:`validate_transcript_input`.

    Attributes:
        allowed: ``False`` when any Tier-1 or Tier-2 blocking error was found.
            The caller must reject the request and must NOT invoke the LLM.
        findings: All findings in evaluation order (errors first, then audit).
    """

    allowed: bool
    findings: list[InputFinding] = field(default_factory=list)

    @property
    def errors(self) -> list[InputFinding]:
        """Blocking findings only (``tier == "error"``)."""
        return [f for f in self.findings if f.tier == "error"]

    @property
    def audit(self) -> list[InputFinding]:
        """Non-blocking audit entries only (``tier == "info"``)."""
        return [f for f in self.findings if f.tier == "info"]
