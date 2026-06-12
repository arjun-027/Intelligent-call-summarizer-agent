"""Pydantic request and response schemas for the Call Summariser API."""

from pydantic import BaseModel, Field

from call_summarizer.summarizer import CHAR_LIMIT


class SummarizeResponse(BaseModel):
    """Response body for ``POST /api/v1/summarize``.

    The summary is returned but not yet persisted.  The client may edit it
    and then call ``POST /api/v1/summaries`` to save.

    Attributes:
        filename: Stem of the uploaded file (no extension), used as the key
            when submitting for save.
        summary: Generated summary text produced by the LLM.
        char_count: Total character count of the summary.
        within_char_limit: True when ``char_count`` is within the 1,500-char limit.
        passed_guardrails: True when no Tier-1 structural errors were found.
            The UI uses this to enable or disable the Submit button.
        errors: Tier-1 blocking error messages.  The summary cannot be saved
            while any errors are present.
        warnings: Tier-2 / Tier-3 advisory messages.  The summary may still be
            saved; the reviewer should inspect these findings.
    """

    filename: str = Field(..., description="Transcript filename stem (no extension)")
    summary: str = Field(..., description="Generated summary text")
    char_count: int = Field(..., description="Total character count of the summary")
    within_char_limit: bool = Field(
        ..., description=f"True when char_count ≤ {CHAR_LIMIT}"
    )
    passed_guardrails: bool = Field(
        ..., description="True when no Tier-1 structural errors were detected"
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Tier-1 blocking error messages — save is blocked until resolved",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Tier-2 / Tier-3 advisory warnings — save is still allowed",
    )
    # ── Evaluation fields (from the agentic eval-feedback loop) ───────────────
    eval_grade: str = Field(
        default="",
        description="Letter grade from the eight-metric evaluator: A / B / C / F",
    )
    eval_score: float = Field(
        default=0.0,
        description="Weighted overall quality score (0.0–1.0)",
    )
    eval_findings: list[str] = Field(
        default_factory=list,
        description="Failing metric descriptions from the final evaluation pass",
    )


class SubmitRequest(BaseModel):
    """Request body for ``POST /api/v1/summaries``.

    Attributes:
        filename: Stem of the original transcript file.  Used to derive the
            output path (``Output_data/<filename>-summary.txt``).
        summary: Summary text to persist.  May differ from the LLM-generated
            version if the user edited it in the UI.
    """

    filename: str = Field(
        ...,
        description="Transcript filename stem — used to derive the output path",
    )
    summary: str = Field(
        ...,
        description="Summary text to save (may be user-edited)",
    )


class SubmitResponse(BaseModel):
    """Response body for a successful ``POST /api/v1/summaries``.

    Attributes:
        output_filename: Name of the file that was written.
        output_path: Relative path to the saved summary file.
        char_count: Character count of the saved summary.
        warnings: Any Tier-2 / Tier-3 warnings detected on the submitted text.
        message: Human-readable confirmation message.
    """

    output_filename: str = Field(..., description="Name of the saved summary file")
    output_path: str = Field(..., description="Relative path to the saved file")
    char_count: int = Field(..., description="Character count of the saved summary")
    warnings: list[str] = Field(
        default_factory=list,
        description="Advisory warnings on the saved summary",
    )
    message: str = Field(..., description="Human-readable success message")
