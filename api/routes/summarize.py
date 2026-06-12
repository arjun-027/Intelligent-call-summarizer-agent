"""Summary generation and submission API endpoints.

Routes
------
POST /api/v1/summarize
    Upload a ``.txt`` transcript file and receive a generated summary together
    with full guardrail findings.  The summary is **not** saved; the client
    reviews it first.

POST /api/v1/summaries
    Persist a (possibly user-edited) summary to the Output_data directory.
    Re-runs Tier-1 guardrails as defence-in-depth before writing to disk.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from call_summarizer.config import Config
from call_summarizer.guardrails import run_guardrails
from call_summarizer.input_guardrails import validate_transcript_input
from call_summarizer.service import generate_summary_from_content
from call_summarizer.utils.storage import derive_output_path, save_summary
from call_summarizer.summarizer import CHAR_LIMIT

from ..schemas import SummarizeResponse, SubmitRequest, SubmitResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_config(request: Request) -> Config:
    """Retrieve the application :class:`~call_summarizer.config.Config` from app state.

    Args:
        request: The current FastAPI :class:`~fastapi.Request` object.

    Returns:
        The :class:`~call_summarizer.config.Config` instance set during startup.
    """
    return request.app.state.config


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Generate a summary from an uploaded transcript",
    description=(
        "Upload a `.txt` call transcript and receive a structured summary with "
        "full guardrail findings. The summary is NOT saved — call "
        "POST /api/v1/summaries to persist after review."
    ),
)
async def generate_summary_endpoint(
    request: Request,
    file: UploadFile = File(..., description="A single .txt call transcript file"),
) -> SummarizeResponse:
    """Upload a ``.txt`` transcript and return a generated summary.

    Runs the three-tier guardrail suite automatically.  Tier-1 errors trigger
    up to two automatic regeneration attempts before the result is returned.

    Args:
        request: FastAPI request (provides access to app state / config).
        file: The uploaded transcript file.  Must be a ``.txt`` file.

    Returns:
        A :class:`~api.schemas.SummarizeResponse` containing the summary,
        character count, and all guardrail findings split into errors and warnings.

    Raises:
        HTTPException 400: File is not ``.txt``, not UTF-8, empty, too long
            (token budget exceeded), or contains injection patterns.
        HTTPException 500: LLM call fails.
    """
    logger.info(
        "POST /summarize — file: %s, content-type: %s", file.filename, file.content_type
    )

    if not (file.filename or "").lower().endswith(".txt"):
        logger.warning("Rejected non-.txt upload: %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only .txt files are accepted. Received: '{file.filename}'.",
        )

    try:
        raw_bytes = await file.read()
        transcript_content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        logger.warning("UTF-8 decode error for %s: %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File could not be decoded as UTF-8: {exc}",
        ) from exc

    if not transcript_content.strip():
        logger.warning("Rejected empty file: %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Input guardrails: token budget (Tier 1) and injection scan (Tier 2).
    # PII audit (Tier 3) runs inside and is logged regardless of the outcome.
    input_result = await run_in_threadpool(
        validate_transcript_input, transcript_content, file.filename
    )
    if not input_result.allowed:
        error_messages = [f.message for f in input_result.errors]
        logger.warning(
            "Input guardrail blocked %s — codes: %s",
            file.filename,
            [f.code for f in input_result.errors],
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Transcript failed input validation.", "errors": error_messages},
        )

    config = _get_config(request)
    result = await run_in_threadpool(
        generate_summary_from_content, transcript_content, config, file.filename
    )

    if not result.success:
        logger.error("Summary generation failed for %s: %s", file.filename, result.error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summary generation failed: {result.error}",
        )

    gr = result.guardrail_result
    errors = [f.message for f in gr.errors] if gr else []
    warnings = [f.message for f in gr.warnings] if gr else []
    passed = gr.passed if gr else True

    ev = result.evaluation_report
    eval_grade = ev.grade if ev else ""
    eval_score = ev.overall_score if ev else 0.0
    eval_findings = (
        [f"{m.name}: {'; '.join(m.findings[:2])}"
         for m in ev.metrics if m.score < 1.0 and m.findings
         and not (len(m.findings) == 1 and "not evaluated" in m.findings[0])]
        if ev else []
    )

    logger.info(
        "Summary ready for %s — %d chars, guardrails: %s, eval: %s (%.0f%%)",
        file.filename,
        len(result.summary),
        "PASS" if passed else "FAIL",
        eval_grade or "N/A",
        eval_score * 100,
    )

    return SummarizeResponse(
        filename=Path(file.filename).stem,
        summary=result.summary,
        char_count=len(result.summary),
        within_char_limit=len(result.summary) <= CHAR_LIMIT,
        passed_guardrails=passed,
        errors=errors,
        warnings=warnings,
        eval_grade=eval_grade,
        eval_score=eval_score,
        eval_findings=eval_findings,
    )


@router.post(
    "/summaries",
    response_model=SubmitResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save a reviewed summary to Output_data",
    description=(
        "Persist a summary (which may have been edited by the user) to "
        "``Output_data/<filename>-summary.txt``. "
        "Re-runs Tier-1 guardrails before writing. "
        "Returns 422 if blocking structural errors are present."
    ),
)
async def save_summary_endpoint(
    request: Request,
    body: SubmitRequest,
) -> SubmitResponse:
    """Validate (Tier-1 + Tier-2) and save the submitted summary.

    Tier-1 guardrails are re-run as defence-in-depth — a client that bypasses
    the UI error gates cannot force an invalid summary onto disk.

    Args:
        request: FastAPI request (provides access to app state / config).
        body: :class:`~api.schemas.SubmitRequest` with filename stem and summary.

    Returns:
        :class:`~api.schemas.SubmitResponse` describing the saved file.

    Raises:
        HTTPException 422: Tier-1 structural errors found in the submitted summary.
        HTTPException 500: File system write failure.
    """
    logger.info(
        "POST /summaries — filename: %s, chars: %d", body.filename, len(body.summary)
    )

    # Defence-in-depth: re-run Tier-1 guardrails even if the UI already checked
    guardrail_result = await run_in_threadpool(run_guardrails, body.summary)

    if not guardrail_result.passed:
        error_messages = [f.message for f in guardrail_result.errors]
        logger.warning(
            "Submit blocked — Tier-1 errors: %s", [f.code for f in guardrail_result.errors]
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Summary has structural errors that must be resolved before saving.",
                "errors": error_messages,
            },
        )

    config = _get_config(request)
    transcript_path = Path(f"{body.filename}.txt")
    output_path = derive_output_path(transcript_path, config.output_dir)

    try:
        await run_in_threadpool(save_summary, body.summary, output_path)
    except OSError as exc:
        logger.error("Failed to write summary to %s: %s", output_path, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to write summary file: {exc}",
        ) from exc

    warnings = [f.message for f in guardrail_result.warnings]
    logger.info(
        "Summary saved: %s (%d chars, %d warnings)", output_path.name, len(body.summary), len(warnings)
    )

    return SubmitResponse(
        output_filename=output_path.name,
        output_path=str(output_path),
        char_count=len(body.summary),
        warnings=warnings,
        message=f"Summary saved successfully to {output_path.name}",
    )
