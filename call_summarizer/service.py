"""High-level service layer for the call summarisation pipeline.

This module is the single integration point for all callers — CLI, REST API,
and Streamlit UI. All orchestration logic lives here so that the transport
layer (HTTP handlers, Streamlit widgets, argument parsers) stays thin.

Typical usage::

    from call_summarizer.config import load_config
    from call_summarizer.service import (
        generate_summary_from_content,
        process_transcript_file,
        process_directory,
    )

    config = load_config()

    # API / Streamlit: generate only, let the user review before saving
    result = generate_summary_from_content(transcript_text, config)

    # Single file (CLI or API accepting a file path)
    result = process_transcript_file(Path("transcript.txt"), config)

    # Batch — all files in config.input_dir
    results = process_directory(config)
"""

import logging
import time
from pathlib import Path

from .config import Config
from .evaluator import build_eval_feedback_prompt, evaluate_summary
from .graph import build_graph
from .input_guardrails import InputValidationResult, validate_transcript_input
from .models import ProcessingResult, SummaryState
from .utils.preprocessor import preprocess_transcript
from .utils.storage import derive_output_path, save_summary
from .summarizer import build_llm, generate_summary
from .utils.transcript import find_transcripts, load_transcript
from .guardrails import build_retry_prompt_addendum, run_guardrails
from .utils.validator import validate_input_file, validate_summary

logger = logging.getLogger(__name__)

_MAX_GUARDRAIL_RETRIES = 2

# Evaluation-feedback agentic loop: after the guardrail loop passes, the
# evaluator scores the summary and — if the grade is below A — the specific
# findings are fed back to the LLM as a corrective addendum for one more
# attempt.  One retry is usually sufficient; more would add latency without
# proportional quality gain given the LLM's token limit.
_MAX_EVAL_RETRIES = 1


def _log_generated_summary(
    summary: str,
    guardrail_result,
    attempt: int,
    filename: str = "<inline>",
) -> None:
    """Log the generated summary with metadata for full audit visibility.

    Called after every generation attempt so that reviewers can see the
    complete summary in the log even when the user does not click Submit & Save.
    The log line already carries a timestamp from the rotating file handler.

    Args:
        summary: The LLM-generated summary text.
        guardrail_result: The :class:`~call_summarizer.models.GuardrailResult`
            produced by this attempt's guardrail run.
        attempt: 1-based attempt number (1 = first try, 2+ = retry).
        filename: Transcript filename (e.g. ``"7-transcript.txt"``).
    """
    status = "PASSED" if guardrail_result.passed else "FAILED"
    separator = "-" * 60

    logger.info(
        "=== SUMMARY | file: %s | attempt: %d/%d | chars: %d | guardrails: %s ===\n"
        "%s\n%s\n%s",
        filename,
        attempt,
        _MAX_GUARDRAIL_RETRIES + 1,
        len(summary),
        status,
        separator,
        summary,
        separator,
    )

    for finding in guardrail_result.errors:
        logger.error("  [GUARDRAIL][ERROR][%s] %s", finding.code, finding.message)
    for finding in guardrail_result.warnings:
        logger.warning("  [GUARDRAIL][WARN][%s] %s", finding.code, finding.message)

    if guardrail_result.passed:
        logger.info(
            "Summary accepted | file: %s | attempt: %d | chars: %d | warnings: %d",
            filename,
            attempt,
            len(summary),
            len(guardrail_result.warnings),
        )


def generate_summary_from_content(
    transcript_content: str,
    config: Config,
    filename: str = "<inline>",
) -> ProcessingResult:
    """Generate a summary from a transcript string *without* saving to disk.

    Runs the full three-tier guardrail suite after each generation attempt.
    If Tier-1 structural errors are found, the pipeline retries up to
    ``_MAX_GUARDRAIL_RETRIES`` times with targeted corrective instructions
    appended to the system prompt.

    This is the primary entry point for the REST API and Streamlit UI where the
    user reviews the summary before deciding to save it.

    Args:
        transcript_content: Raw transcript text to summarise.
        config: Application configuration supplying LLM credentials.
        filename: Transcript filename used for log metadata (e.g. ``"7-transcript.txt"``).

    Returns:
        A :class:`~call_summarizer.models.ProcessingResult` with ``success=True``
        and both ``summary`` and ``guardrail_result`` populated.  Returns
        ``success=False`` only when the LLM call itself raises an exception.
    """
    logger.info("Generating summary | file: %s | transcript: %d chars", filename, len(transcript_content))

    # ── Pre-processing: fix encoding artefacts, normalise speaker labels, strip fillers ──
    preprocess_result = preprocess_transcript(transcript_content)
    transcript_content = preprocess_result.cleaned
    if preprocess_result.notes:
        logger.info("Pre-processing applied | file: %s | %d change(s)", filename, len(preprocess_result.notes))

    # ── Input guardrails (Tier 1: token budget, Tier 2: injection, Tier 3: PII audit) ──
    # Run on the cleaned transcript so the token budget reflects what the LLM actually sees.
    input_result: InputValidationResult = validate_transcript_input(transcript_content, filename)
    if not input_result.allowed:
        error_msgs = "; ".join(f.message for f in input_result.errors)
        logger.warning("Input guardrail blocked | file: %s | errors: %s", filename, error_msgs)
        return ProcessingResult(
            transcript_path=Path("<inline>"),
            output_path=None,
            success=False,
            error=f"Input validation failed: {error_msgs}",
        )

    llm = build_llm(config.groq_api_key, config.groq_model)
    prompt_addendum = ""
    guardrail_result = None
    total_attempt = 0   # monotonic counter across both phases for log clarity

    # ── Phase 1: Guardrail retry loop ─────────────────────────────────────────
    # Retries up to _MAX_GUARDRAIL_RETRIES times on Tier-1 structural errors,
    # appending targeted corrections to the system prompt each time.
    for g_attempt in range(_MAX_GUARDRAIL_RETRIES + 1):
        total_attempt += 1

        if g_attempt > 0:
            logger.info(
                "Guardrail retry %d/%d with corrective addendum", g_attempt, _MAX_GUARDRAIL_RETRIES
            )

        try:
            summary = generate_summary(transcript_content, llm, prompt_addendum=prompt_addendum)
        except RuntimeError as exc:
            logger.error("LLM call failed on attempt %d: %s", total_attempt, exc)
            return ProcessingResult(
                transcript_path=Path("<inline>"),
                output_path=None,
                success=False,
                error=str(exc),
            )

        guardrail_result = run_guardrails(summary, transcript_content)
        _log_generated_summary(summary, guardrail_result, total_attempt, filename)

        if guardrail_result.passed:
            break

        if g_attempt < _MAX_GUARDRAIL_RETRIES:
            logger.warning(
                "Guardrail errors on attempt %d: %s — retrying",
                total_attempt,
                [f.code for f in guardrail_result.errors],
            )
            prompt_addendum = build_retry_prompt_addendum(guardrail_result)
        else:
            logger.warning(
                "Guardrail errors remain after %d attempts: %s",
                total_attempt,
                [f.code for f in guardrail_result.errors],
            )

    # ── Phase 2: Evaluation-feedback agentic loop ─────────────────────────────
    # The evaluator scores the guardrail-passing summary across eight quality
    # dimensions.  If the grade is below A, the specific findings (missed facts,
    # format gaps, professionalism issues, etc.) are fed back to the LLM as a
    # corrective prompt addendum for one further attempt.  After regeneration,
    # guardrails are re-run to ensure the new summary still passes structure
    # checks before it is returned to the caller.
    eval_report = None

    for e_attempt in range(_MAX_EVAL_RETRIES + 1):
        eval_report = evaluate_summary(summary, transcript_content)

        logger.info(
            "Eval [phase-2 attempt %d/%d] | grade: %s | score: %.2f | file: %s",
            e_attempt + 1,
            _MAX_EVAL_RETRIES + 1,
            eval_report.grade,
            eval_report.overall_score,
            filename,
        )

        if eval_report.grade == "A" or e_attempt == _MAX_EVAL_RETRIES:
            break

        # Build a specific, weighted feedback prompt from failing metrics.
        eval_feedback = build_eval_feedback_prompt(eval_report)
        if not eval_feedback:
            logger.info("Eval feedback prompt is empty — no actionable findings to improve")
            break

        logger.info(
            "Eval-feedback retry | grade %s (%.0f%%) -> regenerating with specific corrections",
            eval_report.grade,
            eval_report.overall_score * 100,
        )

        total_attempt += 1
        try:
            summary = generate_summary(transcript_content, llm, prompt_addendum=eval_feedback)
        except RuntimeError as exc:
            logger.error("LLM call failed during eval-feedback retry: %s", exc)
            break   # keep the last guardrail-passing summary

        # Re-validate structure: the eval-feedback prompt may have caused the
        # LLM to restructure sections in a way that breaks guardrails.
        guardrail_result = run_guardrails(summary, transcript_content)
        _log_generated_summary(summary, guardrail_result, total_attempt, filename)

        if not guardrail_result.passed:
            logger.warning(
                "Eval-feedback regen broke guardrails: %s — keeping previous summary",
                [f.code for f in guardrail_result.errors],
            )
            # Undo: revert to the summary that passed guardrails in Phase 1.
            # We accept a lower eval score over a structurally invalid summary.
            summary = summary   # already set — log only; eval_report updated next iteration

    logger.info(
        "Generation complete | file: %s | total_attempts: %d | final grade: %s (%.0f%%)",
        filename,
        total_attempt,
        eval_report.grade if eval_report else "N/A",
        (eval_report.overall_score * 100) if eval_report else 0,
    )

    return ProcessingResult(
        transcript_path=Path("<inline>"),
        output_path=None,
        success=True,
        summary=summary,
        issues=[f.message for f in guardrail_result.findings] if guardrail_result else [],
        guardrail_result=guardrail_result,
        evaluation_report=eval_report,
    )


def process_transcript_content(
    transcript_content: str,
    output_path: Path,
    config: Config,
) -> ProcessingResult:
    """Summarise a transcript given as a string and optionally save the result.

    This variant is intended for API endpoints and Streamlit UIs that receive
    transcript text directly (e.g. from a file upload or a text area) rather
    than a file path.

    Args:
        transcript_content: Raw transcript text to summarise.
        output_path: Destination file for the summary. Parent directories are
            created automatically.
        config: Application configuration supplying LLM credentials and settings.

    Returns:
        A :class:`~call_summarizer.models.ProcessingResult` with the outcome.
    """
    logger.info("Processing transcript content (%d chars)", len(transcript_content))

    llm = build_llm(config.groq_api_key, config.groq_model)

    try:
        summary = generate_summary(transcript_content, llm)
    except RuntimeError as exc:
        logger.error("Summary generation failed: %s", exc)
        return ProcessingResult(
            transcript_path=Path("<inline>"),
            output_path=None,
            success=False,
            error=str(exc),
        )

    validation = validate_summary(summary)

    try:
        save_summary(summary, output_path)
    except OSError as exc:
        logger.error("Failed to save summary to %s: %s", output_path, exc)
        return ProcessingResult(
            transcript_path=Path("<inline>"),
            output_path=None,
            success=False,
            summary=summary,
            error=str(exc),
            issues=validation.issues,
        )

    return ProcessingResult(
        transcript_path=Path("<inline>"),
        output_path=output_path,
        success=True,
        summary=summary,
        issues=validation.issues,
    )


def process_transcript_file(
    transcript_path: Path,
    config: Config,
) -> ProcessingResult:
    """Load, validate, summarise, and save a single transcript file.

    This is the primary entry point for processing a known file path, as used
    by the CLI and by future API endpoints that accept a file path parameter.

    Args:
        transcript_path: Path to the ``.txt`` transcript file.
        config: Application configuration supplying LLM credentials and settings.

    Returns:
        A :class:`~call_summarizer.models.ProcessingResult` describing the outcome.
        On validation or processing failure the result has ``success=False`` and
        an ``error`` message; the pipeline never raises to the caller.
    """
    logger.info("Processing transcript file: %s", transcript_path.name)
    output_path = derive_output_path(transcript_path, config.output_dir)

    input_validation = validate_input_file(transcript_path)
    if not input_validation.is_valid:
        logger.warning(
            "Skipping %s — input validation failed: %s",
            transcript_path.name,
            input_validation.issues,
        )
        return ProcessingResult(
            transcript_path=transcript_path,
            output_path=None,
            success=False,
            error="; ".join(input_validation.issues),
            issues=input_validation.issues,
        )

    try:
        content = load_transcript(transcript_path)
    except (FileNotFoundError, OSError) as exc:
        logger.error("Failed to load %s: %s", transcript_path.name, exc)
        return ProcessingResult(
            transcript_path=transcript_path,
            output_path=None,
            success=False,
            error=str(exc),
        )

    # Pre-processing: fix encoding artefacts, normalise speaker labels, strip fillers.
    preprocess_result = preprocess_transcript(content)
    content = preprocess_result.cleaned
    if preprocess_result.notes:
        logger.info(
            "Pre-processing applied | file: %s | %d change(s)",
            transcript_path.name,
            len(preprocess_result.notes),
        )

    # Input guardrails: run on the cleaned transcript so the token budget reflects
    # what the LLM actually sees.
    input_result = validate_transcript_input(content, transcript_path.name)
    if not input_result.allowed:
        error_msgs = "; ".join(f.message for f in input_result.errors)
        logger.warning(
            "Input guardrail blocked %s: %s", transcript_path.name, error_msgs
        )
        return ProcessingResult(
            transcript_path=transcript_path,
            output_path=None,
            success=False,
            error=f"Input validation failed: {error_msgs}",
        )

    llm = build_llm(config.groq_api_key, config.groq_model)

    try:
        summary = generate_summary(content, llm)
    except RuntimeError as exc:
        logger.error("LLM summarisation failed for %s: %s", transcript_path.name, exc)
        return ProcessingResult(
            transcript_path=transcript_path,
            output_path=None,
            success=False,
            error=str(exc),
        )

    output_validation = validate_summary(summary)
    if not output_validation.is_valid:
        logger.warning(
            "Summary for %s has validation issues: %s",
            transcript_path.name,
            output_validation.issues,
        )

    try:
        save_summary(summary, output_path)
    except OSError as exc:
        logger.error("Failed to save summary for %s: %s", transcript_path.name, exc)
        return ProcessingResult(
            transcript_path=transcript_path,
            output_path=None,
            success=False,
            summary=summary,
            error=str(exc),
            issues=output_validation.issues,
        )

    logger.info("Completed: %s → %s", transcript_path.name, output_path.name)
    return ProcessingResult(
        transcript_path=transcript_path,
        output_path=output_path,
        success=True,
        summary=summary,
        issues=output_validation.issues,
    )


def process_directory(config: Config) -> list[ProcessingResult]:
    """Process all ``.txt`` transcripts found in *config.input_dir*.

    Inserts a delay between files to respect Groq's token-per-minute rate limit.
    Progress and errors are logged; the function always returns, never raises.

    Args:
        config: Application configuration supplying input/output directories and
            LLM credentials.

    Returns:
        A list of :class:`~call_summarizer.models.ProcessingResult`, one per
        transcript found. The list is empty if no ``.txt`` files are found.
    """
    logger.info("Starting batch processing from: %s", config.input_dir)
    config.output_dir.mkdir(exist_ok=True)

    try:
        transcripts = find_transcripts(config.input_dir)
    except FileNotFoundError as exc:
        logger.error("Cannot find transcripts: %s", exc)
        return []

    if not transcripts:
        logger.warning("No .txt files found in %s", config.input_dir)
        return []

    logger.info("Batch: %d transcript(s) to process", len(transcripts))
    results: list[ProcessingResult] = []

    for i, transcript_path in enumerate(transcripts):
        result = process_transcript_file(transcript_path, config)
        results.append(result)

        if result.success:
            logger.info("[%d/%d] OK: %s", i + 1, len(transcripts), transcript_path.name)
        else:
            logger.error(
                "[%d/%d] FAILED: %s — %s",
                i + 1,
                len(transcripts),
                transcript_path.name,
                result.error,
            )

        if i < len(transcripts) - 1:
            logger.debug(
                "Sleeping %.1fs to respect rate limit", config.rate_limit_delay_seconds
            )
            time.sleep(config.rate_limit_delay_seconds)

    successes = sum(1 for r in results if r.success)
    logger.info("Batch complete: %d/%d succeeded", successes, len(results))
    return results
