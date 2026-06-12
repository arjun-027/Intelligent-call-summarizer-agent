#!/usr/bin/env python
"""Generate summaries for all transcripts in a directory, then evaluate them.

End-to-end script: every transcript in *transcripts_dir* is processed through
the full pipeline (input guardrails → Groq LLM → output guardrails) and the
resulting summary is saved to *output_dir*.  Each generated summary is then
scored by the eight-metric evaluation engine and a consolidated quality report
is printed to stdout.

Usage::

    # Default: Sample_data/target_data → Output_data, then evaluate
    python scripts/generate_and_evaluate.py

    # Custom directories
    python scripts/generate_and_evaluate.py \\
        --transcripts-dir Sample_data/target_data \\
        --output-dir Output_data \\
        --verbose

    # Skip generation — evaluate already-saved summaries in Output_data
    python scripts/generate_and_evaluate.py --eval-only

Output_data/ is created automatically if it does not exist.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from call_summarizer.config import load_config
from call_summarizer.evaluator import EvaluationReport, evaluate_summary
from call_summarizer.service import generate_summary_from_content
from call_summarizer.storage import derive_output_path, save_summary

# ── Display constants (ASCII-safe for Windows cp1252) ─────────────────────────
_W_NAME   = 22
_W_GRADE  = 6
_W_SCORE  = 7
_W_COL    = 7

_METRIC_ABBREVIATIONS = [
    ("Factual Groundedness", "Ground"),
    ("Completeness",         "Compl"),
    ("Format Compliance",    "Format"),
    ("Hallucination",        "Hallu"),
    ("Professionalism",      "Prof"),
    ("Handoff Readiness",    "Hndoff"),
    ("Section Precision",    "Sect"),
    ("Redundancy",           "Redu"),
]

_HEADER = (
    f"{'File':<{_W_NAME}} {'Gr':<{_W_GRADE}} {'Score':>{_W_SCORE}}"
    + "".join(f" {abbr:>{_W_COL}}" for _, abbr in _METRIC_ABBREVIATIONS)
)
_SEP = "-" * len(_HEADER)
_EQ  = "=" * len(_HEADER)


# ── Generation ────────────────────────────────────────────────────────────────

def _generate_summaries(
    transcripts_dir: Path,
    output_dir: Path,
    verbose: bool,
) -> list[tuple[Path, Path, str]]:
    """Run the LLM pipeline on every transcript and save results.

    Returns:
        List of (transcript_path, summary_path, summary_text) triples for
        every transcript that was processed successfully.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    transcripts = sorted(transcripts_dir.glob("*.txt"))

    if not transcripts:
        print(f"  No .txt transcripts found in {transcripts_dir}", file=sys.stderr)
        return []

    try:
        config = load_config()
    except EnvironmentError as exc:
        print(f"  Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    results: list[tuple[Path, Path, str]] = []
    total = len(transcripts)

    print(f"\n  Generating summaries for {total} transcript(s) in {transcripts_dir}/\n")

    for i, transcript_path in enumerate(transcripts, start=1):
        transcript_text = transcript_path.read_text(encoding="utf-8")
        summary_path = derive_output_path(transcript_path, output_dir)

        print(f"  [{i:>2}/{total}] {transcript_path.name} ...", end=" ", flush=True)
        t0 = time.perf_counter()

        result = generate_summary_from_content(
            transcript_text, config, transcript_path.name
        )
        elapsed = time.perf_counter() - t0

        if not result.success:
            print(f"FAILED ({elapsed:.1f}s) — {result.error}")
            continue

        save_summary(result.summary, summary_path)
        guardrail_status = "OK" if result.guardrail_result and result.guardrail_result.passed else "WARN"
        print(f"{guardrail_status} ({elapsed:.1f}s) -> {summary_path.name}")

        if verbose and result.guardrail_result:
            for f in result.guardrail_result.findings:
                marker = "ERROR" if f.tier == "error" else "warn "
                print(f"         [{marker}] {f.message}")

        # Prefer the evaluation already computed by the agentic loop inside
        # generate_summary_from_content rather than re-running it.
        results.append((transcript_path, summary_path, result.summary, result.evaluation_report))

        # Respect Groq's 30 RPM rate limit between calls.
        if i < total:
            time.sleep(config.rate_limit_delay_seconds)

    return results


# ── Evaluation ────────────────────────────────────────────────────────────────

def _format_row(name: str, report: EvaluationReport) -> str:
    by_name = {m.name: m for m in report.metrics}
    scores = "".join(
        f" {by_name[mn].score:>{_W_COL}.0%}"
        for mn, _ in _METRIC_ABBREVIATIONS
    )
    return (
        f"{name:<{_W_NAME}} {report.grade:<{_W_GRADE}} "
        f"{report.overall_score:>{_W_SCORE}.1%}{scores}"
    )


def _print_findings(name: str, report: EvaluationReport) -> None:
    print(f"\n{_SEP}")
    print(f"  {name}  --  Grade {report.grade}  ({report.overall_score:.1%})")
    for m in report.metrics:
        if m.score < 1.0 and m.findings:
            print(f"  [{m.name}]  {m.score:.0%}")
            for finding in m.findings[:4]:
                print(f"    * {finding}")


def _evaluate_results(
    results: list[tuple[Path, Path, str, "EvaluationReport | None"]],
    verbose: bool,
) -> None:
    """Score every generated summary and print the quality report.

    Uses the :class:`~call_summarizer.evaluator.EvaluationReport` already
    computed by the agentic eval-feedback loop inside
    :func:`~call_summarizer.service.generate_summary_from_content` when
    available.  Falls back to running the evaluator directly for summaries
    loaded from disk via ``--eval-only``.
    """
    if not results:
        print("\n  No summaries to evaluate.")
        return

    print(f"\n{_EQ}")
    print(f"{'EVALUATION REPORT':^{len(_HEADER)}}")
    print(_EQ)
    print(_HEADER)
    print(_SEP)

    reports: list[tuple[str, EvaluationReport]] = []
    for transcript_path, _summary_path, summary_text, cached_report in results:
        if cached_report is not None:
            report = cached_report   # already computed by the agentic loop
        else:
            transcript_text = transcript_path.read_text(encoding="utf-8")
            report = evaluate_summary(summary_text, transcript_text)
        label = transcript_path.stem          # e.g. "1-transcript"
        reports.append((label, report))
        print(_format_row(label, report))

    print(_EQ)

    # ── Aggregate statistics ──────────────────────────────────────────────────
    all_scores   = [r.overall_score for _, r in reports]
    all_grades   = [r.grade         for _, r in reports]
    avg_score    = sum(all_scores) / len(all_scores)

    print(f"\n  Total   : {len(reports)} summaries")
    print(f"  Average : {avg_score:.1%}")
    for g in ["A", "B", "C", "F"]:
        count = all_grades.count(g)
        if count:
            bar = "*" * count
            print(f"  Grade {g} : {count:>2}  {bar}")

    # ── Per-metric averages ───────────────────────────────────────────────────
    print(f"\n  Per-metric averages:")
    for mname, abbr in _METRIC_ABBREVIATIONS:
        metric_scores = [r.metric(mname).score for _, r in reports if r.metric(mname)]
        if metric_scores:
            avg = sum(metric_scores) / len(metric_scores)
            bar = "#" * int(avg * 20)
            print(f"    {abbr:<8} {avg:5.1%}  [{bar:<20}]")

    # ── Detailed findings for below-A or --verbose ────────────────────────────
    for label, report in reports:
        if verbose or report.grade != "A":
            _print_findings(label, report)


# ── Load pre-generated summaries (--eval-only mode) ───────────────────────────

def _load_existing(
    output_dir: Path,
    transcripts_dir: Path,
) -> list[tuple[Path, Path, str, None]]:
    """Pair existing summaries in *output_dir* with their source transcripts.

    Returns 4-tuples with ``None`` as the cached evaluation report so that
    :func:`_evaluate_results` falls back to running the evaluator directly.
    """
    results = []
    for summary_path in sorted(output_dir.glob("*-summary.txt")):
        stem = summary_path.stem.replace("-summary", "")   # "1-transcript"
        transcript_path = transcripts_dir / f"{stem}.txt"
        if not transcript_path.exists():
            transcript_path = transcripts_dir / f"{stem}-transcript.txt"
        if not transcript_path.exists():
            print(f"  SKIP {summary_path.name}: no matching transcript in {transcripts_dir}")
            continue
        summary_text = summary_path.read_text(encoding="utf-8")
        results.append((transcript_path, summary_path, summary_text, None))
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate summaries for target transcripts, then evaluate quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--transcripts-dir",
        default="Sample_data/target_data",
        metavar="PATH",
        help="Directory containing source transcript .txt files (default: Sample_data/target_data)",
    )
    parser.add_argument(
        "--output-dir",
        default="Output_data",
        metavar="PATH",
        help="Directory where generated summaries are saved (default: Output_data)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip generation — evaluate summaries already in --output-dir",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed findings for all summaries, not just below-A grades",
    )
    args = parser.parse_args()

    transcripts_dir = Path(args.transcripts_dir)
    output_dir      = Path(args.output_dir)

    if not transcripts_dir.exists():
        print(f"Error: transcripts directory not found: {transcripts_dir}", file=sys.stderr)
        sys.exit(1)

    if args.eval_only:
        print(f"\n  [eval-only] loading summaries from {output_dir}/")
        results = _load_existing(output_dir, transcripts_dir)
    else:
        results = _generate_summaries(transcripts_dir, output_dir, args.verbose)

    _evaluate_results(results, args.verbose)


if __name__ == "__main__":
    main()
