#!/usr/bin/env python
"""Batch evaluation script for the insurance call summariser.

Evaluates every transcript/summary pair found in the examples directory and
prints a formatted quality report.  Pairs are grouped by quality tier
(good / okay / bad) so you can verify that the metrics correctly distinguish
summary quality levels.

Usage::

    # Evaluate all examples in the default directory
    python scripts/run_evaluation.py

    # Point at a different directory
    python scripts/run_evaluation.py --examples-dir path/to/examples

    # Show detailed findings for every example (not just below-A grades)
    python scripts/run_evaluation.py --verbose

    # Evaluate generated summaries in Output_data/ against target transcripts
    python scripts/run_evaluation.py \\
        --summaries-dir Output_data \\
        --transcripts-dir Sample_data/target_data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from call_summarizer.evaluator import EvaluationReport, evaluate_summary

# ── Column widths ──────────────────────────────────────────────────────────────
_W_NAME = 22
_W_GRADE = 6
_W_SCORE = 7
_W_COL = 7

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

_HEADER_ROW = (
    f"{'Example':<{_W_NAME}} {'Gr':<{_W_GRADE}} {'Score':>{_W_SCORE}}"
    + "".join(f" {abbr:>{_W_COL}}" for _, abbr in _METRIC_ABBREVIATIONS)
)
_SEPARATOR = "-" * len(_HEADER_ROW)


def _format_row(name: str, report: EvaluationReport) -> str:
    metrics_by_name = {m.name: m for m in report.metrics}
    scores = "".join(
        f" {metrics_by_name[mname].score:>{_W_COL}.0%}"
        for mname, _ in _METRIC_ABBREVIATIONS
    )
    return (
        f"{name:<{_W_NAME}} {report.grade:<{_W_GRADE}} "
        f"{report.overall_score:>{_W_SCORE}.1%}{scores}"
    )


def _print_findings(name: str, report: EvaluationReport, feedback: str) -> None:
    print(f"\n{_SEPARATOR}")
    print(f"  {name}  —  Grade {report.grade}  ({report.overall_score:.1%})")
    if feedback:
        # Trim to 3 lines so it doesn't dominate the output
        fb_lines = [ln.strip() for ln in feedback.splitlines() if ln.strip()][:3]
        print(f"  Human feedback: {' | '.join(fb_lines)}")
    for m in report.metrics:
        if m.score < 1.0 and m.findings:
            print(f"  [{m.name}]  {m.score:.0%}")
            for finding in m.findings[:4]:
                print(f"    * {finding}")


def _evaluate_paired_directory(
    examples_dir: Path,
    verbose: bool,
) -> list[tuple[str, str, EvaluationReport, str]]:
    """Evaluate all transcript/summary pairs in *examples_dir*.

    Returns list of (tier, name, report, feedback).
    """
    results: list[tuple[str, str, EvaluationReport, str]] = []

    for transcript_path in sorted(examples_dir.glob("*-transcript.txt")):
        stem = transcript_path.name.replace("-transcript.txt", "")
        summary_path = examples_dir / f"{stem}-summary.txt"
        feedback_path = examples_dir / f"{stem}-feedback.txt"

        if not summary_path.exists():
            print(f"  SKIP {stem}: no matching summary file")
            continue

        transcript = transcript_path.read_text(encoding="utf-8")
        summary = summary_path.read_text(encoding="utf-8")
        feedback = feedback_path.read_text(encoding="utf-8").strip() if feedback_path.exists() else ""

        report = evaluate_summary(summary, transcript)

        # Derive tier from filename prefix (good/okay/bad)
        tier = stem.split("-")[0] if "-" in stem else "unknown"
        results.append((tier, stem, report, feedback))

    return results


def _evaluate_generated_summaries(
    summaries_dir: Path,
    transcripts_dir: Path,
) -> list[tuple[str, str, EvaluationReport, str]]:
    """Evaluate generated summaries in *summaries_dir* against *transcripts_dir*.

    Matches ``<stem>-summary.txt`` in summaries_dir to ``<stem>-transcript.txt``
    in transcripts_dir (or ``<stem>.txt`` for target_data style files).
    """
    results: list[tuple[str, str, EvaluationReport, str]] = []

    for summary_path in sorted(summaries_dir.glob("*-summary.txt")):
        stem = summary_path.name.replace("-summary.txt", "")

        # Support both naming conventions
        transcript_path = transcripts_dir / f"{stem}-transcript.txt"
        if not transcript_path.exists():
            transcript_path = transcripts_dir / f"{stem}.txt"
        if not transcript_path.exists():
            transcript_path = transcripts_dir / f"{stem}-transcript.txt"

        summary = summary_path.read_text(encoding="utf-8")

        if transcript_path.exists():
            transcript = transcript_path.read_text(encoding="utf-8")
            report = evaluate_summary(summary, transcript)
        else:
            print(f"  NOTE {stem}: no matching transcript — format-only evaluation")
            report = evaluate_summary(summary)

        results.append(("generated", stem, report, ""))

    return results


def _print_report(
    results: list[tuple[str, str, EvaluationReport, str]],
    verbose: bool,
) -> None:
    if not results:
        print("No pairs evaluated.")
        return

    print("\n" + "=" * len(_HEADER_ROW))
    print(f"{'EVALUATION REPORT':^{len(_HEADER_ROW)}}")
    print("=" * len(_HEADER_ROW))
    print(_HEADER_ROW)
    print(_SEPARATOR)

    grades_by_tier: dict[str, list[str]] = {}
    scores_by_tier: dict[str, list[float]] = {}

    prev_tier = None
    for tier, name, report, feedback in results:
        if prev_tier is not None and tier != prev_tier:
            print(_SEPARATOR)
        print(_format_row(name, report))
        prev_tier = tier

        grades_by_tier.setdefault(tier, []).append(report.grade)
        scores_by_tier.setdefault(tier, []).append(report.overall_score)

    print("=" * len(_HEADER_ROW))

    # Per-tier averages
    if len(set(t for t, *_ in results)) > 1:
        print("\nAverage score by tier:")
        for tier in sorted(grades_by_tier):
            tier_scores = scores_by_tier[tier]
            avg = sum(tier_scores) / len(tier_scores)
            print(f"  {tier:<8}: {avg:.1%}  ({', '.join(grades_by_tier[tier])})")

    # Overall totals
    all_grades = [r.grade for _, _, r, _ in results]
    avg_all = sum(r.overall_score for _, _, r, _ in results) / len(results)
    print(f"\nTotal: {len(results)} example(s) | Average: {avg_all:.1%}")
    for g in ["A", "B", "C", "F"]:
        count = all_grades.count(g)
        if count:
            print(f"  Grade {g}: {count}")

    # Detailed findings
    for tier, name, report, feedback in results:
        if verbose or report.grade != "A":
            _print_findings(name, report, feedback)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch evaluate transcript/summary pairs for the call summariser.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--examples-dir",
        default="Sample_data/examples",
        metavar="PATH",
        help="Directory with *-transcript.txt and *-summary.txt pairs (default: Sample_data/examples)",
    )
    mode.add_argument(
        "--summaries-dir",
        metavar="PATH",
        help="Directory of generated *-summary.txt files to evaluate",
    )
    parser.add_argument(
        "--transcripts-dir",
        default="Sample_data/target_data",
        metavar="PATH",
        help="Transcript directory used with --summaries-dir (default: Sample_data/target_data)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed findings for all examples, not just below-A grades",
    )

    args = parser.parse_args()

    if args.summaries_dir:
        summaries_dir = Path(args.summaries_dir)
        transcripts_dir = Path(args.transcripts_dir)
        if not summaries_dir.exists():
            print(f"Error: summaries directory not found: {summaries_dir}", file=sys.stderr)
            sys.exit(1)
        results = _evaluate_generated_summaries(summaries_dir, transcripts_dir)
    else:
        examples_dir = Path(args.examples_dir)
        if not examples_dir.exists():
            print(f"Error: examples directory not found: {examples_dir}", file=sys.stderr)
            sys.exit(1)
        results = _evaluate_paired_directory(examples_dir, args.verbose)

    _print_report(results, args.verbose)


if __name__ == "__main__":
    main()
