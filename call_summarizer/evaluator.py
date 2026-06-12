"""Summary quality evaluation engine for generated insurance call summaries.

Produces a scored, weighted assessment across eight quality dimensions:

1. Factual Groundedness (30%): every verifiable fact in the summary must be
   traceable to the source transcript (amounts, IBANs, emails, reference numbers).
2. Completeness / Recall (20%): key facts present in the transcript must appear
   in the summary — catches the common failure of omitting IBANs, callback
   numbers, settlement confirmations, or claim references.
3. Format Compliance (20%): required sections (Caller, Subject, Executive
   Summary, Next Steps), correct structure, bullets, and character limit.
4. Hallucination (10%): confirmation phrases ("confirmed bank details", "waived
   consideration period", "accepted the offer") must be supported by transcript
   evidence — not asserted when the transcript does not show agreement.
5. Professionalism (5%): tone and language appropriate for external communication.
   Checks for informal markers, internal jargon, placeholder text, and all-caps
   words beyond standard acronyms.
6. Handoff Readiness (5%): another claims agent should be able to continue the
   case without re-listening to the call. Checks actionability of Next Steps,
   descriptiveness of the Subject, and presence of key identifying information.
7. Section Precision (5%): conditional sections (Liability, Negotiation, Vehicle
   Damage, Injury, Property) only appear when the topic was discussed.
8. Redundancy (5%): the same fact should not appear in multiple bullets.

Overall score → grade:
    A  ≥ 0.90  Production-ready
    B  ≥ 0.75  Minor issues — usable with review
    C  ≥ 0.60  Notable gaps — needs correction before filing
    F  < 0.60  Significant errors — block or retry

Public API::

    from call_summarizer.evaluator import evaluate_summary, EvaluationReport

    report = evaluate_summary(summary_text, transcript_text)
    print(report.grade, f"{report.overall_score:.0%}")
    for m in report.metrics:
        print(f"  {m.name}: {m.score:.0%}")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .guardrails import (
    _CONDITIONAL_SECTION_NAMES,
    _CONFIRMATION_PHRASE_CHECKS,
    _check_amounts_in_transcript,
    _check_caller_direction,
    _check_caller_line_present,
    _check_caller_relationship,
    _check_char_limit,
    _check_conditional_section_empty_body,
    _check_conditional_sections_justified,
    _check_duplicate_bullet_content,
    _check_emails_in_transcript,
    _check_executive_summary_bullets,
    _check_ibans_in_transcript,
    _check_missing_executive_summary,
    _check_missing_next_steps,
    _check_missing_subject,
    _check_next_steps_completeness,
    _check_phantom_conditional_sections,
    _check_references_in_transcript,
    _check_subject_multiline,
    _check_unknown_section_headers,
    _check_unverified_confirmations,
    _extract_amounts,
    _extract_emails,
    _extract_ibans,
    _extract_references,
    _extract_next_steps_body,
    _extract_section_body,
)

logger = logging.getLogger(__name__)

# ── Weights — must sum to 1.0 ─────────────────────────────────────────────────
_METRIC_WEIGHTS: dict[str, float] = {
    "Factual Groundedness": 0.30,
    "Completeness":         0.20,
    "Format Compliance":    0.20,
    "Hallucination":        0.10,
    "Professionalism":      0.05,
    "Handoff Readiness":    0.05,
    "Section Precision":    0.05,
    "Redundancy":           0.05,
}

# ── Grade thresholds ──────────────────────────────────────────────────────────
_GRADE_THRESHOLDS: list[tuple[float, str]] = [
    (0.90, "A"),  # Production-ready
    (0.75, "B"),  # Minor issues, usable with review
    (0.60, "C"),  # Notable gaps, needs correction
    (0.00, "F"),  # Significant errors
]

# ── Professionalism word lists ────────────────────────────────────────────────
_INFORMAL_MARKERS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bgonna\b", r"\bwanna\b", r"\bkinda\b", r"\bsorta\b",
        r"\byeah\b",  r"\bnope\b",  r"\byep\b",   r"\bok\b",
        r"\bgotta\b", r"\blemme\b", r"\bcuz\b",
    ]
]

_JARGON_MARKERS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bTBD\b", r"\bFYI\b", r"\bASAP\b",
        r"\bTODO\b", r"\bFIXME\b",
    ]
]

_PLACEHOLDER_RE = re.compile(
    r"\[to\s+be\s+(?:filled|confirmed|updated)\]"
    r"|\[insert\b"
    r"|\[needs?\s+",
    re.IGNORECASE,
)

# Words that are legitimate all-caps in insurance context (excluded from penalty)
_ALLOWED_ACRONYMS: set[str] = {
    "IBAN", "PDF", "VAT", "UK", "IE", "NI", "DOB", "TPR",
    "GDPR", "UTC", "IP", "PII",
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MetricScore:
    """Score for one quality dimension.

    Attributes:
        name: Human-readable dimension name (matches a key in :data:`_METRIC_WEIGHTS`).
        score: 0.0 (worst) to 1.0 (perfect) for this dimension.
        weight: Fraction of the overall score this dimension contributes.
        findings: Explanatory notes for any reduction from 1.0.
    """

    name: str
    score: float
    weight: float
    findings: list[str] = field(default_factory=list)

    @property
    def weighted_score(self) -> float:
        """Contribution of this metric to the overall weighted score."""
        return self.score * self.weight


@dataclass
class EvaluationReport:
    """Aggregated quality report from :func:`evaluate_summary`.

    Attributes:
        metrics: Ordered list of :class:`MetricScore` instances.
        overall_score: Weighted sum across all metrics (0.0–1.0).
        grade: ``"A"``, ``"B"``, ``"C"``, or ``"F"``.
        char_count: Length of the evaluated summary.
    """

    metrics: list[MetricScore]
    overall_score: float
    grade: str
    char_count: int

    def metric(self, name: str) -> MetricScore | None:
        """Return the named metric, or None if not present.

        Args:
            name: Exact metric name (e.g. ``"Factual Groundedness"``).
        """
        return next((m for m in self.metrics if m.name == name), None)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_grade(score: float) -> str:
    """Map a 0–1 score to a letter grade A/B/C/F."""
    for threshold, grade in _GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"


def _no_transcript_metric(name: str) -> MetricScore:
    return MetricScore(
        name=name,
        score=1.0,
        weight=_METRIC_WEIGHTS[name],
        findings=["No transcript provided — metric not evaluated"],
    )


# ── Metric 1: Factual Groundedness ────────────────────────────────────────────

def _score_groundedness(summary: str, transcript: str) -> MetricScore:
    """Factual groundedness: every verifiable fact in summary → exists in transcript.

    Extracts four entity types from the summary (amounts, IBANs, emails,
    reference numbers) and checks each against the transcript using the same
    Tier-3 guardrail checks.  A fact that cannot be matched reduces the score.

    A summary with no verifiable entities scores 1.0 (no hallucination
    *detected*) — but a low Completeness score may signal a sparse summary.

    Args:
        summary: Generated summary text.
        transcript: Original transcript text.

    Returns:
        :class:`MetricScore` with weight 0.30.
    """
    fact_groups = [
        ("amount",    _extract_amounts(summary),    _check_amounts_in_transcript(summary, transcript)),
        ("IBAN",      _extract_ibans(summary),      _check_ibans_in_transcript(summary, transcript)),
        ("email",     _extract_emails(summary),     _check_emails_in_transcript(summary, transcript)),
        ("reference", _extract_references(summary), _check_references_in_transcript(summary, transcript)),
    ]

    total = sum(len(facts) for _, facts, _ in fact_groups)
    ungrounded: list[str] = []
    for label, _facts, bad in fact_groups:
        for f in bad:
            ungrounded.append(f"Ungrounded {label}: '{f.detail}' — {f.message}")

    if total == 0:
        return MetricScore(
            name="Factual Groundedness",
            score=1.0,
            weight=_METRIC_WEIGHTS["Factual Groundedness"],
            findings=["No verifiable facts (amounts/IBANs/emails/refs) found in summary"],
        )

    score = max(0.0, (total - len(ungrounded)) / total)
    logger.debug("[%s] score=%.2f findings=%d", "Factual Groundedness", score, len(ungrounded))
    return MetricScore(
        name="Factual Groundedness",
        score=score,
        weight=_METRIC_WEIGHTS["Factual Groundedness"],
        findings=ungrounded,
    )


# ── Metric 2: Completeness ────────────────────────────────────────────────────

def _score_completeness(summary: str, transcript: str) -> MetricScore:
    """Completeness (recall): key facts in transcript → captured in summary.

    This is the complement of groundedness and is the primary detector of
    the common failure mode of omitting critical details: missing IBANs
    (okay-5), missing callback phone, missing reference numbers (bad-3/4),
    missing settlement/payment confirmations (bad-5, okay-3).

    Args:
        summary: Generated summary text.
        transcript: Original transcript text.

    Returns:
        :class:`MetricScore` with weight 0.20.
    """
    groups = [
        ("amount",    _extract_amounts(transcript),    _extract_amounts(summary)),
        ("IBAN",      _extract_ibans(transcript),      _extract_ibans(summary)),
        ("email",     _extract_emails(transcript),     _extract_emails(summary)),
        ("reference", _extract_references(transcript), _extract_references(summary)),
    ]

    total = sum(len(t) for _, t, _ in groups)
    missed: list[str] = []
    captured = 0

    for label, t_facts, s_facts in groups:
        for fact in t_facts:
            if fact in s_facts:
                captured += 1
            else:
                missed.append(f"Missed {label} from transcript: '{fact}'")

    if total == 0:
        return MetricScore(
            name="Completeness",
            score=1.0,
            weight=_METRIC_WEIGHTS["Completeness"],
            findings=["No verifiable facts found in transcript to check recall against"],
        )

    score = captured / total
    logger.debug("[%s] score=%.2f findings=%d", "Completeness", score, len(missed))
    return MetricScore(
        name="Completeness",
        score=score,
        weight=_METRIC_WEIGHTS["Completeness"],
        findings=missed,
    )


# ── Metric 3: Format Compliance ───────────────────────────────────────────────

def _score_format_compliance(summary: str) -> MetricScore:
    """Format compliance: required sections, structure, and character limit.

    Runs every Tier-1 structural and Tier-2 format guardrail check and
    converts the findings into a fractional score: failed_checks / total_checks.

    Args:
        summary: Generated summary text.

    Returns:
        :class:`MetricScore` with weight 0.20.
    """
    checks: list[tuple[str, bool]] = [
        ("Caller line present",              _check_caller_line_present(summary) is None),
        ("Call direction (inbound/outbound)", _check_caller_direction(summary) is None),
        ("Caller relationship recognized",    _check_caller_relationship(summary) is None),
        ("Subject present",                  _check_missing_subject(summary) is None),
        ("Subject is single line",           _check_subject_multiline(summary) is None),
        ("Executive Summary present",        _check_missing_executive_summary(summary) is None),
        ("Executive Summary has bullets",    _check_executive_summary_bullets(summary) is None),
        ("Next Steps present",               _check_missing_next_steps(summary) is None),
        ("Next Steps complete",              len(_check_next_steps_completeness(summary)) == 0),
        ("No unknown section headers",       len(_check_unknown_section_headers(summary)) == 0),
        ("No phantom conditional sections",  len(_check_phantom_conditional_sections(summary)) == 0),
        ("No empty conditional sections",    len(_check_conditional_section_empty_body(summary)) == 0),
        ("Within 1,500 char limit",          _check_char_limit(summary) is None),
    ]

    failed = [name for name, ok in checks if not ok]
    score = (len(checks) - len(failed)) / len(checks)
    logger.debug("[%s] score=%.2f findings=%d", "Format Compliance", score, len(failed))
    return MetricScore(
        name="Format Compliance",
        score=score,
        weight=_METRIC_WEIGHTS["Format Compliance"],
        findings=[f"FAIL: {name}" for name in failed],
    )


# ── Metric 4: Hallucination ───────────────────────────────────────────────────

def _score_hallucination(summary: str, transcript: str) -> MetricScore:
    """Hallucination: confirmation phrases must be supported by transcript evidence.

    Checks four high-risk confirmation patterns ("confirmed bank details",
    "waived the 10-day consideration period", "accepted the offer",
    "confirmed the settlement") against the transcript.  A phrase that appears
    in the summary but lacks supporting evidence in the transcript is a
    potential hallucination — the most damaging failure type in claims handling.

    Args:
        summary: Generated summary text.
        transcript: Original transcript text.

    Returns:
        :class:`MetricScore` with weight 0.10.
    """
    summary_lower = summary.lower()
    total = sum(
        1 for pattern, _ in _CONFIRMATION_PHRASE_CHECKS
        if re.search(pattern, summary_lower)
    )

    if total == 0:
        return MetricScore(
            name="Hallucination",
            score=1.0,
            weight=_METRIC_WEIGHTS["Hallucination"],
            findings=["No high-risk confirmation phrases detected in summary"],
        )

    bad = _check_unverified_confirmations(summary, transcript)
    unverified = len(bad)
    score = max(0.0, (total - unverified) / total)
    logger.debug("[%s] score=%.2f findings=%d", "Hallucination", score, len(bad))
    return MetricScore(
        name="Hallucination",
        score=score,
        weight=_METRIC_WEIGHTS["Hallucination"],
        findings=[f.message for f in bad],
    )


# ── Metric 5: Professionalism ─────────────────────────────────────────────────

def _score_professionalism(summary: str) -> MetricScore:
    """Professionalism: tone and language appropriate for external communication.

    Insurance summaries may be shared directly with customers or auditors.
    This metric uses lightweight deterministic heuristics as a first-pass
    professionalism screen — it does not replace human review but catches
    obvious failures.

    Checks:
    - Informal speech markers (gonna, wanna, yeah, nope, ok, kinda, …)
    - Internal jargon not suitable for external sharing (TBD, FYI, ASAP, TODO)
    - Unfilled placeholder text ([to be filled], [insert …])
    - All-caps words beyond standard insurance acronyms (IBAN, PDF, VAT, …)

    Each issue type found applies a 0.25 penalty; score is floored at 0.0.

    Args:
        summary: Generated summary text.

    Returns:
        :class:`MetricScore` with weight 0.05.
    """
    issues: list[str] = []

    # Informal markers
    informal_hits = [p.pattern for p in _INFORMAL_MARKERS if p.search(summary)]
    if informal_hits:
        issues.append(f"Informal language detected: {', '.join(informal_hits[:3])}")

    # Internal jargon
    jargon_hits = [p.pattern for p in _JARGON_MARKERS if p.search(summary)]
    if jargon_hits:
        issues.append(f"Internal jargon detected: {', '.join(jargon_hits)}")

    # Placeholder text
    if _PLACEHOLDER_RE.search(summary):
        issues.append("Unfilled placeholder text detected in summary")

    # Rogue all-caps words (≥ 4 letters, not a known acronym)
    caps_words = re.findall(r"\b[A-Z]{4,}\b", summary)
    rogue_caps = [w for w in caps_words if w not in _ALLOWED_ACRONYMS]
    if rogue_caps:
        issues.append(f"Unexpected all-caps word(s): {', '.join(set(rogue_caps[:3]))}")

    score = max(0.0, 1.0 - 0.25 * len(issues))
    logger.debug("[%s] score=%.2f findings=%d", "Professionalism", score, len(issues))
    return MetricScore(
        name="Professionalism",
        score=score,
        weight=_METRIC_WEIGHTS["Professionalism"],
        findings=issues,
    )


# ── Metric 6: Handoff Readiness ───────────────────────────────────────────────

def _score_handoff_readiness(summary: str, transcript: str) -> MetricScore:
    """Handoff readiness: another agent can continue without re-listening.

    Evaluates whether the summary provides enough context for a seamless case
    handoff.  Unlike Format Compliance (which checks structural presence),
    this metric checks the *content quality* of those sections.

    Checks (each worth 1 point):
    1. Next Steps contains at least one concrete action (not just "None" for both).
    2. Subject line is descriptive — at least four words.
    3. Executive Summary narrative paragraph is substantive (≥ 50 characters).
    4. At least one verifiable identifier (amount, IBAN, email, or reference)
       is present in the summary — a completely identifier-free summary cannot
       be linked to a specific case record.
    5. Next Steps specifies a real company name (not a generic placeholder like
       "[COMPANY]" or the literal word "Company").

    Args:
        summary: Generated summary text.
        transcript: Original transcript text (used to check identifier presence).

    Returns:
        :class:`MetricScore` with weight 0.05.
    """
    checks: list[tuple[str, bool]] = []

    # 1 — Next Steps has a non-None action
    ns_body = _extract_next_steps_body(summary)
    none_count = len(re.findall(r":\s*None\b", ns_body, re.IGNORECASE)) if ns_body else 0
    total_action_lines = len([l for l in ns_body.split("\n") if ":" in l]) if ns_body else 0
    checks.append((
        "Next Steps has at least one concrete action",
        total_action_lines > 0 and none_count < total_action_lines,
    ))

    # 2 — Subject is descriptive (≥ 4 words)
    subject_body = _extract_section_body(summary, "Subject") or ""
    subject_words = len(subject_body.split())
    checks.append((
        "Subject is descriptive (>= 4 words)",
        subject_words >= 4,
    ))

    # 3 — Executive Summary has substantive paragraph (≥ 50 chars)
    exec_body = _extract_section_body(summary, "Executive Summary") or ""
    # Look at the narrative part (lines without bullet markers)
    narrative_lines = [
        ln for ln in exec_body.split("\n")
        if ln.strip() and not re.match(r"^\s*[-•*]", ln)
    ]
    narrative_text = " ".join(narrative_lines)
    checks.append((
        "Executive Summary has substantive narrative (>= 50 chars)",
        len(narrative_text) >= 50,
    ))

    # 4 — At least one verifiable identifier in summary
    has_identifier = bool(
        _extract_amounts(summary)
        or _extract_ibans(summary)
        or _extract_emails(summary)
        or _extract_references(summary)
    )
    checks.append((
        "Summary contains at least one verifiable identifier",
        has_identifier,
    ))

    # 5 — Next Steps names a real company (not a placeholder)
    placeholder_company = re.search(
        r"\[COMPANY\]|\bCompany\b(?!\s+name|\s+representative)",
        ns_body,
        re.IGNORECASE,
    )
    checks.append((
        "Next Steps uses a specific company name (not a placeholder)",
        not placeholder_company if ns_body else False,
    ))

    passed = sum(1 for _, ok in checks if ok)
    score = passed / len(checks)
    issues = [name for name, ok in checks if not ok]
    logger.debug("[%s] score=%.2f findings=%d", "Handoff Readiness", score, len(issues))
    return MetricScore(
        name="Handoff Readiness",
        score=score,
        weight=_METRIC_WEIGHTS["Handoff Readiness"],
        findings=[f"FAIL: {name}" for name, ok in checks if not ok],
    )


# ── Metric 7: Section Precision ───────────────────────────────────────────────

def _score_section_precision(summary: str, transcript: str) -> MetricScore:
    """Section precision: conditional sections only when topic was discussed.

    Checks that none of the five conditional sections (Liability, Negotiation,
    Vehicle Damage, Injury, Property) appear in the summary without
    corresponding evidence in the transcript.

    Args:
        summary: Generated summary text.
        transcript: Original transcript text.

    Returns:
        :class:`MetricScore` with weight 0.05.
    """
    present = [
        s for s in _CONDITIONAL_SECTION_NAMES
        if re.search(rf"^{re.escape(s)}:", summary, re.MULTILINE)
    ]

    if not present:
        return MetricScore(
            name="Section Precision",
            score=1.0,
            weight=_METRIC_WEIGHTS["Section Precision"],
            findings=["No conditional sections included"],
        )

    phantom = {f.detail for f in _check_phantom_conditional_sections(summary)}
    unjustified = (
        {f.detail for f in _check_conditional_sections_justified(summary, transcript)}
        if transcript else set()
    )
    bad = (phantom | unjustified) & set(present)

    score = (len(present) - len(bad)) / len(present)
    logger.debug("[%s] score=%.2f findings=%d", "Section Precision", score, len(bad))
    return MetricScore(
        name="Section Precision",
        score=score,
        weight=_METRIC_WEIGHTS["Section Precision"],
        findings=[f"Unjustified conditional section: '{s}'" for s in bad],
    )


# ── Metric 8: Redundancy ──────────────────────────────────────────────────────

def _score_redundancy(summary: str) -> MetricScore:
    """Redundancy: the same fact should not appear in more than one bullet.

    Detects when the same numeric value (amount, day count, reference fragment)
    appears in multiple Executive Summary bullets.  Each duplicated distinct
    value applies a 0.25 penalty.

    Args:
        summary: Generated summary text.

    Returns:
        :class:`MetricScore` with weight 0.05.
    """
    dup_findings = _check_duplicate_bullet_content(summary)
    if not dup_findings:
        return MetricScore(
            name="Redundancy",
            score=1.0,
            weight=_METRIC_WEIGHTS["Redundancy"],
            findings=["No duplicate facts detected in bullet points"],
        )

    dup_values: set[str] = set()
    for f in dup_findings:
        dup_values.update(f.detail.split(", "))

    score = max(0.0, 1.0 - 0.25 * len(dup_values))
    logger.debug("[%s] score=%.2f findings=%d", "Redundancy", score, len(dup_values))
    return MetricScore(
        name="Redundancy",
        score=score,
        weight=_METRIC_WEIGHTS["Redundancy"],
        findings=[f.message for f in dup_findings],
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def build_eval_feedback_prompt(report: EvaluationReport) -> str:
    """Build a targeted corrective prompt from an :class:`EvaluationReport`.

    Used by the agentic eval-feedback loop in ``service.py`` to give the LLM
    specific, actionable instructions for a regeneration attempt.  Only metrics
    that actually failed (score < 1.0 **and** have findings) are included;
    they are sorted by weight descending so the highest-impact corrections
    appear first.

    The output is appended to the system prompt as an addendum — it should be
    concise enough to stay within the token budget while being specific enough
    for the LLM to act on each point.

    Args:
        report: The :class:`EvaluationReport` from the previous generation attempt.

    Returns:
        A multi-line string to append to the system prompt, or an empty string
        if every metric scored 1.0 (nothing to correct).
    """
    failing = [
        m for m in report.metrics
        if m.score < 1.0 and m.findings
        # Skip the "no transcript provided" placeholder finding
        and not (len(m.findings) == 1 and "not evaluated" in m.findings[0])
    ]
    if not failing:
        return ""

    failing.sort(key=lambda m: m.weight, reverse=True)

    lines: list[str] = [
        f"QUALITY EVALUATION FEEDBACK — previous attempt scored "
        f"{report.overall_score:.0%} (Grade {report.grade}).",
        "Address each issue below before regenerating:\n",
    ]

    for m in failing:
        lines.append(f"{m.name} ({m.score:.0%}, weight {m.weight:.0%}):")
        for finding in m.findings[:3]:   # cap at 3 per metric to save tokens
            lines.append(f"  - {finding}")
        lines.append("")

    logger.debug(
        "Built eval feedback prompt — %d failing metrics, %d lines",
        len(failing),
        len(lines),
    )
    return "\n".join(lines)


def evaluate_summary(
    summary: str,
    transcript: str = "",
) -> EvaluationReport:
    """Evaluate a generated summary across eight quality dimensions.

    Transcript-dependent metrics (Groundedness, Completeness, Hallucination,
    Handoff Readiness, Section Precision) default to 1.0 when ``transcript``
    is not provided, since absence of evidence is not evidence of failure.

    Args:
        summary: The generated (or reviewed) summary text.
        transcript: The original call transcript.  Pass an empty string to
            evaluate format-only metrics (Compliance, Professionalism,
            Redundancy) without the source text.

    Returns:
        :class:`EvaluationReport` with per-metric scores, an overall weighted
        score (0.0–1.0), and a letter grade (A / B / C / F).
    """
    logger.debug(
        "Evaluating summary — %d chars, transcript: %s",
        len(summary),
        f"{len(transcript)} chars" if transcript else "not provided",
    )

    if transcript:
        metrics = [
            _score_groundedness(summary, transcript),
            _score_completeness(summary, transcript),
            _score_format_compliance(summary),
            _score_hallucination(summary, transcript),
            _score_professionalism(summary),
            _score_handoff_readiness(summary, transcript),
            _score_section_precision(summary, transcript),
            _score_redundancy(summary),
        ]
    else:
        metrics = [
            _no_transcript_metric("Factual Groundedness"),
            _no_transcript_metric("Completeness"),
            _score_format_compliance(summary),
            _no_transcript_metric("Hallucination"),
            _score_professionalism(summary),
            _no_transcript_metric("Handoff Readiness"),
            _no_transcript_metric("Section Precision"),
            _score_redundancy(summary),
        ]

    overall = round(sum(m.weighted_score for m in metrics), 4)
    grade = _compute_grade(overall)

    logger.info(
        "Evaluation — grade: %s, score: %.2f | %s",
        grade,
        overall,
        " | ".join(f"{m.name[:8]}={m.score:.0%}" for m in metrics),
    )

    return EvaluationReport(
        metrics=metrics,
        overall_score=overall,
        grade=grade,
        char_count=len(summary),
    )
