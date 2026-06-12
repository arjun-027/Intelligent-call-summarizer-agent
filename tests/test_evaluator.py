"""Unit tests for call_summarizer.evaluator.

Coverage
--------
- EvaluationReport / MetricScore dataclasses
- _compute_grade: boundary values
- _score_groundedness: perfect, partial, zero
- _score_completeness: perfect recall, partial, empty transcript
- _score_format_compliance: all-pass, individual failures
- _score_hallucination: no phrases, verified, unverified
- _score_professionalism: clean, informal, jargon, all-caps, placeholder
- _score_handoff_readiness: all checks pass / individual failures
- _score_section_precision: no sections, justified, unjustified
- _score_redundancy: no duplicates, duplicate found
- evaluate_summary: integration, no-transcript mode
"""

import pytest

from call_summarizer.evaluator import (
    EvaluationReport,
    MetricScore,
    _METRIC_WEIGHTS,
    _compute_grade,
    _score_completeness,
    _score_format_compliance,
    _score_groundedness,
    _score_hallucination,
    _score_handoff_readiness,
    _score_professionalism,
    _score_redundancy,
    _score_section_precision,
    evaluate_summary,
)

# ── Shared fixtures ────────────────────────────────────────────────────────────

# A structurally complete, content-rich summary used as the baseline
_GOOD_SUMMARY = (
    "Caller: Jane Doe, policyholder, inbound\n"
    "\n"
    "Subject:\n"
    "Vehicle damage claim following road traffic accident on the M50\n"
    "\n"
    "Executive Summary:\n"
    "The policyholder called to report a road traffic accident on the M50 motorway. "
    "The other party accepted liability and an engineer's report has been arranged.\n"
    "- Claim reference CLM-2024-00123\n"
    "- Settlement offer of €3,500 discussed\n"
    "- Payment to IBAN IE29AIBK93115212345678\n"
    "\n"
    "Next Steps:\n"
    "Pemberton Insurance: Process settlement of €3,500 to IBAN IE29AIBK93115212345678.\n"
    "Other: Policyholder to provide written acceptance of liability waiver."
)

_MATCHING_TRANSCRIPT = (
    "Agent: Good morning, claims team speaking.\n"
    "Caller: Hi, I'm Jane Doe. I had an accident on the M50 motorway.\n"
    "Agent: Can I get your claim reference?\n"
    "Caller: It's CLM-2024-00123.\n"
    "Agent: I can see a settlement offer of €3,500.\n"
    "Caller: Yes that's fine, I accept.\n"
    "Agent: And your IBAN for the payment?\n"
    "Caller: IE29AIBK93115212345678.\n"
    "Agent: Confirmed, we will process that today."
)


# ── _compute_grade ─────────────────────────────────────────────────────────────

class TestComputeGrade:
    def test_a_at_0_90(self):
        assert _compute_grade(0.90) == "A"

    def test_a_at_1_0(self):
        assert _compute_grade(1.0) == "A"

    def test_b_at_0_75(self):
        assert _compute_grade(0.75) == "B"

    def test_b_at_0_89(self):
        assert _compute_grade(0.89) == "B"

    def test_c_at_0_60(self):
        assert _compute_grade(0.60) == "C"

    def test_c_at_0_74(self):
        assert _compute_grade(0.74) == "C"

    def test_f_at_0_59(self):
        assert _compute_grade(0.59) == "F"

    def test_f_at_0_0(self):
        assert _compute_grade(0.0) == "F"


# ── MetricScore / EvaluationReport ────────────────────────────────────────────

class TestDataclasses:
    def test_weighted_score(self):
        m = MetricScore("Test", score=0.8, weight=0.5)
        assert m.weighted_score == pytest.approx(0.4)

    def test_metric_lookup(self):
        m = MetricScore("Factual Groundedness", score=0.9, weight=0.30)
        report = EvaluationReport(metrics=[m], overall_score=0.9, grade="A", char_count=100)
        assert report.metric("Factual Groundedness") is m

    def test_metric_lookup_missing(self):
        report = EvaluationReport(metrics=[], overall_score=1.0, grade="A", char_count=0)
        assert report.metric("Nonexistent") is None

    def test_weights_sum_to_one(self):
        total = sum(_METRIC_WEIGHTS.values())
        assert total == pytest.approx(1.0)


# ── _score_groundedness ────────────────────────────────────────────────────────

class TestScoreGroundedness:
    def test_perfect_when_all_facts_verified(self):
        result = _score_groundedness(_GOOD_SUMMARY, _MATCHING_TRANSCRIPT)
        assert result.score == pytest.approx(1.0)

    def test_no_facts_in_summary_gives_1_0(self):
        summary = (
            "Caller: John, policyholder, inbound\n\n"
            "Subject:\nGeneral inquiry\n\n"
            "Executive Summary:\nThe caller asked about their policy.\n- No specific amounts discussed\n\n"
            "Next Steps:\nPemberton Insurance: Follow up.\nOther: None"
        )
        transcript = "Agent: Hello. Caller: I have a question. Agent: Sure."
        result = _score_groundedness(summary, transcript)
        assert result.score == pytest.approx(1.0)

    def test_ungrounded_amount_reduces_score(self):
        summary = (
            "Caller: Jane, policyholder, inbound\n\n"
            "Subject:\nSettlement discussion\n\n"
            "Executive Summary:\nSettlement of €99,999 was discussed.\n- Amount €99,999 agreed\n\n"
            "Next Steps:\nPemberton Insurance: Process payment.\nOther: None"
        )
        transcript = "Agent: We have an offer. Caller: What is it? Agent: We can offer €500."
        result = _score_groundedness(summary, transcript)
        assert result.score < 1.0

    def test_findings_list_populated_for_failures(self):
        summary = (
            "Caller: Jane, policyholder, inbound\n\n"
            "Subject:\nQuery\n\n"
            "Executive Summary:\nClaim ref CLM-9999-00000 mentioned.\n- Reference CLM-9999-00000\n\n"
            "Next Steps:\nPemberton Insurance: Follow up.\nOther: None"
        )
        transcript = "Agent: Hello. Caller: CLM-1234-56789 is my reference."
        result = _score_groundedness(summary, transcript)
        assert len(result.findings) > 0

    def test_weight_is_correct(self):
        result = _score_groundedness(_GOOD_SUMMARY, _MATCHING_TRANSCRIPT)
        assert result.weight == pytest.approx(_METRIC_WEIGHTS["Factual Groundedness"])


# ── _score_completeness ────────────────────────────────────────────────────────

class TestScoreCompleteness:
    def test_perfect_when_all_transcript_facts_captured(self):
        result = _score_completeness(_GOOD_SUMMARY, _MATCHING_TRANSCRIPT)
        assert result.score == pytest.approx(1.0)

    def test_missed_iban_reduces_score(self):
        summary_no_iban = (
            "Caller: Jane, policyholder, inbound\n\n"
            "Subject:\nSettlement payment query\n\n"
            "Executive Summary:\nPayment of €3,500 to be processed.\n- Settlement €3,500 agreed\n\n"
            "Next Steps:\nPemberton Insurance: Process payment.\nOther: None"
        )
        result = _score_completeness(summary_no_iban, _MATCHING_TRANSCRIPT)
        assert result.score < 1.0
        assert any("IBAN" in f for f in result.findings)

    def test_no_facts_in_transcript_gives_1_0(self):
        transcript = "Agent: Hello. Caller: I have a question. Agent: Sure."
        summary = (
            "Caller: John, policyholder, inbound\n\n"
            "Subject:\nGeneral inquiry\n\n"
            "Executive Summary:\nThe caller had a general question.\n- Policy status discussed\n\n"
            "Next Steps:\nPemberton Insurance: Follow up.\nOther: None"
        )
        result = _score_completeness(summary, transcript)
        assert result.score == pytest.approx(1.0)

    def test_missed_reference_in_findings(self):
        summary_no_ref = (
            "Caller: Jane, policyholder, inbound\n\n"
            "Subject:\nAccount query\n\n"
            "Executive Summary:\nCaller had a question about the claim.\n- Awaiting update\n\n"
            "Next Steps:\nPemberton Insurance: Follow up.\nOther: None"
        )
        result = _score_completeness(summary_no_ref, _MATCHING_TRANSCRIPT)
        assert any("reference" in f.lower() for f in result.findings)


# ── _score_format_compliance ───────────────────────────────────────────────────

class TestScoreFormatCompliance:
    def test_perfect_on_good_summary(self):
        result = _score_format_compliance(_GOOD_SUMMARY)
        assert result.score == pytest.approx(1.0)
        assert result.findings == []

    def test_missing_caller_line_penalised(self):
        summary_no_caller = _GOOD_SUMMARY.replace("Caller: Jane Doe, policyholder, inbound\n\n", "")
        result = _score_format_compliance(summary_no_caller)
        assert result.score < 1.0
        assert any("Caller" in f for f in result.findings)

    def test_missing_bullets_penalised(self):
        summary_no_bullets = (
            "Caller: Jane, policyholder, inbound\n\n"
            "Subject:\nVehicle damage claim\n\n"
            "Executive Summary:\n"
            "The policyholder reported a vehicle accident. No specific action taken yet.\n\n"
            "Next Steps:\nPemberton Insurance: Follow up.\nOther: None"
        )
        result = _score_format_compliance(summary_no_bullets)
        assert result.score < 1.0
        assert any("bullet" in f.lower() for f in result.findings)

    def test_char_limit_exceeded_penalised(self):
        long_summary = _GOOD_SUMMARY + " Additional content " * 80
        result = _score_format_compliance(long_summary)
        assert result.score < 1.0
        assert any("1,500" in f for f in result.findings)

    def test_score_bounded_0_to_1(self):
        result = _score_format_compliance("")
        assert 0.0 <= result.score <= 1.0


# ── _score_hallucination ───────────────────────────────────────────────────────

class TestScoreHallucination:
    def test_no_confirmation_phrases_gives_1_0(self):
        transcript = "Agent: Hello. Caller: Yes please."
        result = _score_hallucination(_GOOD_SUMMARY, transcript)
        # _GOOD_SUMMARY has no "confirmed bank details / waived / accepted offer"
        assert result.score == pytest.approx(1.0)

    def test_unverified_confirmation_reduces_score(self):
        summary_with_hallucination = (
            "Caller: Jane, policyholder, inbound\n\n"
            "Subject:\nSettlement query\n\n"
            "Executive Summary:\nCaller confirmed bank details for payment.\n"
            "- Caller confirmed bank details\n\n"
            "Next Steps:\nPemberton Insurance: Process.\nOther: None"
        )
        transcript = "Agent: Hello. Caller: I want to know about my payment."
        result = _score_hallucination(summary_with_hallucination, transcript)
        assert result.score < 1.0
        assert len(result.findings) > 0

    def test_verified_confirmation_does_not_penalise(self):
        summary = (
            "Caller: Jane, policyholder, inbound\n\n"
            "Subject:\nSettlement query\n\n"
            "Executive Summary:\nCaller confirmed bank details for payment.\n"
            "- Payment to confirmed account\n\n"
            "Next Steps:\nPemberton Insurance: Process.\nOther: None"
        )
        transcript = (
            "Agent: Can you confirm your bank details?\n"
            "Caller: Yes, my account is confirmed, here are the details.\n"
            "Agent: Thank you, confirmed and recorded."
        )
        result = _score_hallucination(summary, transcript)
        # With sufficient evidence keywords, should not be flagged
        assert result.score == pytest.approx(1.0)


# ── _score_professionalism ─────────────────────────────────────────────────────

class TestScoreProfessionalism:
    def test_clean_summary_scores_1_0(self):
        result = _score_professionalism(_GOOD_SUMMARY)
        assert result.score == pytest.approx(1.0)
        assert result.findings == []

    def test_informal_word_penalised(self):
        summary = _GOOD_SUMMARY.replace("called to report", "gonna report")
        result = _score_professionalism(summary)
        assert result.score < 1.0
        assert any("Informal" in f for f in result.findings)

    def test_internal_jargon_penalised(self):
        # Replace text that actually exists in _GOOD_SUMMARY
        summary = _GOOD_SUMMARY.replace("written acceptance", "TBD — FYI confirmation")
        result = _score_professionalism(summary)
        assert result.score < 1.0
        assert any("jargon" in f.lower() for f in result.findings)

    def test_placeholder_text_penalised(self):
        summary = _GOOD_SUMMARY.replace("Policyholder to provide", "[to be filled]")
        result = _score_professionalism(summary)
        assert result.score < 1.0
        assert any("placeholder" in f.lower() for f in result.findings)

    def test_known_acronym_not_penalised(self):
        # IBAN is in the allowed-acronym list
        result = _score_professionalism(_GOOD_SUMMARY)
        assert not any("IBAN" in f for f in result.findings)

    def test_rogue_all_caps_penalised(self):
        summary = _GOOD_SUMMARY.replace("Settlement offer", "SETTLEMENT OFFER")
        result = _score_professionalism(summary)
        assert result.score < 1.0
        assert any("all-caps" in f.lower() or "SETTLEMENT" in f or "OFFER" in f for f in result.findings)

    def test_score_floored_at_zero(self):
        # Multiple issues should not produce a negative score
        summary = "GONNA TBD [to be filled] RESULT PENDING ACTION"
        result = _score_professionalism(summary)
        assert result.score >= 0.0


# ── _score_handoff_readiness ───────────────────────────────────────────────────

class TestScoreHandoffReadiness:
    def test_perfect_on_good_summary(self):
        result = _score_handoff_readiness(_GOOD_SUMMARY, _MATCHING_TRANSCRIPT)
        assert result.score == pytest.approx(1.0)

    def test_both_next_steps_none_reduces_score(self):
        summary_none_steps = (
            "Caller: Jane, policyholder, inbound\n\n"
            "Subject:\nGeneral inquiry about policy details and claim status\n\n"
            "Executive Summary:\nThe policyholder called to ask about the status of "
            "their vehicle damage claim reference CLM-2024-00123.\n"
            "- Claim reference CLM-2024-00123 confirmed\n\n"
            "Next Steps:\nPemberton Insurance: None\nOther: None"
        )
        result = _score_handoff_readiness(summary_none_steps, _MATCHING_TRANSCRIPT)
        assert result.score < 1.0
        assert any("concrete action" in f.lower() for f in result.findings)

    def test_vague_subject_reduces_score(self):
        summary_vague = _GOOD_SUMMARY.replace(
            "Vehicle damage claim following road traffic accident on the M50",
            "Call",  # too short — less than 4 words
        )
        result = _score_handoff_readiness(summary_vague, _MATCHING_TRANSCRIPT)
        assert result.score < 1.0
        assert any("Subject" in f for f in result.findings)

    def test_no_identifier_reduces_score(self):
        summary_no_ids = (
            "Caller: Jane, policyholder, inbound\n\n"
            "Subject:\nGeneral vehicle damage policy query and status update\n\n"
            "Executive Summary:\nThe policyholder called to discuss the status of their claim "
            "and ask about next steps for the repair process going forward.\n"
            "- Discussed claim status in general terms\n\n"
            "Next Steps:\nPemberton Insurance: Follow up with customer on claim progress.\nOther: None"
        )
        transcript = "Agent: Hello. Caller: What is the status of my claim? Agent: I will check."
        result = _score_handoff_readiness(summary_no_ids, transcript)
        assert result.score < 1.0
        assert any("identifier" in f.lower() for f in result.findings)

    def test_short_narrative_reduces_score(self):
        summary_short_narrative = (
            "Caller: Jane, policyholder, inbound\n\n"
            "Subject:\nClaim status update and next steps\n\n"
            "Executive Summary:\nQuery.\n"
            "- Claim reference CLM-2024-00123\n\n"
            "Next Steps:\nPemberton Insurance: Follow up.\nOther: None"
        )
        result = _score_handoff_readiness(summary_short_narrative, _MATCHING_TRANSCRIPT)
        assert result.score < 1.0
        assert any("narrative" in f.lower() or ">= 50" in f for f in result.findings)


# ── _score_section_precision ───────────────────────────────────────────────────

class TestScoreSectionPrecision:
    def test_no_conditional_sections_gives_1_0(self):
        result = _score_section_precision(_GOOD_SUMMARY, _MATCHING_TRANSCRIPT)
        assert result.score == pytest.approx(1.0)
        assert any("No conditional sections" in f for f in result.findings)

    def test_justified_vehicle_section_does_not_penalise(self):
        summary_with_vehicle = (
            _GOOD_SUMMARY
            + "\n\nVehicle Damage:\n"
            + "Vehicle Status: repairable\nTowage: None\nCar hire: None"
        )
        transcript_with_vehicle = _MATCHING_TRANSCRIPT + "\nCaller: The car was damaged."
        result = _score_section_precision(summary_with_vehicle, transcript_with_vehicle)
        assert result.score == pytest.approx(1.0)

    def test_unjustified_liability_section_penalised(self):
        summary_with_liability = (
            _GOOD_SUMMARY + "\n\nLiability Summary:\nLiability was not discussed."
        )
        transcript_no_liability = (
            "Agent: Hello. Caller: I want to check payment status. Agent: Sure."
        )
        result = _score_section_precision(summary_with_liability, transcript_no_liability)
        assert result.score < 1.0

    def test_phantom_none_section_penalised(self):
        summary_phantom = _GOOD_SUMMARY + "\n\nInjury: None"
        result = _score_section_precision(summary_phantom, _MATCHING_TRANSCRIPT)
        assert result.score < 1.0


# ── _score_redundancy ──────────────────────────────────────────────────────────

class TestScoreRedundancy:
    def test_no_duplicates_gives_1_0(self):
        result = _score_redundancy(_GOOD_SUMMARY)
        assert result.score == pytest.approx(1.0)

    def test_duplicate_amount_in_bullets_penalised(self):
        summary_dup = (
            "Caller: Jane, policyholder, inbound\n\n"
            "Subject:\nSettlement discussion\n\n"
            "Executive Summary:\nSettlement of 3500 discussed.\n"
            "- Offer of 3500 accepted\n"
            "- Payment of 3500 to be processed\n\n"
            "Next Steps:\nPemberton Insurance: Process €3,500.\nOther: None"
        )
        result = _score_redundancy(summary_dup)
        assert result.score < 1.0
        assert len(result.findings) > 0

    def test_score_floored_at_zero(self):
        # Many duplicates should not produce negative score
        summary_many_dups = (
            "Caller: Jane, policyholder, inbound\n\n"
            "Subject:\nClaim\n\n"
            "Executive Summary:\nSettlement discussed.\n"
            "- 3500 accepted\n"
            "- 3500 offered\n"
            "- 3500 transferred\n"
            "- 3500 confirmed\n\n"
            "Next Steps:\nPemberton Insurance: Process.\nOther: None"
        )
        result = _score_redundancy(summary_many_dups)
        assert result.score >= 0.0


# ── evaluate_summary (integration) ────────────────────────────────────────────

class TestEvaluateSummary:
    def test_good_summary_scores_high(self):
        report = evaluate_summary(_GOOD_SUMMARY, _MATCHING_TRANSCRIPT)
        assert report.overall_score >= 0.85
        assert report.grade in ("A", "B")

    def test_returns_eight_metrics(self):
        report = evaluate_summary(_GOOD_SUMMARY, _MATCHING_TRANSCRIPT)
        assert len(report.metrics) == 8

    def test_weights_sum_to_1_in_report(self):
        report = evaluate_summary(_GOOD_SUMMARY, _MATCHING_TRANSCRIPT)
        total_weight = sum(m.weight for m in report.metrics)
        assert total_weight == pytest.approx(1.0)

    def test_overall_score_equals_weighted_sum(self):
        report = evaluate_summary(_GOOD_SUMMARY, _MATCHING_TRANSCRIPT)
        expected = sum(m.weighted_score for m in report.metrics)
        assert report.overall_score == pytest.approx(expected, abs=1e-4)

    def test_char_count_correct(self):
        report = evaluate_summary(_GOOD_SUMMARY, _MATCHING_TRANSCRIPT)
        assert report.char_count == len(_GOOD_SUMMARY)

    def test_no_transcript_mode_returns_eight_metrics(self):
        report = evaluate_summary(_GOOD_SUMMARY)
        assert len(report.metrics) == 8

    def test_no_transcript_mode_transcript_metrics_are_1_0(self):
        report = evaluate_summary(_GOOD_SUMMARY)
        for name in ["Factual Groundedness", "Completeness", "Hallucination", "Section Precision"]:
            m = report.metric(name)
            assert m is not None
            assert m.score == pytest.approx(1.0)
            assert "not evaluated" in m.findings[0]

    def test_empty_summary_scores_low(self):
        # An empty summary has several "no-op" metrics (Groundedness, Hallucination,
        # Professionalism) that default to 1.0 because there is nothing to check,
        # but Completeness = 0.0 and Handoff Readiness = 0.0 pull the score below B.
        report = evaluate_summary("", _MATCHING_TRANSCRIPT)
        assert report.overall_score < 0.75
        assert report.grade in ("C", "F")

    def test_grade_matches_score(self):
        report = evaluate_summary(_GOOD_SUMMARY, _MATCHING_TRANSCRIPT)
        # Grade should be consistent with the score thresholds
        if report.overall_score >= 0.90:
            assert report.grade == "A"
        elif report.overall_score >= 0.75:
            assert report.grade == "B"

    def test_metric_lookup_by_name(self):
        report = evaluate_summary(_GOOD_SUMMARY, _MATCHING_TRANSCRIPT)
        m = report.metric("Factual Groundedness")
        assert m is not None
        assert m.weight == pytest.approx(0.30)
