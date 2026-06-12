"""Unit tests for the three-tier guardrail engine.

Each test class targets a single check code. Tests mutate a shared base
summary fixture (which passes all tiers cleanly) to trigger exactly the check
under test — making failures easy to diagnose.
"""

import pytest

from call_summarizer.guardrails import (
    _check_amounts_in_transcript,
    _check_char_count_high,
    _check_char_limit,
    _check_conditional_section_empty_body,
    _check_duplicate_bullet_content,
    _check_emails_in_transcript,
    _check_empty,
    _check_executive_summary_bullets,
    _check_ibans_in_transcript,
    _check_missing_executive_summary,
    _check_missing_next_steps,
    _check_missing_subject,
    _check_next_steps_both_none,
    _check_next_steps_completeness,
    _check_phantom_conditional_sections,
    _check_references_in_transcript,
    _check_subject_multiline,
    _check_unverified_confirmations,
    _check_unknown_section_headers,
    _check_vehicle_damage_subfields,
    _extract_amounts,
    _extract_emails,
    _extract_ibans,
    _extract_references,
    _get_caller_line,
    _get_all_section_header_names,
    build_retry_prompt_addendum,
    run_guardrails,
)
from call_summarizer.models import GuardrailResult
from call_summarizer.summarizer import CHAR_LIMIT


# ── Fixture ────────────────────────────────────────────────────────────────

_BASE_SUMMARY = """\
Caller: Jane Brown, policyholder, inbound

Subject:
Call regarding a vehicle damage claim and repair arrangement

Executive Summary:
Jane Brown contacted Pemberton Insurance to report damage following an accident.
- The vehicle sustained damage to the front bumper and bonnet
- An approved garage will provide a repair estimate
- A claim reference was logged for further handling

Next Steps:
Pemberton Insurance: Send repair instructions to the approved garage
Other: None
"""

_BASE_TRANSCRIPT = """\
Agent: Hello, Pemberton Insurance, how can I help?
Caller: Hi, my name is Jane Brown, I'm calling about my vehicle claim. I had an accident last week.
Agent: Of course, Ms Brown. Can you describe the damage?
Caller: The front bumper and bonnet are damaged. The car needs to go to a garage.
Agent: We'll arrange an estimate at an approved garage.
Caller: Thank you.
"""


# ── Helpers ────────────────────────────────────────────────────────────────


def _find_codes(result: GuardrailResult) -> list[str]:
    return [f.code for f in result.findings]


# ── Public API: run_guardrails ─────────────────────────────────────────────


class TestRunGuardrailsCleanSummary:
    def test_base_summary_passes(self):
        result = run_guardrails(_BASE_SUMMARY, _BASE_TRANSCRIPT)
        assert result.passed is True
        assert result.errors == []

    def test_returns_guardrail_result_type(self):
        result = run_guardrails(_BASE_SUMMARY)
        assert isinstance(result, GuardrailResult)

    def test_char_count_is_accurate(self):
        result = run_guardrails(_BASE_SUMMARY)
        assert result.char_count == len(_BASE_SUMMARY)

    def test_char_within_limit_true_for_short_summary(self):
        result = run_guardrails(_BASE_SUMMARY)
        assert result.char_within_limit is True

    def test_tier3_skipped_when_no_transcript(self):
        # Inject an amount that isn't in any transcript; without a transcript
        # no AMOUNT_NOT_IN_TRANSCRIPT warning should fire.
        summary = _BASE_SUMMARY + "\n- Settlement of €9,999 agreed."
        result = run_guardrails(summary)
        assert "AMOUNT_NOT_IN_TRANSCRIPT" not in _find_codes(result)

    def test_tier3_runs_when_transcript_provided(self):
        summary = _BASE_SUMMARY + "\n- Settlement of €9,999 agreed."
        transcript = "caller asked about their claim"  # no €9,999
        result = run_guardrails(summary, transcript)
        assert "AMOUNT_NOT_IN_TRANSCRIPT" in _find_codes(result)


# ── Tier 1: Structural checks ──────────────────────────────────────────────


class TestEmptySummary:
    def test_empty_string_raises_error(self):
        finding = _check_empty("")
        assert finding is not None
        assert finding.code == "EMPTY_SUMMARY"
        assert finding.tier == "error"

    def test_whitespace_only_raises_error(self):
        finding = _check_empty("   \n\t  ")
        assert finding is not None
        assert finding.code == "EMPTY_SUMMARY"

    def test_non_empty_returns_none(self):
        assert _check_empty("Some content") is None

    def test_run_guardrails_catches_empty(self):
        result = run_guardrails("")
        assert not result.passed
        assert "EMPTY_SUMMARY" in _find_codes(result)


class TestCharLimitExceeded:
    def test_over_limit_raises_error(self):
        finding = _check_char_limit("A" * (CHAR_LIMIT + 1))
        assert finding is not None
        assert finding.code == "CHAR_LIMIT_EXCEEDED"
        assert finding.tier == "error"

    def test_exactly_at_limit_passes(self):
        assert _check_char_limit("A" * CHAR_LIMIT) is None

    def test_under_limit_passes(self):
        assert _check_char_limit("A" * 100) is None

    def test_run_guardrails_catches_over_limit(self):
        long_summary = _BASE_SUMMARY + ("x" * (CHAR_LIMIT + 1))
        result = run_guardrails(long_summary)
        assert not result.passed
        assert "CHAR_LIMIT_EXCEEDED" in _find_codes(result)


class TestMissingSubject:
    def test_no_subject_raises_error(self):
        summary = _BASE_SUMMARY.replace("Subject:", "Topic:")
        finding = _check_missing_subject(summary)
        assert finding is not None
        assert finding.code == "MISSING_SUBJECT"
        assert finding.tier == "error"

    def test_subject_present_returns_none(self):
        assert _check_missing_subject(_BASE_SUMMARY) is None

    def test_run_guardrails_catches_missing_subject(self):
        summary = _BASE_SUMMARY.replace("Subject:", "Topic:")
        result = run_guardrails(summary)
        assert not result.passed
        assert "MISSING_SUBJECT" in _find_codes(result)


class TestMissingExecutiveSummary:
    def test_no_executive_summary_raises_error(self):
        summary = _BASE_SUMMARY.replace("Executive Summary:", "Overview:")
        finding = _check_missing_executive_summary(summary)
        assert finding is not None
        assert finding.code == "MISSING_EXECUTIVE_SUMMARY"
        assert finding.tier == "error"

    def test_executive_summary_present_returns_none(self):
        assert _check_missing_executive_summary(_BASE_SUMMARY) is None


class TestMissingNextSteps:
    def test_no_next_steps_raises_error(self):
        summary = _BASE_SUMMARY.replace("Next Steps:", "Actions:")
        finding = _check_missing_next_steps(summary)
        assert finding is not None
        assert finding.code == "MISSING_NEXT_STEPS"
        assert finding.tier == "error"

    def test_next_steps_present_returns_none(self):
        assert _check_missing_next_steps(_BASE_SUMMARY) is None

    def test_next_steps_without_colon_still_passes(self):
        summary = _BASE_SUMMARY.replace("Next Steps:", "Next Steps")
        assert _check_missing_next_steps(summary) is None


class TestNextStepsCompleteness:
    def test_missing_company_line_raises_error(self):
        # Replace the Pemberton line with an Other: line only
        summary = _BASE_SUMMARY.replace(
            "Pemberton Insurance: Send repair instructions to the approved garage\nOther: None",
            "Other: None",
        )
        findings = _check_next_steps_completeness(summary)
        codes = [f.code for f in findings]
        assert "NEXT_STEPS_INCOMPLETE" in codes

    def test_missing_other_line_raises_error(self):
        summary = _BASE_SUMMARY.replace(
            "Pemberton Insurance: Send repair instructions to the approved garage\nOther: None",
            "Pemberton Insurance: Send repair instructions to the approved garage",
        )
        findings = _check_next_steps_completeness(summary)
        codes = [f.code for f in findings]
        assert "NEXT_STEPS_INCOMPLETE" in codes

    def test_complete_next_steps_returns_no_findings(self):
        findings = _check_next_steps_completeness(_BASE_SUMMARY)
        assert findings == []


class TestPhantomConditionalSections:
    def test_section_set_to_none_raises_error(self):
        summary = _BASE_SUMMARY + "\nLiability Summary: None\n"
        findings = _check_phantom_conditional_sections(summary)
        codes = [f.code for f in findings]
        assert "PHANTOM_CONDITIONAL_SECTION" in codes

    def test_each_conditional_section_name_detected(self):
        for section_name in [
            "Liability Summary",
            "Negotiation Summary",
            "Vehicle Damage",
            "Injury",
            "Property",
        ]:
            summary = _BASE_SUMMARY + f"\n{section_name}: None\n"
            findings = _check_phantom_conditional_sections(summary)
            codes = [f.code for f in findings]
            assert "PHANTOM_CONDITIONAL_SECTION" in codes, f"Not caught for: {section_name}"

    def test_valid_conditional_section_not_flagged(self):
        summary = _BASE_SUMMARY + "\nLiability Summary:\n- Fault accepted by third party.\n"
        findings = _check_phantom_conditional_sections(summary)
        assert findings == []

    def test_no_conditional_sections_returns_empty(self):
        findings = _check_phantom_conditional_sections(_BASE_SUMMARY)
        assert findings == []


class TestUnknownSectionHeaders:
    def test_unknown_header_raises_error(self):
        summary = _BASE_SUMMARY + "\nSummary Of Costs:\nVarious costs incurred.\n"
        findings = _check_unknown_section_headers(summary)
        codes = [f.code for f in findings]
        assert "UNKNOWN_SECTION_HEADER" in codes

    def test_known_headers_not_flagged(self):
        findings = _check_unknown_section_headers(_BASE_SUMMARY)
        assert findings == []

    def test_known_section_names_are_accepted(self):
        for header in ["Subject", "Executive Summary", "Next Steps"]:
            summary = f"{header}:\nSome content here.\n"
            findings = _check_unknown_section_headers(summary)
            assert findings == [], f"Incorrectly flagged known header: {header}"


class TestSubjectMultiline:
    def test_multiline_subject_raises_error(self):
        summary = _BASE_SUMMARY.replace(
            "Call regarding a vehicle damage claim and repair arrangement",
            "Call regarding a vehicle damage claim.\nAlso discussed repair arrangement.",
        )
        finding = _check_subject_multiline(summary)
        assert finding is not None
        assert finding.code == "SUBJECT_MULTILINE"
        assert finding.tier == "error"

    def test_single_line_subject_returns_none(self):
        assert _check_subject_multiline(_BASE_SUMMARY) is None

    def test_absent_subject_returns_none(self):
        # No Subject: header; caught by MISSING_SUBJECT, not here
        summary = _BASE_SUMMARY.replace("Subject:", "Topic:")
        assert _check_subject_multiline(summary) is None


class TestConditionalSectionEmptyBody:
    def test_empty_conditional_section_raises_error(self):
        # The check works when the empty section is the LAST section in the summary
        # (i.e., nothing follows it). The regex uses `\s*\n` which would otherwise
        # consume a blank line and merge the following section into the "body".
        injected = (
            "Caller: Jane Brown, policyholder, inbound\n"
            "\n"
            "Subject:\n"
            "One-line subject here\n"
            "\n"
            "Executive Summary:\n"
            "Some event occurred.\n"
            "- Bullet one\n"
            "- Bullet two\n"
            "\n"
            "Next Steps:\n"
            "Company: Action here\n"
            "Other: None\n"
            "\n"
            "Liability Summary:\n"  # empty body — nothing follows
        )
        findings = _check_conditional_section_empty_body(injected)
        codes = [f.code for f in findings]
        assert "CONDITIONAL_SECTION_EMPTY_BODY" in codes

    def test_no_conditional_sections_returns_empty(self):
        findings = _check_conditional_section_empty_body(_BASE_SUMMARY)
        assert findings == []


# ── Tier 2: Format quality checks ─────────────────────────────────────────


class TestMissingCallerLine:
    def test_missing_caller_line_raises_warning(self):
        from call_summarizer.guardrails import _check_caller_line_present

        summary = _BASE_SUMMARY.replace("Caller: Jane Brown, policyholder, inbound", "")
        finding = _check_caller_line_present(summary)
        assert finding is not None
        assert finding.code == "MISSING_CALLER_LINE"
        assert finding.tier == "warning"

    def test_caller_line_present_returns_none(self):
        from call_summarizer.guardrails import _check_caller_line_present

        assert _check_caller_line_present(_BASE_SUMMARY) is None


class TestCallerDirectionMissing:
    def test_no_direction_keyword_raises_warning(self):
        from call_summarizer.guardrails import _check_caller_direction

        summary = _BASE_SUMMARY.replace("inbound", "unknown")
        finding = _check_caller_direction(summary)
        assert finding is not None
        assert finding.code == "CALLER_DIRECTION_MISSING"
        assert finding.tier == "warning"

    def test_inbound_keyword_passes(self):
        from call_summarizer.guardrails import _check_caller_direction

        assert _check_caller_direction(_BASE_SUMMARY) is None

    def test_outbound_keyword_passes(self):
        from call_summarizer.guardrails import _check_caller_direction

        summary = _BASE_SUMMARY.replace("inbound", "outbound")
        assert _check_caller_direction(summary) is None


class TestCallerRelationshipUnrecognized:
    def test_unknown_relationship_raises_warning(self):
        from call_summarizer.guardrails import _check_caller_relationship

        summary = _BASE_SUMMARY.replace("policyholder", "mystery person")
        finding = _check_caller_relationship(summary)
        assert finding is not None
        assert finding.code == "CALLER_RELATIONSHIP_UNRECOGNIZED"
        assert finding.tier == "warning"

    def test_known_relationship_passes(self):
        from call_summarizer.guardrails import _check_caller_relationship

        for rel in ["policyholder", "solicitor", "family member", "third party"]:
            summary = _BASE_SUMMARY.replace("policyholder", rel)
            assert _check_caller_relationship(summary) is None, f"Incorrectly flagged: {rel}"


class TestExecutiveSummaryNoBullets:
    def test_no_bullets_raises_error(self):
        # Missing bullets is a Tier-1 error so the retry loop can correct it.
        no_bullets = (
            "Caller: Jane Brown, policyholder, inbound\n"
            "\n"
            "Subject:\n"
            "One-line subject\n"
            "\n"
            "Executive Summary:\n"
            "The caller contacted us. Nothing else was discussed.\n"
            "\n"
            "Next Steps:\n"
            "Company: Action here\n"
            "Other: None\n"
        )
        finding = _check_executive_summary_bullets(no_bullets)
        assert finding is not None
        assert finding.code == "EXECUTIVE_SUMMARY_NO_BULLETS"
        assert finding.tier == "error"

    def test_no_bullets_blocks_save(self):
        no_bullets = (
            "Caller: Jane Brown, policyholder, inbound\n"
            "\n"
            "Subject:\n"
            "One-line subject\n"
            "\n"
            "Executive Summary:\n"
            "The caller contacted us. Nothing else was discussed.\n"
            "\n"
            "Next Steps:\n"
            "Company: Action here\n"
            "Other: None\n"
        )
        result = run_guardrails(no_bullets)
        assert not result.passed
        assert "EXECUTIVE_SUMMARY_NO_BULLETS" in [f.code for f in result.errors]

    def test_bullets_present_returns_none(self):
        assert _check_executive_summary_bullets(_BASE_SUMMARY) is None


class TestNextStepsBothNone:
    def test_both_none_raises_warning(self):
        summary = _BASE_SUMMARY.replace(
            "Pemberton Insurance: Send repair instructions to the approved garage\nOther: None",
            "Pemberton Insurance: None\nOther: None",
        )
        finding = _check_next_steps_both_none(summary)
        assert finding is not None
        assert finding.code == "NEXT_STEPS_BOTH_NONE"
        assert finding.tier == "warning"

    def test_one_none_does_not_raise_warning(self):
        assert _check_next_steps_both_none(_BASE_SUMMARY) is None


class TestCharCountHigh:
    def test_summary_above_watermark_below_limit_raises_warning(self):
        # 1201 chars should trigger the high watermark warning
        summary = "A" * 1201
        finding = _check_char_count_high(summary)
        assert finding is not None
        assert finding.code == "CHAR_COUNT_HIGH"
        assert finding.tier == "warning"

    def test_summary_exactly_at_watermark_does_not_raise_warning(self):
        # At exactly 1200 chars the condition is 1200 < 1200 → False → no warning
        assert _check_char_count_high("A" * 1200) is None

    def test_summary_at_char_limit_raises_warning(self):
        # 1500 chars: within limit but above watermark → warning (not a Tier-1 error)
        finding = _check_char_count_high("A" * CHAR_LIMIT)
        assert finding is not None
        assert finding.code == "CHAR_COUNT_HIGH"

    def test_short_summary_does_not_raise_warning(self):
        assert _check_char_count_high(_BASE_SUMMARY) is None


class TestDuplicateBulletContent:
    def test_repeated_numeric_token_raises_warning(self):
        summary = """\
Caller: Jane Brown, policyholder, inbound

Subject:
One-line subject

Executive Summary:
Call details.
- Settlement offer of 500 discussed
- Counter offer raised; original 500 still on table

Next Steps:
Company: Action
Other: None
"""
        findings = _check_duplicate_bullet_content(summary)
        codes = [f.code for f in findings]
        assert "DUPLICATE_BULLET_CONTENT" in codes

    def test_no_duplicate_returns_empty(self):
        findings = _check_duplicate_bullet_content(_BASE_SUMMARY)
        assert findings == []


class TestVehicleDamageTowageMissing:
    def test_vehicle_damage_without_towage_raises_warning(self):
        summary = _BASE_SUMMARY + """\
Vehicle Damage:
Vehicle Status: Repairable
Car hire: None
"""
        finding = _check_vehicle_damage_subfields(summary)
        assert finding is not None
        assert finding.code == "VEHICLE_DAMAGE_TOWAGE_MISSING"
        assert finding.tier == "warning"

    def test_vehicle_damage_with_towage_returns_none(self):
        summary = _BASE_SUMMARY + """\
Vehicle Damage:
Vehicle Status: Repairable
Towage: None
Car hire: None
"""
        assert _check_vehicle_damage_subfields(summary) is None

    def test_no_vehicle_damage_section_returns_none(self):
        assert _check_vehicle_damage_subfields(_BASE_SUMMARY) is None


# ── Tier 3: Content integrity checks ──────────────────────────────────────


class TestAmountNotInTranscript:
    def test_hallucinated_amount_raises_warning(self):
        summary = _BASE_SUMMARY + "\n- Settlement of €9,999 agreed.\n"
        transcript = "caller discussed their claim"  # no €9,999
        findings = _check_amounts_in_transcript(summary, transcript)
        codes = [f.code for f in findings]
        assert "AMOUNT_NOT_IN_TRANSCRIPT" in codes

    def test_amount_present_in_transcript_passes(self):
        summary = _BASE_SUMMARY + "\n- Settlement of €9,999 agreed.\n"
        transcript = "caller discussed settlement of €9,999"
        findings = _check_amounts_in_transcript(summary, transcript)
        assert findings == []

    def test_no_amounts_returns_empty(self):
        findings = _check_amounts_in_transcript(_BASE_SUMMARY, _BASE_TRANSCRIPT)
        assert findings == []


class TestReferenceNotInTranscript:
    def test_hallucinated_reference_raises_warning(self):
        summary = _BASE_SUMMARY + "\n- Reference: CLM-1234-56789\n"
        transcript = "caller asked about their policy"
        findings = _check_references_in_transcript(summary, transcript)
        codes = [f.code for f in findings]
        assert "REFERENCE_NOT_IN_TRANSCRIPT" in codes

    def test_reference_in_transcript_passes(self):
        summary = _BASE_SUMMARY + "\n- Reference: CLM-1234-56789\n"
        transcript = "reference number CLM-1234-56789 was confirmed"
        findings = _check_references_in_transcript(summary, transcript)
        assert findings == []

    def test_no_references_returns_empty(self):
        findings = _check_references_in_transcript(_BASE_SUMMARY, _BASE_TRANSCRIPT)
        assert findings == []


class TestIbanNotInTranscript:
    def test_hallucinated_iban_raises_warning(self):
        summary = _BASE_SUMMARY + "\n- IBAN: IE29AIBK93115212345678\n"
        transcript = "caller confirmed bank account details"
        findings = _check_ibans_in_transcript(summary, transcript)
        codes = [f.code for f in findings]
        assert "IBAN_NOT_IN_TRANSCRIPT" in codes

    def test_iban_in_transcript_passes(self):
        summary = _BASE_SUMMARY + "\n- IBAN: IE29AIBK93115212345678\n"
        transcript = "The IBAN provided was IE29AIBK93115212345678"
        findings = _check_ibans_in_transcript(summary, transcript)
        assert findings == []

    def test_no_ibans_returns_empty(self):
        findings = _check_ibans_in_transcript(_BASE_SUMMARY, _BASE_TRANSCRIPT)
        assert findings == []


class TestEmailNotInTranscript:
    def test_hallucinated_email_raises_warning(self):
        summary = _BASE_SUMMARY + "\n- Documents sent to claims@fake.com\n"
        transcript = "caller discussed their claim"
        findings = _check_emails_in_transcript(summary, transcript)
        codes = [f.code for f in findings]
        assert "EMAIL_NOT_IN_TRANSCRIPT" in codes

    def test_email_in_transcript_passes(self):
        summary = _BASE_SUMMARY + "\n- Documents sent to claims@fake.com\n"
        transcript = "email claims@fake.com was confirmed by caller"
        findings = _check_emails_in_transcript(summary, transcript)
        assert findings == []

    def test_no_emails_returns_empty(self):
        findings = _check_emails_in_transcript(_BASE_SUMMARY, _BASE_TRANSCRIPT)
        assert findings == []


class TestUnverifiedConfirmation:
    def test_bank_detail_confirmation_without_transcript_evidence_raises_warning(self):
        summary = _BASE_SUMMARY + "\n- Caller confirmed their bank details.\n"
        transcript = "caller asked about their claim status"  # no bank/detail evidence
        findings = _check_unverified_confirmations(summary, transcript)
        codes = [f.code for f in findings]
        assert "UNVERIFIED_CONFIRMATION" in codes

    def test_bank_detail_confirmation_with_transcript_evidence_passes(self):
        summary = _BASE_SUMMARY + "\n- Caller confirmed their bank details.\n"
        transcript = "caller confirmed their bank account and provided all details"
        findings = _check_unverified_confirmations(summary, transcript)
        assert findings == []

    def test_no_confirmation_phrases_returns_empty(self):
        findings = _check_unverified_confirmations(_BASE_SUMMARY, _BASE_TRANSCRIPT)
        assert findings == []


class TestConditionalSectionUnjustified:
    # Append the section without Python indentation so `^` anchors work correctly.
    _VEHICLE_DAMAGE_BLOCK = (
        "\nVehicle Damage:\n"
        "Vehicle Status: Repairable\n"
        "Towage: None\n"
        "Car hire: None\n"
    )

    def test_vehicle_damage_without_vehicle_terms_raises_warning(self):
        from call_summarizer.guardrails import _check_conditional_sections_justified

        summary = _BASE_SUMMARY + self._VEHICLE_DAMAGE_BLOCK
        # Transcript contains NO vehicle-domain terms so the section is unjustified.
        transcript = "caller called about a home contents claim. storm damage to roof."
        findings = _check_conditional_sections_justified(summary, transcript)
        codes = [f.code for f in findings]
        assert "CONDITIONAL_SECTION_UNJUSTIFIED" in codes

    def test_vehicle_damage_with_vehicle_terms_passes(self):
        from call_summarizer.guardrails import _check_conditional_sections_justified

        summary = _BASE_SUMMARY + self._VEHICLE_DAMAGE_BLOCK
        transcript = "caller said their car was damaged in an accident"
        findings = _check_conditional_sections_justified(summary, transcript)
        assert findings == []

    def test_no_conditional_sections_returns_empty(self):
        from call_summarizer.guardrails import _check_conditional_sections_justified

        findings = _check_conditional_sections_justified(_BASE_SUMMARY, _BASE_TRANSCRIPT)
        assert findings == []


# ── Extraction helpers ─────────────────────────────────────────────────────


class TestExtractAmounts:
    def test_extracts_euro_amount(self):
        amounts = _extract_amounts("Settlement of €3,150.00 was agreed.")
        assert "3150.00" in amounts

    def test_extracts_pound_amount(self):
        amounts = _extract_amounts("Payment of £500 received.")
        assert "500" in amounts

    def test_extracts_dollar_amount(self):
        amounts = _extract_amounts("Offer of $1,200 made.")
        assert "1200" in amounts

    def test_no_amounts_returns_empty_set(self):
        assert _extract_amounts("No monetary value mentioned here.") == set()

    def test_normalizes_whitespace_between_symbol_and_digits(self):
        amounts = _extract_amounts("€ 3,150.00")
        assert "3150.00" in amounts


class TestExtractIbans:
    def test_extracts_valid_iban(self):
        ibans = _extract_ibans("IBAN: IE29AIBK93115212345678")
        assert "IE29AIBK93115212345678" in ibans

    def test_no_match_for_lowercase_iban(self):
        # The regex requires uppercase letters — lowercase IBANs are not extracted.
        ibans = _extract_ibans("iban ie29aibk93115212345678")
        assert ibans == set()

    def test_no_iban_returns_empty_set(self):
        assert _extract_ibans("No bank details in this text.") == set()


class TestExtractEmails:
    def test_extracts_email_address(self):
        emails = _extract_emails("Send docs to claims@insurer.ie please.")
        assert "claims@insurer.ie" in emails

    def test_lowercases_email(self):
        emails = _extract_emails("Contact Claims@Insurer.IE")
        assert "claims@insurer.ie" in emails

    def test_no_email_returns_empty_set(self):
        assert _extract_emails("No email address here.") == set()


class TestExtractReferences:
    def test_extracts_reference_number(self):
        refs = _extract_references("Claim reference CLM-1234-56789 confirmed.")
        assert "CLM-1234-56789" in refs

    def test_no_match_for_lowercase_reference(self):
        # The regex requires uppercase letters — lowercase references are not extracted.
        refs = _extract_references("ref clm-1234-56789")
        assert refs == set()

    def test_no_reference_returns_empty_set(self):
        assert _extract_references("No reference number found.") == set()


# ── Section extraction helpers ─────────────────────────────────────────────


class TestGetCallerLine:
    def test_extracts_caller_content(self):
        result = _get_caller_line(_BASE_SUMMARY)
        assert result == "Jane Brown, policyholder, inbound"

    def test_returns_none_when_absent(self):
        assert _get_caller_line("No caller line here.") is None


class TestGetAllSectionHeaderNames:
    def test_returns_known_header_names(self):
        headers = _get_all_section_header_names(_BASE_SUMMARY)
        assert "Subject" in headers
        assert "Executive Summary" in headers
        assert "Next Steps" in headers

    def test_does_not_capture_inline_caller_line(self):
        headers = _get_all_section_header_names(_BASE_SUMMARY)
        assert "Caller" not in headers  # Caller: has inline content, not a bare header

    def test_does_not_capture_sub_field_labels(self):
        summary = _BASE_SUMMARY + """\
Vehicle Damage:
Vehicle Status: Repairable
Towage: None
Car hire: None
"""
        headers = _get_all_section_header_names(summary)
        assert "Vehicle Status" not in headers


# ── build_retry_prompt_addendum ────────────────────────────────────────────


class TestBuildRetryPromptAddendum:
    def _make_result_with_codes(self, codes: list[str]) -> GuardrailResult:
        from call_summarizer.models import Finding

        findings = [
            Finding(tier="error", code=code, message=f"Test error: {code}")
            for code in codes
        ]
        return GuardrailResult(
            passed=False,
            findings=findings,
            char_count=100,
            char_within_limit=True,
        )

    def test_char_limit_error_produces_addendum(self):
        result = self._make_result_with_codes(["CHAR_LIMIT_EXCEEDED"])
        addendum = build_retry_prompt_addendum(result)
        assert "concise" in addendum.lower() or str(CHAR_LIMIT) in addendum

    def test_phantom_section_error_produces_addendum(self):
        result = self._make_result_with_codes(["PHANTOM_CONDITIONAL_SECTION"])
        addendum = build_retry_prompt_addendum(result)
        assert "None" in addendum or "omit" in addendum.lower()

    def test_next_steps_error_produces_addendum(self):
        result = self._make_result_with_codes(["MISSING_NEXT_STEPS"])
        addendum = build_retry_prompt_addendum(result)
        assert "Next Steps" in addendum

    def test_unknown_section_error_produces_addendum(self):
        result = self._make_result_with_codes(["UNKNOWN_SECTION_HEADER"])
        addendum = build_retry_prompt_addendum(result)
        assert "section" in addendum.lower()

    def test_subject_multiline_error_produces_addendum(self):
        result = self._make_result_with_codes(["SUBJECT_MULTILINE"])
        addendum = build_retry_prompt_addendum(result)
        assert "Subject" in addendum

    def test_no_errors_returns_empty_string(self):
        result = GuardrailResult(
            passed=True,
            findings=[],
            char_count=100,
            char_within_limit=True,
        )
        assert build_retry_prompt_addendum(result) == ""

    def test_executive_summary_no_bullets_error_produces_addendum(self):
        result = self._make_result_with_codes(["EXECUTIVE_SUMMARY_NO_BULLETS"])
        addendum = build_retry_prompt_addendum(result)
        assert "bullet" in addendum.lower() or "-" in addendum

    def test_multiple_errors_produce_combined_addendum(self):
        result = self._make_result_with_codes(
            ["CHAR_LIMIT_EXCEEDED", "PHANTOM_CONDITIONAL_SECTION"]
        )
        addendum = build_retry_prompt_addendum(result)
        assert len(addendum) > 0
        # Both instructions should appear in the combined output
        assert "CRITICAL" in addendum
