"""Output guardrail engine for generated call summaries.

Package structure
-----------------
constants.py          Shared lookup tables and compiled regex patterns.
helpers.py            Section-extraction and entity-extraction utilities.
tier1_structural.py   Tier-1 structural error checks (BLOCKING).
tier2_format.py       Tier-2 format quality warning checks (ADVISORY).
tier3_content.py      Tier-3 content integrity checks vs. transcript (ADVISORY).
runner.py             Orchestrates all tiers; exposes the public API.

Public API
----------
    from call_summarizer.guardrails import run_guardrails, build_retry_prompt_addendum

    result = run_guardrails(summary, transcript_content)
    if not result.passed:
        addendum = build_retry_prompt_addendum(result)
"""

from .constants import (
    _CONDITIONAL_SECTION_NAMES,
    _CONFIRMATION_PHRASE_CHECKS,
)
from .helpers import (
    _extract_amounts,
    _extract_emails,
    _extract_ibans,
    _extract_references,
    _extract_next_steps_body,
    _extract_section_body,
    _get_caller_line,
    _get_all_section_header_names,
)
from .runner import build_retry_prompt_addendum, run_guardrails
from .tier1_structural import (
    _check_char_limit,
    _check_conditional_section_empty_body,
    _check_empty,
    _check_executive_summary_bullets,
    _check_missing_executive_summary,
    _check_missing_next_steps,
    _check_missing_subject,
    _check_next_steps_completeness,
    _check_phantom_conditional_sections,
    _check_subject_multiline,
    _check_unknown_section_headers,
)
from .tier2_format import (
    _check_caller_direction,
    _check_caller_line_present,
    _check_caller_relationship,
    _check_char_count_high,
    _check_duplicate_bullet_content,
    _check_next_steps_both_none,
    _check_vehicle_damage_subfields,
)
from .tier3_content import (
    _check_amounts_in_transcript,
    _check_conditional_sections_justified,
    _check_emails_in_transcript,
    _check_ibans_in_transcript,
    _check_references_in_transcript,
    _check_unverified_confirmations,
)

__all__ = [
    # Primary public API
    "run_guardrails",
    "build_retry_prompt_addendum",
    # Constants re-exported for evaluator.py
    "_CONDITIONAL_SECTION_NAMES",
    "_CONFIRMATION_PHRASE_CHECKS",
    # Helpers re-exported for evaluator.py / tests
    "_extract_amounts",
    "_extract_emails",
    "_extract_ibans",
    "_extract_references",
    "_extract_next_steps_body",
    "_extract_section_body",
    "_get_caller_line",
    "_get_all_section_header_names",
    # Tier-1 checks re-exported for evaluator.py / tests
    "_check_char_limit",
    "_check_conditional_section_empty_body",
    "_check_empty",
    "_check_executive_summary_bullets",
    "_check_missing_executive_summary",
    "_check_missing_next_steps",
    "_check_missing_subject",
    "_check_next_steps_completeness",
    "_check_phantom_conditional_sections",
    "_check_subject_multiline",
    "_check_unknown_section_headers",
    # Tier-2 checks re-exported for evaluator.py / tests
    "_check_caller_direction",
    "_check_caller_line_present",
    "_check_caller_relationship",
    "_check_char_count_high",
    "_check_duplicate_bullet_content",
    "_check_next_steps_both_none",
    "_check_vehicle_damage_subfields",
    # Tier-3 checks re-exported for evaluator.py
    "_check_amounts_in_transcript",
    "_check_conditional_sections_justified",
    "_check_emails_in_transcript",
    "_check_ibans_in_transcript",
    "_check_references_in_transcript",
    "_check_unverified_confirmations",
]
