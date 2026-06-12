import pytest

from call_summarizer.summarizer import CHAR_LIMIT, SYSTEM_PROMPT, validate_summary_length


class TestValidateSummaryLength:
    def test_returns_true_when_within_limit(self):
        summary = "A" * CHAR_LIMIT
        assert validate_summary_length(summary) is True

    def test_returns_false_when_over_limit(self):
        summary = "A" * (CHAR_LIMIT + 1)
        assert validate_summary_length(summary) is False

    def test_returns_true_for_empty_string(self):
        assert validate_summary_length("") is True

    def test_uses_custom_limit_when_provided(self):
        summary = "A" * 100
        assert validate_summary_length(summary, limit=50) is False
        assert validate_summary_length(summary, limit=200) is True


class TestSystemPrompt:
    def test_prompt_is_not_empty(self):
        assert len(SYSTEM_PROMPT.strip()) > 0

    def test_prompt_mentions_char_limit(self):
        assert "1,500" in SYSTEM_PROMPT

    def test_prompt_instructs_on_conditional_sections(self):
        assert "Liability" in SYSTEM_PROMPT
        assert "Vehicle Damage" in SYSTEM_PROMPT
        assert "Injury" in SYSTEM_PROMPT
        assert "Property" in SYSTEM_PROMPT

    def test_prompt_instructs_on_caller_field(self):
        assert "Caller:" in SYSTEM_PROMPT

    def test_prompt_instructs_on_next_steps(self):
        assert "Next Steps" in SYSTEM_PROMPT
