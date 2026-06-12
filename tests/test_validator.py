"""Unit tests for the validator module."""

import pytest

from call_summarizer.summarizer import CHAR_LIMIT
from call_summarizer.validator import (
    ValidationResult,
    validate_input_file,
    validate_summary,
)


class TestValidateInputFile:
    def test_valid_file_passes(self, tmp_path):
        transcript = tmp_path / "call.txt"
        transcript.write_text("A" * 200, encoding="utf-8")
        result = validate_input_file(transcript)
        assert result.is_valid is True
        assert result.issues == []

    def test_missing_file_fails(self, tmp_path):
        result = validate_input_file(tmp_path / "nonexistent.txt")
        assert result.is_valid is False
        assert any("does not exist" in issue for issue in result.issues)

    def test_non_txt_extension_fails(self, tmp_path):
        csv_file = tmp_path / "call.csv"
        csv_file.write_text("A" * 200, encoding="utf-8")
        result = validate_input_file(csv_file)
        assert result.is_valid is False
        assert any(".txt" in issue for issue in result.issues)

    def test_empty_file_fails(self, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("", encoding="utf-8")
        result = validate_input_file(empty)
        assert result.is_valid is False
        assert any("empty" in issue or "short" in issue for issue in result.issues)

    def test_returns_validation_result_type(self, tmp_path):
        transcript = tmp_path / "call.txt"
        transcript.write_text("A" * 200, encoding="utf-8")
        result = validate_input_file(transcript)
        assert isinstance(result, ValidationResult)


class TestValidateSummary:
    _VALID_SUMMARY = (
        "Caller: John Smith, policyholder, inbound\n\n"
        "Subject:\nCalling to check claim status\n\n"
        "Executive Summary:\nJohn called to follow up on his claim.\n"
        "- Claim received on 3rd January\n"
        "- Assessor to visit within 48 hours\n\n"
        "Next Steps:\nPemberton Insurance: Send confirmation email\n"
        "Other: None"
    )

    def test_valid_summary_passes(self):
        result = validate_summary(self._VALID_SUMMARY)
        assert result.is_valid is True
        assert result.issues == []

    def test_summary_over_limit_fails(self):
        long_summary = self._VALID_SUMMARY + ("x" * (CHAR_LIMIT + 1))
        result = validate_summary(long_summary)
        assert result.is_valid is False
        assert any("limit" in issue for issue in result.issues)

    def test_missing_caller_section_fails(self):
        summary = self._VALID_SUMMARY.replace("Caller:", "Person:")
        result = validate_summary(summary)
        assert result.is_valid is False
        assert any("Caller:" in issue for issue in result.issues)

    def test_missing_next_steps_section_fails(self):
        summary = self._VALID_SUMMARY.replace("Next Steps:", "Actions:")
        result = validate_summary(summary)
        assert result.is_valid is False
        assert any("Next Steps:" in issue for issue in result.issues)

    def test_missing_executive_summary_section_fails(self):
        summary = self._VALID_SUMMARY.replace("Executive Summary:", "Overview:")
        result = validate_summary(summary)
        assert result.is_valid is False

    def test_returns_validation_result_type(self):
        result = validate_summary(self._VALID_SUMMARY)
        assert isinstance(result, ValidationResult)
