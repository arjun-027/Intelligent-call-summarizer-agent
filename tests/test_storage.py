from pathlib import Path

import pytest

from call_summarizer.utils.storage import derive_output_path, save_summary


class TestDeriveOutputPath:
    def test_appends_summary_suffix(self):
        transcript = Path("Input_data/1-transcript.txt")
        output_dir = Path("Output_data")
        result = derive_output_path(transcript, output_dir)
        assert result == Path("Output_data/1-transcript-summary.txt")

    def test_uses_output_dir(self):
        transcript = Path("Input_data/call.txt")
        output_dir = Path("/some/other/dir")
        result = derive_output_path(transcript, output_dir)
        assert result.parent == output_dir

    def test_stem_preserved_in_output_name(self):
        transcript = Path("Input_data/my-call-recording.txt")
        output_dir = Path("Output_data")
        result = derive_output_path(transcript, output_dir)
        assert result.name == "my-call-recording-summary.txt"


class TestSaveSummary:
    def test_creates_file_with_correct_content(self, tmp_path):
        output_path = tmp_path / "output" / "summary.txt"
        save_summary("Test summary content", output_path)
        assert output_path.read_text(encoding="utf-8") == "Test summary content"

    def test_creates_parent_directories(self, tmp_path):
        nested_path = tmp_path / "a" / "b" / "c" / "summary.txt"
        save_summary("content", nested_path)
        assert nested_path.exists()

    def test_overwrites_existing_file(self, tmp_path):
        output_path = tmp_path / "summary.txt"
        output_path.write_text("old content", encoding="utf-8")
        save_summary("new content", output_path)
        assert output_path.read_text(encoding="utf-8") == "new content"
