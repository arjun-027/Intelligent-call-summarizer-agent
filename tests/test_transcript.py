import pytest

from call_summarizer.transcript import find_transcripts, load_transcript


class TestLoadTranscript:
    def test_reads_file_content(self, tmp_path):
        transcript = tmp_path / "call.txt"
        transcript.write_text("Hello transcript", encoding="utf-8")
        assert load_transcript(transcript) == "Hello transcript"

    def test_preserves_multiline_content(self, tmp_path):
        content = "Line one\nLine two\nLine three"
        transcript = tmp_path / "call.txt"
        transcript.write_text(content, encoding="utf-8")
        assert load_transcript(transcript) == content

    def test_raises_file_not_found_for_missing_file(self, tmp_path):
        missing = tmp_path / "nonexistent.txt"
        with pytest.raises(FileNotFoundError, match="Transcript file not found"):
            load_transcript(missing)


class TestFindTranscripts:
    def test_returns_txt_files_only(self, tmp_path):
        (tmp_path / "a.txt").touch()
        (tmp_path / "b.txt").touch()
        (tmp_path / "readme.md").touch()
        result = find_transcripts(tmp_path)
        assert all(p.suffix == ".txt" for p in result)
        assert len(result) == 2

    def test_returns_files_sorted_by_name(self, tmp_path):
        (tmp_path / "3-transcript.txt").touch()
        (tmp_path / "1-transcript.txt").touch()
        (tmp_path / "2-transcript.txt").touch()
        result = find_transcripts(tmp_path)
        names = [p.name for p in result]
        assert names == sorted(names)

    def test_returns_empty_list_when_no_txt_files(self, tmp_path):
        (tmp_path / "notes.md").touch()
        assert find_transcripts(tmp_path) == []

    def test_raises_file_not_found_for_missing_directory(self, tmp_path):
        missing_dir = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="Input directory not found"):
            find_transcripts(missing_dir)
