"""Transcript file discovery and loading utilities.

Provides two helpers that abstract all filesystem interaction for transcript
files.  All callers (CLI, batch script, service layer) use these functions so
changes to encoding or discovery logic only need to be made in one place.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_transcript(transcript_path: Path) -> str:
    """Read and return the full text of a transcript file.

    Args:
        transcript_path: Path to the ``.txt`` transcript file.

    Returns:
        The file contents as a decoded UTF-8 string.

    Raises:
        FileNotFoundError: If *transcript_path* does not exist.
        OSError: If the file exists but cannot be read (e.g. permission error).
    """
    logger.debug("Loading transcript: %s", transcript_path)

    if not transcript_path.exists():
        logger.error("Transcript not found: %s", transcript_path)
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    try:
        content = transcript_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("Cannot read transcript %s: %s", transcript_path.name, exc)
        raise

    logger.info("Loaded transcript: %s (%d chars)", transcript_path.name, len(content))
    return content


def find_transcripts(directory: Path) -> list[Path]:
    """Return all ``.txt`` files in *directory*, sorted alphabetically by name.

    Args:
        directory: Directory to search for transcript files.

    Returns:
        Alphabetically sorted list of :class:`~pathlib.Path` objects pointing
        to each ``.txt`` file found.  Returns an empty list when the directory
        contains no matching files.

    Raises:
        FileNotFoundError: If *directory* does not exist on the filesystem.
    """
    logger.debug("Searching for transcripts in: %s", directory)

    if not directory.exists():
        logger.error("Input directory not found: %s", directory)
        raise FileNotFoundError(f"Input directory not found: {directory}")

    transcript_paths = sorted(directory.glob("*.txt"))
    logger.info(
        "Found %d transcript(s) in %s",
        len(transcript_paths),
        directory,
    )
    return transcript_paths
