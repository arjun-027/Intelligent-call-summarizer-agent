"""Transcript file discovery and loading utilities."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_transcript(path: Path) -> str:
    """Read and return the full text content of a transcript file.

    Args:
        path: Path to the ``.txt`` transcript file.

    Returns:
        The file contents as a UTF-8 decoded string.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        OSError: If the file exists but cannot be read (e.g. permission error).
    """
    logger.debug("Loading transcript from: %s", path)

    if not path.exists():
        logger.error("Transcript file not found: %s", path)
        raise FileNotFoundError(f"Transcript file not found: {path}")

    content = path.read_text(encoding="utf-8")
    logger.info("Loaded transcript: %s (%d chars)", path.name, len(content))
    return content


def find_transcripts(directory: Path) -> list[Path]:
    """Return all ``.txt`` files in *directory*, sorted by filename.

    Args:
        directory: Directory to search for transcript files.

    Returns:
        A list of :class:`~pathlib.Path` objects, sorted alphabetically.
        Returns an empty list if the directory contains no ``.txt`` files.

    Raises:
        FileNotFoundError: If *directory* does not exist.
    """
    logger.debug("Searching for transcripts in: %s", directory)

    if not directory.exists():
        logger.error("Input directory not found: %s", directory)
        raise FileNotFoundError(f"Input directory not found: {directory}")

    transcripts = sorted(directory.glob("*.txt"))
    logger.info("Found %d transcript(s) in %s", len(transcripts), directory)
    return transcripts
