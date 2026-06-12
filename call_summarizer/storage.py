"""Summary file persistence utilities."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def derive_output_path(transcript_path: Path, output_dir: Path) -> Path:
    """Compute the output file path for a given transcript.

    Maps ``<input_dir>/<stem>.txt`` → ``<output_dir>/<stem>-summary.txt``.

    Args:
        transcript_path: Path of the source transcript file.
        output_dir: Directory where output summaries are stored.

    Returns:
        The :class:`~pathlib.Path` where the summary should be written.
    """
    output_path = output_dir / f"{transcript_path.stem}-summary.txt"
    logger.debug("Derived output path: %s → %s", transcript_path.name, output_path)
    return output_path


def save_summary(summary: str, output_path: Path) -> None:
    """Write *summary* text to *output_path*, creating parent directories as needed.

    Args:
        summary: The generated summary text to persist.
        output_path: Destination file path.

    Raises:
        OSError: If the file or its parent directories cannot be created or written.
    """
    logger.debug("Saving summary to: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary, encoding="utf-8")
    logger.info("Summary saved: %s (%d chars)", output_path.name, len(summary))
