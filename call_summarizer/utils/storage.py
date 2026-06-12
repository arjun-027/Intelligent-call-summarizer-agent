"""Summary file persistence utilities.

Provides two focused helpers used by every caller (CLI, API, Streamlit, batch
scripts) to derive a canonical output path and write a summary to disk.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def derive_output_path(transcript_path: Path, output_dir: Path) -> Path:
    """Compute the canonical output path for a given transcript.

    Maps ``<any_dir>/<stem>.txt`` → ``<output_dir>/<stem>-summary.txt`` so
    all callers write to a consistent, predictable location regardless of
    where the source transcript lives.

    Args:
        transcript_path: Path to the source ``.txt`` transcript file.
        output_dir: Directory where generated summaries should be stored.

    Returns:
        The :class:`~pathlib.Path` the summary should be written to.
        The parent directory is NOT created here; call :func:`save_summary`
        which creates it automatically.

    Example::

        derive_output_path(Path("data/1-transcript.txt"), Path("Output_data"))
        # → Path("Output_data/1-transcript-summary.txt")
    """
    output_path = output_dir / f"{transcript_path.stem}-summary.txt"
    logger.debug("Derived output path: %s → %s", transcript_path.name, output_path)
    return output_path


def save_summary(summary_text: str, output_path: Path) -> None:
    """Write *summary_text* to *output_path*, creating parent directories as needed.

    The parent directory is created (with all missing intermediate directories)
    before writing so callers do not need to pre-create it.

    Args:
        summary_text: The generated summary text to persist.
        output_path: Destination file path.

    Raises:
        OSError: If the file or its parent directories cannot be created or
            written (e.g. permission error, disk full).
    """
    logger.debug("Saving summary → %s (%d chars)", output_path, len(summary_text))
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(summary_text, encoding="utf-8")
        logger.info("Summary saved: %s (%d chars)", output_path.name, len(summary_text))
    except OSError as exc:
        logger.error("Failed to save summary to %s: %s", output_path, exc)
        raise
