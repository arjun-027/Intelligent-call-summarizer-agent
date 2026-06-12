"""Streamlit frontend for the Insurance Call Summariser.

Connects to the FastAPI backend at ``API_BASE_URL``
(default: ``http://localhost:8000/api/v1``).

Workflow
--------
1. User uploads a single ``.txt`` transcript file.
2. Clicks **Generate Summary** → calls ``POST /api/v1/summarize``.
3. Reviews the summary and guardrail findings:
   - Red error banners (Tier-1) disable the Submit button until resolved.
   - Yellow warning banners (Tier-2 / Tier-3) are advisory.
4. Optionally edits the summary text area.
5. Clicks **Submit & Save** → calls ``POST /api/v1/summaries`` to persist.
"""

import logging
import os

import requests
import streamlit as st

logger = logging.getLogger(__name__)

_API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
_CHAR_LIMIT: int = 1500
_REQUEST_TIMEOUT: int = 60  # seconds


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------


def _init_session_state() -> None:
    """Initialise all session-state keys on the very first script run.

    Only sets keys that are absent so user edits are never overwritten by
    normal page re-renders.
    """
    defaults: dict = {
        "summary_text": "",
        "filename": "",
        "generated": False,
        "errors": [],           # Tier-1 blocking errors
        "warnings": [],         # Tier-2 / Tier-3 advisory warnings
        "passed_guardrails": True,
        "last_uploaded_filename": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_summary_state() -> None:
    """Clear summary and guardrail state when a new file is selected.

    Prevents stale findings from a previous transcript being shown
    alongside a newly generated summary.
    """
    st.session_state.summary_text = ""
    st.session_state.filename = ""
    st.session_state.generated = False
    st.session_state.errors = []
    st.session_state.warnings = []
    st.session_state.passed_guardrails = True


# ---------------------------------------------------------------------------
# API client helpers
# ---------------------------------------------------------------------------


def _post_summarize(file: st.runtime.uploaded_file_manager.UploadedFile) -> dict | None:
    """POST the uploaded file to ``POST /api/v1/summarize`` and return parsed JSON.

    Args:
        file: The Streamlit uploaded-file object containing the transcript.

    Returns:
        Parsed JSON response dict on HTTP 200, or ``None`` if the request
        failed (error details are shown via ``st.error``).
    """
    try:
        response = requests.post(
            f"{_API_BASE_URL}/summarize",
            files={"file": (file.name, file.getvalue(), "text/plain")},
            timeout=_REQUEST_TIMEOUT,
        )
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API server. Is it running on port 8000?")
        logger.error("Connection error calling POST /summarize")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. The model may be taking too long — please retry.")
        logger.error("Timeout calling POST /summarize")
        return None

    if response.status_code == 200:
        return response.json()

    detail = response.json().get("detail", "Unknown API error")
    st.error(f"API error {response.status_code}: {detail}")
    logger.error("POST /summarize returned %d: %s", response.status_code, detail)
    return None


def _post_submit(filename: str, summary: str) -> dict | None:
    """POST the final summary to ``POST /api/v1/summaries`` for persistence.

    Args:
        filename: Transcript filename stem returned by the summarize endpoint.
        summary: Final summary text (possibly user-edited).

    Returns:
        Parsed JSON response dict on HTTP 201, or ``None`` on failure.
    """
    try:
        response = requests.post(
            f"{_API_BASE_URL}/summaries",
            json={"filename": filename, "summary": summary},
            timeout=_REQUEST_TIMEOUT,
        )
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API server. Is it running on port 8000?")
        logger.error("Connection error calling POST /summaries")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out.")
        logger.error("Timeout calling POST /summaries")
        return None

    if response.status_code == 201:
        return response.json()

    # Surface structured errors from the 422 response
    body = response.json()
    if isinstance(body.get("detail"), dict):
        st.error(body["detail"].get("message", "Validation failed"))
        for err in body["detail"].get("errors", []):
            st.error(f"  • {err}")
    else:
        st.error(f"API error {response.status_code}: {body.get('detail', 'Unknown error')}")

    logger.error("POST /summaries returned %d: %s", response.status_code, body)
    return None


# ---------------------------------------------------------------------------
# UI section renderers
# ---------------------------------------------------------------------------


def _render_upload_section() -> "st.runtime.uploaded_file_manager.UploadedFile | None":
    """Render the file uploader widget.

    Returns:
        The uploaded file object if a file is selected, otherwise ``None``.
    """
    st.subheader("Step 1 — Upload Transcript")
    return st.file_uploader(
        "Select a call transcript",
        type=["txt"],
        accept_multiple_files=False,
        help="Only .txt files are accepted. Upload one file at a time.",
        label_visibility="collapsed",
    )


def _render_generate_section(uploaded_file) -> None:
    """Render the Generate Summary button and trigger the API call on click.

    Args:
        uploaded_file: Uploaded file object, or ``None`` when no file is selected
            (button is disabled in that case).
    """
    st.subheader("Step 2 — Generate Summary")

    clicked = st.button(
        "Generate Summary",
        disabled=uploaded_file is None,
        type="primary",
        use_container_width=True,
    )
    if not clicked:
        return

    with st.spinner("Generating summary — this may take a few seconds…"):
        data = _post_summarize(uploaded_file)

    if not data:
        return

    st.session_state.summary_text = data["summary"]
    st.session_state.filename = data["filename"]
    st.session_state.generated = True
    st.session_state.errors = data.get("errors", [])
    st.session_state.warnings = data.get("warnings", [])
    st.session_state.passed_guardrails = data.get("passed_guardrails", True)

    logger.info(
        "Summary received — %d chars, passed: %s, errors: %d, warnings: %d",
        data.get("char_count", 0),
        data.get("passed_guardrails"),
        len(st.session_state.errors),
        len(st.session_state.warnings),
    )
    st.rerun()


def _render_guardrail_errors() -> None:
    """Render Tier-1 blocking error banners.

    Each error is shown as a distinct ``st.error`` block with the check code
    prefix so the reviewer knows exactly which rule was violated.
    """
    if not st.session_state.errors:
        return

    st.error(
        f"**{len(st.session_state.errors)} blocking error(s) — resolve before saving:**"
    )
    for msg in st.session_state.errors:
        st.error(f"• {msg}")


def _render_guardrail_warnings() -> None:
    """Render Tier-2 / Tier-3 advisory warning banners inside a collapsible panel."""
    if not st.session_state.warnings:
        return

    with st.expander(
        f"⚠ {len(st.session_state.warnings)} advisory warning(s) — review before saving",
        expanded=True,
    ):
        for msg in st.session_state.warnings:
            st.warning(f"• {msg}")


def _render_char_counter() -> None:
    """Display a live character counter tied to the current text area content."""
    char_count = len(st.session_state.summary_text)
    within_limit = char_count <= _CHAR_LIMIT

    if within_limit:
        st.caption(f"✓ {char_count:,} / {_CHAR_LIMIT:,} characters")
    else:
        st.warning(
            f"⚠ {char_count:,} characters — "
            f"{char_count - _CHAR_LIMIT:,} over the {_CHAR_LIMIT:,}-character limit."
        )


def _render_submit_section() -> None:
    """Render the Submit & Save button.

    The button is disabled when:
    - The summary text area is blank.
    - Any Tier-1 blocking errors are present.

    On click, calls ``POST /api/v1/summaries`` and shows the outcome.
    """
    has_errors = bool(st.session_state.errors)
    summary_empty = not st.session_state.summary_text.strip()

    if has_errors:
        st.info("Fix the blocking errors above before you can submit.")

    clicked = st.button(
        "Submit & Save",
        type="primary",
        use_container_width=True,
        disabled=has_errors or summary_empty,
    )
    if not clicked:
        return

    with st.spinner("Saving summary…"):
        data = _post_submit(st.session_state.filename, st.session_state.summary_text)

    if data:
        st.success(f"✓ {data['message']} — {data['char_count']:,} characters saved.")
        if data.get("warnings"):
            with st.expander("Warnings on the saved summary", expanded=False):
                for w in data["warnings"]:
                    st.warning(f"• {w}")
        logger.info("Summary submitted: %s", data.get("output_filename"))


def _render_summary_editor() -> None:
    """Render the full review section: text area, char counter, guardrail findings,
    and the Submit button.

    The text area uses ``key='summary_text'`` so Streamlit keeps
    ``st.session_state.summary_text`` in sync with every keystroke.
    """
    st.subheader("Step 3 — Review & Submit")
    st.caption("Review the generated summary. You may edit it before saving.")

    st.text_area(
        "Summary",
        key="summary_text",
        height=400,
        label_visibility="collapsed",
    )

    _render_char_counter()
    _render_guardrail_errors()
    _render_guardrail_warnings()
    st.divider()
    _render_submit_section()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Configure the Streamlit page and render all UI sections."""
    st.set_page_config(
        page_title="Insurance Call Summariser",
        page_icon="📞",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    _init_session_state()

    st.title("📞 Insurance Call Summariser")
    st.markdown(
        "Upload a call transcript, generate a structured summary, "
        "review it with guardrail feedback, then save it to the output folder."
    )
    st.divider()

    uploaded_file = _render_upload_section()

    if (
        uploaded_file is not None
        and st.session_state.last_uploaded_filename != uploaded_file.name
    ):
        _reset_summary_state()
        st.session_state.last_uploaded_filename = uploaded_file.name

    st.divider()
    _render_generate_section(uploaded_file)

    if st.session_state.generated:
        st.divider()
        _render_summary_editor()


if __name__ == "__main__":
    main()
