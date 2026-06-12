"""Single-command launcher for the Call Summariser Agent.

Starts the FastAPI backend (uvicorn) and the Streamlit frontend as two
sub-processes and blocks until either exits or Ctrl+C is pressed, at which
point both processes are gracefully terminated.

LangSmith observability is activated automatically when LANGSMITH_TRACING=true
is set in .env.  load_dotenv() is called here — before Popen — so that both
child processes inherit the LangSmith environment variables from the start,
ensuring traces are captured from the very first LLM call in each process.

Usage::

    uv run run.py
"""

import os
import subprocess
import sys
import time

# Must be called before Popen so child processes inherit the LangSmith vars.
from dotenv import load_dotenv
load_dotenv()

_API_HOST = "127.0.0.1"
_API_PORT = 8000
_UI_PORT = 8501
_API_STARTUP_DELAY_SECONDS = 2  # give uvicorn time to bind before Streamlit starts


def _start_api_server() -> subprocess.Popen:
    """Start the FastAPI backend via uvicorn.

    Returns:
        The :class:`~subprocess.Popen` handle for the uvicorn process.
    """
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.app:app",
            "--host",
            _API_HOST,
            "--port",
            str(_API_PORT),
            "--reload",
        ]
    )


def _start_ui_server() -> subprocess.Popen:
    """Start the Streamlit frontend.

    Returns:
        The :class:`~subprocess.Popen` handle for the Streamlit process.
    """
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "ui/app.py",
            "--server.port",
            str(_UI_PORT),
            "--server.headless",
            "true",
        ]
    )


def _terminate_processes(*processes: subprocess.Popen) -> None:
    """Send SIGTERM to each process and wait up to 5 seconds for clean exit.

    Falls back to SIGKILL for any process that does not exit in time.

    Args:
        *processes: Any number of :class:`~subprocess.Popen` instances to stop.
    """
    for proc in processes:
        proc.terminate()

    for proc in processes:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _print_langsmith_status() -> None:
    """Print whether LangSmith tracing is active so the operator can confirm observability."""
    tracing = os.getenv("LANGSMITH_TRACING", "false").lower()
    if tracing in ("true", "1", "yes"):
        project = os.getenv("LANGSMITH_PROJECT", "<default>")
        endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        print(f"  LangSmith → ENABLED  project={project!r}  endpoint={endpoint}")
    else:
        print("  LangSmith → DISABLED (set LANGSMITH_TRACING=true in .env to enable)")


def main() -> None:
    """Launch both services and wait until interrupted."""
    print("\n  Starting Call Summariser Agent…")
    _print_langsmith_status()

    api_proc = _start_api_server()
    print(f"\n  FastAPI   → http://{_API_HOST}:{_API_PORT}")
    print(f"  API Docs  → http://{_API_HOST}:{_API_PORT}/docs")

    print(
        f"  Waiting {_API_STARTUP_DELAY_SECONDS}s for API to be ready…",
        flush=True,
    )
    time.sleep(_API_STARTUP_DELAY_SECONDS)

    ui_proc = _start_ui_server()
    print(f"  Streamlit → http://localhost:{_UI_PORT}")
    print("\n  Press Ctrl+C to stop all services.\n")

    try:
        api_proc.wait()
        ui_proc.wait()
    except KeyboardInterrupt:
        print("\n  Shutting down services…")
        _terminate_processes(api_proc, ui_proc)
        print("  Done.\n")


if __name__ == "__main__":
    main()
