"""LangSmith observability: trace-span wrappers for the pipeline stages.

Without explicit spans, LangSmith sees the entire pipeline as a single
undifferentiated block.  Decorating key functions with :func:`traceable`
creates nested child spans so you can drill into exactly which stage is slow
or produces bad output.

Span hierarchy produced
-----------------------
::

    [call_summarizer_pipeline]          ← top-level run
      ├── [validate_input]              ← input guardrails
      ├── [generate_summary]            ← LLM call + retries
      │     └── (groq.chat.completions) ← auto-traced by LangChain integration
      ├── [run_output_guardrails]       ← T1/T2/T3 checks
      ├── [evaluate_summary]            ← 8-metric scoring
      └── [save_summary]               ← file I/O (optional)

Usage
-----
Apply :data:`traceable` as a decorator on functions you want to appear as
distinct spans in the LangSmith UI::

    from call_summarizer.observability.tracing import traceable

    @traceable(name="my_span")
    def my_function(...):
        ...

When ``LANGSMITH_TRACING=false`` (or the environment variable is absent),
:data:`traceable` is a transparent no-op decorator so production code is
unaffected when tracing is disabled.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _tracing_enabled() -> bool:
    """Return ``True`` when LangSmith tracing is explicitly enabled via env var."""
    return os.getenv("LANGSMITH_TRACING", "false").lower() in ("true", "1", "yes")


def traceable(name: str | None = None, **kwargs: Any) -> Callable[[F], F]:
    """Wrap a function in a LangSmith trace span when tracing is enabled.

    When ``LANGSMITH_TRACING`` is ``true``, delegates to
    ``langsmith.traceable`` (which must be installed as part of
    ``langchain-core``).  When tracing is disabled, returns the original
    function unchanged — no LangSmith import is performed at all.

    Args:
        name: Human-readable span name shown in the LangSmith UI.  Defaults
            to the decorated function's ``__name__``.
        **kwargs: Additional keyword arguments forwarded to
            ``langsmith.traceable`` (e.g. ``run_type``, ``tags``).

    Returns:
        The decorated (or unchanged) callable.
    """
    def decorator(func: F) -> F:
        if not _tracing_enabled():
            return func

        try:
            from langsmith import traceable as ls_traceable  # type: ignore[import]
            span_name = name or func.__name__
            logger.debug("Registering LangSmith trace span: %s", span_name)
            return ls_traceable(name=span_name, **kwargs)(func)  # type: ignore[return-value]
        except ImportError:
            logger.warning(
                "langsmith package not installed — tracing disabled for %s",
                name or getattr(func, "__name__", "unknown"),
            )
            return func

    return decorator
