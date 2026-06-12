"""FastAPI application factory.

Creates and configures the ``app`` instance used by uvicorn.  Configuration is
loaded once during the lifespan startup event so every request handler can
access it via ``request.app.state.config`` without re-reading the environment.

Usage (via run.py or directly)::

    uvicorn api.app:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from call_summarizer.config import load_config
from call_summarizer.logging_config import setup_logging

from .middleware.rate_limiter import SlidingWindowRateLimiter, _DEFAULT_MAX_REQUESTS
from .routes.summarize import router as summarize_router

logger = logging.getLogger(__name__)

_STREAMLIT_ORIGIN = "http://localhost:8501"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan context: load config on startup, log on shutdown.

    Args:
        app: The FastAPI application instance whose ``state`` is populated.

    Raises:
        EnvironmentError: Re-raised from :func:`~call_summarizer.config.load_config`
            if ``GROQ_API_KEY`` is missing, so uvicorn exits with a clear message
            rather than failing silently on the first request.
    """
    setup_logging()
    logger.info("FastAPI startup — loading configuration")
    try:
        app.state.config = load_config()
    except EnvironmentError as exc:
        logger.critical("Startup aborted — configuration error: %s", exc)
        raise

    logger.info("FastAPI ready")
    yield
    logger.info("FastAPI shutdown")


def create_app(rate_limit_per_minute: int = _DEFAULT_MAX_REQUESTS) -> FastAPI:
    """Construct and return a configured :class:`~fastapi.FastAPI` instance.

    Registers:
    - Sliding-window rate limiter (default: 30 req/min matching Groq's RPM cap).
    - CORS middleware allowing the Streamlit frontend origin.
    - The ``/api/v1`` router with summarise and submit endpoints.

    Args:
        rate_limit_per_minute: Maximum requests per 60-second window for the
            ``/api/v1/summarize`` endpoint.  Override in tests to use a low
            limit without changing the module-level default.

    Returns:
        A fully configured :class:`~fastapi.FastAPI` application.
    """
    app = FastAPI(
        title="Call Summariser API",
        description=(
            "Insurance call transcript summarisation service. "
            "Upload a transcript to generate a summary, then submit to save it."
        ),
        version="1.0.0",
        lifespan=_lifespan,
    )

    # Rate limiter must be added before CORS so it can short-circuit before
    # the CORS preflight headers are evaluated on rejected requests.
    app.add_middleware(SlidingWindowRateLimiter, max_requests=rate_limit_per_minute)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[_STREAMLIT_ORIGIN],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(summarize_router, prefix="/api/v1", tags=["Summaries"])

    logger.debug(
        "FastAPI app created — router prefix /api/v1, rate limit %d req/min",
        rate_limit_per_minute,
    )
    return app


app = create_app()
