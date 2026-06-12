"""Sliding-window rate limiter middleware for the Call Summariser API.

Why this exists
---------------
Groq's ``llama-3.1-8b-instant`` model is capped at **30 requests per minute**
(RPM).  Without an HTTP-layer guard, a client can trivially send a burst that
exhausts the per-minute quota, causing every subsequent request to fail with a
cryptic ``RuntimeError`` wrapping a Groq 429 response.  This middleware
intercepts requests to ``/api/v1/summarize`` *before* the route handler runs and
returns a proper HTTP 429 with ``Retry-After`` headers, giving clients the
information they need to back off gracefully.

Algorithm — sliding window
--------------------------
A ``deque`` stores the monotonic timestamp of every admitted request.  On each
new request:

1. Evict all timestamps older than ``window_seconds`` from the left of the deque
   (O(evicted), amortised O(1)).
2. If ``len(deque) >= max_requests``, reject with HTTP 429.
3. Otherwise append the current timestamp and admit the request.

This is an approximation of a true sliding-window counter — it gives an exact
count within the current window rather than the leaky-bucket smoothing of a
token bucket.  That matches Groq's own rate-limit semantics.

Thread / async safety
---------------------
FastAPI runs on a single asyncio event loop.  Because the deque operations are
all synchronous (no ``await`` between them), they execute atomically between
event loop ticks — no ``asyncio.Lock`` is required.
"""

import collections
import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Groq llama-3.1-8b-instant hard limits:
#   30 requests per minute (RPM) — enforced here at the HTTP layer
#   6,000 tokens per minute (TPM) — enforced by input_guardrails.py (Tier 1)
_DEFAULT_MAX_REQUESTS: int = 30
_DEFAULT_WINDOW_SECONDS: int = 60

# Only rate-limit the LLM-calling endpoint — health and save endpoints are cheap.
_RATE_LIMITED_PATHS: frozenset[str] = frozenset({"/api/v1/summarize"})


class SlidingWindowRateLimiter(BaseHTTPMiddleware):
    """Per-server sliding-window rate limiter for Groq RPM quota protection.

    Tracks all admitted requests to ``/api/v1/summarize`` within a rolling
    ``window_seconds`` window.  When the window is full the request is rejected
    with HTTP 429 and ``Retry-After``, ``X-RateLimit-*`` headers.

    This is a *server-wide* counter — all callers share a single quota bucket,
    which mirrors Groq's own per-API-key quota.  For multi-tenant deployments
    key the counter by ``request.client.host`` instead.

    Args:
        app: The ASGI application to wrap.
        max_requests: Maximum requests to admit within the window.
            Defaults to ``30`` (Groq RPM limit).
        window_seconds: Duration of the sliding window in seconds.
            Defaults to ``60``.
    """

    def __init__(
        self,
        app: ASGIApp,
        max_requests: int = _DEFAULT_MAX_REQUESTS,
        window_seconds: int = _DEFAULT_WINDOW_SECONDS,
    ) -> None:
        super().__init__(app)
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._timestamps: collections.deque[float] = collections.deque()
        logger.info(
            "Rate limiter active — %d req / %ds window on %s",
            max_requests,
            window_seconds,
            ", ".join(sorted(_RATE_LIMITED_PATHS)),
        )

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path not in _RATE_LIMITED_PATHS:
            return await call_next(request)

        now = time.monotonic()
        cutoff = now - self._window_seconds

        # Evict timestamps that have slid out of the current window.
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

        current_count = len(self._timestamps)
        remaining = self._max_requests - current_count

        if remaining <= 0:
            # Calculate seconds until the oldest in-window request ages out.
            retry_after = max(1, int(self._timestamps[0] + self._window_seconds - now) + 1)
            logger.warning(
                "Rate limit exceeded — %d/%d requests in %ds window, retry after %ds",
                current_count,
                self._max_requests,
                self._window_seconds,
                retry_after,
            )
            return Response(
                content=(
                    f'{{"detail": "Rate limit exceeded. '
                    f"Maximum {self._max_requests} requests per "
                    f"{self._window_seconds}s. "
                    f'Retry after {retry_after}s."}}'
                ),
                status_code=429,
                headers={
                    "Content-Type": "application/json",
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self._max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(retry_after),
                },
            )

        self._timestamps.append(now)
        response = await call_next(request)

        # Surface quota state on every admitted response so clients can pace themselves.
        response.headers["X-RateLimit-Limit"] = str(self._max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining - 1)
        response.headers["X-RateLimit-Reset"] = str(self._window_seconds)
        return response
