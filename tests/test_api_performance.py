"""API performance tests — latency percentiles and rate-limit enforcement.

All tests mock the LLM / service layer so they run **offline**, consume no
Groq quota, and produce deterministic results regardless of network conditions.

What is measured
----------------
Latency percentiles (sequential requests via TestClient):

    mean   — arithmetic mean of all response times
    p90    — 90th percentile  (90 % of requests finish at or below this)
    p95    — 95th percentile  (5 % of requests are slower)
    p99    — 99th percentile  (1 % of requests are slower)

Thresholds reflect **local mocked runs only** (no network, no LLM latency).
They capture FastAPI routing + middleware + (de)serialisation overhead.

    mean  <  100 ms
    p90   <  150 ms
    p95   <  200 ms
    p99   <  500 ms

Rate-limit behaviour (``SlidingWindowRateLimiter`` with a 5-req/60s cap):

    - Requests 1-5  → HTTP 200, X-RateLimit-Remaining counts down
    - Request  6    → HTTP 429, Retry-After header present
    - 429 response body contains "Rate limit exceeded"
"""

from __future__ import annotations

import io
import os
import statistics
import time
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from call_summarizer.models import GuardrailResult, ProcessingResult

# ── Constants ──────────────────────────────────────────────────────────────────

# Number of back-to-back requests used for the latency suite.
# Keep this well below the production rate limit (30 req/min) so the
# performance fixture never triggers a 429.
_N_REQUESTS: int = 20

# Latency thresholds (seconds) — mocked, synchronous TestClient only.
_THRESHOLD_MEAN: float = 0.100   # 100 ms
_THRESHOLD_P90: float  = 0.150   # 150 ms
_THRESHOLD_P95: float  = 0.200   # 200 ms
_THRESHOLD_P99: float  = 0.500   # 500 ms

# A minimal valid transcript that passes all input guardrails.
_SAMPLE_TRANSCRIPT: str = (
    "Agent: Good afternoon, this is Sarah at Bright Insurance. How can I help?\n"
    "Caller: Hi, I'm calling about claim reference CLM-2024-9871. I wanted to confirm "
    "the settlement of 4500 pounds has been processed to my IBAN GB29NWBK60161331926819.\n"
    "Agent: Let me pull that up. Yes, I can confirm the payment of £4,500.00 was "
    "processed on 14 January 2025 to that IBAN. Is there anything else?\n"
    "Caller: No, that's everything. Thank you.\n"
    "Agent: You're welcome. Have a good day.\n"
)

# A minimal valid summary returned by the mock — passes output guardrails.
_MOCK_SUMMARY: str = (
    "Caller: Inbound — John Smith (Policyholder)\n"
    "Subject: Settlement payment confirmation for motor claim CLM-2024-9871\n\n"
    "Executive Summary:\n"
    "Policyholder called to confirm receipt of the £4,500.00 settlement payment.\n"
    "- Settlement of £4,500.00 confirmed processed on 14 January 2025\n"
    "- Payment sent to IBAN GB29NWBK60161331926819\n"
    "- Claim reference CLM-2024-9871\n\n"
    "Next Steps:\n"
    "Bright Insurance: None\n"
    "Claimant: None\n"
)


# ── Shared helpers ────────────────────────────────────────────────────────────


def _make_processing_result(success: bool = True) -> ProcessingResult:
    """Build a :class:`~call_summarizer.models.ProcessingResult` for the mock."""
    gr = GuardrailResult(
        passed=True,
        findings=[],
        char_count=len(_MOCK_SUMMARY),
        char_within_limit=True,
    )
    return ProcessingResult(
        transcript_path=Path("<inline>"),
        output_path=None,
        success=success,
        summary=_MOCK_SUMMARY if success else "",
        error=None if success else "mocked LLM error",
        issues=[],
        guardrail_result=gr if success else None,
    )


def _transcript_file() -> tuple[str, bytes, str]:
    """Return a (filename, content, mime_type) triple for multipart upload."""
    return ("sample.txt", _SAMPLE_TRANSCRIPT.encode(), "text/plain")


def _percentile(data: list[float], pct: float) -> float:
    """Return the *pct*-th percentile of *data* using nearest-rank."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    rank = max(1, int(len(sorted_data) * pct / 100))
    return sorted_data[rank - 1]


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def latency_client() -> Generator[TestClient, None, None]:
    """TestClient backed by a fresh app with a high rate limit (no throttling).

    The service layer is patched so no real LLM calls are made.
    ``GROQ_API_KEY`` is set to a dummy value to satisfy ``load_config()``.
    """
    os.environ.setdefault("GROQ_API_KEY", "test-key-perf")

    from api.app import create_app

    # High limit so latency tests never hit 429.
    app = create_app(rate_limit_per_minute=500)

    mock_result = _make_processing_result(success=True)
    target = "api.routes.summarize.generate_summary_from_content"

    with patch(target, return_value=mock_result):
        with TestClient(app) as client:
            yield client


@pytest.fixture
def rate_limit_client() -> Generator[TestClient, None, None]:
    """Fresh TestClient with a 5-request cap for rate-limit assertions.

    A new app instance is created per test so the sliding-window counter
    starts empty — tests are independent of each other's request history.
    """
    os.environ.setdefault("GROQ_API_KEY", "test-key-ratelimit")

    from api.app import create_app

    app = create_app(rate_limit_per_minute=5)

    mock_result = _make_processing_result(success=True)
    target = "api.routes.summarize.generate_summary_from_content"

    with patch(target, return_value=mock_result):
        with TestClient(app) as client:
            yield client


# ── Latency helper ────────────────────────────────────────────────────────────


def _collect_latencies(client: TestClient, n: int) -> list[float]:
    """Send *n* sequential POST /summarize requests and return per-request latencies."""
    latencies: list[float] = []
    for _ in range(n):
        start = time.perf_counter()
        response = client.post(
            "/api/v1/summarize",
            files={"file": _transcript_file()},
        )
        elapsed = time.perf_counter() - start
        assert response.status_code == 200, (
            f"Expected 200 but got {response.status_code}: {response.text}"
        )
        latencies.append(elapsed)
    return latencies


# ── Test class: latency percentiles ──────────────────────────────────────────


class TestLatencyPercentiles:
    """Verify that sequential API response times stay within acceptable bounds.

    All requests are mocked (no LLM, no disk IO) so the measured latency
    reflects FastAPI routing, middleware, (de)serialisation, and validation
    overhead only.
    """

    @pytest.fixture(autouse=True)
    def _latencies(self, latency_client: TestClient) -> None:
        """Collect latencies once; individual test methods read the result."""
        self._data = _collect_latencies(latency_client, _N_REQUESTS)
        self._mean = statistics.mean(self._data)
        self._p90 = _percentile(self._data, 90)
        self._p95 = _percentile(self._data, 95)
        self._p99 = _percentile(self._data, 99)

    def test_sample_count(self) -> None:
        """Sanity check: all N requests completed."""
        assert len(self._data) == _N_REQUESTS

    def test_all_requests_succeeded(self) -> None:
        """Every request returned a non-zero latency (i.e. completed)."""
        assert all(t > 0 for t in self._data)

    def test_mean_latency(self) -> None:
        """Mean response time is below the threshold."""
        assert self._mean < _THRESHOLD_MEAN, (
            f"Mean latency {self._mean * 1000:.1f}ms exceeds "
            f"{_THRESHOLD_MEAN * 1000:.0f}ms threshold"
        )

    def test_p90_latency(self) -> None:
        """90th-percentile latency is below the threshold."""
        assert self._p90 < _THRESHOLD_P90, (
            f"p90 latency {self._p90 * 1000:.1f}ms exceeds "
            f"{_THRESHOLD_P90 * 1000:.0f}ms threshold"
        )

    def test_p95_latency(self) -> None:
        """95th-percentile latency is below the threshold."""
        assert self._p95 < _THRESHOLD_P95, (
            f"p95 latency {self._p95 * 1000:.1f}ms exceeds "
            f"{_THRESHOLD_P95 * 1000:.0f}ms threshold"
        )

    def test_p99_latency(self) -> None:
        """99th-percentile latency is below the threshold."""
        assert self._p99 < _THRESHOLD_P99, (
            f"p99 latency {self._p99 * 1000:.1f}ms exceeds "
            f"{_THRESHOLD_P99 * 1000:.0f}ms threshold"
        )

    def test_no_outliers_above_1s(self) -> None:
        """No individual request took more than 1 second (mocked, no IO)."""
        outliers = [t for t in self._data if t >= 1.0]
        assert not outliers, (
            f"{len(outliers)} request(s) exceeded 1 000ms: "
            f"{[f'{t * 1000:.0f}ms' for t in outliers]}"
        )

    def test_prints_summary(self, capsys) -> None:
        """Print a human-readable latency report for CI log visibility."""
        print(
            f"\n--- Latency report ({_N_REQUESTS} requests, mocked LLM) ---\n"
            f"  mean : {self._mean * 1000:6.1f} ms   (threshold < {_THRESHOLD_MEAN * 1000:.0f} ms)\n"
            f"  p90  : {self._p90  * 1000:6.1f} ms   (threshold < {_THRESHOLD_P90  * 1000:.0f} ms)\n"
            f"  p95  : {self._p95  * 1000:6.1f} ms   (threshold < {_THRESHOLD_P95  * 1000:.0f} ms)\n"
            f"  p99  : {self._p99  * 1000:6.1f} ms   (threshold < {_THRESHOLD_P99  * 1000:.0f} ms)\n"
            f"  min  : {min(self._data) * 1000:6.1f} ms\n"
            f"  max  : {max(self._data) * 1000:6.1f} ms\n"
        )
        captured = capsys.readouterr()
        assert "mean" in captured.out


# ── Test class: rate limit enforcement ───────────────────────────────────────


class TestRateLimitEnforcement:
    """Verify the SlidingWindowRateLimiter correctly enforces the per-minute cap."""

    def test_requests_within_limit_succeed(self, rate_limit_client: TestClient) -> None:
        """First 5 requests (equal to the cap) all return HTTP 200."""
        for i in range(5):
            resp = rate_limit_client.post(
                "/api/v1/summarize",
                files={"file": _transcript_file()},
            )
            assert resp.status_code == 200, f"Request {i + 1} failed: {resp.text}"

    def test_request_beyond_limit_returns_429(self, rate_limit_client: TestClient) -> None:
        """The (cap + 1)th request within the window returns HTTP 429."""
        for _ in range(5):
            rate_limit_client.post(
                "/api/v1/summarize",
                files={"file": _transcript_file()},
            )
        resp = rate_limit_client.post(
            "/api/v1/summarize",
            files={"file": _transcript_file()},
        )
        assert resp.status_code == 429

    def test_429_response_has_retry_after_header(self, rate_limit_client: TestClient) -> None:
        """HTTP 429 response includes a ``Retry-After`` header with a positive integer."""
        for _ in range(5):
            rate_limit_client.post(
                "/api/v1/summarize",
                files={"file": _transcript_file()},
            )
        resp = rate_limit_client.post(
            "/api/v1/summarize",
            files={"file": _transcript_file()},
        )
        assert "Retry-After" in resp.headers
        retry_after = int(resp.headers["Retry-After"])
        assert retry_after > 0

    def test_429_body_contains_message(self, rate_limit_client: TestClient) -> None:
        """HTTP 429 response body mentions rate limiting."""
        for _ in range(5):
            rate_limit_client.post(
                "/api/v1/summarize",
                files={"file": _transcript_file()},
            )
        resp = rate_limit_client.post(
            "/api/v1/summarize",
            files={"file": _transcript_file()},
        )
        assert "Rate limit exceeded" in resp.text

    def test_429_includes_ratelimit_headers(self, rate_limit_client: TestClient) -> None:
        """HTTP 429 includes ``X-RateLimit-Limit`` and ``X-RateLimit-Remaining: 0``."""
        for _ in range(5):
            rate_limit_client.post(
                "/api/v1/summarize",
                files={"file": _transcript_file()},
            )
        resp = rate_limit_client.post(
            "/api/v1/summarize",
            files={"file": _transcript_file()},
        )
        assert resp.headers.get("X-RateLimit-Limit") == "5"
        assert resp.headers.get("X-RateLimit-Remaining") == "0"

    def test_successful_response_includes_ratelimit_headers(
        self, rate_limit_client: TestClient
    ) -> None:
        """Successful responses surface ``X-RateLimit-Remaining`` so clients can self-pace."""
        resp = rate_limit_client.post(
            "/api/v1/summarize",
            files={"file": _transcript_file()},
        )
        assert resp.status_code == 200
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers
        remaining = int(resp.headers["X-RateLimit-Remaining"])
        assert remaining == 4   # 5 cap − 1 used = 4 remaining

    def test_remaining_decrements_per_request(self, rate_limit_client: TestClient) -> None:
        """``X-RateLimit-Remaining`` decrements by 1 on each admitted request."""
        remainders = []
        for _ in range(5):
            resp = rate_limit_client.post(
                "/api/v1/summarize",
                files={"file": _transcript_file()},
            )
            assert resp.status_code == 200
            remainders.append(int(resp.headers["X-RateLimit-Remaining"]))
        # Should be [4, 3, 2, 1, 0]
        assert remainders == list(range(4, -1, -1))

    def test_non_summarize_paths_are_not_rate_limited(
        self, rate_limit_client: TestClient
    ) -> None:
        """Health check and other paths bypass the rate limiter entirely."""
        # Exhaust the limit on /summarize
        for _ in range(5):
            rate_limit_client.post(
                "/api/v1/summarize",
                files={"file": _transcript_file()},
            )
        # /health should still succeed regardless
        resp = rate_limit_client.get("/health")
        assert resp.status_code == 200


# ── Test class: LLM retry logic ───────────────────────────────────────────────


class TestLLMRetryBehaviour:
    """Verify that transient LLM errors are retried and permanent errors are not."""

    def test_transient_error_is_retried_and_eventually_succeeds(self) -> None:
        """A 429-style error on the first attempt is retried and the second succeeds."""
        from call_summarizer.summarizer import generate_summary, build_llm

        mock_llm = MagicMock()
        transient_exc = Exception("groq 429 rate_limit exceeded")
        success_response = MagicMock()
        success_response.content = _MOCK_SUMMARY

        # First call raises, second returns success.
        mock_llm.invoke.side_effect = [transient_exc, success_response]

        with patch("call_summarizer.summarizer.time.sleep"):  # skip the backoff wait
            result = generate_summary("test transcript", mock_llm)

        # generate_summary calls .strip() on the LLM response, so compare stripped.
        assert result == _MOCK_SUMMARY.strip()
        assert mock_llm.invoke.call_count == 2

    def test_non_retryable_error_raises_immediately(self) -> None:
        """A permanent error (e.g. bad auth) is not retried."""
        from call_summarizer.summarizer import generate_summary

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("401 invalid api key")

        with pytest.raises(RuntimeError, match="LLM call failed"):
            generate_summary("test transcript", mock_llm)

        # Should have been called only once — no retry for permanent errors.
        assert mock_llm.invoke.call_count == 1

    def test_all_retries_exhausted_raises_runtime_error(self) -> None:
        """When every retry attempt fails, RuntimeError is raised."""
        from call_summarizer.summarizer import generate_summary, _MAX_LLM_RETRIES

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("503 service unavailable")

        with patch("call_summarizer.summarizer.time.sleep"):
            with pytest.raises(RuntimeError, match="LLM call failed"):
                generate_summary("test transcript", mock_llm)

        # Should have been called max_retries + 1 times total.
        assert mock_llm.invoke.call_count == _MAX_LLM_RETRIES + 1

    def test_retry_sleeps_with_exponential_backoff(self) -> None:
        """Each retry sleeps longer than the previous one (exponential backoff)."""
        from call_summarizer.summarizer import generate_summary, _RETRY_BASE_DELAY

        mock_llm = MagicMock()
        transient_exc = Exception("timeout connection reset")
        success_response = MagicMock()
        success_response.content = _MOCK_SUMMARY
        # Fail twice, then succeed.
        mock_llm.invoke.side_effect = [transient_exc, transient_exc, success_response]

        sleep_calls: list[float] = []
        with patch(
            "call_summarizer.summarizer.time.sleep",
            side_effect=lambda t: sleep_calls.append(t),
        ):
            generate_summary("test transcript", mock_llm)

        assert len(sleep_calls) == 2, "Expected exactly 2 sleep calls for 2 retries"
        # Second sleep should be longer than the first (base doubles each attempt).
        assert sleep_calls[1] > sleep_calls[0], (
            f"Expected increasing backoff: {sleep_calls}"
        )
        # Both delays should be at least the base delay for their attempt.
        assert sleep_calls[0] >= _RETRY_BASE_DELAY * 1   # attempt 0: base * 2^0 = 1s
        assert sleep_calls[1] >= _RETRY_BASE_DELAY * 2   # attempt 1: base * 2^1 = 2s
