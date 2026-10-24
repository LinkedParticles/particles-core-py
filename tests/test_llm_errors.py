# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The rate-limit half of ``particles.llm.errors``."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
import pytest

from particles.llm.errors import is_rate_limited, rate_limit_wait_seconds, retry_on_rate_limit


class _StatusError(Exception):
    """A provider error carrying ``status_code`` and, optionally, a response."""

    def __init__(self, status_code: int, headers: dict[str, str] | None = None) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.response = httpx.Response(status_code, headers=headers or {})


class TestIsRateLimited:
    def test_429_is_rate_limited(self) -> None:
        assert is_rate_limited(_StatusError(429))

    @pytest.mark.parametrize("status", [400, 401, 403, 500, 529])
    def test_other_statuses_are_not(self, status: int) -> None:
        assert not is_rate_limited(_StatusError(status))

    def test_an_error_without_a_status_is_not(self) -> None:
        assert not is_rate_limited(RuntimeError("connection reset"))

    def test_the_anthropic_sdk_error_matches(self) -> None:
        import anthropic

        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        exc = anthropic.RateLimitError(
            "rate limited", response=httpx.Response(429, request=request), body=None
        )
        assert is_rate_limited(exc)


class TestWaitSeconds:
    def test_retry_after_seconds_are_honoured(self) -> None:
        exc = _StatusError(429, {"retry-after": "7"})
        assert rate_limit_wait_seconds(exc, 0, backoff_seconds=10, max_wait_seconds=60) == 7.0

    def test_retry_after_is_capped(self) -> None:
        exc = _StatusError(429, {"retry-after": "600"})
        assert rate_limit_wait_seconds(exc, 0, backoff_seconds=10, max_wait_seconds=60) == 60.0

    def test_backoff_doubles_per_attempt_with_bounded_jitter(self) -> None:
        exc = _StatusError(429)
        for attempt in range(4):
            wait = rate_limit_wait_seconds(exc, attempt, backoff_seconds=2, max_wait_seconds=1000)
            assert 2 * 2**attempt <= wait <= 2 * 2**attempt * 1.25

    def test_an_http_date_falls_back_to_the_backoff(self) -> None:
        exc = _StatusError(429, {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
        wait = rate_limit_wait_seconds(exc, 0, backoff_seconds=3, max_wait_seconds=60)
        assert 3 <= wait <= 3.75


class TestRetryOnRateLimit:
    @staticmethod
    def _flaky(failures: list[BaseException]) -> tuple[Any, list[int]]:
        calls: list[int] = []

        async def call() -> str:
            calls.append(1)
            if failures:
                raise failures.pop(0)
            return "ok"

        return call, calls

    @pytest.mark.asyncio
    async def test_a_429_waits_and_retries(self) -> None:
        call, calls = self._flaky([_StatusError(429), _StatusError(429)])
        sleeps: list[float] = []

        async def _sleep(seconds: float) -> None:
            sleeps.append(seconds)

        with patch("particles.llm.errors.asyncio.sleep", _sleep):
            result = await retry_on_rate_limit(
                call, retries=4, backoff_seconds=1, max_wait_seconds=60, label="t"
            )
        assert result == "ok"
        assert len(calls) == 3
        assert len(sleeps) == 2

    @pytest.mark.asyncio
    async def test_the_last_429_propagates(self) -> None:
        call, calls = self._flaky([_StatusError(429)] * 3)
        with pytest.raises(_StatusError):
            await retry_on_rate_limit(
                call, retries=2, backoff_seconds=0, max_wait_seconds=0, label="t"
            )
        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_other_failures_are_not_retried(self) -> None:
        call, calls = self._flaky([_StatusError(500)])
        with pytest.raises(_StatusError):
            await retry_on_rate_limit(
                call, retries=4, backoff_seconds=0, max_wait_seconds=0, label="t"
            )
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_zero_retries_is_one_attempt(self) -> None:
        call, calls = self._flaky([_StatusError(429)])
        with pytest.raises(_StatusError):
            await retry_on_rate_limit(
                call, retries=0, backoff_seconds=0, max_wait_seconds=0, label="t"
            )
        assert len(calls) == 1
