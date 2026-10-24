# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Account-level LLM failure classification — the shared predicate.

An *account-level* failure fails **every** call until an operator fixes the
account: a bad or missing key (401), no permission (403), or an out-of-credits
billing error (a 400 whose message names a credit-balance / billing problem).
A *per-call* failure — one malformed prompt, a transient 429 / 5xx, a network
blip — does not.

The distinction has two consumers with different needs, which is why the
predicate lives here (the Client-layer ``llm`` package) rather than in either
of them:

* ``particles.operations._llm`` (Engine) trips the circuit breaker on
  it, so the semantic seam stops probing a dead API.
* ``particles.extraction`` (Client) raises :class:`AccountLevelLLMError` on it,
  so a bulk run stops instead of repeating the same failure per snapshot.

Before this module the predicate existed only in the first, so extraction —
which cannot import Engine code — had no way to tell "your key has no credit"
from "this one page confused the model", and ``extract --all-pending`` walked
every pending snapshot re-issuing a request that could not succeed.

A rate limit (HTTP 429) is the third kind: per-call, and certain to clear with
time. :func:`retry_on_rate_limit` waits it out at the extraction seam rather
than letting it fail the snapshot.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


class AccountLevelLLMError(RuntimeError):
    """An LLM failure that will recur until the operator fixes the account.

    Raised by the extraction seam so a bulk caller aborts rather than repeating
    a doomed call per snapshot. Deliberately **not** caught as a transient
    extraction error: the snapshot's content is fine, so the pipeline's
    interrupt handler resets it ``IN_PROGRESS → PENDING`` and re-raises, leaving
    the work queued for a retry once the account is fixed.
    """

    def __init__(self, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.cause = cause


def is_account_level_failure(exc: BaseException) -> bool:
    """True if ``exc`` is an account-level LLM failure that will recur.

    Duck-typed on ``status_code`` plus the message, so it is not tied to one
    provider's SDK — a local endpoint's 401 / 403 trips it too. A per-call 400
    (one malformed prompt) and a transient 429 / 5xx deliberately do not.
    """
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return True
    msg = str(exc).lower()
    return status == 400 and ("credit balance" in msg or "billing" in msg)


def describe_account_level_failure(exc: BaseException) -> str:
    """A short, operator-readable cause class for an account-level failure.

    The raw exception text is a provider's JSON error body, useful in a log and
    illegible in a report headline. A surface that must say *why* the LLM was
    unavailable names the class instead: ``credit balance too low`` rather than
    ``Error code: 400 - {'type': 'error', …}``.
    """
    status = getattr(exc, "status_code", None)
    if status == 401:
        return "API key rejected"
    if status == 403:
        return "API key lacks permission"
    msg = str(exc).lower()
    if "credit balance" in msg:
        return "credit balance too low"
    if "billing" in msg:
        return "billing error"
    return "account-level error"


def is_rate_limited(exc: BaseException) -> bool:
    """True if ``exc`` is an HTTP 429, a rate limit that clears with time.

    Duck-typed on ``status_code`` like :func:`is_account_level_failure`, so the
    Anthropic SDK's ``RateLimitError`` and an OpenAI-compatible endpoint's 429
    both match. It is per-call, never account-level: the same request succeeds
    once the window resets.
    """
    return getattr(exc, "status_code", None) == 429


def rate_limit_wait_seconds(
    exc: BaseException, attempt: int, *, backoff_seconds: float, max_wait_seconds: float
) -> float:
    """How long to wait before retry ``attempt`` (0-based) of a rate-limited call.

    The reply's ``retry-after`` header when it carries a number of seconds,
    else ``backoff_seconds`` doubled per attempt with up to a quarter added at
    random, so calls limited together do not all return together. Never more
    than ``max_wait_seconds``.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    retry_after: str | None = None
    if headers is not None:
        try:
            retry_after = headers.get("retry-after")
        except Exception:  # noqa: BLE001 — a header we cannot read is a header we do not have
            retry_after = None
    if retry_after is not None:
        try:
            return min(max(float(retry_after), 0.0), max_wait_seconds)
        except ValueError:
            pass  # an HTTP date, not seconds: fall back to the backoff
    wait = backoff_seconds * float(2**attempt) * (1.0 + random.random() / 4)
    return min(wait, max_wait_seconds)


async def retry_on_rate_limit(
    call: Callable[[], Awaitable[T]],
    *,
    retries: int,
    backoff_seconds: float,
    max_wait_seconds: float,
    label: str,
) -> T:
    """Await ``call()``, sending it again after a wait each time it is rate limited.

    At most ``retries`` further attempts. Any other failure, and a
    429 after the last attempt, propagates unchanged, so a caller's handling of
    a failed call is the same as before.
    """
    attempt = 0
    while True:
        try:
            return await call()
        except Exception as exc:
            if attempt >= retries or not is_rate_limited(exc):
                raise
            wait = rate_limit_wait_seconds(
                exc, attempt, backoff_seconds=backoff_seconds, max_wait_seconds=max_wait_seconds
            )
            attempt += 1
            log.warning(
                "%s rate limited (HTTP 429); retry %d of %d in %.1f s",
                label,
                attempt,
                retries,
                wait,
            )
            await asyncio.sleep(wait)
