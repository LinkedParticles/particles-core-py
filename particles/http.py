"""Shared HTTP client and retry helpers for all outbound requests."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from particles.url_safety import ValidatingTransport

log = logging.getLogger(__name__)


@asynccontextmanager
async def particles_client(
    timeout: float | None = None,
    extra_headers: dict[str, str] | None = None,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Async context manager yielding a configured httpx.AsyncClient.

    All traffic routes through a :class:`~particles.url_safety.ValidatingTransport`,
    so the SSRF connect-time gate covers the initial request and
    every redirect hop — ``follow_redirects=True`` stays safe because the
    transport, not a manual loop, is what makes following safe.
    """
    from particles.config import get_config

    cfg = get_config().http
    headers = {"User-Agent": cfg.user_agent, **(extra_headers or {})}
    async with httpx.AsyncClient(
        headers=headers,
        follow_redirects=True,
        timeout=timeout if timeout is not None else cfg.timeout_seconds,
        transport=ValidatingTransport(),
    ) as client:
        yield client


class ResponseTooLarge(Exception):
    """Raised when a fetched body exceeds ``config.http.max_bytes``."""


class SourceFetchError(RuntimeError):
    """An outbound fetch of a deposit's source failed at the origin.

    Distinguishes an **expected, external** fetch failure — the origin returned
    an error response (e.g. an HTTP 4xx/5xx, a Cloudflare bot-wall) or the
    fetch otherwise could not complete — from a genuine SDK bug. It carries the
    upstream ``status_code`` when one could be determined, so the API layer can
    map it to a ``502 Bad Gateway`` with an actionable message (and log it
    without a stack trace) rather than an opaque ``400`` + traceback.

    Subclasses ``RuntimeError`` for backward compatibility: fetch helpers used
    to raise bare ``RuntimeError``, and any ``except RuntimeError`` still catches
    this.
    """

    def __init__(
        self, message: str, *, url: str | None = None, status_code: int | None = None
    ) -> None:
        super().__init__(message)
        self.url = url
        self.status_code = status_code


async def get_capped(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_bytes: int | None = None,
    **kwargs: Any,
) -> httpx.Response:
    """GET ``url``, enforcing a hard cap on the (decompressed) body size.

    httpx transparently decompresses ``gzip`` / ``deflate`` / ``br`` /
    ``zstd``, so a small compressed payload can expand without bound — a
    "compression bomb" that exhausts memory before the caller ever sees the
    bytes. This streams the decompressed body, keeping a running total and
    aborting once it exceeds ``max_bytes``. A declared ``Content-Length``
    over the cap is rejected before any body is read (a cheap fast-path; the
    real protection is the streaming total, because ``Content-Length`` counts
    *compressed* bytes when an encoding is applied).

    The returned response is fully read — ``.content`` is populated and the
    stream is closed — so call sites that already do ``resp.content`` /
    ``resp.headers`` / ``resp.raise_for_status()`` work unchanged.

    Args:
        client: An ``httpx.AsyncClient`` (typically from ``particles_client()``).
        url: Absolute URL to GET.
        max_bytes: Override the cap. ``None`` reads ``config.http.max_bytes``
            at call time (so ``reset_config()`` and tests take effect).
        **kwargs: Forwarded to ``client.stream("GET", ...)``.

    Returns:
        The fully-read ``httpx.Response``.

    Raises:
        ResponseTooLarge: The declared Content-Length or the streamed body
            exceeded ``max_bytes``.
    """
    if max_bytes is None:
        from particles.config import get_config

        max_bytes = get_config().http.max_bytes

    async with client.stream("GET", url, **kwargs) as resp:
        declared = resp.headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > max_bytes:
                    raise ResponseTooLarge(
                        f"Content-Length {declared} exceeds cap of {max_bytes} bytes: {url}"
                    )
            except ValueError:
                # Malformed header — ignore and rely on the streaming total.
                pass

        chunks: list[bytes] = []
        total = 0
        async for chunk in resp.aiter_bytes():
            total += len(chunk)
            if total > max_bytes:
                raise ResponseTooLarge(
                    f"Response body exceeded cap of {max_bytes} bytes (decompressed): {url}"
                )
            chunks.append(chunk)
        # Mark the streamed response as read so ``.content`` is available
        # after the stream context closes.
        resp._content = b"".join(chunks)
    return resp


# Default transient codes worth retrying. 502/503/504 are upstream/load-balancer
# edge errors that routinely resolve within seconds. 500 is treated as
# permanent (likely a real server bug, not a transient hop failure).
DEFAULT_RETRY_STATUSES: frozenset[int] = frozenset({502, 503, 504})
# Backoff seconds between attempts. Total attempts = 1 + len(DEFAULT_RETRY_BACKOFFS).
DEFAULT_RETRY_BACKOFFS: tuple[float, ...] = (1.0, 2.0)


class TransientHttpError(Exception):
    """Raised by ``get_with_retry`` when every attempt landed in the retry set."""


async def get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    retry_statuses: frozenset[int] | None = None,
    backoffs: tuple[float, ...] | None = None,
    label: str = "HTTP",
    max_bytes: int | None = None,
    **kwargs: Any,
) -> httpx.Response:
    """GET ``url`` with retry on transient status codes, body size-capped.

    The default retry set (502/503/504) covers most upstream edge errors —
    GitHub API hiccups, Cloudflare-fronted services, etc. After
    ``len(backoffs)`` retries (a total of ``1 + len(backoffs)`` attempts),
    if the final response is still in ``retry_statuses``,
    ``TransientHttpError`` is raised.

    Every attempt routes through :func:`get_capped`, so the decompressed body
    is bounded by ``max_bytes`` (default ``config.http.max_bytes``) on the
    success path *and* on each retried error response — a compression bomb is
    rejected with :class:`ResponseTooLarge` rather than buffered unbounded.

    Non-transient codes (incl. 4xx and 500) are returned as-is so the caller
    can apply its own error mapping or ``raise_for_status()``. Network
    exceptions are not retried — they surface as the underlying httpx
    exception.

    Args:
        client: An ``httpx.AsyncClient`` (typically from ``particles_client()``).
        url: Absolute URL to GET.
        retry_statuses: Response codes that trigger a retry. ``None`` uses
            ``DEFAULT_RETRY_STATUSES`` (read at call time so tests may
            monkeypatch the module attribute).
        backoffs: Sleep seconds between attempts; ``len(backoffs)`` retries
            occur. ``None`` uses ``DEFAULT_RETRY_BACKOFFS`` (read at call time).
        label: Short prefix used in log lines and the raised error message
            (e.g. ``"GitHub API"``).
        max_bytes: Body cap forwarded to :func:`get_capped`. ``None`` reads
            ``config.http.max_bytes`` at call time.
        **kwargs: Forwarded to :func:`get_capped` (and thence to
            ``client.stream("GET", ...)``).

    Returns:
        The successful (non-retry-status) response, fully read.

    Raises:
        TransientHttpError: All attempts returned a status in ``retry_statuses``.
        ResponseTooLarge: An attempt's body exceeded ``max_bytes``.
    """
    if retry_statuses is None:
        retry_statuses = DEFAULT_RETRY_STATUSES
    if backoffs is None:
        backoffs = DEFAULT_RETRY_BACKOFFS

    log.debug("%s GET %s", label, url)
    resp = await get_capped(client, url, max_bytes=max_bytes, **kwargs)
    log.debug("%s ← %d %s", label, resp.status_code, url)
    for delay in backoffs:
        if resp.status_code not in retry_statuses:
            return resp
        log.warning(
            "%s transient %d for %s; retrying in %.1fs",
            label,
            resp.status_code,
            url,
            delay,
        )
        await asyncio.sleep(delay)
        resp = await get_capped(client, url, max_bytes=max_bytes, **kwargs)
        log.debug("%s ← %d %s (retry)", label, resp.status_code, url)
    if resp.status_code in retry_statuses:
        raise TransientHttpError(
            f"{label} unavailable ({resp.status_code}) after {len(backoffs) + 1} attempts: {url}"
        )
    return resp
