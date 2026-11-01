# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Shared HTTP client and retry helpers for all outbound requests."""

from __future__ import annotations

import asyncio
import logging
import ssl
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx

from particles import egress
from particles.url_safety import ValidatingTransport

log = logging.getLogger(__name__)


def tls_context() -> ssl.SSLContext:
    """The TLS context every client built here verifies with, writing no key log.

    httpx's default context comes from :func:`ssl.create_default_context`, which
    honours an inherited ``SSLKEYLOGFILE`` and appends each session's secrets to
    the file it names, so whoever reads that file decrypts the engine token or a
    provider key the connection carries (G15, SECURITY_REVIEW_20261027-1). The
    context is otherwise httpx's own: the same certificate authorities, and
    ``SSL_CERT_FILE`` / ``SSL_CERT_DIR`` still apply.
    """
    context = httpx.create_ssl_context()
    context.keylog_filename = None  # type: ignore[assignment]
    return context


def httpx2_tls_context() -> ssl.SSLContext:
    """httpx2's default TLS context, writing no key log.

    The ``httpx2`` clients (the Anthropic SDK's, the Hugging Face Hub's) verify
    with :func:`ssl.create_default_context` when ``SSL_CERT_FILE`` or
    ``SSL_CERT_DIR`` is set, which honours an inherited ``SSLKEYLOGFILE`` as
    httpx's does (security review 2026-10-28-1).
    """
    import httpx2

    context = httpx2.create_ssl_context()
    context.keylog_filename = None  # type: ignore[assignment]
    return context


def proxy_mounts_writing_no_key_log(
    proxies: Mapping[str, str | None], **transport_kwargs: Any
) -> dict[str, Any]:
    """``httpx2`` mounts for the environment's proxies whose TLS writes no key log.

    ``proxies`` maps an httpx mount pattern to a proxy URL, as the client
    would build it from the environment (``None`` entries, the ``NO_PROXY``
    exemptions, are left to the client). A client's own environment mounts
    give the hop to an ``https://`` proxy no TLS context, and httpcore then
    opens it with :func:`ssl.create_default_context` under ``SSL_CERT_FILE`` or
    ``SSL_CERT_DIR``, so the proxy credentials and the ``CONNECT`` target
    reached the key log (security review 2026-10-30-1, E28). Mounts a caller
    passes replace the client's; each here verifies the proxy and the tunnelled
    session with :func:`httpx2_tls_context`. ``transport_kwargs`` go to each
    ``httpx2.HTTPTransport``.
    """
    import httpx2

    return {
        key: httpx2.HTTPTransport(
            proxy=httpx2.Proxy(url, ssl_context=httpx2_tls_context()),
            verify=httpx2_tls_context(),
            **transport_kwargs,
        )
        for key, url in proxies.items()
        if url is not None
    }


def hub_client() -> Any:
    """The ``httpx2.Client`` the Hugging Face Hub downloads the encoder through.

    ``huggingface_hub``'s default client is the same apart from its TLS
    contexts: the session with the Hub and the hop to an ``https://`` proxy
    write no key log here (security review 2026-10-30-1, the E28 class).
    Installed with ``huggingface_hub.set_client_factory`` before the load.
    """
    import httpx2
    from httpx2._utils import get_environment_proxies
    from huggingface_hub.utils._http import hf_request_event_hook

    return httpx2.Client(
        event_hooks={"request": [hf_request_event_hook]},
        follow_redirects=True,
        timeout=None,
        verify=httpx2_tls_context(),
        mounts=proxy_mounts_writing_no_key_log(get_environment_proxies()),
    )


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

    httpx drops ``Authorization`` when a redirect leaves the request's origin,
    and nothing else. Every other header the first request of a chain carries,
    from ``extra_headers`` or passed on the request itself (an API key header,
    say), is dropped on such a hop too, unless it is one of the
    content-negotiation and conditional headers that carry no credential or
    one httpx sets per hop (security review 2026-10-25-2, R39; 2026-10-26-1).
    """
    from particles.config import get_config

    cfg = get_config().http
    headers = {"User-Agent": cfg.user_agent, **(extra_headers or {})}
    async with httpx.AsyncClient(
        headers=headers,
        follow_redirects=True,
        timeout=timeout if timeout is not None else cfg.timeout_seconds,
        transport=ValidatingTransport(verify=tls_context()),
        event_hooks={"request": [_origin_bound_headers_hook()]},
    ) as client:
        yield client


#: Caller headers :func:`particles_client` keeps on a cross-origin redirect hop.
_NON_CREDENTIAL_HEADERS = frozenset(
    {
        "accept",
        "accept-language",
        "content-type",
        "if-none-match",
        "if-modified-since",
        "user-agent",
    }
)

#: Headers httpx itself sets or rebuilds on every hop: the cookie jar decides
#: ``Cookie`` per destination, and the rest describe the connection or body.
_PER_HOP_HEADERS = frozenset(
    {
        "accept-encoding",
        "connection",
        "content-length",
        "cookie",
        "host",
        "transfer-encoding",
    }
)

#: The ``request.extensions`` key that remembers where a redirect chain began.
#: httpx hands a redirect hop the same ``extensions`` mapping as the request
#: it follows, so the first hop's origin survives the whole chain.
_CHAIN_ORIGIN = "particles.chain_origin"
#: The ``request.extensions`` key holding the header names bound to that origin.
_CHAIN_BOUND = "particles.chain_bound"


def _origin(url: httpx.URL) -> tuple[str, str, int | None]:
    port = url.port or {"http": 80, "https": 443}.get(url.scheme)
    return (url.scheme, url.host, port)


def _same_origin(first: tuple[str, str, int | None], url: httpx.URL) -> bool:
    """Whether ``url`` keeps the chain's origin, an http-to-https upgrade included.

    The upgrade is the one exception httpx makes for ``Authorization`` too.
    """
    here = _origin(url)
    if here == first:
        return True
    return first == ("http", here[1], 80) and here == ("https", first[1], 443)


def _origin_bound_headers_hook() -> Any:
    """A request hook that keeps the first hop's credential headers on its origin.

    On a chain's first request it binds the name of every header that request
    carries, except the non-credential and per-hop ones, and drops those names
    from every later hop off the chain's origin.
    """

    async def hook(request: httpx.Request) -> None:
        first = request.extensions.get(_CHAIN_ORIGIN)
        if first is None:
            request.extensions[_CHAIN_ORIGIN] = _origin(request.url)
            request.extensions[_CHAIN_BOUND] = (
                frozenset(name.lower() for name in request.headers)
                - _NON_CREDENTIAL_HEADERS
                - _PER_HOP_HEADERS
            )
            return
        if not _same_origin(first, request.url):
            for name in request.extensions.get(_CHAIN_BOUND, ()):
                request.headers.pop(name, None)

    return hook


def _egress_check(request: httpx.Request, purpose: str) -> None:
    port = request.url.port or egress.host_port(str(request.url))[1]
    egress.check(request.url.host, port, purpose)


class EgressTransport(httpx.AsyncHTTPTransport):
    """Async transport that enforces the no-egress posture and nothing else.

    For the clients that must not use :class:`ValidatingTransport`, because
    their destination is operator-configured and may be loopback or a LAN host:
    the OpenAI-compatible adapter and the remote-engine client.
    Every request, including each redirect hop, is checked against
    ``network.egress`` before it connects; no SSRF blocklist applies.
    """

    def __init__(self, *, purpose: str = "connection", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._purpose = purpose

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        _egress_check(request, self._purpose)
        return await super().handle_async_request(request)


def egress_request_hook(purpose: str) -> Callable[[Any], None]:
    """A request event hook that enforces the posture on a client we do not build.

    The Anthropic SDK runs on ``httpx2`` and refuses an ``httpx`` transport, and
    any custom transport would drop its connection limits, keepalive options and
    proxy mounts. Its own ``anthropic.DefaultHttpxClient`` takes
    ``event_hooks`` unchanged, and a request hook runs before every request and
    redirect hop, so the check rides that instead. The request is
    duck-typed: ``httpx`` and ``httpx2`` share the ``url.host`` / ``url.port`` /
    ``url.scheme`` shape.
    """

    def _hook(request: Any) -> None:
        url = request.url
        egress.check(url.host, url.port or egress.host_port(str(url))[1], purpose)

    return _hook


def egress_client(*, purpose: str = "connection", **kwargs: Any) -> httpx.AsyncClient:
    """An ``httpx.AsyncClient`` for an operator-configured endpoint, egress-checked.

    The one constructor for every async client outside :func:`particles_client`:
    ``tests/test_egress_chokepoint.py`` fails on an
    ``httpx.AsyncClient(`` / ``httpx.Client(`` anywhere else in the package.
    ``kwargs`` go to ``httpx.AsyncClient`` unchanged.
    """
    return httpx.AsyncClient(
        transport=EgressTransport(purpose=purpose, verify=tls_context()), **kwargs
    )


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
