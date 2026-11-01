# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Shared Anthropic SDK client (the ``anthropic`` adapter's transport).

The ``AnthropicProvider`` adapter (``particles/llm/adapters/anthropic.py``) is
the only production caller of ``get_client()``; every completion call site
reaches the model through the :class:`CompletionProvider` port, not
this client directly. ``ANTHROPIC_API_KEY`` is read at first use and the
client is cached for the process lifetime.

Tests override the client via ``set_client()``; passing ``None`` clears the
cache so the next ``get_client()`` rebuilds against the current environment.
This remains the canonical Anthropic mocking seam — patching it reaches every
purpose that resolves to the ``anthropic`` provider. See ``tests/AGENTS.md``.
"""

from __future__ import annotations

import logging
import socket
import ssl
import sys
from functools import cache

import anthropic
from anthropic._utils._httpx import get_environment_proxies

from particles.config import get_config
from particles.http import egress_request_hook, httpx2_tls_context, proxy_mounts_writing_no_key_log
from particles.llm.registry import PlaintextKeyRefused
from particles.secrets import get_anthropic_api_key, get_anthropic_base_url_optional
from particles.url_safety import is_cleartext_remote, url_without_userinfo

log = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


@cache
def _warn_plaintext_key_once(base_url: str) -> None:
    """One warning per URL when ``llm.anthropic_allow_plaintext_key`` lets the key through."""
    log.warning(
        "The Anthropic API key is being sent over an unencrypted connection to %s "
        "(llm.anthropic_allow_plaintext_key is set). Anyone on the network path "
        "can read it.",
        url_without_userinfo(base_url),
    )


def _check_plaintext_base_url() -> None:
    """Refuse the key for a cleartext, non-loopback ``ANTHROPIC_BASE_URL`` (Q14).

    The rule the OpenAI-compatible adapter applies to an entry's key (N12) and
    the engine client to its token: a no-op for ``https://``, for
    a loopback host, and when the variable is unset (the SDK's https default).
    Otherwise raises :class:`PlaintextKeyRefused` before the client is built,
    unless the operator set ``llm.anthropic_allow_plaintext_key``, in which case
    it warns once.
    """
    base_url = get_anthropic_base_url_optional()
    if base_url is None or not is_cleartext_remote(base_url):
        return
    if get_config().llm.anthropic_allow_plaintext_key:
        _warn_plaintext_key_once(base_url)
        return
    raise PlaintextKeyRefused(
        f"Refusing to send the Anthropic API key to {url_without_userinfo(base_url)} "
        "over unencrypted "
        "http:// (ANTHROPIC_BASE_URL). Use https://, or reach the gateway through "
        "a loopback hop (an SSH or Tailscale forward to http://127.0.0.1). To send "
        "it anyway on a network you trust end to end, set "
        "llm.anthropic_allow_plaintext_key: true."
    )


def get_client() -> anthropic.Anthropic:
    """Return the process-level Anthropic client, creating it on first use.

    Raises ``ValueError`` if ``ANTHROPIC_API_KEY`` is unset and no client has
    been injected via ``set_client()``. Without this pre-flight check the
    Anthropic SDK fails much later with the cryptic message
    "Could not resolve authentication method…". Raises
    :class:`PlaintextKeyRefused` (also a ``ValueError``) if
    ``ANTHROPIC_BASE_URL`` would carry the key over cleartext to a remote host.
    """
    global _client
    if _client is None:
        _check_plaintext_base_url()
        # The SDK's own client factory, so every SDK default (timeout, limits,
        # keepalive, proxies) stands; the request hook adds the no-egress check
        # on every request. Redirects are not followed: anthropic
        # 1.x sends the key as ``X-Api-Key``, which httpx's cross-origin strip
        # leaves in place, so a 3xx from the endpoint would carry it to
        # whatever host the redirect names, over http included (P9). anthropic
        # 2.x sends ``Authorization: Bearer``, which the strip removes, but a
        # custom header from ``ANTHROPIC_CUSTOM_HEADERS`` would still follow, so
        # the rule holds on every major. The API never redirects, so a 3xx
        # surfaces as an error instead.
        #
        # The SDK's own spans take their options from the environment
        # (``ANTHROPIC_OPEN_TELEMETRY_*``) when the client is given none, and
        # content mode would put prompts, which carry stored and harvested
        # text, on spans Particles' exporter ships. A complete option ignores
        # the environment: metadata only, and no trace context sent to the
        # provider (security review 2026-10-26-1).
        #
        # With ``SSL_CERT_FILE`` or ``SSL_CERT_DIR`` set, httpx2 verifies with
        # ``ssl.create_default_context``, which honours an inherited
        # ``SSLKEYLOGFILE`` and would write each session's secrets, the key's
        # included, to the file it names; the context passed here is httpx2's
        # own with the key log turned off (security review 2026-10-28-1). The
        # SDK builds its environment proxy mounts with no context for the hop
        # to an ``https://`` proxy, which then falls back to that same default,
        # so the mounts passed here, which replace the SDK's, carry the
        # key-log-free context for both halves (2026-10-30-1, E28).
        _client = anthropic.Anthropic(
            api_key=get_anthropic_api_key(),
            open_telemetry={"traces": {"content_mode": "metadata_only"}, "propagation": False},
            http_client=anthropic.DefaultHttpxClient(
                follow_redirects=False,
                verify=_tls_context(),
                mounts=proxy_mounts_writing_no_key_log(
                    get_environment_proxies(), socket_options=_keepalive_socket_options()
                ),
                event_hooks={"request": [egress_request_hook("llm purpose (anthropic)")]},
            ),
        )
    return _client


def _tls_context() -> ssl.SSLContext:
    """httpx2's default TLS context for the SDK's pool, writing no key log."""
    return httpx2_tls_context()


def _keepalive_socket_options() -> list[tuple[int, int, int | bool]]:
    """The TCP keepalive options the SDK gives its own transports.

    A proxy mount passed in replaces the SDK's, so it carries the same options
    (``DefaultHttpxClient``): a long request through the proxy stays alive.
    """
    options: list[tuple[int, int, int | bool]] = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, True)]
    interval = getattr(socket, "TCP_KEEPINTVL", None)
    if interval is not None:
        options.append((socket.IPPROTO_TCP, interval, 60))
    elif sys.platform == "darwin":
        options.append((socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPALIVE", 0x10), 60))
    count = getattr(socket, "TCP_KEEPCNT", None)
    if count is not None:
        options.append((socket.IPPROTO_TCP, count, 5))
    idle = getattr(socket, "TCP_KEEPIDLE", None)
    if idle is not None:
        options.append((socket.IPPROTO_TCP, idle, 60))
    return options


def sent_credentials() -> list[str | None]:
    """The credentials the cached client sends, for redacting an error it raised.

    The SDK also picks up ``ANTHROPIC_AUTH_TOKEN`` and the headers in
    ``ANTHROPIC_CUSTOM_HEADERS`` on its own, so an error is scrubbed of whatever
    the client actually sent (the Q13 residual): its key, its token, and each
    custom header's value and every word of it, so an echo of a bare token
    from ``Bearer <token>`` is caught too (security review 2026-10-25-2, R9).
    Empty when no client is built, or when an injected test double holds no
    strings.
    """
    client = _client
    if client is None:
        return []
    values: list[object] = [getattr(client, "api_key", None), getattr(client, "auth_token", None)]
    custom = getattr(client, "_custom_headers", None)
    if isinstance(custom, dict):
        for value in custom.values():
            if isinstance(value, str):
                values.append(value)
                values.extend(value.split())
    return [v for v in values if isinstance(v, str)]


def set_client(client: anthropic.Anthropic | None) -> None:
    """Override or clear the cached client.

    Used by tests; passing ``None`` clears the cache so the next ``get_client()``
    rebuilds. See ``tests/AGENTS.md`` for the mocking pattern.
    """
    global _client
    _client = client
