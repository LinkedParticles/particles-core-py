# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""SSRF protection for outbound URL fetches.

Defends against an attacker handing the system a URL that resolves to
internal infrastructure — cloud metadata endpoints (169.254.169.254),
loopback (127.0.0.1, ::1), private RFC 1918 ranges, link-local, RFC 6598
carrier-grade NAT, etc.

Three layers, sharing one blocklist (``_is_blocked_ip``):

* ``validate_fetch_url`` is the cheap pre-flight on the user-supplied URL —
  scheme allow-list, hostname presence, and a first DNS-resolution check. The
  CLI deposit flow and the FastAPI ``/corpus/deposit/url`` endpoint call it
  before a client is even built; ``corpus/fetch.py`` re-validates on refetch.
* ``ValidatingTransport`` is the authoritative connect-time gate.
  Wired inside ``particles_client``, it re-resolves and re-validates the host
  on **every** request httpx makes — the initial GET and each redirect hop —
  then connects to the vetted IP with the original ``Host`` header and TLS
  SNI preserved. Because the validated IP *is* the connected IP, this closes
  three gaps the pre-flight alone cannot: DNS-rebinding / TOCTOU (the
  pre-flight and httpx no longer resolve independently), redirect SSRF
  (``follow_redirects=True`` re-validates every hop, not just the first), and
  the ``urlparse``/httpx host-parser differential (the transport reads the
  authority httpx itself parsed).
* ``resolve_and_pin`` reaches the same guarantee for fetches that never enter
  ``httpx`` — the ``curl`` and ``git`` subprocesses. It performs
  the transport's first two steps (resolve, vet every address) and returns
  the vetted set instead of connecting, so the caller can hand it to
  ``curl --resolve`` / ``git -c http.curloptResolve``. Same resolver, same
  blocklist, same fail-closed rule; only the connecting differs.

Failure mode is fail-closed: an unresolvable hostname raises
``UnsafeUrlError`` rather than being allowed through.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import httpx

ALLOWED_SCHEMES = frozenset({"http", "https"})

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
Resolver = Callable[[str], list[IPAddress]]

# RFC 6598 carrier-grade NAT. ``ipaddress.is_private`` does not include these,
# yet in NAT / hosted environments they can reach internal infrastructure
# (security review F-4-ip). Block both the bare IPv4 range and its
# IPv4-mapped IPv6 form (``::ffff:100.64.0.0`` + the 10-bit prefix → /106).
_CGNAT_V4 = ipaddress.IPv4Network("100.64.0.0/10")
_CGNAT_V6_MAPPED = ipaddress.IPv6Network("::ffff:100.64.0.0/106")


class UnsafeUrlError(ValueError):
    """Raised when a URL is rejected by ``validate_fetch_url`` for safety reasons.

    Subclass of ValueError so existing ``except ValueError`` handlers in
    deposit endpoints (which already turn ValueError into HTTP 400) catch it.
    """


def validate_fetch_url(url: str) -> None:
    """Raise UnsafeUrlError if the URL should not be fetched.

    Rejects:
      - non-http(s) schemes (file://, ftp://, gopher://, …)
      - URLs with no hostname
      - URLs whose hostname resolves to a loopback, private, link-local,
        multicast, unspecified, CGNAT, or otherwise reserved IP

    DNS resolution failures are treated as failure-closed (unsafe).

    This is the cheap pre-flight on the user-supplied URL. The authoritative
    connect-time gate — which re-checks every redirect hop and pins the
    connection to the validated IP — is ``ValidatingTransport``.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise UnsafeUrlError(f"URL scheme {parsed.scheme!r} not allowed; use http or https")
    if not parsed.hostname:
        raise UnsafeUrlError(f"URL {url!r} has no hostname")

    for ip in _resolve_all(parsed.hostname):
        if _is_blocked_ip(ip):
            raise UnsafeUrlError(
                f"Refusing to fetch {parsed.hostname!r}: resolves to a "
                f"private/reserved address ({ip}). This protects against SSRF "
                "(cloud metadata, internal services, loopback)."
            )


def _resolve_all(hostname: str) -> list[IPAddress]:
    """Resolve ``hostname`` to all of its IPv4 + IPv6 addresses.

    If ``hostname`` is already an IP literal, return it directly without DNS.
    Raises ``UnsafeUrlError`` (fail-closed) if the hostname does not resolve.
    """
    try:
        return [ipaddress.ip_address(hostname)]
    except ValueError:
        pass  # not a literal IP; fall through to DNS

    try:
        addrinfo = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise UnsafeUrlError(
            f"Could not resolve hostname {hostname!r}: {exc}. Refusing to fetch (fail-closed)."
        ) from exc

    seen: set[str] = set()
    addrs: list[IPAddress] = []
    for _family, _type, _proto, _canon, sockaddr in addrinfo:
        ip_str = sockaddr[0]
        if not isinstance(ip_str, str) or ip_str in seen:
            continue
        seen.add(ip_str)
        try:
            addrs.append(ipaddress.ip_address(ip_str))
        except ValueError:
            continue  # skip junk results
    return addrs


def _is_blocked_ip(ip: IPAddress) -> bool:
    """Return True for IPs we refuse to fetch from."""
    if (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return True
    # RFC 6598 CGNAT — not covered by ``is_private`` (F-4-ip).
    if isinstance(ip, ipaddress.IPv4Address):
        return ip in _CGNAT_V4
    return ip in _CGNAT_V6_MAPPED


def resolve_and_pin(host: str, *, resolve: Resolver | None = None) -> list[IPAddress]:
    """Resolve ``host`` and return every address, all of them vetted.

    The subprocess-egress counterpart to :class:`ValidatingTransport`:
    it runs the transport's first two steps — resolve, then check every address
    against ``_is_blocked_ip`` — and hands the vetted set back instead of
    connecting, so a ``curl`` / ``git`` subprocess can be pinned to it. Sharing
    the resolver and the blocklist is the point: a fetch outside ``httpx`` is
    held to the same rule as one inside it.

    The **whole** vetted set is returned, not just the first address, so
    libcurl keeps its own failover across a multi-A-record host — every
    candidate has already passed the blocklist, so widening from one address to
    all cannot admit a blocked one.

    Raises:
        UnsafeUrlError: fail-closed, if the host does not resolve or if ANY
            resolved address is blocked. That all-or-nothing rule is the one
            ``validate_fetch_url`` already applies, and it is what stops a
            split A/AAAA answer smuggling an internal address past a public one.
    """
    resolver: Resolver = resolve if resolve is not None else _resolve_all
    addrs = resolver(host)
    if not addrs:
        raise UnsafeUrlError(
            f"Refusing to connect to {host!r}: hostname did not resolve (fail-closed)."
        )
    for ip in addrs:
        if _is_blocked_ip(ip):
            raise UnsafeUrlError(
                f"Refusing to connect to {host!r}: resolves to a private/reserved "
                f"address ({ip}). This protects against SSRF (cloud metadata, "
                "internal services, loopback) across DNS rebinding."
            )
    return addrs


def format_connect_pin(host: str, port: int, addrs: list[IPAddress]) -> str:
    """Render a ``<host>:<port>:<addr>[,<addr>…]`` connect-pin spec.

    The wire format shared by ``curl --resolve`` and libcurl's
    ``CURLOPT_RESOLVE`` (which git exposes as ``http.curloptResolve``), so both
    subprocess paths format their pin identically. IPv6 addresses are bracketed
    — the form curl documents — so the colons inside them can't be misread as
    field separators.
    """
    rendered = [f"[{ip}]" if isinstance(ip, ipaddress.IPv6Address) else str(ip) for ip in addrs]
    return f"{host}:{port}:{','.join(rendered)}"


class ValidatingTransport(httpx.AsyncHTTPTransport):
    """httpx transport that re-validates the connect target on every request.

    httpx issues the initial request and each redirect follow as fresh
    requests through the same transport, so intercepting here checks **every
    hop**. For each request we:

    1. Resolve the host exactly as httpx parsed it (``request.url.host``) —
       this is what eliminates the parser differential: validation and
       connection now read the same authority.
    2. Run every resolved address through ``_is_blocked_ip``; raise
       ``UnsafeUrlError`` (fail-closed) if any is blocked or the host does not
       resolve.
    3. Pin the connection to a vetted address while preserving the ``Host``
       header (set by httpx at request-build time) and TLS SNI / certificate
       verification against the original hostname.

    Because the check runs at connect time, the validated address *is* the
    connected address (closes DNS-rebinding / TOCTOU), and it runs on every
    hop including redirects (closes redirect SSRF).
    """

    def __init__(self, *, resolve: Resolver | None = None, **kwargs: Any) -> None:
        # ``resolve`` is injectable for unit tests ("stub resolver");
        # ``None`` uses the real DNS path shared with ``validate_fetch_url``.
        super().__init__(**kwargs)
        self._resolve: Resolver = resolve if resolve is not None else _resolve_all

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        addrs = self._resolve(host)
        if not addrs:
            raise UnsafeUrlError(
                f"Refusing to connect to {host!r}: hostname did not resolve (fail-closed)."
            )
        for ip in addrs:
            if _is_blocked_ip(ip):
                raise UnsafeUrlError(
                    f"Refusing to connect to {host!r}: resolves to a "
                    f"private/reserved address ({ip}). This protects against "
                    "SSRF (cloud metadata, internal services, loopback) across "
                    "DNS rebinding and redirects."
                )
        # Pin to a vetted address; keep the Host header and bind TLS SNI to the
        # real hostname so certificate verification is unaffected.
        request.url = request.url.copy_with(host=str(addrs[0]))
        request.extensions["sni_hostname"] = host
        return await super().handle_async_request(request)
