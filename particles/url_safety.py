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

All three first ask the no-egress posture (:mod:`particles.egress`),
before anything is resolved, so a refused host is never looked up. That check
is composed with the SSRF blocklist, never merged into it: allowing a LAN host
for a model server does not let a deposited URL reach it.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
import time
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlparse

import httpx

from particles import egress

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = frozenset({"http", "https"})

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
Resolver = Callable[[str], list[IPAddress]]

# RFC 6598 carrier-grade NAT. ``ipaddress.is_private`` does not include these,
# yet in NAT / hosted environments they can reach internal infrastructure
# (security review F-4-ip). Block both the bare IPv4 range and its
# IPv4-mapped IPv6 form (``::ffff:100.64.0.0`` + the 10-bit prefix → /106).
_CGNAT_V4 = ipaddress.IPv4Network("100.64.0.0/10")
_CGNAT_V6_MAPPED = ipaddress.IPv6Network("::ffff:100.64.0.0/106")


def is_network_uri(uri: str) -> bool:
    """Whether ``uri`` names a network resource (``http`` / ``https``), not a local one.

    The test a remote deposit's ``fetch_policy=LAZY`` must pass. A ``LAZY``
    entry is re-read by the engine later, and only a network URI-R is re-read
    through the guarded fetch path: a ``file://`` URI-R (or any other scheme)
    names a path on the engine host, so a remote caller who could enrol one
    could make the engine read its own files and store them in the corpus
    (security review 2026-10-23, finding D2).
    """
    return urlparse(uri).scheme.lower() in ALLOWED_SCHEMES


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
    egress.check_url(url, "fetch")

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


def resolve_and_pin(
    host: str, *, port: int = 443, resolve: Resolver | None = None
) -> list[IPAddress]:
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

    ``port`` is the one the subprocess will connect to; the no-egress posture
    is checked against ``host:port`` before anything is resolved.

    Raises:
        EgressRefused: ``network.egress`` does not allow ``host:port``.
        UnsafeUrlError: fail-closed, if the host does not resolve or if ANY
            resolved address is blocked. That all-or-nothing rule is the one
            ``validate_fetch_url`` already applies, and it is what stops a
            split A/AAAA answer smuggling an internal address past a public one.
    """
    egress.check(host, port, "subprocess fetch")
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


# ---------------------------------------------------------------------------
# Credentials over cleartext (N12, SECURITY_REVIEW_20261024-1)
# ---------------------------------------------------------------------------


def host_is_loopback(host: str | None) -> bool:
    """True iff ``host`` is loopback (``127.0.0.0/8`` / ``::1`` / ``localhost``).

    Fail-closed like ``particles.api.auth._is_loopback_host``: a non-IP
    hostname (other than the literal ``localhost``) is treated as non-loopback.
    No DNS is resolved, so a custom name aliasing loopback counts as remote,
    the safe direction for a credential check.
    """
    if not host:
        return False
    h = host.strip().lower()
    if h == "localhost":
        return True
    h = h.removeprefix("[").removesuffix("]")
    h = h.split("%", 1)[0]
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def is_cleartext_remote(url: str) -> bool:
    """True iff a credential sent to ``url`` would cross the network unencrypted.

    That is any scheme other than ``https`` to a host that is not loopback. The
    documented tunnel deployments (an SSH or Tailscale forward to
    ``http://127.0.0.1``) stay ``False``. The engine bearer, the
    OpenAI-compatible provider key and the OTLP exporter headers are each
    refused on a ``True`` here unless the operator opted in for that one
    endpoint.
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() == "https":
        return False
    return not host_is_loopback(parsed.hostname)


# ---------------------------------------------------------------------------
# Git subprocesses: no program a repository or the user's config names
# (N13 / D22, SECURITY_REVIEW_20261024-1)
# ---------------------------------------------------------------------------

#: ``-c`` options every local-only git subprocess passes (class 3).
#: A repository's own ``.git/config`` can name a program git runs by itself,
#: and ``core.fsmonitor`` runs on any index refresh, ``ls-files`` and
#: ``rev-parse`` included, so a planted config in a checkout the operator names
#: would otherwise execute. ``diff`` callers also pass ``--no-ext-diff
#: --no-textconv``. A caller that also stages or commits adds its own
#: hardening on top (the memory projection's ``_HARDENING_CONFIG`` disables
#: hooks and skips a repository whose config names a filter driver).
GIT_LOCAL_ARGS: tuple[str, ...] = ("-c", "core.fsmonitor=false")

#: Environment variables that point git at a repository other than the one its
#: arguments name. A git hook exports several of them, so a subprocess run from
#: inside one would otherwise reach the hook's repository.
GIT_REDIRECT_VARS = frozenset(
    {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_NAMESPACE",
        "GIT_PREFIX",
    }
)


def local_git_env(environ: Mapping[str, str]) -> dict[str, str]:
    """``environ`` without the repository-redirect variables, for a local git query."""
    return {k: v for k, v in environ.items() if k not in GIT_REDIRECT_VARS}


def isolated_git_env(environ: Mapping[str, str], *, home: str) -> dict[str, str]:
    """An environment for git over a repository fetched from elsewhere (N13).

    Every ``GIT_*`` variable is dropped, so neither a hook's redirect variables
    nor an inherited ``GIT_CONFIG_PARAMETERS`` / ``GIT_SSL_NO_VERIFY`` applies.
    The system and global config are switched off (``GIT_CONFIG_NOSYSTEM``,
    ``GIT_CONFIG_GLOBAL``), and ``HOME`` / ``XDG_CONFIG_HOME`` point at
    ``home``, an empty directory, for a git older than ``GIT_CONFIG_GLOBAL``.
    With no config defining a filter driver, a ``.gitattributes`` in the
    fetched tree names nothing git can run: a stock ``git lfs install`` defines
    ``filter.lfs.*`` globally, and git-lfs would otherwise fetch from whatever a
    committed ``.lfsconfig`` names, unpinned and unchecked by
    ``network.egress``. ``GIT_LFS_SKIP_SMUDGE`` covers a git-lfs that is
    configured some other way. Only https is allowed as a transport.
    """
    env = {k: v for k, v in environ.items() if not k.startswith("GIT_")}
    env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ALLOW_PROTOCOL": "https",
            "HOME": home,
            "XDG_CONFIG_HOME": home,
        }
    )
    return env


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

    The pin covers the **whole** vetted set, in resolver order: when the
    connection to one address cannot be established, the next is tried, the
    failover httpx would have done itself over a multi-A-record host and which
    ``curl --resolve`` keeps on the subprocess paths. Every
    candidate has already passed ``_is_blocked_ip``, so widening from one
    address to all cannot admit a blocked one. Only a failure to *connect* moves
    on — nothing has been sent yet, so any method is safe to retry — and the
    connect-timeout budget is shared rather than multiplied: each address but
    the last gets half of what remains and the last gets the rest (libcurl's
    rule), so a dead host still fails within one connect timeout.
    """

    def __init__(self, *, resolve: Resolver | None = None, **kwargs: Any) -> None:
        # ``resolve`` is injectable for unit tests ("stub resolver");
        # ``None`` uses the real DNS path shared with ``validate_fetch_url``.
        super().__init__(**kwargs)
        self._resolve: Resolver = resolve if resolve is not None else _resolve_all

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        # The no-egress posture first, before any lookup; every
        # redirect hop passes here, so each hop's host is checked.
        egress.check(host, request.url.port or egress.host_port(str(request.url))[1], "fetch")
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
        request.extensions["sni_hostname"] = host
        url = request.url
        timeouts: dict[str, float | None] = dict(request.extensions.get("timeout") or {})
        budget = timeouts.get("connect")
        deadline = None if budget is None else time.monotonic() + budget
        for i, ip in enumerate(addrs):
            last = i == len(addrs) - 1
            request.url = url.copy_with(host=str(ip))
            if deadline is not None:
                remaining = max(deadline - time.monotonic(), 0.0)
                request.extensions["timeout"] = {
                    **timeouts,
                    "connect": remaining if last else remaining / 2,
                }
            try:
                return await super().handle_async_request(request)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                if last:
                    raise
                logger.debug("connect to %s (%s) failed; trying the next vetted address", host, ip)
        raise AssertionError("unreachable: addrs is non-empty")  # pragma: no cover
