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
import re
import socket
import stat
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


_USERINFO_RE = re.compile(r"^(\s*[A-Za-z][A-Za-z0-9+.-]*://)[^/?#]*@")
_SCHEME_RE = re.compile(r"^(\s*[A-Za-z][A-Za-z0-9+.-]*://)([^/?#]*)")
_AUTHORITY_RE = re.compile(r"^(?:[A-Za-z0-9._~%-]*|\[[0-9A-Fa-f:.]*\])(?::\d*)?$")


def url_without_userinfo(url: str) -> str:
    """``url`` with any ``user:password@`` removed, for printing or logging it.

    An operator can write a credential into an endpoint's userinfo
    (``http://user:token@gateway``), and a refusal, a warning or a diagnostic
    that names the endpoint would echo it (security review 2026-10-25-2, R36).
    Textual, so it never fails on a URL ``urllib`` cannot parse: everything
    from the scheme's ``//`` to the authority's last ``@`` goes.

    A password written unescaped can hold ``/``, ``?`` or ``#``, which end the
    authority for a parser. When the text before the first of them is no host
    and port (``user:pa`` of ``user:pa/ss@host``), it is taken as the start of
    a userinfo, and everything up to the URL's last ``@`` goes; that can hide
    more than the userinfo, never less (security review 2026-10-26-1, R36).

    A password of digits up to its first ``/``, ``?`` or ``#`` reads as a
    port (``user:4417`` of ``user:4417/rest@engine``): when the authority holds
    a ``:`` and an ``@`` follows anywhere in the URL, everything up to the last
    ``@`` goes, whatever the host after it looks like, unless the authority
    names an IP literal. A URL written without a
    scheme (``user:token@host``) loses everything up to the last ``@`` before
    its first whitespace (2026-10-27-1, R36, W12; security review
    2026-10-28-1, O31).
    """
    stripped = _USERINFO_RE.sub(r"\1", url, count=1)
    if stripped != url:
        return stripped
    match = _SCHEME_RE.match(url)
    if match is None:
        return _schemeless_without_userinfo(url)
    if _AUTHORITY_RE.match(match.group(2)):
        return _port_shaped_password_removed(url, match)
    at = url.rfind("@", match.end(1))
    if at < 0:
        return url
    return match.group(1) + url[at + 1 :]


def _port_shaped_password_removed(url: str, match: re.Match[str]) -> str:
    """``url`` without a ``user:<digits>...@`` prefix whose password reads as a port.

    An authority whose name is an IP literal (``127.0.0.1:8000``, ``[::1]:8000``)
    is a host and port, since no user name is one, so an ``@`` later in its path
    or query is kept. Any other name may be a user's, and the text up to the
    last ``@`` goes.
    """
    authority = match.group(2)
    if ":" not in authority or authority.startswith("["):
        return url
    try:
        ipaddress.ip_address(authority.split(":", 1)[0])
    except ValueError:
        pass
    else:
        return url
    at = url.rfind("@", match.end(2))
    if at < 0:
        return url
    return match.group(1) + url[at + 1 :]


def _schemeless_without_userinfo(url: str) -> str:
    """``url``, written without ``scheme://``, without the userinfo it holds."""
    lead = len(url) - len(url.lstrip())
    space = re.search(r"\s", url[lead:])
    end = lead + space.start() if space else len(url)
    at = url.rfind("@", lead, end)
    if at < 0:
        return url
    return url[:lead] + url[at + 1 :]


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


#: Variables that send a subprocess's connection through a proxy or swap the
#: trust anchors it verifies with, compared case-insensitively. Through a
#: proxy the proxy resolves the host, so the vetted-address pin (``--resolve``,
#: ``http.curloptResolve``) and the ``network.egress`` check of the target no
#: longer decide the peer, and the proxy itself is never checked (P43).
PINNED_SUBPROCESS_DROPPED_VARS = frozenset(
    {
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "ftp_proxy",
        "no_proxy",
        "curl_ca_bundle",
        "ssl_cert_file",
        "ssl_cert_dir",
        # Writes the TLS session keys to a file, so anyone who can read it can
        # decrypt the connection; and the program ssh runs to ask for a
        # password (security review 2026-10-26-1, K10).
        "sslkeylogfile",
        "ssh_askpass",
    }
)


def pinned_subprocess_env(environ: Mapping[str, str]) -> dict[str, str]:
    """``environ`` without the proxy and CA variables, for a pinned curl or git fetch.

    ``PATH`` keeps only the absolute entries no other user may write: the
    subprocess's program, and any program it runs, is looked up there, so a
    relative entry would find one in the working directory (security review
    2026-10-26-1, K10), and a directory writable by its group or by others, or
    owned by neither root nor this user, would find one another local user
    planted (2026-10-27-1, G52). An entry that is another user's link, or that
    sits under a directory others may write without the sticky bit, is dropped
    too, since that user could repoint or rename it after the check. When no
    entry is left, ``PATH`` falls back to the entries of ``os.defpath`` that
    pass, and failing those to a path that holds no program: an empty ``PATH``
    would make the lookup search the working directory (2026-10-28-1, G52).
    """
    env = {k: v for k, v in environ.items() if k.lower() not in PINNED_SUBPROCESS_DROPPED_VARS}
    if "PATH" in env:
        kept = [p for p in env["PATH"].split(os.pathsep) if _only_trusted_users_write(p)]
        if not kept:
            kept = [p for p in os.defpath.split(os.pathsep) if _only_trusted_users_write(p)]
        env["PATH"] = os.pathsep.join(kept) if kept else _NO_PROGRAM_PATH
    return env


#: A ``PATH`` under which no program can be found (``/dev/null`` is no directory).
_NO_PROGRAM_PATH = os.devnull

#: Groups whose members can already act as root, so a directory they may
#: write (Homebrew's ``/opt/homebrew/bin`` is ``admin``-writable) gives no
#: other user more than they hold.
_ADMINISTRATIVE_GROUPS = frozenset({"root", "wheel", "admin", "sudo"})


def _only_trusted_users_write(directory: str) -> bool:
    """Whether ``directory`` is an absolute path to a directory no other ordinary user may write.

    Refused: a relative or missing entry, one others may write, one owned by
    neither root nor this user, and one its group may write unless that group
    is administrative. A link is judged as itself (its owner) and as its
    target, and every directory above the entry, as written and as resolved,
    is refused when another user may write it without the sticky bit. Every
    link the entry passes through on the way to its target is judged the same
    way, since another user who owns a middle link, or may rename entries in
    its directory, can repoint it after the check (security review
    2026-10-28-2). A middle link's directories are judged as resolved too: a
    hop reached through a directory link lies in that link's target, and the
    directories above the target are the ones that can rename it (2026-10-30-1).
    """
    if not os.path.isabs(directory):
        return False
    try:
        hops = _link_hops(directory)
        st = os.stat(directory)
    except OSError:
        return False
    for path, link in hops:
        resolved_hop = os.path.join(os.path.realpath(os.path.dirname(path)), os.path.basename(path))
        if (
            not _owned_by_trusted_user(link)
            or _an_ancestor_is_open(path)
            or _an_ancestor_is_open(resolved_hop)
        ):
            return False
    if not stat.S_ISDIR(st.st_mode) or _others_may_write(st) or not _owned_by_trusted_user(st):
        return False
    written, resolved = os.path.abspath(directory), os.path.realpath(directory)
    return not (_an_ancestor_is_open(written) or _an_ancestor_is_open(resolved))


#: The longest link chain a ``PATH`` entry may sit behind.
_MAX_LINK_HOPS = 40


def _link_hops(path: str) -> list[tuple[str, os.stat_result]]:
    """Each link ``path`` passes through, with its own ``lstat``, one ``readlink`` at a time.

    Raises ``OSError`` when the chain is longer than :data:`_MAX_LINK_HOPS`.
    """
    hops: list[tuple[str, os.stat_result]] = []
    current = os.path.abspath(path)
    link = os.lstat(current)
    while stat.S_ISLNK(link.st_mode):
        if len(hops) >= _MAX_LINK_HOPS:
            raise OSError(f"{path}: too many levels of symbolic links")
        hops.append((current, link))
        # Joined, not normalised: the system resolves a ``..`` in a link
        # against the directory the link is in, wherever that leads.
        current = os.path.join(os.path.dirname(current), os.readlink(current))
        link = os.lstat(current)
    return hops


def _owned_by_trusted_user(st: os.stat_result) -> bool:
    geteuid = getattr(os, "geteuid", None)
    return geteuid is None or st.st_uid in (0, geteuid())


def _others_may_write(st: os.stat_result) -> bool:
    """Whether a user other than the owner, outside an administrative group, may write it."""
    if st.st_mode & stat.S_IWOTH:
        return True
    return bool(st.st_mode & stat.S_IWGRP) and not _is_administrative_group(st.st_gid)


def _an_ancestor_is_open(path: str) -> bool:
    """Whether a directory above ``path`` lets another user rename its entries.

    A directory owned by neither root nor this user is refused whatever its
    mode bits, since its owner may change them (security review 2026-10-28-2).
    """
    parent = os.path.dirname(path)
    while True:
        try:
            link = os.lstat(parent)
            st = os.stat(parent)
        except OSError:
            return True
        if stat.S_ISLNK(link.st_mode) and not _owned_by_trusted_user(link):
            return True
        if not _owned_by_trusted_user(st):
            return True
        if _others_may_write(st) and not st.st_mode & stat.S_ISVTX:
            return True
        up = os.path.dirname(parent)
        if up == parent:
            return False
        parent = up


def _is_administrative_group(gid: int) -> bool:
    if gid == 0:
        return True
    try:
        import grp
    except ImportError:  # not POSIX: no group to look up
        return False
    try:
        return grp.getgrgid(gid).gr_name in _ADMINISTRATIVE_GROUPS
    except KeyError:
        return False


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
    configured some other way. Only https is allowed as a transport. The proxy
    and CA variables are dropped too (:func:`pinned_subprocess_env`), so the
    clone connects to the pinned address and verifies it with the system trust
    store.
    """
    env = {k: v for k, v in pinned_subprocess_env(environ).items() if not k.startswith("GIT_")}
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
