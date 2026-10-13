# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The no-egress posture: one policy every outbound path is held to.

``network.egress`` is ``open`` (no check), ``none`` (refuse every outbound
connection and name lookup, loopback included) or ``allowlist`` (refuse all
but the destinations in ``network.allow``). This module compiles that setting
into an immutable :class:`EgressPolicy` and raises :class:`EgressRefused` for a
destination it does not allow.

Local is named, never inferred: no rule here consults ``is_loopback``,
``is_private`` or the SSRF blocklist in :mod:`particles.url_safety`. The two
gates are composed (this one runs first), never merged.

Enforcement lives at the transports, not in callers:

* :class:`particles.url_safety.ValidatingTransport`,
  :func:`~particles.url_safety.validate_fetch_url` and
  :func:`~particles.url_safety.resolve_and_pin` call :func:`check` before they
  resolve anything;
* :class:`particles.http.EgressTransport` / ``SyncEgressTransport`` carry the
  check for the clients that bypass the SSRF transport (the LLM adapters and
  the remote-engine client);
* :func:`ensure_backstop` installs a process-wide audit hook for anything else.

Callers that know their destination up front ask :func:`allows` first and take
a graceful branch; the transports are the guarantee behind them. A refusal is
never cached and never counted as a miss.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit

from particles.config import NetworkConfig, get_config

log = logging.getLogger(__name__)

Posture = Literal["open", "none", "allowlist"]
IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network

_DEFAULT_PORTS = {"http": 80, "https": 443}

#: Compiled-in destinations, named once so the posture report and the
#: pre-checks agree on what each path would reach.
ANTHROPIC_HOST = ("api.anthropic.com", 443)
WIKIDATA_HOST = ("www.wikidata.org", 443)
HUGGINGFACE_HOST = ("huggingface.co", 443)
NOTION_HOST = ("api.notion.com", 443)


class EgressRefused(Exception):
    """An outbound connection the no-egress posture does not allow.

    Deliberately not a ``ValueError`` and not an ``httpx`` error, so no broad
    handler for either turns a policy refusal into an SSRF block or an outage.
    """

    def __init__(self, host: str, port: int | None, purpose: str, posture: str) -> None:
        self.host = host
        self.port = port
        self.purpose = purpose
        self.posture = posture
        where = host if port is None else f"{host}:{port}"
        if posture == "none":
            why = "network.egress is none"
        else:
            why = f"network.egress is allowlist and {where} is not in network.allow"
        super().__init__(
            f"{why}: refusing to connect to {where} ({purpose}). "
            "Add the destination to network.allow to permit it."
        )


def normalize_host(host: str) -> str:
    """Lower-case, strip IPv6 brackets and a trailing dot, IDNA-encode a name."""
    value = host.strip().strip("[]").rstrip(".").lower()
    if not value:
        return value
    try:
        return str(ipaddress.ip_address(value.split("%", 1)[0]))
    except ValueError:
        pass
    try:
        return value.encode("idna").decode("ascii")
    except UnicodeError:
        return value


def _ip_or_none(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host.strip("[]").split("%", 1)[0])
    except ValueError:
        return None


@dataclass(frozen=True)
class _Entry:
    kind: Literal["ip", "net", "name"]
    value: str
    network: IPNetwork | None
    port: int | None

    def port_matches(self, port: int | None) -> bool:
        return self.port is None or port is None or self.port == port


@dataclass(frozen=True)
class EgressPolicy:
    """The compiled posture. Pure: no I/O, no config read."""

    posture: Posture
    entries: tuple[_Entry, ...] = ()
    #: Addresses a listed name resolved to in this process, with the ports its
    #: entries allow (None for any). Filled by the backstop's resolver wrapper.
    _admitted: dict[str, set[int | None]] = field(
        default_factory=dict, compare=False, hash=False, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, compare=False, hash=False, repr=False
    )

    @property
    def refusing(self) -> bool:
        """True for ``none`` and ``allowlist``: some destination may be refused."""
        return self.posture != "open"

    def allows(self, host: str, port: int | None) -> bool:
        """Whether ``host:port`` is a destination this posture permits."""
        if self.posture == "open":
            return True
        name = normalize_host(host)
        ip = _ip_or_none(name)
        for entry in self.entries:
            if not entry.port_matches(port):
                continue
            if ip is not None:
                if entry.kind == "ip" and entry.value == str(ip):
                    return True
                if entry.kind == "net" and entry.network is not None and ip in entry.network:
                    return True
            elif entry.kind == "name" and entry.value == name:
                return True
        return False

    def allows_lookup(self, host: str) -> bool:
        """Whether resolving ``host`` is permitted: a listed name, or an IP literal."""
        if self.posture == "open":
            return True
        name = normalize_host(host)
        if _ip_or_none(name) is not None:
            return True
        return any(e.kind == "name" and e.value == name for e in self.entries)

    def record_resolution(self, host: str, addresses: list[str]) -> None:
        """Admit the addresses a listed name resolved to, for that name's ports."""
        name = normalize_host(host)
        ports = {e.port for e in self.entries if e.kind == "name" and e.value == name}
        if not ports:
            return
        with self._lock:
            for address in addresses:
                ip = _ip_or_none(address)
                if ip is not None:
                    self._admitted.setdefault(str(ip), set()).update(ports)

    def admits(self, address: str, port: int | None) -> bool:
        """The backstop's connect check: a listed literal/CIDR, or an admitted address."""
        if self.allows(address, port):
            return True
        ip = _ip_or_none(address)
        if ip is None:
            return False
        ports = self._admitted.get(str(ip))
        return ports is not None and (None in ports or port in ports)


def compile_policy(network: NetworkConfig) -> EgressPolicy:
    """Compile ``config.network`` into an :class:`EgressPolicy`."""
    entries: list[_Entry] = []
    for item in network.allow:
        host = item.host
        if "/" in host:
            net = ipaddress.ip_network(host, strict=False)
            entries.append(_Entry("net", str(net), net, item.port))
            continue
        name = normalize_host(host)
        if _ip_or_none(name) is not None:
            entries.append(_Entry("ip", name, None, item.port))
        else:
            entries.append(_Entry("name", name, None, item.port))
    return EgressPolicy(posture=network.egress, entries=tuple(entries))


def entry_labels(policy: EgressPolicy) -> tuple[str, ...]:
    """The ``network.allow`` entries as the reports print them."""
    return tuple(e.value if e.port is None else f"{e.value}:{e.port}" for e in policy.entries)


_policy_cache: tuple[int, EgressPolicy] | None = None


def current_policy() -> EgressPolicy:
    """The policy for the current config, compiled once per config object."""
    global _policy_cache
    cfg = get_config()
    cached = _policy_cache
    if cached is not None and cached[0] == id(cfg.network):
        return cached[1]
    policy = compile_policy(cfg.network)
    _policy_cache = (id(cfg.network), policy)
    return policy


def allows(host: str, port: int | None) -> bool:
    """Whether the current posture permits ``host:port``."""
    return current_policy().allows(host, port)


def check(host: str, port: int | None, purpose: str) -> None:
    """Raise :class:`EgressRefused` unless the current posture permits ``host:port``."""
    policy = current_policy()
    if not policy.allows(host, port):
        raise EgressRefused(normalize_host(host), port, purpose, policy.posture)


def host_port(url: str) -> tuple[str, int | None]:
    """The ``(host, port)`` a URL connects to, with the scheme's default port."""
    parts = urlsplit(url if "://" in url else f"//{url}")
    host = parts.hostname or ""
    try:
        port = parts.port
    except ValueError:
        port = None
    if port is None:
        port = _DEFAULT_PORTS.get(parts.scheme.lower())
    return host, port


def check_url(url: str, purpose: str) -> None:
    """:func:`check` for the destination a URL connects to."""
    host, port = host_port(url)
    check(host, port, purpose)


def allows_url(url: str) -> bool:
    """:func:`allows` for the destination a URL connects to."""
    host, port = host_port(url)
    return allows(host, port)


# ---------------------------------------------------------------------------
# The process-wide backstop
# ---------------------------------------------------------------------------

#: The policy the audit hook enforces. None ⇒ the hook is a no-op. Swapped by
#: :func:`ensure_backstop`; the hook never calls ``get_config()``, because a
#: config read opens files and file opens raise audit events.
_backstop_policy: EgressPolicy | None = None
_backstop_installed = False
_install_lock = threading.Lock()
_real_getaddrinfo: Callable[..., Any] | None = None


def _audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event != "socket.connect" and event != "socket.getaddrinfo":
        return
    policy = _backstop_policy
    if policy is None:
        return
    if event == "socket.getaddrinfo":
        host = args[0]
        if isinstance(host, bytes):
            host = host.decode("ascii", "replace")
        if isinstance(host, str) and host and not policy.allows_lookup(host):
            raise EgressRefused(normalize_host(host), None, "name lookup", policy.posture)
        return
    sock, address = args[0], args[1]
    if getattr(sock, "family", None) not in (socket.AF_INET, socket.AF_INET6):
        return
    if not isinstance(address, tuple) or len(address) < 2:
        return
    host, port = str(address[0]), address[1]
    port = port if isinstance(port, int) else None
    if _ip_or_none(host) is None:
        if not policy.allows(host, port):
            raise EgressRefused(normalize_host(host), port, "connection", policy.posture)
        return
    if not policy.admits(host, port):
        raise EgressRefused(normalize_host(host), port, "connection", policy.posture)


def _recording_getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
    assert _real_getaddrinfo is not None
    result = _real_getaddrinfo(host, port, *args, **kwargs)
    policy = _backstop_policy
    if policy is not None and isinstance(host, str | bytes) and result:
        name = host.decode("ascii", "replace") if isinstance(host, bytes) else host
        policy.record_resolution(name, [str(info[4][0]) for info in result])
    return result


def ensure_backstop() -> None:
    """Arm the process-wide backstop for the current posture.

    Called by each process entry point beside ``setup_observability``. Under
    ``open`` it disarms (the hook stays installed, since an audit hook cannot be
    removed, but does nothing). Under a refusing posture it installs, once, an
    audit hook refusing lookups and connects the policy does not admit, and a
    wrapper around :func:`socket.getaddrinfo` that admits the addresses a
    listed name resolves to.
    """
    global _backstop_policy, _backstop_installed, _real_getaddrinfo
    policy = current_policy()
    if not policy.refusing:
        _backstop_policy = None
        return
    with _install_lock:
        if not _backstop_installed:
            _real_getaddrinfo = socket.getaddrinfo
            socket.getaddrinfo = _recording_getaddrinfo
            sys.addaudithook(_audit_hook)
            _backstop_installed = True
        _backstop_policy = policy
    log.debug("egress backstop armed: network.egress=%s", policy.posture)


def disarm_backstop() -> None:
    """Make the backstop a no-op (tests, and a reload to ``open``)."""
    global _backstop_policy
    _backstop_policy = None
