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

import asyncio
import ipaddress
import logging
import socket
import sys
import threading
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar
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


# Keyed on the config object itself, compared with ``is``. An ``id()`` key
# is reused once the old object is collected, so a fresh ``NetworkConfig``
# allocated at the freed address would be served the stale policy. Holding
# the reference keeps the key alive, so its address can never be reused.
_policy_cache: tuple[NetworkConfig, EgressPolicy] | None = None


def current_policy() -> EgressPolicy:
    """The policy for the current config, compiled once per config object."""
    global _policy_cache
    network = get_config().network
    cached = _policy_cache
    if cached is not None and cached[0] is network:
        return cached[1]
    policy = compile_policy(network)
    _policy_cache = (network, policy)
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


#: Forward lookups: the first argument is the name the resolver is asked for.
_LOOKUP_EVENTS = frozenset({"socket.getaddrinfo", "socket.gethostbyname"})
#: Reverse lookups: the argument is an address (or, for ``gethostbyaddr``, a
#: name it resolves first), and the query goes to the resolver all the same.
_REVERSE_EVENTS = frozenset({"socket.gethostbyaddr", "socket.getnameinfo"})
#: Traffic to a destination: a TCP/UDP connect, or a datagram sent to an
#: address without a connect (``sendto`` / ``sendmsg`` on an unconnected socket).
_SEND_EVENTS = frozenset({"socket.connect", "socket.sendto", "socket.sendmsg"})
_WATCHED_EVENTS = _LOOKUP_EVENTS | _REVERSE_EVENTS | _SEND_EVENTS


def _text(host: Any) -> str | None:
    if isinstance(host, bytes):
        return host.decode("ascii", "replace")
    return host if isinstance(host, str) else None


def _audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event not in _WATCHED_EVENTS:
        return
    policy = _backstop_policy
    if policy is None:
        return
    if event in _LOOKUP_EVENTS:
        host = _text(args[0])
        if host and not policy.allows_lookup(host):
            raise EgressRefused(normalize_host(host), None, "name lookup", policy.posture)
        return
    if event in _REVERSE_EVENTS:
        target = args[0]
        if isinstance(target, tuple) and target:
            target = target[0]
        name = _text(target)
        if name is None or not name:
            return
        # A reverse lookup of an address the posture admits (or of a listed
        # name) is the resolver traffic an admitted destination already
        # implies; any other is a query the posture never allowed.
        admitted = (
            policy.admits(name, None)
            if _ip_or_none(name) is not None
            else policy.allows_lookup(name)
        )
        if not admitted:
            raise EgressRefused(normalize_host(name), None, "reverse lookup", policy.posture)
        return
    sock, address = args[0], args[1]
    if getattr(sock, "family", None) not in (socket.AF_INET, socket.AF_INET6):
        return
    # ``sendto`` / ``sendmsg`` on a connected socket carry no address; the
    # connect that fixed the destination was checked already.
    if not isinstance(address, tuple) or len(address) < 2:
        return
    purpose = "connection" if event == "socket.connect" else "datagram"
    host, port = str(address[0]), address[1]
    port = port if isinstance(port, int) else None
    if _ip_or_none(host) is None:
        if not policy.allows(host, port):
            raise EgressRefused(normalize_host(host), port, purpose, policy.posture)
        return
    if not policy.admits(host, port):
        raise EgressRefused(normalize_host(host), port, purpose, policy.posture)


def _refuse_named_destination(sock: Any, address: Any, purpose: str) -> None:
    """Judge a connect or send that names a host, before the name is resolved (Q44).

    CPython resolves the host of a ``connect`` / ``sendto`` address inside the
    C call, before it raises the audit event and without raising a lookup
    event, so by the time :func:`_audit_hook` sees a named destination its DNS
    query has already left the host. This runs first, from the Python-level
    wrappers :func:`ensure_backstop` puts on :class:`socket.socket`. An address
    literal is left to the audit hook, which judges it after the call.
    """
    policy = _backstop_policy
    if policy is None:
        return
    if getattr(sock, "family", None) not in (socket.AF_INET, socket.AF_INET6):
        return
    if not isinstance(address, tuple) or len(address) < 2:
        return
    host = _text(address[0])
    if not host or _ip_or_none(host) is not None:
        return
    port = address[1] if isinstance(address[1], int) else None
    if not policy.allows(host, port):
        raise EgressRefused(normalize_host(host), port, purpose, policy.posture)


def _guard_socket_methods() -> None:
    """Wrap the ``socket.socket`` calls that take a destination (Q44)."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_sendto = socket.socket.sendto
    # Windows has no sendmsg.
    real_sendmsg: Callable[..., int] | None = getattr(socket.socket, "sendmsg", None)

    def connect(self: socket.socket, address: Any) -> None:
        _refuse_named_destination(self, address, "connection")
        real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        _refuse_named_destination(self, address, "connection")
        return real_connect_ex(self, address)

    def sendto(self: socket.socket, data: Any, *args: Any) -> int:
        # sendto(data, address) or sendto(data, flags, address).
        if args:
            _refuse_named_destination(self, args[-1], "datagram")
        return real_sendto(self, data, *args)

    def sendmsg(self: socket.socket, buffers: Any, *args: Any) -> int:
        # sendmsg(buffers, ancdata, flags, address): the address is fourth.
        if len(args) >= 3:
            _refuse_named_destination(self, args[2], "datagram")
        assert real_sendmsg is not None
        return real_sendmsg(self, buffers, *args)

    wrappers: list[tuple[str, Callable[..., Any]]] = [
        ("connect", connect),
        ("connect_ex", connect_ex),
        ("sendto", sendto),
    ]
    if real_sendmsg is not None:
        wrappers.append(("sendmsg", sendmsg))
    for name, wrapper in wrappers:
        setattr(socket.socket, name, wrapper)


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
    audit hook refusing the forward and reverse lookups, connects and unconnected
    datagram sends the policy does not admit, a wrapper around
    :func:`socket.getaddrinfo` that admits the addresses a listed name resolves
    to, and wrappers on the :class:`socket.socket` calls that take a destination,
    which refuse a named host before CPython resolves it (Q44).
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
            _guard_socket_methods()
            sys.addaudithook(_audit_hook)
            _backstop_installed = True
        _backstop_policy = policy
    _prefer_selector_loop()
    log.debug("egress backstop armed: network.egress=%s", policy.posture)


def _prefer_selector_loop() -> None:
    """On Windows, make the loops this process creates selector loops (Q15).

    Windows' default loop is the proactor loop, whose connects go through
    ``ConnectEx`` / ``WSAConnect`` on the raw descriptor rather than
    ``socket.socket.connect``, so the audit hook never sees them. Every entry
    point arms the backstop before it starts its loop, so setting the selector
    policy here keeps a refusing posture on a loop the hook covers. A no-op on
    every other platform, whose default loop is already a selector loop.
    """
    if sys.platform == "win32":
        import warnings

        with warnings.catch_warnings():
            # The policy API is deprecated from Python 3.14 and still honoured.
            warnings.simplefilter("ignore", DeprecationWarning)
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class UncoveredEventLoop(RuntimeError):
    """A refusing posture is armed on an event loop the backstop cannot see."""


def require_covered_loop() -> None:
    """Refuse to run a refusing posture on an event loop whose sockets skip the hook.

    The backstop sees connects made through Python's ``socket`` module. A
    stdlib selector loop makes them there. uvloop opens its sockets in libuv,
    which raises no audit event, and Windows' proactor loop connects through
    ``ConnectEx`` on the raw descriptor, so under either the posture would hold
    only at the checked transports (security review 2026-10-25, Q15). Only a
    selector loop is admitted. ``engine serve`` picks one itself; a raw
    ``uvicorn`` launch with ``uvicorn[standard]`` installed picks uvloop, and
    this is what stops it (B49). A no-op under ``open`` and outside a running
    loop.
    """
    policy = current_policy()
    if not policy.refusing:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    if isinstance(loop, asyncio.selector_events.BaseSelectorEventLoop):
        return
    kind = f"{type(loop).__module__}.{type(loop).__qualname__}"
    raise UncoveredEventLoop(
        f"network.egress is {policy.posture}, and the event loop is {kind}, whose "
        "sockets the no-egress backstop cannot see. Start the engine with "
        "`particles engine serve`, or pass `--loop asyncio:SelectorEventLoop` "
        "to uvicorn."
    )


_T = TypeVar("_T")


async def on_covered_loop(coro: Coroutine[Any, Any, _T]) -> _T:
    """Await ``coro`` once :func:`require_covered_loop` admits the running loop.

    A refused loop closes ``coro`` unstarted and raises :class:`UncoveredEventLoop`.
    """
    try:
        require_covered_loop()
    except UncoveredEventLoop:
        coro.close()
        raise
    return await coro


def run_armed(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run ``coro`` under :func:`asyncio.run` with the backstop armed and its loop covered.

    For every process entry point that starts its own loop (the CLI's ``run()``,
    the hooks, the diagnostics' probes; security review 2026-10-26-1, D18, U13).
    Raises :class:`UncoveredEventLoop` before ``coro`` starts when a refusing
    posture would run on a loop the backstop cannot see, and whatever loading
    the config raises; each caller handles both under its own error contract.
    """
    try:
        ensure_backstop()
    except BaseException:
        coro.close()
        raise
    return asyncio.run(on_covered_loop(coro))


def disarm_backstop() -> None:
    """Make the backstop a no-op (tests, and a reload to ``open``)."""
    global _backstop_policy
    _backstop_policy = None
