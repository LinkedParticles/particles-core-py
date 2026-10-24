# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The no-egress policy, its config, and the process-wide backstop.

Store-free: this file rides the Client distribution's tests too.
"""

from __future__ import annotations

import gc
import socket
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import ValidationError

from particles import egress
from particles.config import EgressAllowEntry, NetworkConfig, get_config


def _policy(posture: str, *entries: tuple[str, int | None]) -> egress.EgressPolicy:
    return egress.compile_policy(
        NetworkConfig(
            egress=posture,  # type: ignore[arg-type]
            allow=[EgressAllowEntry(host=h, port=p) for h, p in entries],
        )
    )


def _set_network(monkeypatch: pytest.MonkeyPatch, network: NetworkConfig) -> None:
    monkeypatch.setattr(get_config(), "network", network)


class TestPolicyTable:
    def test_open_allows_everything(self) -> None:
        policy = _policy("open")
        assert policy.allows("api.anthropic.com", 443)
        assert policy.allows("10.0.0.5", 80)
        assert not policy.refusing

    def test_none_refuses_loopback_too(self) -> None:
        policy = _policy("none")
        for host in ("127.0.0.1", "::1", "localhost", "10.0.0.5", "api.anthropic.com"):
            assert not policy.allows(host, 443), host

    @pytest.mark.parametrize(
        ("host", "port", "allowed"),
        [
            ("127.0.0.1", 11434, True),
            ("127.0.0.1", 8080, False),  # wrong port
            ("localhost", 11434, False),  # a name is not its address: never inferred
            ("::1", 11434, False),
            ("10.0.0.5", 443, True),  # inside the CIDR, any port
            ("10.0.1.5", 443, False),
            ("100.64.0.1", 443, False),  # private-ish ranges are not local by default
            ("models.lan", 8000, True),
            ("MODELS.LAN.", 8000, True),  # case and trailing dot normalised
            ("models.lan", 8001, False),
            ("other.lan", 8000, False),
        ],
    )
    def test_allowlist(self, host: str, port: int, allowed: bool) -> None:
        policy = _policy(
            "allowlist", ("127.0.0.1", 11434), ("10.0.0.0/24", None), ("models.lan", 8000)
        )
        assert policy.allows(host, port) is allowed

    def test_localhost_entry_does_not_admit_its_address(self) -> None:
        policy = _policy("allowlist", ("localhost", None))
        assert policy.allows("localhost", 1234)
        assert not policy.allows("127.0.0.1", 1234)

    def test_idna_name(self) -> None:
        policy = _policy("allowlist", ("bücher.example", None))
        assert policy.allows("xn--bcher-kva.example", 443)
        assert policy.allows("Bücher.example", 443)

    def test_lookup_rule(self) -> None:
        policy = _policy("allowlist", ("models.lan", 8000))
        assert policy.allows_lookup("models.lan")
        assert policy.allows_lookup("192.0.2.1")  # a literal needs no lookup
        assert not policy.allows_lookup("www.wikidata.org")


class TestConfig:
    def test_default_is_open(self) -> None:
        assert NetworkConfig().egress == "open"

    def test_none_with_entries_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="use egress: allowlist"):
            NetworkConfig(egress="none", allow=[EgressAllowEntry(host="127.0.0.1")])

    @pytest.mark.parametrize("host", ["*.lan", "", "10.0.0.0/33", "models.lan:8000"])
    def test_bad_hosts_are_rejected(self, host: str) -> None:
        with pytest.raises(ValidationError):
            EgressAllowEntry(host=host)

    def test_port_range(self) -> None:
        with pytest.raises(ValidationError):
            EgressAllowEntry(host="127.0.0.1", port=70000)


class TestChecks:
    def test_check_raises_with_the_knob_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_network(monkeypatch, NetworkConfig(egress="none"))
        with pytest.raises(egress.EgressRefused) as info:
            egress.check_url("https://www.wikidata.org/w/api.php", "fetch")
        assert info.value.host == "www.wikidata.org"
        assert info.value.port == 443
        assert "network.egress is none" in str(info.value)
        assert "network.allow" in str(info.value)

    def test_refusal_is_not_a_value_error(self) -> None:
        # A broad ``except ValueError`` (the SSRF path) must not swallow it.
        assert not issubclass(egress.EgressRefused, ValueError)

    def test_host_port_defaults(self) -> None:
        assert egress.host_port("http://localhost:11434/v1") == ("localhost", 11434)
        assert egress.host_port("https://api.notion.com/v1") == ("api.notion.com", 443)
        assert egress.host_port("http://[::1]/x") == ("::1", 80)

    def test_policy_follows_config_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert egress.allows("example.org", 443)
        _set_network(monkeypatch, NetworkConfig(egress="none"))
        assert not egress.allows("example.org", 443)

    def test_a_fresh_config_is_never_served_a_released_ones_policy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Each round releases the previous config before building the next, so
        # CPython tends to allocate the new one at the freed address. A cache
        # keyed on ``id()`` then serves the released config's policy (a CI
        # flake, 2026-10-22). The swaps bypass monkeypatch, which would hold
        # every replaced value alive; one monkeypatch call up front restores
        # the original.
        cfg = get_config()
        original = cfg.network
        _set_network(monkeypatch, original)
        for i in range(50):
            cfg.network = original
            gc.collect()
            posture = "none" if i % 2 else "open"
            cfg.network = NetworkConfig(egress=posture)
            assert egress.current_policy().posture == posture
        cached = egress._policy_cache
        assert cached is not None
        assert cached[0] is get_config().network


@pytest.fixture
def backstop(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Arm the backstop for one test over a stub resolver, then disarm it.

    An audit hook cannot be removed, so the hook stays installed for the rest of
    the session; disarming makes it a no-op, which is the production behaviour
    under ``open``.
    """
    stub = {"models.test": "127.0.0.1"}
    real = egress._real_getaddrinfo or socket.getaddrinfo

    def fake(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(host, str) and host in stub:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (stub[host], port or 0))]
        return real(host, port, *args, **kwargs)

    _set_network(
        monkeypatch,
        NetworkConfig(
            egress="allowlist",
            allow=[EgressAllowEntry(host="models.test", port=8000)],
        ),
    )
    egress.ensure_backstop()
    monkeypatch.setattr(egress, "_real_getaddrinfo", fake)
    try:
        yield
    finally:
        egress.disarm_backstop()


class TestBackstop:
    def test_refuses_a_lookup_of_an_unlisted_name(self, backstop: None) -> None:
        with pytest.raises(egress.EgressRefused):
            socket.getaddrinfo("www.wikidata.org", 443)

    def test_refuses_a_connect_to_an_unadmitted_address(self, backstop: None) -> None:
        with (
            socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
            pytest.raises(egress.EgressRefused),
        ):
            sock.connect(("127.0.0.1", 8000))

    def test_a_listed_name_admits_what_it_resolved_to(self, backstop: None) -> None:
        """The name-entry case: the connect sees only the address."""
        socket.getaddrinfo("models.test", 8000)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            # Reaching the OS (a refused or accepted connect) proves the hook
            # admitted the address; only a refusal from the hook fails here.
            try:
                sock.connect(("127.0.0.1", 8000))
            except egress.EgressRefused:
                pytest.fail("the backstop refused an address its listed name resolved to")
            except OSError:
                pass
        # The admission is port-scoped like the entry.
        with (
            socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
            pytest.raises(egress.EgressRefused),
        ):
            sock.connect(("127.0.0.1", 8001))

    def test_unix_sockets_and_disarmed_hook_are_untouched(self, backstop: None) -> None:
        a, b = socket.socketpair()
        a.close()
        b.close()
        egress.disarm_backstop()
        assert socket.getaddrinfo("127.0.0.1", 80)
