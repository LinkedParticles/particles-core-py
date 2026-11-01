# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""The native Anthropic client's redirect and egress guard on whatever SDK is installed.

``particles.llm.client.get_client`` builds ``DefaultHttpxClient`` with
``follow_redirects=False`` and the egress request hook. The published core
admits a range of anthropic releases, so CI's SDK-range job runs this file on
the core's floor and on the newest admitted 2.x, beside the default suite's
locked release (security review 2026-10-28-1, Z45, R48).

Each case drives the client in a child process, since the suite's own guard
refuses every SDK request in process, against two loopback servers the child
starts: the endpoint answers a retryable 500 and then a 307 to the second
server. The key is a dummy literal only the loopback server receives.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

CHILD = textwrap.dedent(
    """
    import json, os, sys, threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    hits = {"endpoint": 0, "redirect_target": 0}

    class Target(BaseHTTPRequestHandler):
        def do_POST(self):
            hits["redirect_target"] += 1
            self.send_response(200); self.send_header("content-length", "0"); self.end_headers()
        do_GET = do_POST
        def log_message(self, *a): pass

    target = HTTPServer(("127.0.0.1", 0), Target)

    class Endpoint(BaseHTTPRequestHandler):
        def do_POST(self):
            hits["endpoint"] += 1
            self.rfile.read(int(self.headers.get("content-length") or 0))
            if hits["endpoint"] == 1:
                self.send_response(500); self.send_header("content-length", "0"); self.end_headers()
                return
            self.send_response(307)
            self.send_header("location", f"http://127.0.0.1:{target.server_port}/v1/messages")
            self.send_header("content-length", "0"); self.end_headers()
        def log_message(self, *a): pass

    endpoint = HTTPServer(("127.0.0.1", 0), Endpoint)
    for s in (target, endpoint):
        threading.Thread(target=s.serve_forever, daemon=True).start()

    os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{endpoint.server_port}"
    import anthropic
    import particles.llm.client as lc

    seen = []
    real_hook = lc.egress_request_hook
    def counting_hook(purpose):
        inner = real_hook(purpose)
        def _hook(request):
            seen.append(str(request.url))
            inner(request)
        return _hook
    lc.egress_request_hook = counting_hook

    out = {"version": anthropic.__version__}
    try:
        client = lc.get_client()
    except Exception as exc:
        out["build_error"] = type(exc).__name__ + ": " + str(exc)[:200]
        print(json.dumps(out)); sys.exit(0)
    try:
        client.with_options(max_retries=1).messages.create(
            model="m", max_tokens=1, messages=[{"role": "user", "content": "x"}]
        )
        out["outcome"] = "returned"
    except Exception as exc:
        out["outcome"] = type(exc).__name__
    out["hook_calls"] = len(seen)
    out.update(hits)
    print(json.dumps(out))
    """
)

_LOOPBACK_ONLY = "network:\n  egress: allowlist\n  allow:\n    - host: 127.0.0.1\n"


def _run(tmp_path: Path, egress_yaml: str) -> dict[str, object]:
    home = tmp_path / "home"
    home.mkdir()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(egress_yaml, encoding="utf-8")
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("ANTHROPIC_", "PARTICLES_LLM_API_KEY_"))
        and k.lower() not in {"http_proxy", "https_proxy", "all_proxy"}
    }
    env.update(
        HOME=str(home),
        PARTICLES_CONFIG=str(cfg),
        ANTHROPIC_API_KEY="dummy-loopback-only",
        NO_PROXY="*",
    )
    child = subprocess.run(
        [sys.executable, "-c", CHILD],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode == 0, child.stderr[-2000:]
    return json.loads(child.stdout.strip().splitlines()[-1])  # type: ignore[no-any-return]


def test_the_native_client_follows_no_redirect_and_checks_every_send(tmp_path: Path) -> None:
    """The client builds, follows no 3xx from the endpoint, and runs the egress
    hook on every send, the retry included."""
    out = _run(tmp_path, _LOOPBACK_ONLY)
    assert "build_error" not in out, out
    assert out["redirect_target"] == 0, out
    assert out["endpoint"] == 2, out
    assert out["hook_calls"] == out["endpoint"], out


def test_the_native_client_sends_nothing_when_egress_is_none(tmp_path: Path) -> None:
    """Under ``network.egress: none`` the client sends no request at all."""
    out = _run(tmp_path, "network:\n  egress: none\n")
    assert "build_error" not in out, out
    assert out["endpoint"] == 0 and out["redirect_target"] == 0, out
    assert out["hook_calls"] >= 1, out


@pytest.mark.parametrize("name", ["ANTHROPIC_OPEN_TELEMETRY_CONTENT_MODE"])
def test_the_client_is_built_with_its_own_telemetry_options(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The client passes a complete telemetry option, so the environment cannot
    turn on content mode or trace propagation on any SDK in the range."""
    import anthropic

    from particles.llm import client as client_mod

    seen: dict[str, object] = {}
    real = anthropic.Anthropic

    def _capture(**kwargs: object) -> object:
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setenv(name, "full")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-not-sent")
    monkeypatch.setattr(anthropic, "Anthropic", _capture)
    client_mod.set_client(None)
    try:
        client_mod.get_client()
    finally:
        client_mod.set_client(None)
    assert seen["open_telemetry"] == {
        "traces": {"content_mode": "metadata_only"},
        "propagation": False,
    }
