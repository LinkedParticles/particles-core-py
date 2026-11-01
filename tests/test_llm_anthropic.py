# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The native Anthropic adapter scrubs the SDK's own errors (security review 2026-10-25).

Q13 redacted provider error text on the OpenAI-compatible path; the native
adapter raised the Anthropic SDK's exceptions unchanged, so a key the API
echoed reached the extraction log, the breaker log and a quality note. Every
transport here is an in-process ``MockTransport``; nothing leaves the process.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

import anthropic
import httpx2
import pytest

from particles.llm import client as client_mod
from particles.llm.adapters.anthropic import AnthropicProvider
from particles.llm.registry import CompletionRequest

_KEY = "sk-ant-echoed-key-7d1e4c2a9b"
_TOKEN = "auth-token-echoed-5c3b1a0f8e"


def _install(
    handler: Callable[[httpx2.Request], httpx2.Response],
    *,
    api_key: str | None = _KEY,
    auth_token: str | None = None,
) -> None:
    client = anthropic.Anthropic(
        api_key=api_key,
        auth_token=auth_token,
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    client_mod.set_client(client)


@pytest.fixture(autouse=True)
def _clear_client() -> Iterator[None]:
    yield
    client_mod.set_client(None)


def _echo_key(status: int) -> Callable[[httpx2.Request], httpx2.Response]:
    def handler(request: httpx2.Request) -> httpx2.Response:
        sent = request.headers.get("x-api-key") or request.headers.get("authorization", "")
        return httpx2.Response(
            status,
            json={
                "type": "error",
                "error": {"type": "authentication_error", "message": f"invalid key: {sent}"},
            },
        )

    return handler


def _assert_scrubbed(exc: BaseException, secret: str) -> None:
    assert secret not in str(exc)
    assert secret not in repr(exc.args)
    assert secret not in str(getattr(exc, "message", ""))
    assert secret not in repr(getattr(exc, "body", ""))
    assert "[REDACTED]" in str(exc)


@pytest.mark.asyncio
async def test_a_single_call_error_is_raised_without_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", _KEY)
    _install(_echo_key(401))
    with pytest.raises(anthropic.AuthenticationError) as caught:
        await AnthropicProvider(model="claude-test").complete("hi", max_tokens=16)
    _assert_scrubbed(caught.value, _KEY)
    # The type and status survive, so the breaker still classifies it.
    assert caught.value.status_code == 401


@pytest.mark.asyncio
async def test_a_token_the_environment_scan_misses_is_scrubbed_too() -> None:
    """The SDK reads ``ANTHROPIC_AUTH_TOKEN`` itself; the client's own credentials are redacted."""
    _install(_echo_key(401), api_key=None, auth_token=_TOKEN)
    with pytest.raises(anthropic.AuthenticationError) as caught:
        await AnthropicProvider(model="claude-test").complete("hi", max_tokens=16)
    _assert_scrubbed(caught.value, _TOKEN)


@pytest.mark.asyncio
async def test_a_batch_error_is_raised_without_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", _KEY)
    _install(_echo_key(403))
    with pytest.raises(anthropic.PermissionDeniedError) as caught:
        await AnthropicProvider(model="claude-test").complete_many(
            [CompletionRequest(prompt="hi")], max_tokens=16
        )
    _assert_scrubbed(caught.value, _KEY)


@pytest.mark.asyncio
async def test_the_temperature_retry_warning_carries_no_key(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", _KEY)
    calls: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx2.Response(
                400,
                json={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"temperature is deprecated (key {_KEY})",
                    },
                },
            )
        return httpx2.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    _install(handler)
    with caplog.at_level(logging.WARNING):
        text = await AnthropicProvider(model="claude-test-temp").complete(
            "hi", max_tokens=16, temperature=0.0
        )
    assert text == "ok"
    assert "temperature" in caplog.text
    assert _KEY not in caplog.text


@pytest.mark.asyncio
async def test_the_breaker_log_carries_no_key(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """End to end through the semantic seam, which logs the failure it trips on."""
    from particles.llm.breaker import _llm_call

    monkeypatch.setenv("ANTHROPIC_API_KEY", _KEY)
    _install(_echo_key(401))
    with caplog.at_level(logging.INFO):
        assert await _llm_call("hi") is None
    assert "401" in caplog.text or "invalid key" in caplog.text
    assert _KEY not in caplog.text


_GATEWAY_SECRET = "gw-header-secret-0123456789abcdef"


def _built_client(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx2.Request], httpx2.Response]
) -> None:
    """The production client (``get_client``), its transport swapped for a mock."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", _KEY)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    client_mod.set_client(None)
    sdk = client_mod.get_client()
    inner = sdk._client
    inner._transport = httpx2.MockTransport(handler)
    inner._mounts = {}
    client_mod.set_client(sdk.with_options(max_retries=0))


def _echo_all_sent_credentials(request: httpx2.Request) -> httpx2.Response:
    echoed = " ".join(
        v
        for k, v in request.headers.items()
        if k.lower() in ("x-api-key", "authorization", "x-gateway-key")
    )
    return httpx2.Response(
        401,
        json={
            "type": "error",
            "error": {"type": "authentication_error", "message": f"rejected: {echoed}"},
        },
    )


@pytest.mark.asyncio
async def test_a_custom_header_credential_is_not_in_the_raised_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R9 (SECURITY_REVIEW_20261025-2): a gateway credential from ANTHROPIC_CUSTOM_HEADERS."""
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", f"x-gateway-key: Bearer {_GATEWAY_SECRET}")
    _built_client(monkeypatch, _echo_all_sent_credentials)
    with pytest.raises(anthropic.AuthenticationError) as caught:
        await AnthropicProvider(model="claude-test").complete("hi", max_tokens=16)
    exc = caught.value
    _assert_scrubbed(exc, _KEY)
    _assert_scrubbed(exc, _GATEWAY_SECRET)
    assert _GATEWAY_SECRET not in exc.response.text


def test_the_environment_scan_lists_the_variables_the_sdk_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R9: the AccountLevelLLMError and hook-log scrub sees the SDK's own credentials too."""
    from particles.secrets import redact_known_secrets

    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", _TOKEN)
    monkeypatch.setenv(
        "ANTHROPIC_CUSTOM_HEADERS", f"x-gateway-key: Bearer {_GATEWAY_SECRET}\nx-trace: on"
    )
    text = redact_known_secrets(f"token {_TOKEN} header {_GATEWAY_SECRET} trace on")
    assert _TOKEN not in text
    assert _GATEWAY_SECRET not in text
    # A value shorter than the redaction floor stays: "on" is no credential.
    assert text.endswith("trace on")


@pytest.mark.asyncio
async def test_the_raised_error_response_body_carries_no_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R37 (SECURITY_REVIEW_20261025-2): no attribute of the raised error holds the echo."""
    _built_client(monkeypatch, _echo_all_sent_credentials)
    with pytest.raises(anthropic.AuthenticationError) as caught:
        await AnthropicProvider(model="claude-test").complete("hi", max_tokens=16)
    exc = caught.value
    _assert_scrubbed(exc, _KEY)
    assert _KEY not in exc.response.text
    assert _KEY.encode() not in exc.response.content
    assert "[REDACTED]" in exc.response.text
    assert exc.response.status_code == 401


@pytest.mark.asyncio
async def test_a_completion_request_carries_no_tool_definitions() -> None:
    """The model has no tools: a caller's tool options are never forwarded, and a
    tool-use block in the reply is never read as the answer (security review
    2026-10-26-1, J11)."""
    from types import SimpleNamespace

    sent: list[dict[str, object]] = []

    class _Messages:
        def create(self, **kwargs: object) -> object:
            sent.append(kwargs)
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="tool_use", name="x", input={}),
                    SimpleNamespace(type="text", text="ok"),
                ],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            )

    client_mod.set_client(SimpleNamespace(messages=_Messages()))  # type: ignore[arg-type]
    try:
        out = await AnthropicProvider(model="claude-test").complete(
            "prompt",
            max_tokens=16,
            system="instructions",
            tools=[{"name": "shell"}],
            tool_choice={"type": "any"},
        )
    finally:
        client_mod.set_client(None)
    assert out == "ok"
    assert sent
    for kwargs in sent:
        assert "tools" not in kwargs and "tool_choice" not in kwargs, kwargs
        assert "tools" not in (kwargs.get("extra_body") or {}), kwargs  # type: ignore[operator]
