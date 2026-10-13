# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the opt-in log-probability side channel.

Covers the default (no scope: nothing requested, nothing recorded), the
request members and recording inside a scope, the unchanged return contract,
scope isolation, payload parsing, and span-to-token alignment. The endpoint is
mocked per ``tests/AGENTS.md``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from particles.llm.adapters import openai_compat as oc_mod
from particles.llm.adapters.openai_compat import OpenAICompatProvider
from particles.llm.logprobs import (
    CapturedCompletion,
    TokenLogprob,
    active_capture,
    capture_logprobs,
    parse_openai_logprobs,
    span_tokens,
)


def _payload(tokens: list[tuple[str, float]], *, top: int = 2) -> dict[str, object]:
    content = "".join(t for t, _ in tokens)
    return {
        "choices": [
            {
                "message": {"content": content},
                "finish_reason": "stop",
                "logprobs": {
                    "content": [
                        {
                            "token": t,
                            "logprob": lp,
                            "top_logprobs": [
                                {"token": t, "logprob": lp},
                                {"token": "x", "logprob": lp - 2.0},
                            ][:top],
                        }
                        for t, lp in tokens
                    ]
                },
            }
        ]
    }


def _install(monkeypatch: pytest.MonkeyPatch, payload: dict[str, object]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.post = AsyncMock(return_value=resp)
    monkeypatch.setattr(oc_mod.httpx, "AsyncClient", lambda **_kw: client)
    monkeypatch.delenv("PARTICLES_LOCAL_LLM_API_KEY", raising=False)
    monkeypatch.delenv("PARTICLES_LLM_API_KEY_LOCAL", raising=False)
    return client


_TOKENS = [(' {"a"', -0.1), (": ", -0.01), ("0", -0.2), (".9", -1.5), ("} ", -0.05)]


def test_no_scope_requests_nothing_and_records_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _install(monkeypatch, _payload(_TOKENS))
    out = asyncio.run(OpenAICompatProvider(name="local", model="m").complete("p", max_tokens=8))
    body = client.post.call_args.kwargs["json"]
    assert "logprobs" not in body
    assert "top_logprobs" not in body
    assert out == '{"a": 0.9}'
    assert active_capture() is None


def test_scope_requests_and_records_without_changing_the_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _install(monkeypatch, _payload(_TOKENS))
    with capture_logprobs(top_logprobs=2) as cap:
        out = asyncio.run(OpenAICompatProvider(name="local", model="m").complete("p", max_tokens=8))
    body = client.post.call_args.kwargs["json"]
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 2
    # The return contract is unchanged: plain, stripped text.
    assert out == '{"a": 0.9}'
    [rec] = cap.completions
    assert rec.provider_model == "local:m"
    # The captured text is the raw token concatenation, unstripped.
    assert rec.text == ' {"a": 0.9} '
    assert rec.tokens[3] == TokenLogprob(token=".9", logprob=-1.5, top=((".9", -1.5), ("x", -3.5)))
    assert active_capture() is None


def test_top_logprobs_zero_asks_for_chosen_tokens_only(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _install(monkeypatch, _payload(_TOKENS))
    with capture_logprobs(top_logprobs=0):
        asyncio.run(OpenAICompatProvider(name="local", model="m").complete("p", max_tokens=8))
    body = client.post.call_args.kwargs["json"]
    assert body["logprobs"] is True
    assert "top_logprobs" not in body


def test_reply_without_logprobs_records_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"choices": [{"message": {"content": "ok"}}]})
    with capture_logprobs() as cap:
        out = asyncio.run(OpenAICompatProvider(name="local", model="m").complete("p", max_tokens=8))
    assert out == "ok"
    assert cap.completions == []


def test_concurrent_scopes_do_not_bleed() -> None:
    from particles.llm.logprobs import record_logprobs

    async def one(model: str) -> list[CapturedCompletion]:
        with capture_logprobs() as cap:
            await asyncio.sleep(0)
            record_logprobs(model, _payload(_TOKENS))
            await asyncio.sleep(0)
        return cap.completions

    async def both() -> tuple[list[CapturedCompletion], list[CapturedCompletion]]:
        a, b = await asyncio.gather(one("p:a"), one("p:b"))
        return a, b

    a, b = asyncio.run(both())
    assert [c.provider_model for c in a] == ["p:a"]
    assert [c.provider_model for c in b] == ["p:b"]


def test_top_logprobs_out_of_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="top_logprobs"), capture_logprobs(top_logprobs=21):
        pass


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"choices": []},
        {"choices": [{"logprobs": None}]},
        {"choices": [{"logprobs": {"content": "nope"}}]},
    ],
)
def test_parse_tolerates_missing_logprobs(payload: object) -> None:
    assert parse_openai_logprobs(payload) == ()


def test_parse_skips_malformed_entries() -> None:
    payload = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {"token": "a", "logprob": -1},
                        {"token": 3, "logprob": -1.0},
                        "junk",
                        {"token": "b", "logprob": -0.5, "top_logprobs": [{"token": "c"}]},
                    ]
                }
            }
        ]
    }
    assert parse_openai_logprobs(payload) == (
        TokenLogprob("a", -1.0),
        TokenLogprob("b", -0.5),
    )


def test_span_tokens_covers_straddling_tokens() -> None:
    comp = CapturedCompletion(
        provider_model="p:m",
        tokens=tuple(TokenLogprob(t, lp) for t, lp in _TOKENS),
    )
    text = comp.text
    start = text.index("0.9")
    got = span_tokens(comp, start, start + 3)
    assert [t.token for t in got] == ["0", ".9"]
    # A span starting inside a token includes that token.
    got = span_tokens(comp, text.index('"a"') + 1, text.index('"a"') + 2)
    assert [t.token for t in got] == [' {"a"']
    assert span_tokens(comp, 5, 5) == []
    assert span_tokens(comp, 100, 120) == []
