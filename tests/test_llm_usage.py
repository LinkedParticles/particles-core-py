# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-run LLM usage accumulator (``particles/llm/usage.py``).

The Anthropic client is mocked through the ``set_client`` seam to return known
``usage`` values, so every total below is exact: sums across calls and
purposes, isolation between concurrent scopes, the ``max_tokens`` stop count,
the batch path, the OpenAI-compatible ``usage`` field, and list-price pricing
with the batch and prompt-cache multipliers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import anthropic
import pytest

from particles import llm
from particles.config import ProviderSelection, TokenPrice, get_config
from particles.llm.adapters.anthropic import AnthropicProvider
from particles.llm.adapters.openai_compat import _extract_text
from particles.llm.usage import (
    UNATTRIBUTED,
    LLMUsage,
    UsageRow,
    format_tokens,
    record_usage,
    render_usage_line,
    track_usage,
)


@pytest.fixture(autouse=True)
def _reset_client() -> Generator[None, None, None]:
    llm.set_client(None)
    yield
    llm.set_client(None)


def _message(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_creation: int = 0,
    cache_read: int = 0,
    stop_reason: str = "end_turn",
    text: str = "ok",
) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        ),
    )


def _client(*responses: SimpleNamespace) -> MagicMock:
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages = MagicMock()
    client.messages.create = MagicMock(side_effect=list(responses))
    return client


def _route(purpose: str, model: str) -> None:
    setattr(get_config().llm, purpose, ProviderSelection(provider="anthropic", model=model))


class TestAccumulator:
    async def test_sums_across_calls_per_purpose_and_model(self) -> None:
        _route("extraction", "claude-sonnet-5")
        _route("semantic_lint", "claude-haiku-4-5")
        llm.set_client(
            _client(
                _message(input_tokens=1_000, output_tokens=200, cache_read=500),
                _message(input_tokens=3_000, output_tokens=800, cache_creation=100),
                _message(input_tokens=50, output_tokens=5),
            )
        )
        with track_usage() as usage:
            await llm.complete("extraction", "a", max_tokens=8192)
            await llm.complete("extraction", "b", max_tokens=8192)
            await llm.complete("semantic_lint", "c", max_tokens=100)

        snap = usage.snapshot()
        rows = {(r.purpose, r.model): r for r in snap.rows}
        extraction = rows[("extraction", "claude-sonnet-5")]
        assert (extraction.calls, extraction.input_tokens, extraction.output_tokens) == (
            2,
            4_000,
            1_000,
        )
        assert extraction.cache_read_tokens == 500
        assert extraction.cache_creation_tokens == 100
        probe = rows[("semantic_lint", "claude-haiku-4-5")]
        assert (probe.calls, probe.input_tokens, probe.output_tokens) == (1, 50, 5)
        assert snap.calls == 3
        assert snap.max_tokens_stops == 0

    async def test_counts_max_tokens_stops(self) -> None:
        llm.set_client(
            _client(
                _message(input_tokens=10, output_tokens=8192, stop_reason="max_tokens"),
                _message(input_tokens=10, output_tokens=20),
                _message(input_tokens=10, output_tokens=8192, stop_reason="max_tokens"),
            )
        )
        with track_usage() as usage:
            for _ in range(3):
                await llm.complete("extraction", "x", max_tokens=8192)
        snap = usage.snapshot()
        assert snap.max_tokens_stops == 2
        assert snap.rows[0].max_tokens_stops == 2
        assert "2 replies stopped at max_tokens" in render_usage_line(snap)

    async def test_a_refused_reply_is_still_counted(self) -> None:
        # Billed even though no text comes back, so it must be in the total.
        llm.set_client(_client(_message(input_tokens=400, output_tokens=0, stop_reason="refusal")))
        with track_usage() as usage, pytest.raises(llm.CompletionError):
            await llm.complete("extraction", "x", max_tokens=100)
        assert usage.snapshot().rows[0].input_tokens == 400

    async def test_two_concurrent_scopes_do_not_bleed(self) -> None:
        llm.set_client(
            MagicMock(
                spec=anthropic.Anthropic,
                messages=MagicMock(
                    create=MagicMock(
                        side_effect=lambda **kw: _message(
                            input_tokens=len(kw["messages"][0]["content"]), output_tokens=1
                        )
                    )
                ),
            )
        )

        async def run(prompt: str, n: int) -> LLMUsage:
            with track_usage() as usage:
                for _ in range(n):
                    await llm.complete("extraction", prompt, max_tokens=10)
                    await asyncio.sleep(0)
            return usage.snapshot()

        first, second = await asyncio.gather(run("a" * 10, 3), run("b" * 1_000, 2))
        assert (first.calls, first.rows[0].input_tokens) == (3, 30)
        assert (second.calls, second.rows[0].input_tokens) == (2, 2_000)

    async def test_nested_scopes_both_count_and_outside_is_a_no_op(self) -> None:
        llm.set_client(
            _client(
                _message(input_tokens=1, output_tokens=1),
                _message(input_tokens=2, output_tokens=2),
                _message(input_tokens=4, output_tokens=4),
            )
        )
        await llm.complete("extraction", "before", max_tokens=10)  # no scope: not counted
        with track_usage() as outer:
            await llm.complete("extraction", "outer", max_tokens=10)
            with track_usage() as inner:
                await llm.complete("extraction", "inner", max_tokens=10)
        assert outer.snapshot().rows[0].input_tokens == 6
        assert inner.snapshot().rows[0].input_tokens == 4

    async def test_direct_adapter_call_is_unattributed(self) -> None:
        llm.set_client(_client(_message(input_tokens=7, output_tokens=1)))
        with track_usage() as usage:
            await AnthropicProvider(model="claude-haiku-4-5").complete("x", max_tokens=10)
        assert usage.snapshot().rows[0].purpose == UNATTRIBUTED

    async def test_batch_results_are_recorded_as_batched(self) -> None:
        results = [
            SimpleNamespace(
                custom_id=str(i),
                result=SimpleNamespace(
                    type="succeeded",
                    message=_message(input_tokens=1_000_000, output_tokens=100_000),
                ),
            )
            for i in range(2)
        ]
        client = MagicMock(spec=anthropic.Anthropic)
        client.messages = MagicMock()
        client.messages.batches.results = MagicMock(return_value=results)
        llm.set_client(client)
        with track_usage() as usage:
            AnthropicProvider._collect("b1", 100, "anthropic:claude-haiku-4-5")
        snap = usage.snapshot()
        row = snap.rows[0]
        assert (row.batched, row.calls, row.input_tokens) == (True, 2, 2_000_000)
        # $1/$5 per MTok: 2M in + 0.2M out = $3.00 at list price, halved.
        assert row.cost_usd == pytest.approx(1.50)
        assert "2 batched" in render_usage_line(snap)


class TestOpenAICompatUsage:
    def test_usage_field_is_recorded_with_the_cached_share_split_out(self) -> None:
        payload: dict[str, Any] = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "length"}],
            "usage": {
                "prompt_tokens": 1_000,
                "completion_tokens": 64,
                "prompt_tokens_details": {"cached_tokens": 300},
            },
        }
        with track_usage() as usage:
            _extract_text(payload, provider_model="local:llama3.1", max_tokens=64)
        row = usage.snapshot().rows[0]
        assert (row.provider, row.model) == ("local", "llama3.1")
        assert (row.input_tokens, row.cache_read_tokens, row.output_tokens) == (700, 300, 64)
        assert row.max_tokens_stops == 1

    def test_a_reply_without_usage_still_counts_the_call(self) -> None:
        payload = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
        with track_usage() as usage:
            _extract_text(payload, provider_model="local:llama3.1")
        row = usage.snapshot().rows[0]
        assert (row.calls, row.input_tokens, row.output_tokens) == (1, 0, 0)


class TestPricing:
    def test_cache_multipliers_apply_to_the_input_price(self) -> None:
        with track_usage() as usage:
            record_usage(
                "anthropic:claude-sonnet-5",
                input_tokens=1_000_000,
                output_tokens=100_000,
                cache_creation_tokens=1_000_000,
                cache_read_tokens=1_000_000,
            )
        # $2/$10 per MTok: 2.00 input + 2.50 cache write (1.25x) + 0.20 cache
        # read (0.1x) + 1.00 output.
        assert usage.snapshot().cost_usd == pytest.approx(5.70)

    def test_an_unpriced_model_prices_nothing(self) -> None:
        with track_usage() as usage:
            record_usage("anthropic:claude-sonnet-5", input_tokens=1_000, output_tokens=10)
            record_usage("local:qwen3", input_tokens=5_000, output_tokens=50)
        snap = usage.snapshot()
        assert snap.cost_usd is None
        assert snap.unpriced == ["local:qwen3"]
        line = render_usage_line(snap)
        assert "$" not in line
        assert "5k input, 50 output tokens" in line
        assert "local:qwen3 has no entry in llm.price_per_mtok" in line

    def test_an_operator_price_entry_prices_a_provider_route(self) -> None:
        get_config().llm.price_per_mtok["local:qwen3"] = TokenPrice(input=1.0, output=2.0)
        with track_usage() as usage:
            record_usage("local:qwen3", input_tokens=1_000_000, output_tokens=1_000_000)
        assert usage.snapshot().cost_usd == pytest.approx(3.0)


class TestRendering:
    def test_line_shape(self) -> None:
        usage = LLMUsage(
            rows=[
                UsageRow(
                    purpose="extraction",
                    provider="anthropic",
                    model="claude-sonnet-5",
                    calls=96,
                    input_tokens=312_000,
                    output_tokens=640_000,
                ),
                UsageRow(
                    purpose="semantic_lint",
                    provider="anthropic",
                    model="claude-haiku-4-5",
                    calls=1,
                    input_tokens=300,
                    output_tokens=5,
                ),
            ],
            cost_usd=9.2,
        )
        assert render_usage_line(usage) == (
            "LLM usage: 96 extraction calls (claude-sonnet-5), 312k input, 640k output "
            "tokens; 1 semantic lint call (claude-haiku-4-5), 300 input, 5 output tokens; "
            "≈ $9.20 at list price."
        )

    def test_no_calls(self) -> None:
        assert render_usage_line(LLMUsage()) == "LLM usage: no LLM calls."

    def test_token_formatting(self) -> None:
        assert [format_tokens(n) for n in (999, 1_000, 312_400, 1_250_000)] == [
            "999",
            "1k",
            "312k",
            "1.2M",
        ]
