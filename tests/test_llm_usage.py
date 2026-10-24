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
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import anthropic
import pytest

from particles import llm
from particles.config import ProviderSelection, TokenPrice, get_config, reset_config
from particles.llm.adapters.anthropic import AnthropicProvider
from particles.llm.adapters.openai_compat import _extract_text
from particles.llm.usage import (
    UNATTRIBUTED,
    LLMUsage,
    UsageRow,
    format_tokens,
    observe_sampling,
    purpose_scope,
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


_TEMPERATURE_REJECTION = "`temperature` is deprecated for this model."


def _rejecting_client() -> MagicMock:
    """A client that 400s on any call carrying ``temperature``."""

    def _create(**kwargs: object) -> object:
        if kwargs.get("extra_body") is not None:
            exc = RuntimeError(_TEMPERATURE_REJECTION)
            exc.status_code = 400  # type: ignore[attr-defined]
            raise exc
        return _message(input_tokens=10, output_tokens=2)

    client = MagicMock(spec=anthropic.Anthropic)
    client.messages = MagicMock()
    client.messages.create = MagicMock(side_effect=_create)
    return client


class TestSampling:
    """What each response sent, counted per slice and observed per block."""

    @pytest.fixture(autouse=True)
    def _forget_temperature_memo(self) -> Generator[None, None, None]:
        from particles.llm.adapters import anthropic as adapter

        adapter._TEMPERATURE_UNSUPPORTED.clear()
        yield
        adapter._TEMPERATURE_UNSUPPORTED.clear()

    async def test_a_sent_temperature_is_requested_and_not_dropped(self) -> None:
        llm.set_client(_client(_message(input_tokens=10, output_tokens=2)))
        with track_usage() as usage, observe_sampling() as seen:
            await llm.complete("semantic_lint", "p", max_tokens=8, temperature=0.0)
        row = usage.snapshot().rows[0]
        assert (row.temperature_requested, row.temperature_dropped) == (1, 0)
        assert seen.value() == "sent"
        assert "Fixed temperature not sent" not in render_usage_line(usage.snapshot())

    async def test_a_refused_temperature_is_counted_once_as_dropped(self) -> None:
        llm.set_client(_rejecting_client())
        with track_usage() as usage, observe_sampling() as seen:
            await llm.complete("semantic_lint", "p", max_tokens=8, temperature=0.0)
            await llm.complete("semantic_lint", "q", max_tokens=8, temperature=0.0)
        snap = usage.snapshot()
        row = snap.rows[0]
        # The refused attempt is not billed and not counted; the retry is.
        assert (row.calls, row.temperature_requested, row.temperature_dropped) == (2, 2, 2)
        assert snap.temperature_dropped == 2
        assert seen.value() == "dropped"
        assert "Fixed temperature not sent (semantic lint: 2 of 2 calls)" in render_usage_line(snap)

    async def test_a_call_that_asks_for_none_counts_in_neither(self) -> None:
        llm.set_client(_client(_message(input_tokens=10, output_tokens=2)))
        with track_usage() as usage, observe_sampling() as seen:
            await llm.complete("extraction", "p", max_tokens=8)
        row = usage.snapshot().rows[0]
        assert (row.temperature_requested, row.temperature_dropped) == (0, 0)
        assert seen.value() == "unset"

    async def test_a_call_outside_the_registry_is_not_claimed(self) -> None:
        llm.set_client(_client(_message(input_tokens=10, output_tokens=2)))
        with observe_sampling() as seen:
            await AnthropicProvider(model="claude-haiku-4-5").complete(
                "p", max_tokens=8, temperature=0.0
            )
        assert seen.value() is None

    def test_no_call_and_disagreeing_calls_observe_nothing(self) -> None:
        with observe_sampling() as empty:
            pass
        assert empty.value() is None
        with (
            observe_sampling() as mixed,
            purpose_scope("semantic_lint", temperature_requested=True),
        ):
            record_usage("anthropic:m", temperature_sent=True)
            record_usage("anthropic:m", temperature_sent=False)
        assert mixed.value() is None
        with (
            observe_sampling() as silent,
            purpose_scope("semantic_lint", temperature_requested=True),
        ):
            record_usage("anthropic:m")
        assert silent.value() is None

    def test_concurrent_blocks_see_only_their_own_calls(self) -> None:
        async def one(sent: bool) -> str | None:
            with (
                observe_sampling() as seen,
                purpose_scope("semantic_lint", temperature_requested=True),
            ):
                await asyncio.sleep(0)
                record_usage("anthropic:m", temperature_sent=sent)
            return seen.value()

        async def both() -> list[str | None]:
            return list(await asyncio.gather(one(True), one(False)))

        assert asyncio.run(both()) == ["sent", "dropped"]

    async def test_a_batch_rerun_without_temperature_is_dropped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PARTICLES_CONFIG", str(_batch_config_file(tmp_path)))
        reset_config()
        first = [_errored(str(i)) for i in range(2)]
        second = [
            SimpleNamespace(
                custom_id=str(i),
                result=SimpleNamespace(
                    type="succeeded", message=_message(input_tokens=10, output_tokens=2)
                ),
            )
            for i in range(2)
        ]
        client = MagicMock(spec=anthropic.Anthropic)
        client.messages = MagicMock()
        client.messages.batches.create.side_effect = [
            SimpleNamespace(id="b1"),
            SimpleNamespace(id="b2"),
        ]
        client.messages.batches.retrieve.return_value = SimpleNamespace(processing_status="ended")
        by_batch = {"b1": first, "b2": second}
        client.messages.batches.results.side_effect = lambda batch_id: iter(by_batch[batch_id])
        llm.set_client(client)
        requests = [llm.CompletionRequest(prompt=f"p{i}") for i in range(2)]
        with track_usage() as usage, observe_sampling() as seen:
            out = await llm.complete_many(
                "semantic_lint", requests, max_tokens=8, temperature=0.0, latency_tolerant=True
            )
        assert out == ["ok", "ok"]
        row = usage.snapshot().rows[0]
        assert (row.batched, row.temperature_requested, row.temperature_dropped) == (True, 2, 2)
        assert seen.value() == "dropped"

    def test_openai_compat_reports_what_it_sent(self) -> None:
        payload = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
        with (
            track_usage() as usage,
            observe_sampling() as seen,
            purpose_scope("semantic_lint", temperature_requested=True),
        ):
            _extract_text(payload, provider_model="local:m", temperature_sent=True)
            _extract_text(payload, provider_model="local:m", temperature_sent=False)
        row = usage.snapshot().rows[0]
        assert (row.temperature_requested, row.temperature_dropped) == (2, 1)
        assert seen.value() is None

    def test_a_replay_entry_keeps_what_was_sent(self, tmp_path: Path) -> None:
        from particles.llm.replay import _read_entry, _summed, _Usage, _write_entry

        rows = [
            UsageRow(
                purpose="p",
                provider="a",
                model="m",
                calls=1,
                temperature_requested=1,
                temperature_dropped=1,
            )
        ]
        usage = _summed(rows)
        assert usage.temperature_sent is False
        _write_entry(tmp_path, "ab" * 32, purpose="p", provider_model="a:m", text="t", usage=usage)
        entry = _read_entry(tmp_path, "ab" * 32)
        assert entry is not None and entry[1].temperature_sent is False
        # An entry recorded before the field existed says nothing.
        legacy = _Usage()
        _write_entry(tmp_path, "cd" * 32, purpose="p", provider_model="a:m", text="t", usage=legacy)
        entry = _read_entry(tmp_path, "cd" * 32)
        assert entry is not None and entry[1].temperature_sent is None

    def test_rows_recorded_before_the_counts_load(self) -> None:
        row = UsageRow.model_validate({"purpose": "p", "provider": "a", "model": "m", "calls": 3})
        assert (row.temperature_requested, row.temperature_dropped) == (0, 0)


def _batch_config_file(tmp_path: Path) -> Path:
    config = tmp_path / "config.yaml"
    config.write_text("llm:\n  batch:\n    min_requests: 2\n    poll_interval_seconds: 0.01\n")
    return config


def _errored(custom_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(
            type="errored",
            error=SimpleNamespace(
                type="error",
                error=SimpleNamespace(type="invalid_request_error", message=_TEMPERATURE_REJECTION),
            ),
        ),
    )


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

    def test_a_model_billing_cache_reads_lower_uses_its_multiplier(self) -> None:
        # The pricing page bills claude-sonnet-5-5's cache reads at 0.05x
        # input (llm.cache_read_price_multiplier_by_model), not the standard 0.1x.
        with track_usage() as usage:
            record_usage("anthropic:claude-sonnet-5-5", cache_read_tokens=1_000_000)
        assert usage.snapshot().cost_usd == pytest.approx(0.10)
        get_config().llm.cache_read_price_multiplier_by_model = {"claude-sonnet-5-5": 0.2}
        with track_usage() as usage:
            record_usage("anthropic:claude-sonnet-5-5", cache_read_tokens=1_000_000)
        assert usage.snapshot().cost_usd == pytest.approx(0.40)

    def test_haiku_5_5_ships_priced(self) -> None:
        # $0.10/$0.50 per MTok, its rate for prompts of up to 100k tokens.
        with track_usage() as usage:
            record_usage(
                "anthropic:claude-haiku-5-5", input_tokens=1_000_000, output_tokens=1_000_000
            )
        assert usage.snapshot().cost_usd == pytest.approx(0.60)

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
