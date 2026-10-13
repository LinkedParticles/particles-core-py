# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the completion record/replay cache (``particles/llm/replay.py``).

The providers are in-process fakes patched in behind the registry's config
resolution, so every call the cache lets through is counted and every token
figure is exact. Covered: record then replay of single calls, the nonce
normalisation that lets a fenced call hit at all, the key's sensitivity to
each request field, miss-falls-through-and-records, failures never recorded,
the batch path (partial hits, a batch of the misses, the even usage split),
the sequential path below ``min_requests``, the untouched override and
``off`` paths, and the usage line's replayed accounting.
"""

from __future__ import annotations

import re
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any

import pytest

from particles.config import TokenPrice, get_config, reset_config
from particles.llm import (
    CompletionRequest,
    ReplayMode,
    RequestFailure,
    complete,
    complete_many_with_provider_model,
    fenced_prompt,
    get_provider,
    override_providers,
    registry,
    render_replay_line,
    render_usage_line,
    replay_scope,
    track_usage,
)
from particles.llm.replay import ReplayingBatchProvider, ReplayingProvider, request_key
from particles.llm.usage import record_usage


class FakeProvider:
    """Answers ``echo:<prompt>`` and reports 100 input / 10 output tokens a call."""

    def __init__(self, model: str = "claude-sonnet-5") -> None:
        self.model = model
        self.calls: list[str] = []
        self.fail_next = False

    @property
    def provider_model(self) -> str:
        return f"anthropic:{self.model}"

    async def complete(self, prompt: str, *, max_tokens: int, **_: Any) -> str:
        self.calls.append(prompt)
        if self.fail_next:
            self.fail_next = False
            raise registry.CompletionError("boom")
        record_usage(self.provider_model, input_tokens=100, output_tokens=10)
        return f"echo:{prompt}"


class FakeBatchProvider(FakeProvider):
    """Adds a batch path reporting 90 input / 9 output tokens per answered request."""

    def __init__(self) -> None:
        super().__init__()
        self.batches: list[list[str]] = []

    async def complete_many(
        self,
        requests: Sequence[CompletionRequest],
        *,
        max_tokens: int,
        failures_out: list[RequestFailure | None] | None = None,
        **_: Any,
    ) -> list[str | None]:
        self.batches.append([r.prompt for r in requests])
        out: list[str | None] = []
        failures: list[RequestFailure | None] = []
        for r in requests:
            if r.prompt.startswith("empty"):
                out.append(None)
                failures.append(RequestFailure.EMPTY)
                continue
            record_usage(
                self.provider_model,
                input_tokens=90,
                output_tokens=9,
                batched=True,
                hit_max_tokens=r.prompt.startswith("trunc"),
            )
            out.append(f"batch:{r.prompt}")
            failures.append(None)
        if failures_out is not None:
            failures_out.extend(failures)
        return out


@pytest.fixture(autouse=True)
def _fresh_config() -> Generator[None, None, None]:
    reset_config()
    yield
    reset_config()


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeProvider:
    provider = FakeProvider()
    monkeypatch.setattr(registry, "_resolve_from_config", lambda purpose: provider)
    return provider


@pytest.fixture
def fake_batch(monkeypatch: pytest.MonkeyPatch) -> FakeBatchProvider:
    provider = FakeBatchProvider()
    monkeypatch.setattr(registry, "_resolve_from_config", lambda purpose: provider)
    return provider


def _batch_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, min_requests: int) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(f"llm:\n  batch:\n    enabled: true\n    min_requests: {min_requests}\n")
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    reset_config()


class TestSingleCalls:
    async def test_record_then_replay_makes_no_second_call(
        self, fake: FakeProvider, tmp_path: Path
    ) -> None:
        with replay_scope(ReplayMode.RECORD, tmp_path) as stats:
            first = await complete("extraction", "hello", max_tokens=64)
        assert (stats.misses, stats.recorded, stats.hits) == (1, 1, 0)

        with replay_scope(ReplayMode.REPLAY, tmp_path) as stats, track_usage() as usage:
            second = await complete("extraction", "hello", max_tokens=64)
        assert second == first == "echo:hello"
        assert fake.calls == ["hello"]
        assert (stats.hits, stats.misses) == (1, 0)
        snap = usage.snapshot()
        assert snap.calls == 0
        assert snap.replayed_calls == 1
        (row,) = snap.rows
        assert row.replayed and (row.input_tokens, row.output_tokens) == (100, 10)

    async def test_record_mode_always_calls(self, fake: FakeProvider, tmp_path: Path) -> None:
        with replay_scope(ReplayMode.RECORD, tmp_path):
            await complete("extraction", "hello", max_tokens=64)
            await complete("extraction", "hello", max_tokens=64)
        assert len(fake.calls) == 2

    async def test_a_replay_miss_calls_and_records(
        self, fake: FakeProvider, tmp_path: Path
    ) -> None:
        with replay_scope(ReplayMode.REPLAY, tmp_path) as stats:
            await complete("extraction", "a", max_tokens=64)
            await complete("extraction", "a", max_tokens=64)
        assert fake.calls == ["a"]
        assert (stats.hits, stats.misses, stats.recorded) == (1, 1, 1)

    async def test_a_failure_is_not_recorded(self, fake: FakeProvider, tmp_path: Path) -> None:
        fake.fail_next = True
        with replay_scope(ReplayMode.REPLAY, tmp_path) as stats:
            with pytest.raises(registry.CompletionError):
                await complete("extraction", "a", max_tokens=64)
            assert await complete("extraction", "a", max_tokens=64) == "echo:a"
        assert fake.calls == ["a", "a"]
        assert stats.recorded == 1

    async def test_fenced_calls_hit_despite_a_fresh_nonce(
        self, fake: FakeProvider, tmp_path: Path
    ) -> None:
        with replay_scope(ReplayMode.REPLAY, tmp_path) as stats:
            for _ in range(2):
                system, user = fenced_prompt("Extract claims.", "the session text")
                await complete("extraction", user, system=system, max_tokens=64)
        assert len(fake.calls) == 1
        assert stats.hits == 1

    async def test_an_unreadable_entry_reads_as_a_miss(
        self, fake: FakeProvider, tmp_path: Path
    ) -> None:
        with replay_scope(ReplayMode.RECORD, tmp_path):
            await complete("extraction", "a", max_tokens=64)
        for entry in tmp_path.rglob("*.json"):
            entry.write_text("{not json")
        with replay_scope(ReplayMode.REPLAY, tmp_path):
            assert await complete("extraction", "a", max_tokens=64) == "echo:a"
        assert len(fake.calls) == 2

    async def test_off_opens_no_scope(self, fake: FakeProvider, tmp_path: Path) -> None:
        with replay_scope(ReplayMode.OFF, tmp_path / "cache") as stats:
            assert get_provider("extraction") is fake
            await complete("extraction", "a", max_tokens=64)
        assert not (tmp_path / "cache").exists()
        assert render_replay_line(stats) == "LLM replay: off."


class TestKey:
    BASE: dict[str, Any] = {
        "purpose": "extraction",
        "provider_model": "anthropic:claude-sonnet-5",
        "prompt": "p",
        "system": "s",
        "cache_prefix": None,
        "images": None,
        "response_schema": None,
        "max_tokens": 64,
        "temperature": None,
    }

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("purpose", "semantic_lint"),
            ("provider_model", "anthropic:claude-haiku-4-5"),
            ("prompt", "q"),
            ("system", "t"),
            ("cache_prefix", "prefix"),
            ("response_schema", {"type": "object"}),
            ("max_tokens", 65),
            ("temperature", 0.0),
        ],
    )
    def test_every_request_field_is_in_the_key(self, field: str, value: object) -> None:
        assert request_key(**self.BASE) != request_key(**{**self.BASE, field: value})

    def test_images_are_keyed_by_content(self) -> None:
        a = request_key(**{**self.BASE, "images": [registry.VisionImage("image/png", b"1")]})
        b = request_key(**{**self.BASE, "images": [registry.VisionImage("image/png", b"2")]})
        assert a != b != request_key(**self.BASE)


class TestBatch:
    async def test_hits_are_served_and_only_misses_are_batched(
        self, fake_batch: FakeBatchProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _batch_on(tmp_path, monkeypatch, min_requests=2)
        cache = tmp_path / "cache"
        first = [CompletionRequest(prompt=p) for p in ("a", "b")]
        with replay_scope(ReplayMode.REPLAY, cache):
            await complete_many_with_provider_model(
                "semantic_lint", first, max_tokens=64, latency_tolerant=True
            )
        assert fake_batch.batches == [["a", "b"]]

        second = [CompletionRequest(prompt=p) for p in ("a", "c", "empty", "b", "d")]
        failures: list[RequestFailure | None] = []
        with replay_scope(ReplayMode.REPLAY, cache) as stats, track_usage() as usage:
            results, pairing = await complete_many_with_provider_model(
                "semantic_lint",
                second,
                max_tokens=64,
                latency_tolerant=True,
                failures_out=failures,
            )
        assert pairing == "anthropic:claude-sonnet-5"
        assert fake_batch.batches[1] == ["c", "empty", "d"]
        assert results == ["batch:a", "batch:c", None, "batch:b", "batch:d"]
        assert failures == [None, None, RequestFailure.EMPTY, None, None]
        assert (stats.hits, stats.misses, stats.recorded) == (2, 3, 2)
        snap = usage.snapshot()
        assert snap.calls == 2 and snap.replayed_calls == 2
        replayed = [r for r in snap.rows if r.replayed]
        assert replayed[0].batched and replayed[0].input_tokens == 180

    async def test_a_batch_share_preserves_the_job_total(
        self, fake_batch: FakeBatchProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _batch_on(tmp_path, monkeypatch, min_requests=2)
        cache = tmp_path / "cache"
        requests = [CompletionRequest(prompt=p) for p in ("a", "b", "c")]
        with replay_scope(ReplayMode.RECORD, cache), track_usage() as paid:
            await complete_many_with_provider_model(
                "semantic_lint", requests, max_tokens=64, latency_tolerant=True
            )
        with replay_scope(ReplayMode.REPLAY, cache), track_usage() as replayed:
            await complete_many_with_provider_model(
                "semantic_lint", requests, max_tokens=64, latency_tolerant=True
            )
        (paid_row,) = paid.snapshot().rows
        (replayed_row,) = replayed.snapshot().rows
        assert replayed_row.replayed
        assert replayed_row.input_tokens == paid_row.input_tokens == 270
        assert replayed_row.output_tokens == paid_row.output_tokens == 27

    async def test_a_truncated_batch_reply_replays_as_truncated(
        self, fake_batch: FakeBatchProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _batch_on(tmp_path, monkeypatch, min_requests=2)
        cache = tmp_path / "cache"
        requests = [CompletionRequest(prompt=p) for p in ("a", "trunc", "c")]
        with replay_scope(ReplayMode.RECORD, cache), track_usage() as paid:
            await complete_many_with_provider_model(
                "semantic_lint", requests, max_tokens=64, latency_tolerant=True
            )
        with replay_scope(ReplayMode.REPLAY, cache), track_usage() as replayed:
            await complete_many_with_provider_model(
                "semantic_lint", requests, max_tokens=64, latency_tolerant=True
            )
        assert paid.snapshot().max_tokens_stops == 1
        assert replayed.snapshot().max_tokens_stops == 1

    async def test_extra_options_are_in_the_batch_key(
        self, fake_batch: FakeBatchProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _batch_on(tmp_path, monkeypatch, min_requests=2)
        requests = [CompletionRequest(prompt=p) for p in ("a", "b")]
        with replay_scope(ReplayMode.REPLAY, tmp_path / "cache"):
            provider = get_provider("semantic_lint")
            assert isinstance(provider, ReplayingBatchProvider)
            await provider.complete_many(requests, max_tokens=64, effort="low")
            await provider.complete_many(requests, max_tokens=64, effort="low")
            await provider.complete_many(requests, max_tokens=64, effort="high")
        assert len(fake_batch.batches) == 2

    async def test_a_rejected_batch_job_is_not_counted_twice(
        self, fake_batch: FakeBatchProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _batch_on(tmp_path, monkeypatch, min_requests=2)

        async def reject(*_: Any, **__: Any) -> list[str | None]:
            raise registry.CompletionError("temperature is deprecated for this model")

        monkeypatch.setattr(fake_batch, "complete_many", reject)
        with replay_scope(ReplayMode.REPLAY, tmp_path / "cache") as stats:
            results, _ = await complete_many_with_provider_model(
                "semantic_lint",
                [CompletionRequest(prompt=p) for p in ("a", "b", "c")],
                max_tokens=64,
                latency_tolerant=True,
            )
        assert results == ["echo:a", "echo:b", "echo:c"]
        assert (stats.misses, stats.recorded) == (3, 3)

    async def test_misses_below_min_requests_run_one_at_a_time(
        self, fake_batch: FakeBatchProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _batch_on(tmp_path, monkeypatch, min_requests=3)
        cache = tmp_path / "cache"
        with replay_scope(ReplayMode.REPLAY, cache):
            await complete("semantic_lint", "a", max_tokens=64)
            results, _ = await complete_many_with_provider_model(
                "semantic_lint",
                [CompletionRequest(prompt=p) for p in ("a", "b", "c")],
                max_tokens=64,
                latency_tolerant=True,
            )
        assert fake_batch.batches == []
        assert fake_batch.calls == ["a", "b", "c"]
        assert results == ["echo:a", "echo:b", "echo:c"]

    def test_the_wrapper_keeps_the_batch_capability(
        self, fake: FakeProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with replay_scope(ReplayMode.REPLAY, tmp_path):
            assert type(get_provider("extraction")) is ReplayingProvider
        batch = FakeBatchProvider()
        monkeypatch.setattr(registry, "_resolve_from_config", lambda purpose: batch)
        with replay_scope(ReplayMode.REPLAY, tmp_path):
            wrapped = get_provider("extraction")
        assert isinstance(wrapped, ReplayingBatchProvider)
        assert isinstance(wrapped, registry.BatchCompletionProvider)


class TestScope:
    async def test_an_override_is_not_wrapped(self, fake: FakeProvider, tmp_path: Path) -> None:
        scripted = FakeProvider(model="scripted")
        with (
            replay_scope(ReplayMode.REPLAY, tmp_path),
            override_providers({"semantic_lint": scripted}),
        ):
            assert get_provider("semantic_lint") is scripted

    async def test_the_scope_ends_with_its_block(self, fake: FakeProvider, tmp_path: Path) -> None:
        with replay_scope(ReplayMode.REPLAY, tmp_path):
            pass
        assert get_provider("extraction") is fake


class TestUsageLine:
    async def test_replayed_calls_are_counted_and_priced_apart(
        self, fake: FakeProvider, tmp_path: Path
    ) -> None:
        get_config().llm.price_per_mtok["anthropic:claude-sonnet-5"] = TokenPrice(
            input=1_000.0, output=10_000.0
        )
        with replay_scope(ReplayMode.RECORD, tmp_path):
            await complete("extraction", "a", max_tokens=64)
        with replay_scope(ReplayMode.REPLAY, tmp_path), track_usage() as usage:
            await complete("extraction", "a", max_tokens=64)
            await complete("extraction", "b", max_tokens=64)
        snap = usage.snapshot()
        # One paid call: 100 in at $1000/MTok + 10 out at $10000/MTok = $0.20.
        assert snap.cost_usd == pytest.approx(0.2)
        assert snap.priced_cost_usd == pytest.approx(0.2)
        assert snap.replayed_cost_usd == pytest.approx(0.2)
        line = render_usage_line(snap)
        assert "2 extraction calls (claude-sonnet-5), 100 input, 10 output tokens" in line
        assert ", 1 replayed" in line
        assert "1 call replayed from the cache (≈ $0.20 not re-bought)." in line


#: Modules allowed to open a replay scope. A new benchmark verb
#: that takes ``--llm-replay`` adds itself here; nothing on a product path may.
_REPLAY_CALLERS = {"particles/api/cli/benchmark.py"}


def test_only_allowlisted_modules_open_a_replay_scope() -> None:
    root = Path(__file__).resolve().parents[1]
    callers = {
        path.relative_to(root).as_posix()
        for path in (root / "particles").rglob("*.py")
        if path.name != "replay.py" and re.search(r"\breplay_scope\(", path.read_text())
    }
    # A subset, not equality: a published tree without the surface layer has
    # no caller at all, which is still within the allowlist.
    assert callers <= _REPLAY_CALLERS, sorted(callers - _REPLAY_CALLERS)
