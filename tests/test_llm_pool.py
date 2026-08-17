"""Tests for the ``CompletionPool`` (quiescence-dispatched fan-in).

The pool is pure asyncio coordination over
``complete_many_with_provider_model``; that seam is patched at the pool
module's own binding (``particles.llm.pool``), so no Anthropic mock and no
config file are needed. The gate behaviour itself is covered by
``tests/test_llm_batch.py`` — here we pin the pooling contract: one merged
dispatch per quiescent wave, positional slicing per group, and job-level
failures raised in every parked group.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from particles.llm import CompletionPool, CompletionRequest
from particles.llm import pool as pool_mod


def _requests(prefix: str, count: int) -> list[CompletionRequest]:
    return [
        CompletionRequest(prompt=f"{prefix} {i}", system=f"sys {prefix} {i}") for i in range(count)
    ]


def _install_fake_many(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pairing: str = "anthropic:test-model",
    fail_with: Exception | None = None,
) -> list[dict[str, Any]]:
    """Patch the pool's dispatch seam; returns the recorded call list."""
    calls: list[dict[str, Any]] = []

    async def fake(
        purpose: str,
        requests: list[CompletionRequest],
        *,
        max_tokens: int,
        temperature: float | None = None,
        response_schema: dict[str, Any] | None = None,
        latency_tolerant: bool = False,
    ) -> tuple[list[str | None], str]:
        calls.append(
            {
                "purpose": purpose,
                "prompts": [r.prompt for r in requests],
                "max_tokens": max_tokens,
                "latency_tolerant": latency_tolerant,
            }
        )
        if fail_with is not None:
            raise fail_with
        return [f"reply:{r.prompt}" for r in requests], pairing

    monkeypatch.setattr(pool_mod, "complete_many_with_provider_model", fake)
    return calls


@pytest.mark.asyncio
async def test_concurrent_groups_merge_into_one_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two participants' request sets ride ONE merged latency-tolerant call."""
    calls = _install_fake_many(monkeypatch)
    pool = CompletionPool("extraction", expected_participants=2)

    async def worker(prefix: str, count: int) -> tuple[list[str | None], str]:
        async with pool.participant():
            reqs = _requests(prefix, count)
            return await pool.complete_group(reqs, max_tokens=64)

    (results_a, pairing_a), (results_b, pairing_b) = await asyncio.gather(
        worker("a", 2), worker("b", 3)
    )

    assert len(calls) == 1
    assert calls[0]["latency_tolerant"] is True
    assert calls[0]["purpose"] == "extraction"
    assert len(calls[0]["prompts"]) == 5
    # Positional slicing: each group gets exactly its own replies, in order.
    assert results_a == ["reply:a 0", "reply:a 1"]
    assert results_b == ["reply:b 0", "reply:b 1", "reply:b 2"]
    assert pairing_a == pairing_b == "anthropic:test-model"


@pytest.mark.asyncio
async def test_dispatch_waits_for_the_unparked_participant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quiescence: no dispatch while a registered participant is still working."""
    calls = _install_fake_many(monkeypatch)
    pool = CompletionPool("extraction", expected_participants=2)
    b_may_park = asyncio.Event()

    async def worker_a() -> None:
        async with pool.participant():
            await pool.complete_group(_requests("a", 1), max_tokens=64)

    async def worker_b() -> None:
        async with pool.participant():
            await b_may_park.wait()  # simulates DB / planning work
            await pool.complete_group(_requests("b", 1), max_tokens=64)

    async def release_after_checking() -> None:
        # Give A ample turns to park; B is still active, so nothing dispatches.
        for _ in range(10):
            await asyncio.sleep(0)
        assert calls == []
        b_may_park.set()

    await asyncio.gather(worker_a(), worker_b(), release_after_checking())
    assert len(calls) == 1
    assert len(calls[0]["prompts"]) == 2


@pytest.mark.asyncio
async def test_participant_exit_without_parking_triggers_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A participant with no LLM work (cache-hit-only) must not hold the wave."""
    calls = _install_fake_many(monkeypatch)
    pool = CompletionPool("extraction", expected_participants=2)
    a_parked = asyncio.Event()

    async def worker_a() -> list[str | None]:
        async with pool.participant():
            a_parked.set()
            results, _ = await pool.complete_group(_requests("a", 2), max_tokens=64)
            return results

    async def worker_b() -> None:
        async with pool.participant():
            await a_parked.wait()  # exits without ever calling complete_group

    results, _ = await asyncio.gather(worker_a(), worker_b())
    assert results == ["reply:a 0", "reply:a 1"]
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_empty_group_returns_immediately_and_does_not_hold_the_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_many(monkeypatch)
    pool = CompletionPool("extraction", expected_participants=2)

    async def worker_a() -> list[str | None]:
        async with pool.participant():
            results, _ = await pool.complete_group(_requests("a", 1), max_tokens=64)
            return results

    async def worker_empty() -> tuple[list[str | None], str]:
        async with pool.participant():
            return await pool.complete_group([], max_tokens=64)

    results, empty = await asyncio.gather(worker_a(), worker_empty())
    assert empty == ([], "")
    assert results == ["reply:a 0"]
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_unregistered_caller_dispatches_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outside a participant context there is nothing to wait for."""
    calls = _install_fake_many(monkeypatch)
    pool = CompletionPool("extraction")

    results, pairing = await pool.complete_group(_requests("solo", 2), max_tokens=64)

    assert results == ["reply:solo 0", "reply:solo 1"]
    assert pairing == "anthropic:test-model"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_distinct_kwargs_shapes_dispatch_as_separate_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Merging never flattens differing uniform kwargs onto one job."""
    calls = _install_fake_many(monkeypatch)
    pool = CompletionPool("extraction", expected_participants=2)

    async def worker(prefix: str, max_tokens: int) -> list[str | None]:
        async with pool.participant():
            results, _ = await pool.complete_group(_requests(prefix, 1), max_tokens=max_tokens)
            return results

    results_a, results_b = await asyncio.gather(worker("a", 64), worker("b", 128))

    assert results_a == ["reply:a 0"]
    assert results_b == ["reply:b 0"]
    assert sorted(c["max_tokens"] for c in calls) == [64, 128]


@pytest.mark.asyncio
async def test_job_level_failure_raises_in_every_parked_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An account-level failure must reach each worker's own failure handling."""
    boom = RuntimeError("credit balance is too low")
    _install_fake_many(monkeypatch, fail_with=boom)
    pool = CompletionPool("extraction", expected_participants=2)

    async def worker(prefix: str) -> str:
        async with pool.participant():
            try:
                await pool.complete_group(_requests(prefix, 1), max_tokens=64)
            except RuntimeError as exc:
                return str(exc)
            return "no error"

    outcomes = await asyncio.gather(worker("a"), worker("b"))
    assert outcomes == ["credit balance is too low", "credit balance is too low"]
