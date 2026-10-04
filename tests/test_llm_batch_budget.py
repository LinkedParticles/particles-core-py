# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the run-level batch-wait budget, and for what a
cancelled batch keeps, re-runs and loses.

The Anthropic SDK is mocked through the ``particles.llm.set_client`` seam
(tests/AGENTS.md § Mocking strategy), so there is no network. Waiting is either
a few hundredths of a second of real time or simulated by advancing the
budget's ``spent_seconds`` from inside the mocked ``batches.retrieve``, so no
test sleeps for long.

The invariant carried over: a spent budget changes the bill and
the wall clock, never what a call site computes.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import pydantic
import pytest

from particles import llm
from particles.config import get_config, reset_config
from particles.llm import CompletionRequest
from particles.llm.batch_budget import (
    BatchWaitBudget,
    batch_wait_budget,
    current_batch_wait_budget,
)
from particles.llm.registry import complete_many_with_provider_model


@pytest.fixture(autouse=True)
def _reset_client_around_each_test() -> Generator[None, None, None]:
    llm.set_client(None)
    yield
    llm.set_client(None)
    reset_config()


def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(body)
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    reset_config()


def _batch_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **knobs: object) -> None:
    lines = "\n".join(f"    {key}: {value}" for key, value in knobs.items())
    _config(tmp_path, monkeypatch, f"llm:\n  batch:\n{lines}\n")


def _text_message(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(text=text)], stop_reason="end_turn")


def _succeeded(custom_id: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(type="succeeded", message=_text_message(text)),
    )


def _client(
    *,
    results: list[list[SimpleNamespace]] | None = None,
    status: Callable[[], str] | None = None,
    sequential: list[str] | None = None,
) -> MagicMock:
    """A mocked client: successive batches read ``results``; ``complete`` reads ``sequential``."""
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.batches.create.side_effect = [
        SimpleNamespace(id=f"msgbatch_{i}") for i in range(len(results or [[]]))
    ]
    client.messages.batches.retrieve.side_effect = lambda _id: SimpleNamespace(
        processing_status=status() if status is not None else "ended"
    )
    client.messages.batches.results.side_effect = [iter(r) for r in (results or [])]
    client.messages.create.side_effect = [_text_message(t) for t in (sequential or [])]
    return client


_REQUESTS = [CompletionRequest(prompt=f"probe {i}", system=f"sys {i}") for i in range(4)]


# ---------------------------------------------------------------------------
# The budget object
# ---------------------------------------------------------------------------


def test_budget_caps_the_wait_at_the_balance_left() -> None:
    budget = BatchWaitBudget(budget_seconds=1000, min_remaining_seconds=300)
    assert budget.wait_cap(3600) == (1000, True)
    budget.charge(400, requests=5, cut_short=False)
    assert budget.remaining == 600
    assert budget.wait_cap(60) == (60, False)
    assert budget.can_submit()


def test_budget_records_the_pass_it_ran_out_in() -> None:
    budget = BatchWaitBudget(budget_seconds=1000, min_remaining_seconds=300)
    budget.pass_name = "extract"
    budget.charge(800, requests=10, cut_short=False)
    budget.pass_name = "census"
    budget.record_sequential(4)
    assert budget.exhausted_in_pass == "extract"
    assert not budget.can_submit()
    assert budget.counters() == (0, 0, 1, 4)


def test_no_budget_outside_a_scope() -> None:
    assert current_batch_wait_budget() is None
    budget = BatchWaitBudget(budget_seconds=10, min_remaining_seconds=1)
    with batch_wait_budget(budget):
        assert current_batch_wait_budget() is budget
    assert current_batch_wait_budget() is None


def test_an_open_wait_is_reported_and_counted_against_the_balance() -> None:
    """the balance is charged when a batch returns, so a watcher
    needs the open wait subtracted to see what is really left."""
    seen: list[tuple[float, float] | None] = []
    budget = BatchWaitBudget(budget_seconds=3600, min_remaining_seconds=300)
    budget.on_wait = lambda b: seen.append(b.pending_wait())

    assert budget.pending_wait() is None
    token = budget.begin_wait(1800)
    pending = budget.pending_wait()
    assert pending is not None
    assert pending[1] == 1800
    assert budget.live_remaining() <= 3600
    budget.note_wait()
    budget.end_wait(token)
    budget.end_wait(token)  # a second close is a no-op

    assert budget.pending_wait() is None
    assert [s is not None for s in seen] == [True, True, False]


def test_a_failing_watcher_never_fails_the_batch() -> None:
    budget = BatchWaitBudget(budget_seconds=10, min_remaining_seconds=1)

    def _boom(_b: BatchWaitBudget) -> None:
        raise RuntimeError("renderer broke")

    budget.on_wait = _boom
    budget.end_wait(budget.begin_wait(5))


# ---------------------------------------------------------------------------
# Dispatch under a budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_path_unchanged_when_the_batch_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch that ends inside the budget is served, charged, and moves nothing."""
    _batch_config(tmp_path, monkeypatch, min_requests=2)
    client = _client(results=[[_succeeded(str(i), f"reply {i}") for i in range(4)]])
    llm.set_client(client)
    budget = BatchWaitBudget(budget_seconds=3600, min_remaining_seconds=300)

    with batch_wait_budget(budget):
        out = await llm.complete_many(
            "semantic_lint", _REQUESTS, max_tokens=10, latency_tolerant=True
        )

    assert out == ["reply 0", "reply 1", "reply 2", "reply 3"]
    assert client.messages.batches.create.call_count == 1
    client.messages.create.assert_not_called()
    assert budget.batches == 1
    assert budget.counters() == (0, 0, 0, 0)
    assert budget.exhausted_in_pass is None


@pytest.mark.asyncio
async def test_a_batch_is_cut_short_at_the_balance_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """With max_wait an hour and 0.05 s of budget left, the batch is cancelled at 0.05 s."""
    _batch_config(
        tmp_path, monkeypatch, min_requests=2, poll_interval_seconds=0.01, cancel_grace_seconds=0
    )
    client = _client(results=[[]], status=lambda: "in_progress")
    llm.set_client(client)
    budget = BatchWaitBudget(budget_seconds=0.05, min_remaining_seconds=0.0)

    with batch_wait_budget(budget):
        out = await llm.complete_many(
            "semantic_lint", _REQUESTS, max_tokens=10, latency_tolerant=True
        )

    assert out == [None, None, None, None]
    client.messages.batches.cancel.assert_called_once_with("msgbatch_0")
    assert budget.batches == 1
    assert budget.batches_cut_short == 1
    assert budget.cut_short_requests == 4
    assert 0.05 <= budget.spent_seconds < 5
    assert "remaining batch-wait budget" in caplog.text


@pytest.mark.asyncio
async def test_below_the_floor_a_set_never_reaches_the_batch_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spent budget serves the set sequentially: same results, full price, disclosed."""
    _batch_config(tmp_path, monkeypatch, min_requests=2)
    client = _client(sequential=["a", "b", "c", "d"])
    llm.set_client(client)
    budget = BatchWaitBudget(budget_seconds=1000, min_remaining_seconds=300)
    budget.charge(800, requests=1, cut_short=False)

    with batch_wait_budget(budget):
        out = await llm.complete_many(
            "semantic_lint", _REQUESTS, max_tokens=10, latency_tolerant=True
        )

    assert out == ["a", "b", "c", "d"]
    client.messages.batches.create.assert_not_called()
    assert client.messages.create.call_count == 4
    assert (budget.sequential_sets, budget.sequential_requests) == (1, 4)


@pytest.mark.asyncio
async def test_no_scope_waits_the_full_max_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Outside a budget, the per-batch ceiling alone decides."""
    _batch_config(
        tmp_path,
        monkeypatch,
        min_requests=2,
        poll_interval_seconds=0.01,
        max_wait_seconds=0.03,
        cancel_grace_seconds=0,
    )
    client = _client(results=[[]], status=lambda: "in_progress")
    llm.set_client(client)

    out = await llm.complete_many("semantic_lint", _REQUESTS, max_tokens=10, latency_tolerant=True)

    assert out == [None, None, None, None]
    client.messages.batches.cancel.assert_called_once()
    assert "llm.batch.max_wait_seconds" in caplog.text
    assert "batch-wait budget" not in caplog.text


@pytest.mark.asyncio
async def test_the_budget_is_kept_per_chunk_not_per_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second chunk sees the first chunk's charge and falls to sequential below the floor."""
    _batch_config(tmp_path, monkeypatch, min_requests=2, max_requests_per_batch=2)
    budget = BatchWaitBudget(budget_seconds=1000, min_remaining_seconds=300)

    def _status() -> str:
        # The first chunk's batch "takes" 900 s: simulated by spending it on
        # the budget while the poll loop is waiting.
        if budget.spent_seconds < 900:
            budget.spent_seconds += 900
        return "ended"

    client = _client(
        results=[[_succeeded("0", "reply 0"), _succeeded("1", "reply 1")]],
        status=_status,
        sequential=["reply 2", "reply 3"],
    )
    llm.set_client(client)

    with batch_wait_budget(budget):
        out = await llm.complete_many(
            "semantic_lint", _REQUESTS, max_tokens=10, latency_tolerant=True
        )

    assert out == ["reply 0", "reply 1", "reply 2", "reply 3"]
    assert client.messages.batches.create.call_count == 1
    assert client.messages.create.call_count == 2
    assert budget.batches == 1
    assert budget.spent_seconds >= 900
    assert (budget.sequential_sets, budget.sequential_requests) == (1, 2)


# ---------------------------------------------------------------------------
# A cancelled batch keeps what it finished
# ---------------------------------------------------------------------------


def _canceled(custom_id: str) -> SimpleNamespace:
    return SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type="canceled"))


def _errored(custom_id: str) -> SimpleNamespace:
    return SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type="errored"))


def _ends_once_cancelled(results: list[SimpleNamespace], **kw: object) -> MagicMock:
    """A batch that runs past its wait, then reports ``ended`` once it is cancelled."""
    holder: dict[str, MagicMock] = {}

    def _status() -> str:
        return "ended" if holder["client"].messages.batches.cancel.called else "in_progress"

    client = _client(results=[results], status=_status, **kw)  # type: ignore[arg-type]
    holder["client"] = client
    return client


def _cancel_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **knobs: object) -> None:
    base: dict[str, object] = {
        "min_requests": 2,
        "poll_interval_seconds": 0.01,
        "max_wait_seconds": 0.03,
        "cancel_grace_seconds": 1,
    }
    _batch_config(tmp_path, monkeypatch, **{**base, **knobs})


@pytest.mark.asyncio
async def test_a_cancelled_batch_keeps_the_requests_it_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Succeeded results are read back by custom_id; canceled ones are unavailable."""
    _cancel_config(tmp_path, monkeypatch)
    client = _ends_once_cancelled(
        [_canceled("3"), _succeeded("2", "reply 2"), _canceled("1"), _succeeded("0", "reply 0")]
    )
    llm.set_client(client)
    failures: list[llm.RequestFailure | None] = []

    out, _ = await complete_many_with_provider_model(
        "semantic_lint",
        _REQUESTS,
        max_tokens=10,
        latency_tolerant=True,
        failures_out=failures,
    )

    assert out == ["reply 0", None, "reply 2", None]
    assert failures == [
        None,
        llm.RequestFailure.UNAVAILABLE,
        None,
        llm.RequestFailure.UNAVAILABLE,
    ]
    client.messages.batches.cancel.assert_called_once_with("msgbatch_0")
    # No budget scope: nothing is re-run outside the consolidation cycle.
    client.messages.create.assert_not_called()
    assert "2 of 4 request(s) had finished and are kept; 0 re-run sequentially; 2 unavailable" in (
        caplog.text
    )


@pytest.mark.asyncio
async def test_the_grace_is_bounded_and_charged_to_the_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A batch that never ends: the grace stops, is charged, and nothing is read or re-run."""
    _cancel_config(tmp_path, monkeypatch, max_wait_seconds=3600, cancel_grace_seconds=0.05)
    client = _client(results=[[]], status=lambda: "in_progress", sequential=["x"] * 4)
    llm.set_client(client)
    budget = BatchWaitBudget(budget_seconds=0.05, min_remaining_seconds=0.0)

    with batch_wait_budget(budget):
        out = await llm.complete_many(
            "semantic_lint", _REQUESTS, max_tokens=10, latency_tolerant=True
        )

    assert out == [None, None, None, None]
    client.messages.batches.cancel.assert_called_once_with("msgbatch_0")
    client.messages.batches.results.assert_not_called()
    client.messages.create.assert_not_called()
    # The wait (capped at the 0.05 s balance) plus the 0.05 s grace, and no more.
    assert 0.1 <= budget.spent_seconds < 5
    assert budget.batches_cut_short == 1
    assert budget.cancellations() == (1, 0, 0, 4)
    assert "did not end within 0s of being cancelled" in caplog.text


@pytest.mark.asyncio
async def test_a_batch_that_never_ends_returns_cleanly_without_a_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cancel_config(tmp_path, monkeypatch, cancel_grace_seconds=0.03)
    client = _client(results=[[]], status=lambda: "in_progress")
    llm.set_client(client)

    out = await llm.complete_many("semantic_lint", _REQUESTS, max_tokens=10, latency_tolerant=True)

    assert out == [None, None, None, None]
    client.messages.batches.results.assert_not_called()


@pytest.mark.asyncio
async def test_a_small_canceled_remainder_is_re_run_under_a_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Only the canceled requests are re-run, and their answers land in position."""
    _cancel_config(tmp_path, monkeypatch, cancel_rerun_max_output_tokens=20)
    client = _ends_once_cancelled(
        [_succeeded("0", "reply 0"), _canceled("1"), _errored("2"), _canceled("3")],
        sequential=["rerun 1", "rerun 3"],
    )
    llm.set_client(client)
    budget = BatchWaitBudget(budget_seconds=3600, min_remaining_seconds=300)

    with batch_wait_budget(budget):
        out = await llm.complete_many(
            "semantic_lint", _REQUESTS, max_tokens=10, latency_tolerant=True
        )

    assert out == ["reply 0", "rerun 1", None, "rerun 3"]
    assert client.messages.create.call_count == 2
    prompts = [c.kwargs["messages"][0]["content"] for c in client.messages.create.call_args_list]
    assert prompts == ["probe 1", "probe 3"]
    assert budget.cancellations() == (1, 1, 2, 1)
    # A max_wait expiry, not the budget, cancelled it.
    assert budget.batches_cut_short == 0
    assert "1 of 4 request(s) had finished and are kept; 2 re-run sequentially; 1 unavailable" in (
        caplog.text
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", [19, 0])
async def test_a_remainder_over_the_bound_is_not_re_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound: int
) -> None:
    """Two canceled requests at max_tokens=10 need 20 output tokens of room."""
    _cancel_config(tmp_path, monkeypatch, cancel_rerun_max_output_tokens=bound)
    client = _ends_once_cancelled(
        [_succeeded("0", "reply 0"), _canceled("1"), _succeeded("2", "reply 2"), _canceled("3")]
    )
    llm.set_client(client)
    budget = BatchWaitBudget(budget_seconds=3600, min_remaining_seconds=300)

    with batch_wait_budget(budget):
        out = await llm.complete_many(
            "semantic_lint", _REQUESTS, max_tokens=10, latency_tolerant=True
        )

    assert out == ["reply 0", None, "reply 2", None]
    client.messages.create.assert_not_called()
    assert budget.cancellations() == (1, 2, 0, 2)


@pytest.mark.asyncio
async def test_a_batch_that_ends_on_time_is_never_cancelled_or_re_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The on-time path: a canceled entry in an ended batch is just unavailable."""
    _cancel_config(tmp_path, monkeypatch, max_wait_seconds=3600)
    client = _client(results=[[_succeeded("0", "a"), _canceled("1"), _succeeded("2", "c")]])
    llm.set_client(client)
    budget = BatchWaitBudget(budget_seconds=3600, min_remaining_seconds=300)

    with batch_wait_budget(budget):
        out = await llm.complete_many(
            "semantic_lint", _REQUESTS, max_tokens=10, latency_tolerant=True
        )

    assert out == ["a", None, "c", None]
    client.messages.batches.cancel.assert_not_called()
    client.messages.create.assert_not_called()
    assert client.messages.batches.retrieve.call_count == 1
    assert budget.cancellations() == (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_cancel_knobs_load_from_yaml_at_call_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert get_config().llm.batch.cancel_grace_seconds == 180
    assert get_config().llm.batch.cancel_rerun_max_output_tokens == 15000
    _batch_config(tmp_path, monkeypatch, cancel_grace_seconds=0, cancel_rerun_max_output_tokens=0)
    assert get_config().llm.batch.cancel_grace_seconds == 0
    assert get_config().llm.batch.cancel_rerun_max_output_tokens == 0


@pytest.mark.parametrize("knob", ["cancel_grace_seconds: -1", "cancel_rerun_max_output_tokens: -1"])
def test_negative_cancel_knobs_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, knob: str
) -> None:
    _config(tmp_path, monkeypatch, f"llm:\n  batch:\n    {knob}\n")
    with pytest.raises(pydantic.ValidationError):
        get_config()


def test_batch_wait_knobs_load_from_yaml_at_call_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert get_config().consolidation.batch_wait.budget_seconds == 3600
    assert get_config().consolidation.batch_wait.min_remaining_seconds == 300
    _config(
        tmp_path,
        monkeypatch,
        "consolidation:\n  batch_wait:\n    budget_seconds: 900\n    min_remaining_seconds: 60\n",
    )
    assert get_config().consolidation.batch_wait.budget_seconds == 900
    assert get_config().consolidation.batch_wait.min_remaining_seconds == 60


def test_a_non_positive_budget_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config(tmp_path, monkeypatch, "consolidation:\n  batch_wait:\n    budget_seconds: -1\n")
    with pytest.raises(pydantic.ValidationError):
        get_config()


@pytest.mark.asyncio
async def test_a_pending_batch_reports_its_wait_on_every_poll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """the wait is visible while the batch is pending, capped at the
    balance left, and closed when it returns."""
    _batch_config(tmp_path, monkeypatch, min_requests=2, poll_interval_seconds=0.01)
    polls = iter(["in_progress", "in_progress", "ended"])
    client = _client(
        results=[[_succeeded(str(i), f"reply {i}") for i in range(4)]],
        status=lambda: next(polls),
    )
    llm.set_client(client)
    budget = BatchWaitBudget(budget_seconds=1200, min_remaining_seconds=0)
    seen: list[tuple[float, float] | None] = []
    budget.on_wait = lambda b: seen.append(b.pending_wait())

    with batch_wait_budget(budget):
        out = await llm.complete_many(
            "semantic_lint", _REQUESTS, max_tokens=10, latency_tolerant=True
        )

    assert out == ["reply 0", "reply 1", "reply 2", "reply 3"]
    pending = [s for s in seen if s is not None]
    # Opened on submission, then once per poll that found the batch pending.
    assert len(pending) == 3
    assert all(cap == 1200 for _elapsed, cap in pending)
    assert seen[-1] is None
    assert budget.pending_wait() is None
