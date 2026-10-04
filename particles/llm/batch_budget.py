# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""A run-level budget for the time spent waiting on Message Batches.

Each submitted batch is bounded by ``llm.batch.max_wait_seconds``; a
caller that submits many batches one after another can still wait many times
that. A :class:`BatchWaitBudget` bounds the *sum*. The consolidation cycle opens
one with :func:`batch_wait_budget` around its passes, and the two places that
decide how a set is served consult it:

- ``registry._complete_many_scoped`` sends a set down the sequential path
  instead of the batch path once less than the floor is left, and
- ``AnthropicProvider._run_batch`` caps each chunk's wait at the balance left,
  charges what it waited, and serves a chunk sequentially when the floor is
  reached between chunks.

The adapter also counts every batch it cancels, with how many of its requests
were kept, re-run and lost, and a scope is the marker under which it re-runs a
cancelled batch's small remainder sequentially.

While a batch is pending the adapter also reports the wait as it happens:
:meth:`BatchWaitBudget.begin_wait` on submission, :meth:`~BatchWaitBudget.note_wait`
on every poll, :meth:`~BatchWaitBudget.end_wait` when it returns. The run owner
installs ``on_wait`` to hear about it, which is how the consolidation cycle
shows ``batch waiting 23m of 60m`` without this module knowing who is watching.
The balance is charged only when a batch returns, so
:meth:`~BatchWaitBudget.live_remaining` also subtracts the waits still open.

Outside a scope there is no budget and both behave exactly as before. The value
is a ``ContextVar`` rather than a keyword because it belongs to the run, not to
any call site: the pool, the census probes and the utility matcher sit layers
below the orchestrator and make no decision of their own about it. The object
is mutable and shared by reference, so tasks spawned inside the scope (the
pooled extract pass) charge the same budget.
"""

from __future__ import annotations

import contextlib
import itertools
import time
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class BatchWaitBudget:
    """Seconds a run may spend waiting on batches, and what it did with them.

    ``pass_name`` is the orchestrator's label for the work in progress; it is
    what :attr:`exhausted_in_pass` records when the balance first falls below
    the floor.
    """

    budget_seconds: float
    min_remaining_seconds: float
    spent_seconds: float = 0.0
    batches: int = 0
    batches_cut_short: int = 0
    cut_short_requests: int = 0
    sequential_sets: int = 0
    sequential_requests: int = 0
    cancelled_batches: int = 0
    cancelled_kept: int = 0
    cancelled_rerun: int = 0
    cancelled_lost: int = 0
    exhausted_in_pass: str | None = None
    pass_name: str | None = None
    #: Called on every change to a pending batch's wait: submitted, polled,
    #: returned. Installed by the run owner; ``None`` reports nothing.
    on_wait: Callable[[BatchWaitBudget], None] | None = field(default=None, repr=False)
    #: Open waits by token: ``(started, cap)`` on the monotonic clock.
    _waits: dict[int, tuple[float, float]] = field(default_factory=dict, repr=False)
    _tokens: itertools.count[int] = field(default_factory=itertools.count, repr=False)

    @property
    def remaining(self) -> float:
        """Seconds of batch waiting left; never negative."""
        return max(0.0, self.budget_seconds - self.spent_seconds)

    def can_submit(self) -> bool:
        """True while enough is left for a batch to be worth submitting."""
        return self.remaining >= self.min_remaining_seconds

    def wait_cap(self, max_wait: float) -> tuple[float, bool]:
        """The wait for the next batch, and whether the budget (not ``max_wait``) set it."""
        remaining = self.remaining
        if remaining < max_wait:
            return remaining, True
        return max_wait, False

    def charge(self, waited: float, *, requests: int, cut_short: bool) -> None:
        """Record one submitted batch that waited ``waited`` seconds."""
        self.spent_seconds += max(0.0, waited)
        self.batches += 1
        if cut_short:
            self.batches_cut_short += 1
            self.cut_short_requests += requests
        self._note_exhaustion()

    def record_sequential(self, requests: int) -> None:
        """Record one set served sequentially because the budget was spent."""
        self.sequential_sets += 1
        self.sequential_requests += requests
        self._note_exhaustion()

    def record_cancellation(self, *, kept: int, rerun: int, lost: int) -> None:
        """Record one cancelled batch, whatever cancelled it.

        ``kept`` requests had finished before the cancellation and were read
        back, ``rerun`` were answered by the sequential re-run, and ``lost``
        came back unavailable.
        """
        self.cancelled_batches += 1
        self.cancelled_kept += kept
        self.cancelled_rerun += rerun
        self.cancelled_lost += lost

    def cancellations(self) -> tuple[int, int, int, int]:
        """``(cancelled_batches, cancelled_kept, cancelled_rerun, cancelled_lost)``."""
        return (
            self.cancelled_batches,
            self.cancelled_kept,
            self.cancelled_rerun,
            self.cancelled_lost,
        )

    def counters(self) -> tuple[int, int, int, int]:
        """``(batches_cut_short, cut_short_requests, sequential_sets, sequential_requests)``."""
        return (
            self.batches_cut_short,
            self.cut_short_requests,
            self.sequential_sets,
            self.sequential_requests,
        )

    def begin_wait(self, cap: float) -> int:
        """Open a pending batch's wait, capped at ``cap`` seconds; returns its token."""
        token = next(self._tokens)
        self._waits[token] = (time.monotonic(), cap)
        self.note_wait()
        return token

    def end_wait(self, token: int) -> None:
        """Close the wait ``token`` names. :meth:`charge` still records what it cost."""
        if self._waits.pop(token, None) is not None:
            self.note_wait()

    def note_wait(self) -> None:
        """Tell ``on_wait`` the wait state moved; its failure never fails the batch."""
        if self.on_wait is None:
            return
        with contextlib.suppress(Exception):
            self.on_wait(self)

    def pending_wait(self) -> tuple[float, float] | None:
        """``(elapsed, cap)`` of the longest-open pending batch, or ``None``."""
        if not self._waits:
            return None
        started, cap = min(self._waits.values())
        return time.monotonic() - started, cap

    def live_remaining(self) -> float:
        """:attr:`remaining` less the waits still open, which are not charged yet."""
        now = time.monotonic()
        open_waits = sum(now - started for started, _cap in self._waits.values())
        return max(0.0, self.remaining - open_waits)

    def _note_exhaustion(self) -> None:
        if self.exhausted_in_pass is None and not self.can_submit():
            self.exhausted_in_pass = self.pass_name


_BUDGET: ContextVar[BatchWaitBudget | None] = ContextVar(
    "particles_llm_batch_wait_budget", default=None
)


@contextlib.contextmanager
def batch_wait_budget(budget: BatchWaitBudget) -> Iterator[BatchWaitBudget]:
    """Install ``budget`` for every batch submitted inside the block."""
    token = _BUDGET.set(budget)
    try:
        yield budget
    finally:
        _BUDGET.reset(token)


def current_batch_wait_budget() -> BatchWaitBudget | None:
    """The budget in scope, or ``None`` when the caller set none."""
    return _BUDGET.get()
