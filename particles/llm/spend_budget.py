# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""A run-level dollar budget, consulted between Message Batches chunks.

The consolidation cycle bounds its spend in US$ (``consolidation.budget_usd``).
The orchestrator checks the budget between passes; inside a pass the one place
that submits work in large units is the batch adapter, which sends a set as
chunks of ``llm.batch.max_requests_per_batch``. Before each chunk it asks the
budget in scope whether the chunk still fits, and a chunk that does not is not
submitted: its requests come back unavailable, exactly as a cancelled batch's
lost requests do, and every caller already degrades on that.

The value is a ``ContextVar`` for the reason :mod:`particles.llm.batch_budget`
gives: it belongs to the run, not to any call site, and tasks spawned inside the
scope (the pooled extract pass) see the same object. Outside a scope there is
no budget and the adapter behaves exactly as before.

What the run has spent is read from the run's open
:func:`~particles.llm.usage.track_usage` accumulator through the ``spent``
callable; what a chunk would cost is the run owner's estimate through
``estimate_chunk``, so this module holds no pricing assumption of its own. The
decision itself is :func:`particles.core.spend_budget.decide_spend`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

from particles.core.spend_budget import decide_spend

if TYPE_CHECKING:
    from particles.llm.registry import CompletionRequest

#: ``(purpose, provider_model, requests, max_tokens)`` → estimated US$ of one
#: batched chunk, or ``None`` when it cannot be priced.
ChunkEstimator = Callable[[str | None, str, "Sequence[CompletionRequest]", int], float | None]


@dataclass
class SpendBudget:
    """US$ a run may spend, and the batch chunks it declined to submit."""

    budget_usd: float
    #: The run's spend so far in US$ at list price (priced rows only).
    spent: Callable[[], float]
    #: The run owner's estimate of one chunk's cost; ``None`` checks spend only.
    estimate_chunk: ChunkEstimator | None = None
    #: The orchestrator's label for the work in progress.
    pass_name: str | None = None
    #: Chunks not submitted because they would exceed the budget, and their requests.
    skipped_chunks: int = 0
    skipped_requests: int = 0
    #: The pass during which the first chunk was declined.
    exhausted_in_pass: str | None = None

    def allows_chunk(
        self,
        *,
        purpose: str | None,
        provider_model: str,
        requests: Sequence[CompletionRequest],
        max_tokens: int,
    ) -> bool:
        """True when the chunk fits; records the skip when it does not."""
        estimate = (
            self.estimate_chunk(purpose, provider_model, requests, max_tokens)
            if self.estimate_chunk is not None
            else None
        )
        verdict = decide_spend(
            spent_usd=self.spent(), budget_usd=self.budget_usd, estimate_usd=estimate
        )
        if verdict == "run":
            return True
        self.skipped_chunks += 1
        self.skipped_requests += len(requests)
        if self.exhausted_in_pass is None:
            self.exhausted_in_pass = self.pass_name
        return False


_BUDGET: ContextVar[SpendBudget | None] = ContextVar("particles_llm_spend_budget", default=None)


@contextlib.contextmanager
def spend_budget(budget: SpendBudget | None) -> Iterator[SpendBudget | None]:
    """Install ``budget`` for every batch chunk submitted inside the block.

    ``None`` installs nothing, so a caller with no budget opens the same block.
    """
    token = _BUDGET.set(budget)
    try:
        yield budget
    finally:
        _BUDGET.reset(token)


def current_spend_budget() -> SpendBudget | None:
    """The budget in scope, or ``None`` when the caller set none."""
    return _BUDGET.get()
