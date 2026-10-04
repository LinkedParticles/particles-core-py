# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Per-run LLM token usage, measured at the completion port.

A run that pays for completions (``particles audit``, ``particles memory
consolidate``) opens :func:`track_usage` around its body and reads the totals
back afterwards; every adapter reports each response's ``usage`` through
:func:`record_usage`. The registry tags each call with its purpose
(:func:`purpose_scope`), so the totals come out per purpose and per model
without the adapters knowing why they were called.

Both scopes are ``ContextVar``-held, the pattern ``override_providers`` uses:
an accumulator sees the calls made by the task that opened it and by every task
or worker thread spawned inside it, and nothing else, so two concurrent runs
never count each other's tokens. Nested scopes each see the inner calls. With
no scope open, :func:`record_usage` is a no-op.

The totals are what the provider *reported*, which is what it bills. The
dollar figure is list price from ``llm.price_per_mtok`` with the batch and
prompt-cache multipliers applied; it is never read from a billing API, which
would need an Admin key no user should have to hold to learn what a run cost.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass

from pydantic import BaseModel, Field

from particles.core.schema import StoreLLMSpend

#: Purpose recorded for a call made outside the registry (a directly-constructed
#: adapter). Kept visible rather than dropped: it is still spend.
UNATTRIBUTED = "unattributed"


class UsageRow(BaseModel):
    """Totals for one (purpose, provider, model, batched) slice of a run."""

    purpose: str
    provider: str
    model: str
    #: True for requests that rode a batch API (billed at the batch discount).
    batched: bool = False
    #: Responses received. A call that failed before the provider answered is
    #: not billed and is not counted.
    calls: int = 0
    #: Uncached input tokens (Anthropic ``input_tokens``; for an
    #: OpenAI-compatible endpoint, ``prompt_tokens`` less its cached share).
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    #: Responses that stopped at the token ceiling (``stop_reason=max_tokens``
    #: or ``finish_reason=length``): the reply was cut off mid-text.
    max_tokens_stops: int = 0
    #: List price of this slice in US$, ``None`` when the model has no entry.
    cost_usd: float | None = None


class LLMUsage(BaseModel):
    """A snapshot of one run's measured LLM usage, priced at list price."""

    rows: list[UsageRow] = Field(default_factory=list)
    #: Sum of every row's list price; ``None`` when any row is unpriced (a
    #: partial figure would read as a total).
    cost_usd: float | None = 0.0
    #: ``provider:model`` of each unpriced model.
    unpriced: list[str] = Field(default_factory=list)

    @property
    def calls(self) -> int:
        """Responses received across every row."""
        return sum(row.calls for row in self.rows)

    @property
    def max_tokens_stops(self) -> int:
        """Responses cut off at the token ceiling across every row."""
        return sum(row.max_tokens_stops for row in self.rows)

    @property
    def priced_cost_usd(self) -> float:
        """List price of the priced rows only.

        What a dollar budget can see: :attr:`cost_usd` is ``None`` as soon as
        one row is unpriced, so a budget check sums the rows it can price and
        the usage line discloses the rest.
        """
        return sum(row.cost_usd for row in self.rows if row.cost_usd is not None)


@dataclass
class _Counts:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    max_tokens_stops: int = 0


_Key = tuple[str, str, str, bool]  # (purpose, provider, model, batched)


class UsageAccumulator:
    """Running token totals for one :func:`track_usage` scope.

    Thread-safe: the Anthropic adapter reads a batch's results in a worker
    thread, and that thread records into the same accumulator.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[_Key, _Counts] = {}

    def add(
        self,
        *,
        purpose: str,
        provider: str,
        model: str,
        batched: bool,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int,
        cache_read_tokens: int,
        hit_max_tokens: bool,
    ) -> None:
        """Add one response's usage to the slice it belongs to."""
        key = (purpose, provider, model, batched)
        with self._lock:
            counts = self._counts.setdefault(key, _Counts())
            counts.calls += 1
            counts.input_tokens += input_tokens
            counts.output_tokens += output_tokens
            counts.cache_creation_tokens += cache_creation_tokens
            counts.cache_read_tokens += cache_read_tokens
            counts.max_tokens_stops += int(hit_max_tokens)

    def snapshot(self) -> LLMUsage:
        """The totals so far, priced at the configured list prices (read now)."""
        with self._lock:
            items = sorted(self._counts.items())
        rows: list[UsageRow] = []
        unpriced: list[str] = []
        total: float | None = 0.0
        for (purpose, provider, model, batched), counts in items:
            cost = _price_row(provider, model, batched, counts)
            if cost is None:
                key = f"{provider}:{model}"
                if key not in unpriced:
                    unpriced.append(key)
                total = None
            elif total is not None:
                total += cost
            rows.append(
                UsageRow(
                    purpose=purpose,
                    provider=provider,
                    model=model,
                    batched=batched,
                    calls=counts.calls,
                    input_tokens=counts.input_tokens,
                    output_tokens=counts.output_tokens,
                    cache_creation_tokens=counts.cache_creation_tokens,
                    cache_read_tokens=counts.cache_read_tokens,
                    max_tokens_stops=counts.max_tokens_stops,
                    cost_usd=cost,
                )
            )
        return LLMUsage(rows=rows, cost_usd=total, unpriced=unpriced)


_ACCUMULATORS: ContextVar[tuple[UsageAccumulator, ...]] = ContextVar(
    "particles_llm_usage_accumulators", default=()
)
_PURPOSE: ContextVar[str | None] = ContextVar("particles_llm_usage_purpose", default=None)


@contextlib.contextmanager
def track_usage() -> Iterator[UsageAccumulator]:
    """Count every completion made inside the block; yields the accumulator.

    Call :meth:`UsageAccumulator.snapshot` on the yielded value (inside or
    after the block) for the totals. An enclosing scope keeps counting too.
    """
    accumulator = UsageAccumulator()
    token = _ACCUMULATORS.set((*_ACCUMULATORS.get(), accumulator))
    try:
        yield accumulator
    finally:
        _ACCUMULATORS.reset(token)


@contextlib.contextmanager
def purpose_scope(purpose: str) -> Iterator[None]:
    """Attribute the completions made inside the block to ``purpose``.

    Set by the registry around each provider call, never by a call site.
    """
    token = _PURPOSE.set(purpose)
    try:
        yield
    finally:
        _PURPOSE.reset(token)


def current_purpose() -> str | None:
    """The LLM purpose of the completion in flight, or None outside a :func:`purpose_scope`."""
    return _PURPOSE.get()


def record_usage(
    provider_model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    hit_max_tokens: bool = False,
    batched: bool = False,
) -> None:
    """Report one response's usage to every open :func:`track_usage` scope.

    ``provider_model`` is the adapter's ``"<provider>:<model>"`` key. Called by
    the adapters; a no-op when no scope is open.
    """
    accumulators = _ACCUMULATORS.get()
    if not accumulators:
        return
    provider, _, model = provider_model.partition(":")
    purpose = _PURPOSE.get() or UNATTRIBUTED
    for accumulator in accumulators:
        accumulator.add(
            purpose=purpose,
            provider=provider,
            model=model,
            batched=batched,
            input_tokens=max(0, input_tokens),
            output_tokens=max(0, output_tokens),
            cache_creation_tokens=max(0, cache_creation_tokens),
            cache_read_tokens=max(0, cache_read_tokens),
            hit_max_tokens=hit_max_tokens,
        )


def _price_row(provider: str, model: str, batched: bool, counts: _Counts) -> float | None:
    """List price of one slice in US$, or ``None`` when the model is unpriced."""
    from particles.config import ProviderSelection, get_config

    config = get_config()
    price = config.llm.price_for(ProviderSelection(provider=provider, model=model))
    if price is None:
        return None
    llm = config.llm
    cost = (
        counts.input_tokens * price.input
        + counts.cache_creation_tokens * price.input * llm.cache_write_price_multiplier
        + counts.cache_read_tokens * price.input * llm.cache_read_price_multiplier
        + counts.output_tokens * price.output
    ) / 1_000_000
    if batched:
        cost *= 1.0 - llm.batch_discount
    return cost


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def format_usd(value: float) -> str:
    """Dollars at a readable precision: cents below $10, whole dollars above."""
    if value < 0.005:
        return "$0.00"
    if value < 10:
        return f"${value:,.2f}"
    return f"${value:,.0f}"


def format_tokens(count: int) -> str:
    """``312k``, ``1.2M``, or the bare count below a thousand."""
    if count < 1_000:
        return str(count)
    if count < 1_000_000:
        return f"{count / 1_000:.0f}k"
    return f"{count / 1_000_000:.1f}M"


def _plural(n: int, word: str, plural: str | None = None) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {plural or word + 's'}"


def render_usage_line(usage: LLMUsage) -> str:
    """One line of measured usage, for the end of a run's report.

    ``LLM usage: 96 extraction calls (claude-sonnet-5), 312k input, 640k output
    tokens; 200 semantic lint calls (claude-haiku-4-5), 60k input, 4k output
    tokens; ≈ $9.20 at list price.`` An unpriced model gets tokens only.
    """
    if not usage.rows:
        return "LLM usage: no LLM calls."
    grouped: dict[tuple[str, str, str], list[UsageRow]] = {}
    for row in usage.rows:
        grouped.setdefault((row.purpose, row.provider, row.model), []).append(row)
    parts: list[str] = []
    for (purpose, provider, model), rows in grouped.items():
        calls = sum(r.calls for r in rows)
        label = model if provider == "anthropic" else f"{provider}:{model}"
        text = (
            f"{_plural(calls, purpose.replace('_', ' ') + ' call')} ({label}), "
            f"{format_tokens(sum(r.input_tokens for r in rows))} input, "
            f"{format_tokens(sum(r.output_tokens for r in rows))} output tokens"
        )
        cache_read = sum(r.cache_read_tokens for r in rows)
        cache_write = sum(r.cache_creation_tokens for r in rows)
        if cache_read:
            text += f", {format_tokens(cache_read)} cache read"
        if cache_write:
            text += f", {format_tokens(cache_write)} cache write"
        batched = sum(r.calls for r in rows if r.batched)
        if batched:
            text += f", {batched} batched"
        parts.append(text)
    line = "LLM usage: " + "; ".join(parts)
    if usage.cost_usd is not None:
        line += f"; ≈ {format_usd(usage.cost_usd)} at list price."
    else:
        verb = "has no entry" if len(usage.unpriced) == 1 else "have no entries"
        line += (
            f"; no dollar total, because {', '.join(usage.unpriced)} {verb} in llm.price_per_mtok."
        )
    stops = usage.max_tokens_stops
    if stops:
        line += (
            f" {_plural(stops, 'reply', 'replies')} stopped at max_tokens and may be cut off; "
            "raise the token budget."
        )
    return line


def render_store_spend_line(spend: StoreLLMSpend) -> str:
    """One line of a store's cumulative recorded spend.

    ``LLM spend recorded for this store: US$41.20 across 37 runs since
    2026-09-30.`` The figure is list price over recorded token counts, and
    runs from before usage was recorded are not in it, hence "since".
    """
    line = (
        f"LLM spend recorded for this store: US{format_usd(spend.cost_usd)} across "
        f"{_plural(spend.runs, 'run')} since {spend.since.date().isoformat()}"
    )
    if spend.unpriced_runs:
        verb = "is" if spend.unpriced_runs == 1 else "are"
        line += (
            f" ({_plural(spend.unpriced_runs, 'run')} used a model with no "
            f"llm.price_per_mtok entry and {verb} not in the total)"
        )
    return line + "."
