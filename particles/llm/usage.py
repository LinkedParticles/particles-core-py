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

A completion served from the replay cache (:mod:`particles.llm.replay`) is
reported too, flagged ``replayed`` with the usage it was recorded with. It
lands in a row of its own, priced at what it would have cost, and is kept out
of the run's spend so the usage line shows the saving instead of undercounting
the work.

**Sampling**. The registry records whether the caller asked for a
fixed temperature (:func:`purpose_scope`), and each adapter says whether it
sent one. A slice counts the calls that asked (``temperature_requested``) and,
of those, the calls the provider received without it (``temperature_dropped``):
a model that refuses the parameter, or an OpenAI-compatible entry with
``send_temperature: false``. A call that asked for none counts in neither.
:func:`observe_sampling` reads the same fact for the calls inside one block,
which is how a probe-ledger row names what its answer was sampled under.
"""

from __future__ import annotations

import contextlib
import threading
from collections import Counter
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from particles.core.schema import StoreLLMSpend

#: Purpose recorded for a call made outside the registry (a directly-constructed
#: adapter). Kept visible rather than dropped: it is still spend.
UNATTRIBUTED = "unattributed"

#: What one call sent, as a probe-ledger row records it.
#: ``sent``: the caller asked for a fixed temperature and the provider received
#: it, which does not make the reply reproducible. ``dropped``:
#: the caller asked and the provider did not receive it. ``unset``: the caller
#: asked for none. ``none`` is written by a rung that makes no model call and
#: is never observed here.
Sampling = Literal["sent", "dropped", "unset", "none"]

#: The ledger value of a row a non-model rung wrote.
NO_SAMPLING: Sampling = "none"


class UsageRow(BaseModel):
    """Totals for one (purpose, provider, model, batched, replayed) slice of a run."""

    purpose: str
    provider: str
    model: str
    #: True for requests that rode a batch API (billed at the batch discount).
    batched: bool = False
    #: True for completions served from the replay cache: nothing was billed,
    #: and the tokens are the ones recorded with the original call.
    replayed: bool = False
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
    #: Responses whose caller asked for a fixed temperature.
    temperature_requested: int = 0
    #: Of those, the responses the provider received without it: the model
    #: refuses the parameter, or the endpoint is configured not to send it.
    temperature_dropped: int = 0
    #: List price of this slice in US$, ``None`` when the model has no entry.
    #: For a replayed slice this is what the calls would have cost, not spend.
    cost_usd: float | None = None


class LLMUsage(BaseModel):
    """A snapshot of one run's measured LLM usage, priced at list price."""

    rows: list[UsageRow] = Field(default_factory=list)
    #: Sum of every billed row's list price; ``None`` when any billed row is
    #: unpriced (a partial figure would read as a total). Replayed rows are
    #: not spend and are not in it.
    cost_usd: float | None = 0.0
    #: ``provider:model`` of each unpriced model.
    unpriced: list[str] = Field(default_factory=list)
    #: List price the replayed rows would have cost: what the replay
    #: cache saved. ``None`` when a replayed row is unpriced.
    replayed_cost_usd: float | None = 0.0

    @property
    def calls(self) -> int:
        """Responses received from a provider across every row (replays excluded)."""
        return sum(row.calls for row in self.rows if not row.replayed)

    @property
    def replayed_calls(self) -> int:
        """Completions served from the replay cache across every row."""
        return sum(row.calls for row in self.rows if row.replayed)

    @property
    def max_tokens_stops(self) -> int:
        """Responses cut off at the token ceiling across every row."""
        return sum(row.max_tokens_stops for row in self.rows)

    @property
    def temperature_dropped(self) -> int:
        """Responses whose requested temperature was not sent, across every row."""
        return sum(row.temperature_dropped for row in self.rows)

    @property
    def priced_cost_usd(self) -> float:
        """List price of the priced rows only.

        What a dollar budget can see: :attr:`cost_usd` is ``None`` as soon as
        one row is unpriced, so a budget check sums the rows it can price and
        the usage line discloses the rest.
        """
        return sum(
            row.cost_usd for row in self.rows if row.cost_usd is not None and not row.replayed
        )


@dataclass
class _Counts:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    max_tokens_stops: int = 0
    temperature_requested: int = 0
    temperature_dropped: int = 0


_Key = tuple[str, str, str, bool, bool]  # (purpose, provider, model, batched, replayed)


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
        replayed: bool = False,
        sampling: Sampling | None = None,
        temperature_requested: bool = False,
    ) -> None:
        """Add one response's usage to the slice it belongs to."""
        key = (purpose, provider, model, batched, replayed)
        with self._lock:
            counts = self._counts.setdefault(key, _Counts())
            counts.calls += 1
            counts.input_tokens += input_tokens
            counts.output_tokens += output_tokens
            counts.cache_creation_tokens += cache_creation_tokens
            counts.cache_read_tokens += cache_read_tokens
            counts.max_tokens_stops += int(hit_max_tokens)
            counts.temperature_requested += int(temperature_requested)
            counts.temperature_dropped += int(sampling == "dropped")

    def take(self) -> LLMUsage:
        """The totals so far, then start counting from zero.

        For a caller that runs passes one after another inside one scope and
        wants each pass's usage on its own (``extractor benchmark --runs N``
        writes one run file per pass).
        """
        with self._lock:
            counts, self._counts = self._counts, {}
        return self._priced(sorted(counts.items()))

    def snapshot(self) -> LLMUsage:
        """The totals so far, priced at the configured list prices (read now)."""
        with self._lock:
            items = sorted(self._counts.items())
        return self._priced(items)

    @staticmethod
    def _priced(items: list[tuple[_Key, _Counts]]) -> LLMUsage:
        rows: list[UsageRow] = []
        unpriced: list[str] = []
        total: float | None = 0.0
        saved: float | None = 0.0
        for (purpose, provider, model, batched, replayed), counts in items:
            cost = _price_row(provider, model, batched, counts)
            if replayed:
                saved = None if cost is None or saved is None else saved + cost
            elif cost is None:
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
                    replayed=replayed,
                    calls=counts.calls,
                    input_tokens=counts.input_tokens,
                    output_tokens=counts.output_tokens,
                    cache_creation_tokens=counts.cache_creation_tokens,
                    cache_read_tokens=counts.cache_read_tokens,
                    max_tokens_stops=counts.max_tokens_stops,
                    temperature_requested=counts.temperature_requested,
                    temperature_dropped=counts.temperature_dropped,
                    cost_usd=cost,
                )
            )
        return LLMUsage(rows=rows, cost_usd=total, unpriced=unpriced, replayed_cost_usd=saved)


_ACCUMULATORS: ContextVar[tuple[UsageAccumulator, ...]] = ContextVar(
    "particles_llm_usage_accumulators", default=()
)
_PURPOSE: ContextVar[str | None] = ContextVar("particles_llm_usage_purpose", default=None)
#: Whether the call in flight asked for a fixed temperature; ``None`` outside
#: the registry, where nobody said.
_TEMPERATURE_REQUESTED: ContextVar[bool | None] = ContextVar(
    "particles_llm_usage_temperature_requested", default=None
)


@dataclass
class SamplingObservation:
    """What the calls inside one :func:`observe_sampling` block sent.

    Filled at each response by :func:`record_usage`, so the value describes
    the calls that answered, whichever provider served them: an override, a
    replayed entry, or the configured adapter. Thread-safe for the same reason
    the accumulator is.
    """

    counts: Counter[str] = field(default_factory=Counter)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _add(self, sampling: Sampling | None) -> None:
        with self._lock:
            self.counts[sampling or "unknown"] += 1

    def value(self) -> Sampling | None:
        """The one value every observed call shares, or ``None``.

        ``None`` when no call reported (an in-process provider records no
        usage, or every answer came from the ledger), when one did not say,
        or when the calls disagree. A ledger row then records ``NULL`` rather
        than a value no call established.
        """
        with self._lock:
            seen = set(self.counts)
        if len(seen) != 1:
            return None
        only = seen.pop()
        return None if only == "unknown" else only  # type: ignore[return-value]


_OBSERVATIONS: ContextVar[tuple[SamplingObservation, ...]] = ContextVar(
    "particles_llm_usage_observations", default=()
)


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
def observe_sampling(
    observation: SamplingObservation | None = None,
) -> Iterator[SamplingObservation]:
    """Record what the completions made inside the block sent.

    Opened by a write site around the calls whose answers it records, the
    :func:`track_usage` pattern: tasks and threads spawned inside see it,
    concurrent blocks never see each other's calls. Read
    :meth:`SamplingObservation.value` after the calls. Pass ``observation``
    to keep adding to one already opened, for a loop that records its answers
    between calls.
    """
    observation = observation if observation is not None else SamplingObservation()
    token = _OBSERVATIONS.set((*_OBSERVATIONS.get(), observation))
    try:
        yield observation
    finally:
        _OBSERVATIONS.reset(token)


def current_observations() -> tuple[SamplingObservation, ...]:
    """The :func:`observe_sampling` blocks open in this context."""
    return _OBSERVATIONS.get()


@contextlib.contextmanager
def observing(observations: tuple[SamplingObservation, ...]) -> Iterator[None]:
    """Report the calls made inside the block to exactly ``observations``.

    For a call made on several callers' behalf in a task none of them runs:
    the completion pool's merged batch reports to each parked
    caller's own blocks, instead of to whichever caller woke the dispatch.
    """
    unique = tuple({id(o): o for o in observations}.values())
    token = _OBSERVATIONS.set(unique)
    try:
        yield
    finally:
        _OBSERVATIONS.reset(token)


@contextlib.contextmanager
def purpose_scope(purpose: str, *, temperature_requested: bool | None = None) -> Iterator[None]:
    """Attribute the completions made inside the block to ``purpose``.

    ``temperature_requested`` says whether the caller asked for a fixed
    temperature, so a response can be counted as sent or dropped.
    Set by the registry around each provider call, never by a call site.
    """
    token = _PURPOSE.set(purpose)
    requested = _TEMPERATURE_REQUESTED.set(temperature_requested)
    try:
        yield
    finally:
        _TEMPERATURE_REQUESTED.reset(requested)
        _PURPOSE.reset(token)


def sampling_of(temperature_sent: bool | None) -> Sampling | None:
    """What a response sent, given whether its adapter sent a temperature.

    ``None`` when nobody said: the call was made outside the registry, or
    the caller asked and the adapter did not report what it sent.
    """
    requested = _TEMPERATURE_REQUESTED.get()
    if requested is None:
        return None
    if not requested:
        return "unset"
    if temperature_sent is None:
        return None
    return "sent" if temperature_sent else "dropped"


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
    replayed: bool = False,
    temperature_sent: bool | None = None,
) -> None:
    """Report one response's usage to every open :func:`track_usage` scope.

    ``provider_model`` is the adapter's ``"<provider>:<model>"`` key. Called by
    the adapters, and by the replay cache with ``replayed=True`` for a
    completion it served without a call; a no-op when no scope is open.
    ``temperature_sent`` is whether the request that produced this response
    carried a temperature; ``None`` when the reporter cannot say.
    """
    sampling = sampling_of(temperature_sent)
    for observation in _OBSERVATIONS.get():
        observation._add(sampling)
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
            replayed=replayed,
            sampling=sampling,
            temperature_requested=bool(_TEMPERATURE_REQUESTED.get()),
        )


def _price_row(provider: str, model: str, batched: bool, counts: _Counts) -> float | None:
    """List price of one slice in US$, or ``None`` when the model is unpriced."""
    from particles.config import ProviderSelection, get_config

    config = get_config()
    selection = ProviderSelection(provider=provider, model=model)
    price = config.llm.price_for(selection)
    if price is None:
        return None
    llm = config.llm
    cost = (
        counts.input_tokens * price.input
        + counts.cache_creation_tokens * price.input * llm.cache_write_price_multiplier
        + counts.cache_read_tokens * price.input * llm.cache_read_multiplier_for(selection)
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

    Replayed completions count in a group's calls and are named
    (``90 replayed``), but their tokens are not in its input and output: those
    are what this run bought. What they would have cost closes the line.
    """
    if not usage.rows:
        return "LLM usage: no LLM calls."
    grouped: dict[tuple[str, str, str], list[UsageRow]] = {}
    for row in usage.rows:
        grouped.setdefault((row.purpose, row.provider, row.model), []).append(row)
    parts: list[str] = []
    for (purpose, provider, model), rows in grouped.items():
        calls = sum(r.calls for r in rows)
        billed = [r for r in rows if not r.replayed]
        label = model if provider == "anthropic" else f"{provider}:{model}"
        text = (
            f"{_plural(calls, purpose.replace('_', ' ') + ' call')} ({label}), "
            f"{format_tokens(sum(r.input_tokens for r in billed))} input, "
            f"{format_tokens(sum(r.output_tokens for r in billed))} output tokens"
        )
        cache_read = sum(r.cache_read_tokens for r in billed)
        cache_write = sum(r.cache_creation_tokens for r in billed)
        if cache_read:
            text += f", {format_tokens(cache_read)} cache read"
        if cache_write:
            text += f", {format_tokens(cache_write)} cache write"
        batched = sum(r.calls for r in billed if r.batched)
        if batched:
            text += f", {batched} batched"
        replayed = sum(r.calls for r in rows if r.replayed)
        if replayed:
            text += f", {replayed} replayed"
        parts.append(text)
    line = "LLM usage: " + "; ".join(parts)
    if usage.cost_usd is not None:
        line += f"; ≈ {format_usd(usage.cost_usd)} at list price."
    else:
        verb = "has no entry" if len(usage.unpriced) == 1 else "have no entries"
        line += (
            f"; no dollar total, because {', '.join(usage.unpriced)} {verb} in llm.price_per_mtok."
        )
    if usage.replayed_calls:
        line += f" {_plural(usage.replayed_calls, 'call')} replayed from the cache"
        if usage.replayed_cost_usd is not None:
            line += f" (≈ {format_usd(usage.replayed_cost_usd)} not re-bought)"
        line += "."
    stops = usage.max_tokens_stops
    if stops:
        line += (
            f" {_plural(stops, 'reply', 'replies')} stopped at max_tokens and may be cut off; "
            "raise the token budget."
        )
    line += _dropped_clause(usage)
    return line


def _dropped_clause(usage: LLMUsage) -> str:
    """`` semantic lint: 200 of 200 calls asked for a fixed temperature …``.

    Empty when every requested temperature was sent. One clause per purpose
    that dropped one, so a reader sees which checks sampled at the default.
    """
    by_purpose: dict[str, list[int]] = {}
    for row in usage.rows:
        if row.temperature_requested:
            counts = by_purpose.setdefault(row.purpose, [0, 0])
            counts[0] += row.temperature_requested
            counts[1] += row.temperature_dropped
    parts = [
        f"{purpose.replace('_', ' ')}: {dropped} of {_plural(requested, 'call')}"
        for purpose, (requested, dropped) in by_purpose.items()
        if dropped
    ]
    if not parts:
        return ""
    return (
        f" Fixed temperature not sent ({'; '.join(parts)}): "
        "the model or endpoint does not take one, so these calls sampled at the default."
    )


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
