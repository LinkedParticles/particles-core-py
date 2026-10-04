# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Per-run completion pooling for latency-tolerant fan-in batching.

A :class:`CompletionPool` is the aggregation point that lets *concurrent*
workers — one asyncio task per pending snapshot in the consolidation extract
pass — merge their independent completion requests into one
:func:`~particles.llm.registry.complete_many` job, so the whole night's
request set is what the batching gate (``llm.batch.min_requests``,
``max_requests_per_batch``, the 50 % Message Batches price) sees, instead of
each worker's fragment.

The pool adds **no second batching policy**. Dispatch always routes through
``complete_many_with_provider_model(..., latency_tolerant=True)`` — a pool's
existence *is* the caller's latency-tolerance assertion (it is threaded as a
parameter, never sniffed) — and every knob applies to the
merged set unchanged. With batching disabled or unavailable the merged
set degrades to the same sequential calls at full price.

Dispatch is **quiescence-triggered, not timed**: the pool fires when every
registered participant is either parked in :meth:`CompletionPool.complete_group`
or has deregistered. That is deterministic and testable, and it stays correct
for a hypothetical genuinely-chained caller, which would simply park one
request per wave. There is no deadlock by construction: the write
lock is never held across an LLM call (``ingest/pipeline.py`` holds it around
the write phase only), so a participant waiting on the lock always finishes
its DB work and either parks or deregisters.

A participant may name what it gives back while it waits (``idle``). The
consolidation extract pass uses it to end its session's transaction before a
batch wait of minutes to an hour, so no pooled database connection is held
across it, and to bound how many tasks touch the store at once.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from particles.llm.registry import (
    CompletionRequest,
    LLMPurpose,
    RequestFailure,
    complete_many_with_provider_model,
)

log = logging.getLogger(__name__)

#: What a parked group resolves to: (positionally aligned results, pairing).
GroupResult = tuple[list[str | None], str]

#: What a participant enters while parked: an async context manager factory,
#: called once per park (a participant may park more than once).
IdleHook = Callable[[], AbstractAsyncContextManager[object]]

# The enclosing participant's idle hook, keyed by the pool it registered on.
# A ContextVar because the parking call is made deep inside the worker (the
# extractor), which knows the pool but not the participant around it; each
# asyncio task carries its own copy, so sibling workers never see each other's.
_IDLE: ContextVar[tuple[CompletionPool, IdleHook] | None] = ContextVar(
    "particles_pool_idle", default=None
)


@dataclass
class _ParkedGroup:
    """One participant's request set, awaiting the wave dispatch."""

    requests: list[CompletionRequest]
    key: tuple[Any, ...]
    max_tokens: int
    temperature: float | None
    response_schema: dict[str, Any] | None
    future: asyncio.Future[GroupResult]
    failures: list[RequestFailure | None] = field(default_factory=list)


class CompletionPool:
    """Merge concurrent participants' completion requests into one batch job.

    Usage (the consolidation extract pass is the one shipped caller)::

        pool = CompletionPool("extraction")

        async def worker(...):
            async with pool.participant():
                ...                                   # DB work, planning
                results, pairing = await pool.complete_group(requests, ...)
                ...                                   # parse, persist

        await asyncio.gather(*(worker(...) for ...))

    ``complete_group`` parks the caller until the wave dispatches; results
    slice back positionally per group, with ``None`` marking a per-request
    failure exactly as ``complete_many`` reports it. A job-level failure
    (e.g. an account-level error re-raised by the sequential fallback)
    is raised **in every parked group**, so each worker's own
    failure handling — for extraction, the IN_PROGRESS → PENDING reset —
    runs unchanged.

    A caller that is not registered as a participant dispatches immediately
    (no pooling across callers); groups whose uniform kwargs differ are
    dispatched as separate ``complete_many`` calls (defensive — the
    extraction path's kwargs are uniform by construction).
    """

    def __init__(self, purpose: LLMPurpose, *, expected_participants: int = 0) -> None:
        """``expected_participants`` closes the startup race.

        A worker created but not yet scheduled has not registered, so a
        sibling that parks quickly could otherwise satisfy quiescence alone
        and dispatch a premature single-group wave. The driver declares how
        many workers it is about to start; the pool holds every wave until
        that many have entered :meth:`participant`. Zero (the default) means
        "whoever registers" — correct only when no coordinated fan-out is
        expected, e.g. an ad-hoc unregistered caller.
        """
        self._purpose: LLMPurpose = purpose
        self._unstarted = expected_participants
        self._participants = 0
        self._parked: list[_ParkedGroup] = []
        # Strong refs so in-flight dispatch tasks cannot be garbage-collected.
        self._dispatch_tasks: set[asyncio.Task[None]] = set()

    @asynccontextmanager
    async def participant(self, *, idle: IdleHook | None = None) -> AsyncIterator[None]:
        """Register the enclosing task as a pool participant.

        The pool waits for every registered participant before dispatching a
        wave; exiting the context (normally or by exception) deregisters and
        may itself trigger the dispatch the remaining parked participants are
        waiting on. The trigger is synchronous, so it is safe from this
        ``finally`` even while a cancellation is propagating.

        ``idle`` is entered around every wait in :meth:`complete_group`, before
        the group parks and until its result arrives: the participant's chance
        to give back what the wait must not hold. Its exit may itself wait (to
        re-acquire), which is safe because the participant is no longer
        counted as parked by then, so it cannot hold a wave open.
        """
        if self._unstarted > 0:
            self._unstarted -= 1
        self._participants += 1
        token = _IDLE.set((self, idle)) if idle is not None else None
        try:
            yield
        finally:
            if token is not None:
                _IDLE.reset(token)
            self._participants -= 1
            self._maybe_dispatch()

    async def complete_group(
        self,
        requests: Sequence[CompletionRequest],
        *,
        max_tokens: int,
        temperature: float | None = None,
        response_schema: dict[str, Any] | None = None,
        failures_out: list[RequestFailure | None] | None = None,
    ) -> GroupResult:
        """Submit this participant's whole request set and await the wave.

        Returns ``(results, provider_model)`` with ``results`` positionally
        aligned to ``requests`` (``None`` = that request failed) and
        ``provider_model`` the ``"<provider>:<model>"`` stamp
        pairing. Raises whatever the merged ``complete_many`` call raised —
        for the Anthropic path that is only an account-level failure
        surfaced through the sequential fallback.

        An empty request set returns ``([], "")`` immediately without
        parking — no work is not a reason to hold the wave open.

        ``failures_out`` is extended with this group's slice of the per-request
        failure kinds (:class:`~particles.llm.registry.RequestFailure`), aligned
        with the results, so a caller can retry a budget failure without
        re-submitting a request that merely expired.
        """
        if not requests:
            return [], ""
        future: asyncio.Future[GroupResult] = asyncio.get_running_loop().create_future()
        group = _ParkedGroup(
            requests=list(requests),
            key=self._kwargs_key(max_tokens, temperature, response_schema),
            max_tokens=max_tokens,
            temperature=temperature,
            response_schema=response_schema,
            future=future,
        )
        registered = _IDLE.get()
        idle = registered[1] if registered is not None and registered[0] is self else None
        async with idle() if idle is not None else nullcontext():
            self._parked.append(group)
            try:
                self._maybe_dispatch()
                result = await future
            finally:
                # Cancelled before dispatch: withdraw the group so a later wave
                # does not try to resolve a dead future's requests.
                if group in self._parked:
                    self._parked.remove(group)
        if failures_out is not None:
            failures_out.extend(group.failures)
        return result

    @staticmethod
    def _kwargs_key(
        max_tokens: int,
        temperature: float | None,
        response_schema: dict[str, Any] | None,
    ) -> tuple[Any, ...]:
        schema_key = (
            json.dumps(response_schema, sort_keys=True) if response_schema is not None else None
        )
        return (max_tokens, temperature, schema_key)

    def _maybe_dispatch(self) -> None:
        """Fire the wave iff every live participant is parked (quiescence).

        "Parked" means a group still awaiting dispatch: a participant's one
        group leaves ``_parked`` the moment its wave is taken, not when its
        task next runs. Counting until the task resumed let the first
        participant to re-park after a wave (a retry group) find its answered
        but not-yet-resumed siblings still "parked" and dispatch alone, so
        retries that should share one follow-up batch went out one by one.

        Synchronous by design: callable from ``finally`` blocks and from the
        parking path without suspending. The dispatch itself runs in its own
        task, so a participant that exits mid-cancellation never carries the
        batch call in its dying frame.
        """
        if self._unstarted > 0 or not self._parked or len(self._parked) < self._participants:
            return
        groups, self._parked = self._parked, []
        task = asyncio.get_running_loop().create_task(self._dispatch(groups))
        self._dispatch_tasks.add(task)
        task.add_done_callback(self._dispatch_tasks.discard)

    async def _dispatch(self, groups: list[_ParkedGroup]) -> None:
        """Merge one wave's groups per kwargs key and resolve their futures."""
        by_key: dict[tuple[Any, ...], list[_ParkedGroup]] = {}
        for group in groups:
            by_key.setdefault(group.key, []).append(group)
        if len(by_key) > 1:
            log.info(
                "Completion pool wave holds %d distinct kwargs shapes; "
                "dispatching them as separate jobs.",
                len(by_key),
            )
        for key_groups in by_key.values():
            merged: list[CompletionRequest] = []
            for group in key_groups:
                merged.extend(group.requests)
            first = key_groups[0]
            log.info(
                "Completion pool dispatching %d pooled request(s) from %d group(s) for purpose %r.",
                len(merged),
                len(key_groups),
                self._purpose,
            )
            failures: list[RequestFailure | None] = []
            try:
                results, provider_model = await complete_many_with_provider_model(
                    self._purpose,
                    merged,
                    max_tokens=first.max_tokens,
                    temperature=first.temperature,
                    response_schema=first.response_schema,
                    latency_tolerant=True,
                    failures_out=failures,
                )
            except Exception as exc:  # noqa: BLE001 — routed into every group's future
                for group in key_groups:
                    if not group.future.done():
                        group.future.set_exception(exc)
                continue
            offset = 0
            for group in key_groups:
                count = len(group.requests)
                group.failures = failures[offset : offset + count]
                if not group.future.done():
                    group.future.set_result((results[offset : offset + count], provider_model))
                offset += count
