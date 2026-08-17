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
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from particles.llm.registry import (
    CompletionRequest,
    LLMPurpose,
    complete_many_with_provider_model,
)

log = logging.getLogger(__name__)

#: What a parked group resolves to: (positionally aligned results, pairing).
GroupResult = tuple[list[str | None], str]


@dataclass
class _ParkedGroup:
    """One participant's request set, awaiting the wave dispatch."""

    requests: list[CompletionRequest]
    key: tuple[Any, ...]
    max_tokens: int
    temperature: float | None
    response_schema: dict[str, Any] | None
    future: asyncio.Future[GroupResult]


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
    (e.g. an account-level error re-raised by the sequential fallback) is raised **in every parked group**, so each worker's own
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
        self._parked_participants = 0
        self._parked: list[_ParkedGroup] = []
        # Strong refs so in-flight dispatch tasks cannot be garbage-collected.
        self._dispatch_tasks: set[asyncio.Task[None]] = set()

    @asynccontextmanager
    async def participant(self) -> AsyncIterator[None]:
        """Register the enclosing task as a pool participant.

        The pool waits for every registered participant before dispatching a
        wave; exiting the context (normally or by exception) deregisters and
        may itself trigger the dispatch the remaining parked participants are
        waiting on. The trigger is synchronous, so it is safe from this
        ``finally`` even while a cancellation is propagating.
        """
        if self._unstarted > 0:
            self._unstarted -= 1
        self._participants += 1
        try:
            yield
        finally:
            self._participants -= 1
            self._maybe_dispatch()

    async def complete_group(
        self,
        requests: Sequence[CompletionRequest],
        *,
        max_tokens: int,
        temperature: float | None = None,
        response_schema: dict[str, Any] | None = None,
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
        self._parked.append(group)
        self._parked_participants += 1
        try:
            self._maybe_dispatch()
            return await future
        finally:
            self._parked_participants -= 1
            # Cancelled before dispatch: withdraw the group so a later wave
            # does not try to resolve a dead future's requests.
            if group in self._parked:
                self._parked.remove(group)

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

        Synchronous by design: callable from ``finally`` blocks and from the
        parking path without suspending. The dispatch itself runs in its own
        task, so a participant that exits mid-cancellation never carries the
        batch call in its dying frame.
        """
        if (
            self._unstarted > 0
            or not self._parked
            or self._parked_participants < self._participants
        ):
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
            try:
                results, provider_model = await complete_many_with_provider_model(
                    self._purpose,
                    merged,
                    max_tokens=first.max_tokens,
                    temperature=first.temperature,
                    response_schema=first.response_schema,
                    latency_tolerant=True,
                )
            except Exception as exc:  # noqa: BLE001 — routed into every group's future
                for group in key_groups:
                    if not group.future.done():
                        group.future.set_exception(exc)
                continue
            offset = 0
            for group in key_groups:
                count = len(group.requests)
                if not group.future.done():
                    group.future.set_result((results[offset : offset + count], provider_model))
                offset += count
