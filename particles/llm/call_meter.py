# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""A scoped meter every completion is charged to (security review 2026-10-25-2, R2).

The HTTP engine's rate limiter counted requests, and one admitted request could
run a model pass over the whole store: a semantic lint, a semantic curation
queue, an ``LLM_JUDGE`` link suggestion, a reconcile dry run. A
:class:`CallMeter` opened with :func:`metered_calls` is charged once per
completion by :func:`particles.llm.registry.get_provider`, so a caller that
admits requests by the token can admit model calls by the same token.

The meter is a ``ContextVar`` holding a mutable object, the
:mod:`particles.llm.usage` pattern: every task spawned inside the scope shares
the one meter, and nothing outside the scope is charged. No scope, no change.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from contextvars import ContextVar
from typing import Any, Protocol

from particles.llm.registry import CompletionError, CompletionProvider, VisionImage

__all__ = [
    "CallMeter",
    "ModelCallBudgetSpent",
    "current_meter",
    "metered_calls",
    "metered_provider",
]


class ModelCallBudgetSpent(CompletionError):
    """The scope's meter refused a completion: its caller's allowance is spent.

    A :class:`CompletionError`, so each call site degrades exactly as it does
    for any other failed call (the pair stays unjudged, the finding is not
    raised), and no request is sent.
    """


class CallMeter(Protocol):
    """What a scope charges each completion to."""

    def charge(self) -> None:
        """Admit one completion, or raise :class:`ModelCallBudgetSpent`."""
        ...


_METER: ContextVar[CallMeter | None] = ContextVar("particles_llm_call_meter", default=None)


def current_meter() -> CallMeter | None:
    """The meter of the enclosing :func:`metered_calls` scope, else ``None``."""
    return _METER.get()


@contextlib.contextmanager
def metered_calls(meter: CallMeter) -> Iterator[CallMeter]:
    """Charge every completion made inside the block to ``meter``."""
    token = _METER.set(meter)
    try:
        yield meter
    finally:
        _METER.reset(token)


class _MeteredProvider:
    """A provider whose every completion is charged to a meter before it is sent.

    It implements only :class:`CompletionProvider`, so a batch-capable adapter
    behind it is served one charged ``complete()`` at a time.
    """

    def __init__(self, inner: CompletionProvider, meter: CallMeter) -> None:
        self._inner = inner
        self._meter = meter

    @property
    def provider_model(self) -> str:
        return self._inner.provider_model

    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        system: str | None = None,
        temperature: float | None = None,
        images: Sequence[VisionImage] | None = None,
        response_schema: dict[str, Any] | None = None,
        cache_prefix: str | None = None,
        **opts: object,
    ) -> str:
        self._meter.charge()
        return await self._inner.complete(
            prompt,
            max_tokens=max_tokens,
            system=system,
            temperature=temperature,
            images=images,
            response_schema=response_schema,
            cache_prefix=cache_prefix,
            **opts,
        )


def metered_provider(provider: CompletionProvider) -> CompletionProvider:
    """``provider`` charged to the enclosing scope's meter, or unchanged outside one."""
    meter = _METER.get()
    if meter is None:
        return provider
    return _MeteredProvider(provider, meter)
