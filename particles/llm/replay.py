# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Record and replay completions at the port, for benchmark ablations.

A benchmark ablation that changes ranking, decay or top-k re-runs extraction
and answer prompts that are byte-identical to the last run's, and pays for
them again. :func:`replay_scope` opens a run-scoped, content-addressed cache
over every completion made inside it: :func:`~particles.llm.registry.get_provider`
wraps the resolved adapter in a :class:`ReplayingProvider`, so the single
call, the batch, the registry's sequential fallback and the pooled dispatch
all go through it with no call-site change.

Three modes (:class:`ReplayMode`): ``off`` (no scope, the default), ``record``
(every call goes to the provider and its reply is written), and ``replay``
(a recorded reply is returned without a call; a miss falls through to the
provider and is recorded). The scope is a ``ContextVar``, the
``override_providers`` pattern: tasks spawned inside the block see it, and
nothing outside it does. It is opened by a benchmark verb's flag and never by
config, so no product write path can replay a completion by accident.

**The key** is a SHA-256 over the purpose, the ``provider:model`` pairing, the
prompt, system and cache prefix, the images (by digest), the response schema,
``max_tokens`` and ``temperature``. Prompt-injection fences carry a per-call
random nonce (:mod:`particles.llm.fencing`), so the nonce is replaced by a
placeholder before hashing; without that no fenced call would ever hit. Two
calls that differ only in their nonce are the same request.

**The entry** is one JSON file under ``<dir>/<key[:2]>/<key>.json`` holding
the reply text and the usage the provider reported for it. A replayed call
reports that usage through :func:`~particles.llm.usage.record_usage` with
``replayed=True``, so the run's usage line counts the work and prices the
saving instead of showing a run that did nothing. A batch reports usage for
the whole job, not per request, so a reply recorded from a batch carries an
even share of the batch's measured tokens (the job total is preserved
exactly). Failures are never recorded: a ``None`` or an exception is retried
on the next run.

The cache lives outside the store DB: it is a development artefact, not state,
and deleting the directory loses nothing but money.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from collections.abc import Iterator, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from particles.llm.usage import UsageRow, record_usage, track_usage

if TYPE_CHECKING:
    from particles.llm.registry import (
        CompletionProvider,
        CompletionRequest,
        RequestFailure,
        VisionImage,
    )

log = logging.getLogger(__name__)

#: Version of the key derivation and the entry format. Bumping it orphans every
#: recorded entry, which is the intended effect of a change to either.
_FORMAT = 1

#: A fence nonce as :func:`particles.llm.fencing.fence` writes it (32 hex
#: characters inside ``nonce="…"``).
_NONCE_RE = re.compile(r'nonce="([0-9a-f]{32})"')
_NONCE_PLACEHOLDER = "<nonce>"


class ReplayMode(StrEnum):
    """What a :func:`replay_scope` does with each completion."""

    OFF = "off"
    """No cache: every call goes to the provider and nothing is written."""
    RECORD = "record"
    """Every call goes to the provider; each reply is written to the cache."""
    REPLAY = "replay"
    """A recorded reply is returned without a call; a miss calls and records."""


@dataclass
class ReplayStats:
    """Counts for one :func:`replay_scope`, read by the verb that opened it."""

    mode: ReplayMode
    directory: Path
    #: Completions served from the cache.
    hits: int = 0
    #: Completions sent to the provider (every call under ``record``). A batch
    #: job the provider rejects is not counted: the registry re-sends its
    #: requests one at a time, and those are.
    misses: int = 0
    #: Replies written to the cache.
    recorded: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _bump(self, *, hits: int = 0, misses: int = 0, recorded: int = 0) -> None:
        with self._lock:
            self.hits += hits
            self.misses += misses
            self.recorded += recorded


@dataclass(frozen=True)
class _Scope:
    mode: ReplayMode
    directory: Path
    stats: ReplayStats


_SCOPE: ContextVar[_Scope | None] = ContextVar("particles_llm_replay_scope", default=None)


def default_replay_dir() -> Path:
    """``~/.particles/llm-replay/``: beside the benchmark caches, outside any store."""
    return Path.home() / ".particles" / "llm-replay"


@contextlib.contextmanager
def replay_scope(mode: ReplayMode, directory: Path | None = None) -> Iterator[ReplayStats]:
    """Record or replay every completion made inside the block.

    ``mode=OFF`` opens no scope and yields stats that stay at zero, so a verb
    can wrap its body unconditionally. ``directory`` defaults to
    :func:`default_replay_dir`. A nested scope replaces the outer one for its
    block.
    """
    target = directory if directory is not None else default_replay_dir()
    stats = ReplayStats(mode=mode, directory=target)
    if mode is ReplayMode.OFF:
        yield stats
        return
    target.mkdir(parents=True, exist_ok=True)
    token = _SCOPE.set(_Scope(mode=mode, directory=target, stats=stats))
    try:
        yield stats
    finally:
        _SCOPE.reset(token)


def current_replay_scope() -> ReplayStats | None:
    """The open scope's stats, or ``None`` when no replay scope is open."""
    scope = _SCOPE.get()
    return scope.stats if scope is not None else None


def maybe_wrap(purpose: str, provider: CompletionProvider) -> CompletionProvider:
    """Wrap ``provider`` in the open replay scope, or return it unchanged.

    Called by :func:`~particles.llm.registry.get_provider` on every config
    resolution. The batch capability is preserved: a batch-capable provider
    gets a wrapper that is batch-capable too, so the registry's
    ``isinstance(provider, BatchCompletionProvider)`` test reads the same.
    """
    scope = _SCOPE.get()
    if scope is None:
        return provider
    from particles.llm.registry import BatchCompletionProvider

    if isinstance(provider, BatchCompletionProvider):
        return ReplayingBatchProvider(purpose, provider, scope)
    return ReplayingProvider(purpose, provider, scope)


def render_replay_line(stats: ReplayStats) -> str:
    """One line for the end of a run: what the replay scope did."""
    if stats.mode is ReplayMode.OFF:
        return "LLM replay: off."
    return (
        f"LLM replay ({stats.mode.value}, {stats.directory}): {stats.hits} served from "
        f"the cache, {stats.misses} sent to the provider, {stats.recorded} recorded."
    )


# ---------------------------------------------------------------------------
# Keys and entries
# ---------------------------------------------------------------------------


def _normalise(texts: Sequence[str | None]) -> list[str | None]:
    """Replace every fence nonce in ``texts`` with one placeholder.

    The nonces are collected across all the texts first and each is replaced
    wherever it appears, so the data-fence instruction in the system turn
    (which names the nonce outside a ``nonce="…"`` attribute too) normalises
    with the fences in the user turn.
    """
    nonces = {m for text in texts if text for m in _NONCE_RE.findall(text)}
    out: list[str | None] = []
    for text in texts:
        if text is not None:
            for nonce in nonces:
                text = text.replace(nonce, _NONCE_PLACEHOLDER)
        out.append(text)
    return out


def request_key(
    *,
    purpose: str,
    provider_model: str,
    prompt: str,
    system: str | None,
    cache_prefix: str | None,
    images: Sequence[VisionImage] | None,
    response_schema: dict[str, Any] | None,
    max_tokens: int,
    temperature: float | None,
    opts: dict[str, object] | None = None,
) -> str:
    """The content address of one completion request (hex SHA-256)."""
    prompt_n, system_n, prefix_n = _normalise([prompt, system, cache_prefix])
    material = {
        "format": _FORMAT,
        "purpose": purpose,
        "provider_model": provider_model,
        "prompt": prompt_n,
        "system": system_n,
        "cache_prefix": prefix_n,
        "images": [[img.media_type, hashlib.sha256(img.data).hexdigest()] for img in images or ()],
        "response_schema": response_schema,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "opts": opts or {},
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    hit_max_tokens: bool = False
    batched: bool = False


def _entry_path(directory: Path, key: str) -> Path:
    return directory / key[:2] / f"{key}.json"


def _read_entry(directory: Path, key: str) -> tuple[str, _Usage] | None:
    """The recorded reply and usage for ``key``, or ``None`` (absent or unreadable)."""
    path = _entry_path(directory, key)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        log.warning("Ignoring unreadable replay entry %s: %s", path, exc)
        return None
    if not isinstance(raw, dict) or raw.get("format") != _FORMAT:
        return None
    text = raw.get("text")
    usage = raw.get("usage")
    if not isinstance(text, str) or not isinstance(usage, dict):
        return None
    return text, _Usage(
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        cache_creation_tokens=int(usage.get("cache_creation_tokens", 0)),
        cache_read_tokens=int(usage.get("cache_read_tokens", 0)),
        hit_max_tokens=bool(usage.get("hit_max_tokens", False)),
        batched=bool(usage.get("batched", False)),
    )


def _write_entry(
    directory: Path, key: str, *, purpose: str, provider_model: str, text: str, usage: _Usage
) -> None:
    """Write one entry atomically (temp file + rename), so a reader never sees half."""
    path = _entry_path(directory, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "format": _FORMAT,
        "purpose": purpose,
        "provider_model": provider_model,
        "text": text,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation_tokens": usage.cache_creation_tokens,
            "cache_read_tokens": usage.cache_read_tokens,
            "hit_max_tokens": usage.hit_max_tokens,
            "batched": usage.batched,
        },
    }
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(body, fh)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _summed(rows: Sequence[UsageRow]) -> _Usage:
    """One usage figure from a nested accumulator's rows."""
    return _Usage(
        input_tokens=sum(r.input_tokens for r in rows),
        output_tokens=sum(r.output_tokens for r in rows),
        cache_creation_tokens=sum(r.cache_creation_tokens for r in rows),
        cache_read_tokens=sum(r.cache_read_tokens for r in rows),
        hit_max_tokens=any(r.max_tokens_stops for r in rows),
        batched=any(r.batched for r in rows),
    )


def _shares(total: _Usage, n: int, stops: int = 0) -> list[_Usage]:
    """Split a batch's usage into ``n`` even shares whose sum is ``total``.

    ``stops`` is the batch's count of replies cut off at ``max_tokens``. Which
    replies they were is not in the usage, so the flag goes on the first
    ``stops`` shares: the replayed run's ``max_tokens_stops`` then matches the
    recorded run's, though not reply by reply.
    """

    def split(value: int) -> list[int]:
        base, extra = divmod(value, n)
        return [base + (1 if i < extra else 0) for i in range(n)]

    columns = [
        split(total.input_tokens),
        split(total.output_tokens),
        split(total.cache_creation_tokens),
        split(total.cache_read_tokens),
    ]
    return [
        _Usage(
            input_tokens=columns[0][i],
            output_tokens=columns[1][i],
            cache_creation_tokens=columns[2][i],
            cache_read_tokens=columns[3][i],
            hit_max_tokens=i < stops,
            batched=total.batched,
        )
        for i in range(n)
    ]


def _report_replayed(provider_model: str, usage: _Usage) -> None:
    record_usage(
        provider_model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_creation_tokens=usage.cache_creation_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        hit_max_tokens=usage.hit_max_tokens,
        batched=usage.batched,
        replayed=True,
    )


# ---------------------------------------------------------------------------
# The wrapping providers
# ---------------------------------------------------------------------------


class ReplayingProvider:
    """A :class:`~particles.llm.registry.CompletionProvider` behind the replay cache."""

    def __init__(self, purpose: str, inner: CompletionProvider, scope: _Scope) -> None:
        self.purpose = purpose
        self.inner = inner
        self._scope = scope

    @property
    def provider_model(self) -> str:
        """The wrapped provider's pairing: a replay is the same model's reply."""
        return self.inner.provider_model

    def _key(
        self,
        prompt: str,
        *,
        system: str | None,
        cache_prefix: str | None,
        images: Sequence[VisionImage] | None,
        response_schema: dict[str, Any] | None,
        max_tokens: int,
        temperature: float | None,
        opts: dict[str, object] | None = None,
    ) -> str:
        return request_key(
            purpose=self.purpose,
            provider_model=self.provider_model,
            prompt=prompt,
            system=system,
            cache_prefix=cache_prefix,
            images=images,
            response_schema=response_schema,
            max_tokens=max_tokens,
            temperature=temperature,
            opts=opts,
        )

    async def _lookup(self, key: str) -> str | None:
        """The recorded reply for ``key`` under ``replay`` (reported as replayed)."""
        if self._scope.mode is not ReplayMode.REPLAY:
            return None
        entry = await asyncio.to_thread(_read_entry, self._scope.directory, key)
        if entry is None:
            return None
        text, usage = entry
        _report_replayed(self.provider_model, usage)
        self._scope.stats._bump(hits=1)
        return text

    async def _store(self, key: str, text: str, usage: _Usage) -> None:
        try:
            await asyncio.to_thread(
                _write_entry,
                self._scope.directory,
                key,
                purpose=self.purpose,
                provider_model=self.provider_model,
                text=text,
                usage=usage,
            )
        except OSError as exc:
            # The cache is a cost lever, never a capability: a full disk loses
            # the saving on the next run, not this run's answer.
            log.warning("Could not record a replay entry under %s: %s", self._scope.directory, exc)
            return
        self._scope.stats._bump(recorded=1)

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
        """Return the recorded reply on a hit; otherwise call, record, and return."""
        key = self._key(
            prompt,
            system=system,
            cache_prefix=cache_prefix,
            images=images,
            response_schema=response_schema,
            max_tokens=max_tokens,
            temperature=temperature,
            opts=dict(opts),
        )
        cached = await self._lookup(key)
        if cached is not None:
            return cached
        self._scope.stats._bump(misses=1)
        # A nested usage scope measures this one call; the enclosing scopes
        # still count it as usual.
        with track_usage() as measured:
            text = await self.inner.complete(
                prompt,
                max_tokens=max_tokens,
                system=system,
                temperature=temperature,
                images=images,
                response_schema=response_schema,
                cache_prefix=cache_prefix,
                **opts,
            )
        await self._store(key, text, _summed(measured.snapshot().rows))
        return text


class ReplayingBatchProvider(ReplayingProvider):
    """A :class:`ReplayingProvider` over a batch-capable provider.

    Serves the recorded requests of a set from the cache and sends only the
    rest on: as a batch when they still clear ``llm.batch.min_requests``, one
    ``complete()`` at a time otherwise (where each reply's usage is exact).
    """

    async def complete_many(
        self,
        requests: Sequence[CompletionRequest],
        *,
        max_tokens: int,
        temperature: float | None = None,
        response_schema: dict[str, Any] | None = None,
        failures_out: list[RequestFailure | None] | None = None,
        **opts: object,
    ) -> list[str | None]:
        """Run ``requests`` with recorded replies filled in from the cache."""
        from particles.config import get_config
        from particles.llm.registry import BatchCompletionProvider, run_sequential

        keys = [
            self._key(
                r.prompt,
                system=r.system,
                cache_prefix=r.cache_prefix,
                images=None,
                response_schema=response_schema,
                max_tokens=max_tokens,
                temperature=temperature,
                opts=dict(opts),
            )
            for r in requests
        ]
        results: list[str | None] = [None] * len(requests)
        failures: list[RequestFailure | None] = [None] * len(requests)
        pending: list[int] = []
        for i, key in enumerate(keys):
            cached = await self._lookup(key)
            if cached is None:
                pending.append(i)
            else:
                results[i] = cached

        if pending:
            misses = [requests[i] for i in pending]
            inner = self.inner
            if len(misses) >= get_config().llm.batch.min_requests and isinstance(
                inner, BatchCompletionProvider
            ):
                reported: list[RequestFailure | None] = []
                with track_usage() as measured:
                    got = await inner.complete_many(
                        misses,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        response_schema=response_schema,
                        failures_out=reported,
                        **opts,
                    )
                self._scope.stats._bump(misses=len(misses))
                answered = [j for j, text in enumerate(got) if text is not None]
                rows = measured.snapshot().rows
                shares = (
                    _shares(
                        _summed(rows),
                        len(answered),
                        stops=sum(r.max_tokens_stops for r in rows),
                    )
                    if answered
                    else []
                )
                for share, j in zip(shares, answered, strict=True):
                    text = got[j]
                    assert text is not None
                    await self._store(keys[pending[j]], text, share)
                for j, i in enumerate(pending):
                    results[i] = got[j] if j < len(got) else None
                    failures[i] = reported[j] if j < len(reported) else None
            else:
                # ``self.complete`` measures and records each reply exactly; a
                # second lookup for a key that just missed is a cheap miss.
                got, seq_failures = await run_sequential(
                    self,
                    misses,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_schema=response_schema,
                )
                for j, i in enumerate(pending):
                    results[i] = got[j]
                    failures[i] = seq_failures[j]
        if failures_out is not None:
            failures_out.extend(failures)
        return results
