# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Chunked LLM extraction with content-hash carry-forward.

An extractor renders source content into one or more ``ChunkUnit``s — each
representing the exact text that would be sent to the LLM. The helper
hashes every chunk, looks up existing ACTIVE particles whose recorded
``chunk_hash`` matches (filtered by the same ``extractor_id`` and
``extractor_version``, and excluding any particle in the caller's
``supersede_ids`` set — reindex marks in-scope particles for replacement,
and a marked particle must not veto its own re-extraction), and either:

* **Cache hit:** skips the LLM call. The existing particles' IDs are
  recorded in ``ExtractionResult.carry_forward_ids`` so the reindex
  operation can exclude them from supersession. Nothing is re-pointed —
  the carry-forward particle still names the snapshot it was originally
  extracted from — and the Engine appends one ref naming the re-observing
  snapshot, as it does for a suppressed duplicate. This
  module stays store-free: it only reports the ids.

* **Cache miss:** calls ``_call_llm`` and stamps every resulting
  ``CandidateParticle`` with the chunk's hash so the next re-extraction
  can carry it forward.

The total number of LLM calls per source is bounded by ``max_llm_calls``;
chunks beyond the cap emit a ``CHUNK_TRUNCATION`` quality note.

This helper is the shared carry-forward mechanism. The gist and reddit
extractors both route their chunked-extraction work through it.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession

from particles.extraction.general import (
    CandidateParticle,
    ChunkOutcome,
    ExtractionResult,
    _build_llm_request,
    _call_llm,
    _pooled_extract,
)
from particles.extraction.tool_turns import TOOL_TURN_LABEL

if TYPE_CHECKING:
    from particles.llm import CompletionPool

log = logging.getLogger(__name__)

# Carry-forward is the one graph-aware step of chunked extraction: it consults
# the store to skip re-extracting chunks whose content hash already has ACTIVE
# particles. To keep this module on the Client layer (§4/§6), the store
# lookup is *injected*, not imported — the Engine registers the real
# implementation (``particle_store.get_active_particles_for_chunk_hash``) at
# import time via :func:`register_carry_forward_lookup`. When unregistered
# (pure Client, store-free), carry-forward is skipped — identical to the
# ``session is None`` path. This mirrors the inverted config↔db coupling.
CarryForwardLookup = Callable[..., Awaitable[list[Any]]]
_carry_forward_lookup: CarryForwardLookup | None = None

# A per-chunk LLM call: chunk text → (candidates, quality notes, transient-error).
# Defaults to the general extractor's ``_call_llm``; the journal extractor injects
# its own journal-prompt caller so the carry-forward machinery
# (chunk hashing, cache lookup, the call cap, transient aggregation) is reused
# verbatim while only the inner prompt changes.
ChunkLLMCall = Callable[[str], Awaitable[tuple[list[CandidateParticle], list[str], bool]]]


def register_carry_forward_lookup(lookup: CarryForwardLookup) -> None:
    """Register the Engine-side carry-forward store lookup."""
    global _carry_forward_lookup
    _carry_forward_lookup = lookup


@dataclass
class ChunkUnit:
    """One unit of LLM extraction work.

    Attributes:
        chunk_id: Human-readable identifier used in log lines and quality
            notes (e.g. ``"body"`` or ``"comments_3_of_5"``). Not stored
            anywhere persistent.
        chunk_text: Exact text the LLM would see. The SHA-256 of this
            string is the cache key.
        context: Already-extracted text shown before ``chunk_text`` on an
            append-only delta read, as its own labelled section
            that the model extracts nothing from. ``None`` on every other
            chunk. When set, the cache key covers it too.
    """

    chunk_id: str
    chunk_text: str
    context: str | None = None

    @property
    def prompt_text(self) -> str:
        """The exact text this chunk puts in front of the model, nonce aside.

        ``chunk_text`` alone for an ordinary chunk; the context section and
        the chunk, each under its label, for a delta chunk.
        A gist chunk is hashed with its header the same way.
        """
        if self.context is None:
            return self.chunk_text
        return f"[context]\n{self.context}\n\n[source]\n{self.chunk_text}"

    @property
    def prompt_hash(self) -> str:
        """SHA-256 of :attr:`prompt_text`: the chunk's carry-forward key and ``chunk_hash``."""
        return _hash_chunk(self.prompt_text)


def _own_carried(
    chunk_hash: str,
    resume_carried: Mapping[str, Sequence[str]] | None,
    supersede_ids: frozenset[str],
) -> list[str]:
    """The ids of this snapshot's own claims carrying ``chunk_hash``, if any.

    The fallback when the version-matched lookup found nothing: on a partial
    read's retry, a chunk an earlier attempt already wrote is carried forward
    whatever extractor version wrote it.
    """
    if not resume_carried:
        return []
    return [pid for pid in resume_carried.get(chunk_hash, ()) if pid not in supersede_ids]


def _hash_chunk(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk_has_tool_turns(chunk: ChunkUnit, tool_turns_present: bool) -> bool:
    """Whether this chunk's prompt needs the tool-turn rule.

    Gated on the caller's flag so a non-transcript source that merely quotes
    the label never gets the rule, then narrowed per chunk so a chunk with no
    relabelled turn keeps the no-tool-turn prompt byte-identical. A delta
    chunk's context counts, since the model reads it too.
    """
    return tool_turns_present and TOOL_TURN_LABEL in chunk.prompt_text


def _llm_kwargs(
    chunk: ChunkUnit, reference_published_at: datetime | None, tool_turns_present: bool
) -> dict[str, Any]:
    """The optional keywords one chunk's default ``_call_llm`` carries, each only when set.

    The reference anchor when the source exposes a publication instant,
    the tool-turn flag when this chunk carries a relabelled turn,
    the context section of a delta chunk. The
    common call stays byte-identical to the bare one, and no test double sees
    a keyword it did not ask for.
    """
    kwargs: dict[str, Any] = {}
    if reference_published_at is not None:
        kwargs["reference_published_at"] = reference_published_at
    if _chunk_has_tool_turns(chunk, tool_turns_present):
        kwargs["tool_turns_present"] = True
    if chunk.context is not None:
        kwargs["context"] = chunk.context
    return kwargs


async def extract_with_carry_forward(
    session: AsyncSession | None,
    chunks: list[ChunkUnit],
    corpus_entry_id: str | None,
    extractor_id: str,
    extractor_version: str,
    max_llm_calls: int | None = None,
    call_llm: ChunkLLMCall | None = None,
    reference_published_at: datetime | None = None,
    completion_pool: CompletionPool | None = None,
    supersede_ids: frozenset[str] = frozenset(),
    tool_turns_present: bool = False,
    resume_carried: Mapping[str, Sequence[str]] | None = None,
) -> ExtractionResult:
    """Run chunked LLM extraction with chunk-hash-based carry-forward.

    Args:
        session: Active async session — used for the carry-forward lookup.
        chunks: Ordered list of ``ChunkUnit``s. Order is preserved both for
            logging and — for the journal extractor — as
            whole-document order: candidates are appended chunk-by-chunk, so
            the returned list reflects source order across chunks.
        corpus_entry_id: The corpus entry being extracted. Used as part of
            the carry-forward lookup key.
        extractor_id: The current extractor's ``EXTRACTOR_ID``. Matched
            against existing particles' ``extractor_ref`` JSON.
        extractor_version: The current extractor's ``EXTRACTOR_VERSION``.
            A version mismatch is treated as a cache miss so that
            ``EXTRACTOR_VERSION`` bumps still force re-extraction.
        max_llm_calls: Optional cap on LLM calls (cache hits don't count
            toward it). Chunks beyond the cap emit a ``CHUNK_TRUNCATION``
            quality note and are skipped.
        call_llm: The per-chunk LLM call. Defaults to the general extractor's
            ``_call_llm`` (general prompt); the journal extractor injects its
            own journal-prompt caller so this machinery is reused
            with only the prompt swapped.
        completion_pool: When set (only the latency-tolerant
            consolidation extract pass passes one) and no ``call_llm`` is
            injected, cache-miss chunks are built up front and submitted as
            one pooled batch group instead of the sequential per-chunk loop —
            same lookups, same cap, same notes, same candidate order. An
            injected ``call_llm`` has no build/parse halves and keeps the
            sequential loop.
        supersede_ids: Particle IDs already marked for replacement by the
            caller (reindex threads its supersede set here). These are never
            eligible as carry-forward matches: a chunk whose only ACTIVE
            particles are in this set is a cache **miss** and is re-sent to
            the LLM. Without the exclusion, ``reindex --provider-model``
            is defeated by the cache — the chunk text and
            extractor version are unchanged by design, so every in-scope
            particle would cache-hit and keep its old model's output.
        tool_turns_present: The caller relabelled tool turns in the source.
            Each chunk that carries the label gets the
            tool-turn rule in its prompt, on both the sequential and the
            pooled path. Ignored for an injected ``call_llm``, which owns its
            own prompt.
        resume_carried: On the retry of a snapshot holding a partial read,
            the chunk hashes its own ACTIVE claims carry, each with those
            claims' ids. A chunk whose hash is here is carried
            forward even when the version match finds nothing: the claims came
            from an earlier attempt at this same snapshot, possibly under an
            older extractor version, and re-reading the chunk would write them
            a second time. Ids in ``supersede_ids`` never count.

    Returns:
        An ``ExtractionResult`` whose ``candidates`` are newly LLM-extracted
        CandidateParticles (each stamped with its chunk's hash) and whose
        ``carry_forward_ids`` are existing particle IDs eligible for
        carry-forward. ``chunk_outcomes`` holds one entry per chunk, in
        order, saying whether it was carried, answered, failed or capped;
        the caller fills in each chunk's start. Caller is
        responsible for the usual downstream processing (subject resolution,
        conflict resolution, persistence) on candidates and for honouring
        ``carry_forward_ids`` during supersession.
    """
    if completion_pool is not None and call_llm is None:
        # the chunk calls are a set, not a chain — the per-chunk
        # prompt is built from the chunk text alone — so the whole cache-miss
        # request set is enumerable up front and rides one pooled batch group.
        return await _extract_chunks_pooled(
            session=session,
            chunks=chunks,
            corpus_entry_id=corpus_entry_id,
            extractor_id=extractor_id,
            extractor_version=extractor_version,
            max_llm_calls=max_llm_calls,
            reference_published_at=reference_published_at,
            completion_pool=completion_pool,
            supersede_ids=supersede_ids,
            tool_turns_present=tool_turns_present,
            resume_carried=resume_carried,
        )

    outcomes: list[ChunkOutcome] = []
    new_candidates: list[CandidateParticle] = []
    carry_forward_ids: list[str] = []
    unread: list[str] = []
    notes: list[str] = []
    transient_errors = 0
    llm_calls_made = 0
    for chunk in chunks:
        h = chunk.prompt_hash
        # Skip the cache lookup when carry-forward is unregistered (pure Client,
        # store-free), when we have no session (e.g. unit tests that drive the
        # helper directly), or no corpus entry yet (importer-time extraction
        # before the entry is persisted).
        existing: list[Any] = []
        if (
            _carry_forward_lookup is not None
            and session is not None
            and corpus_entry_id is not None
        ):
            existing = await _carry_forward_lookup(
                session,
                corpus_entry_id=corpus_entry_id,
                chunk_hash=h,
                extractor_id=extractor_id,
                extractor_version=extractor_version,
            )
            # Particles the caller is replacing (the reindex supersede set)
            # must not satisfy the cache — see the supersede_ids docstring.
            existing = [p for p in existing if p.id not in supersede_ids]
        carried = [p.id for p in existing] or _own_carried(h, resume_carried, supersede_ids)
        if carried:
            log.info(
                "Carry-forward: chunk %s (hash %s…) reuses %d existing particle(s)",
                chunk.chunk_id,
                h[:8],
                len(carried),
            )
            carry_forward_ids.extend(carried)
            notes.append(
                f"CHUNK_CARRY_FORWARD: {chunk.chunk_id} "
                f"reused {len(carried)} particles (hash {h[:8]}…)"
            )
            outcomes.append(ChunkOutcome(chunk.chunk_id, h, "carried"))
            continue

        if max_llm_calls is not None and llm_calls_made >= max_llm_calls:
            notes.append(
                f"CHUNK_TRUNCATION: {chunk.chunk_id} skipped (LLM call cap {max_llm_calls} reached)"
            )
            unread.append(chunk.chunk_id)
            outcomes.append(ChunkOutcome(chunk.chunk_id, h, "capped"))
            continue

        if call_llm is not None:
            # An injected caller (the journal extractor) owns its own
            # prompt and takes no anchor or tool-turn flag.
            candidates, chunk_notes, transient = await call_llm(chunk.chunk_text)
        else:
            # ``_call_llm`` is looked up here, at call time, so a patched
            # ``incremental._call_llm`` is the one called.
            candidates, chunk_notes, transient = await _call_llm(
                chunk.chunk_text,
                **_llm_kwargs(chunk, reference_published_at, tool_turns_present),
            )
        llm_calls_made += 1
        if transient:
            transient_errors += 1
        outcomes.append(ChunkOutcome(chunk.chunk_id, h, "failed" if transient else "answered"))
        for c in candidates:
            c.chunk_hash = h
        new_candidates.extend(candidates)
        if chunk_notes:
            notes.extend(f"{chunk.chunk_id}: {n}" for n in chunk_notes)
        log.info(
            "Chunk %s (hash %s…): %d new particles from LLM (%d/%s calls used)",
            chunk.chunk_id,
            h[:8],
            len(candidates),
            llm_calls_made,
            max_llm_calls if max_llm_calls is not None else "∞",
        )

    return ExtractionResult(
        candidates=new_candidates,
        quality_notes=notes,
        carry_forward_ids=carry_forward_ids,
        transient_error_count=transient_errors,
        answered_calls=llm_calls_made - transient_errors,
        unread_chunk_ids=unread,
        chunk_outcomes=outcomes,
    )


async def _extract_chunks_pooled(
    *,
    session: AsyncSession | None,
    chunks: list[ChunkUnit],
    corpus_entry_id: str | None,
    extractor_id: str,
    extractor_version: str,
    max_llm_calls: int | None,
    reference_published_at: datetime | None,
    completion_pool: CompletionPool,
    supersede_ids: frozenset[str] = frozenset(),
    tool_turns_present: bool = False,
    resume_carried: Mapping[str, Sequence[str]] | None = None,
) -> ExtractionResult:
    """Pooled-batch twin of the sequential chunk loop.

    Phase 1 runs the carry-forward lookups for every chunk in order — the
    same guard, the same cap accounting (cache hits don't count; chunks past
    the cap get ``CHUNK_TRUNCATION``) — and builds the cache-miss requests.
    Phase 2 submits them as one pooled group and parses per chunk, stamping
    ``chunk_hash`` exactly as the loop does. Candidates and quality notes
    keep chunk order, so the output is ordered as the sequential path's.
    """
    carry_forward_ids: list[str] = []
    unread: list[str] = []
    notes_by_chunk: list[list[str]] = []
    outcomes: list[ChunkOutcome] = []
    pending: list[tuple[int, ChunkUnit, str]] = []
    llm_calls_planned = 0

    for chunk in chunks:
        h = chunk.prompt_hash
        chunk_notes_slot: list[str] = []
        notes_by_chunk.append(chunk_notes_slot)
        # Same lookup guard as the sequential loop: skip when carry-forward is
        # unregistered (pure Client), sessionless, or pre-entry (importer-time).
        existing: list[Any] = []
        if (
            _carry_forward_lookup is not None
            and session is not None
            and corpus_entry_id is not None
        ):
            existing = await _carry_forward_lookup(
                session,
                corpus_entry_id=corpus_entry_id,
                chunk_hash=h,
                extractor_id=extractor_id,
                extractor_version=extractor_version,
            )
            # Same supersede-set exclusion as the sequential loop.
            existing = [p for p in existing if p.id not in supersede_ids]
        carried = [p.id for p in existing] or _own_carried(h, resume_carried, supersede_ids)
        if carried:
            log.info(
                "Carry-forward: chunk %s (hash %s…) reuses %d existing particle(s)",
                chunk.chunk_id,
                h[:8],
                len(carried),
            )
            carry_forward_ids.extend(carried)
            chunk_notes_slot.append(
                f"CHUNK_CARRY_FORWARD: {chunk.chunk_id} "
                f"reused {len(carried)} particles (hash {h[:8]}…)"
            )
            outcomes.append(ChunkOutcome(chunk.chunk_id, h, "carried"))
            continue

        if max_llm_calls is not None and llm_calls_planned >= max_llm_calls:
            chunk_notes_slot.append(
                f"CHUNK_TRUNCATION: {chunk.chunk_id} skipped (LLM call cap {max_llm_calls} reached)"
            )
            unread.append(chunk.chunk_id)
            outcomes.append(ChunkOutcome(chunk.chunk_id, h, "capped"))
            continue

        outcomes.append(ChunkOutcome(chunk.chunk_id, h, "answered"))
        pending.append((len(outcomes) - 1, chunk, h))
        llm_calls_planned += 1

    planned = [
        _build_llm_request(
            chunk.chunk_text,
            reference_published_at=reference_published_at,
            tool_turns_present=_chunk_has_tool_turns(chunk, tool_turns_present),
            context=chunk.context,
        )
        for _, chunk, _ in pending
    ]
    settled = await _pooled_extract(completion_pool, planned)

    new_candidates: list[CandidateParticle] = []
    transient_errors = 0
    for (slot, chunk, h), (candidates, chunk_notes, transient) in zip(
        pending, settled, strict=True
    ):
        if transient:
            transient_errors += 1
            outcomes[slot].status = "failed"
        for c in candidates:
            c.chunk_hash = h
        new_candidates.extend(candidates)
        if chunk_notes:
            notes_by_chunk[slot].extend(f"{chunk.chunk_id}: {n}" for n in chunk_notes)
        log.info(
            "Chunk %s (hash %s…): %d new particles from pooled batch (%d planned call(s))",
            chunk.chunk_id,
            h[:8],
            len(candidates),
            llm_calls_planned,
        )

    return ExtractionResult(
        candidates=new_candidates,
        quality_notes=[note for slot in notes_by_chunk for note in slot],
        carry_forward_ids=carry_forward_ids,
        transient_error_count=transient_errors,
        answered_calls=len(pending) - transient_errors,
        unread_chunk_ids=unread,
        chunk_outcomes=outcomes,
    )
