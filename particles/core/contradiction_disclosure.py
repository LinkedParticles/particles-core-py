# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Pure decisions of the nightly contradiction disclosure.

A contradiction the nightly census's second reading confirms, between claims
from two different sources, is disclosed to the agent as one ``INCONSISTENCY``
record per **disagreement**: the confirmed pairs that share a claim, split
into two sides. This module holds every decision that needs no store, so each
is testable on plain values (D2):

* the record's properties, and its constructor
  (:func:`build_census_record`, the sibling of the retired-value
  branch of :func:`particles.core.conflict_resolution.build_inconsistency_particle`);
* grouping kept pairs by shared claim, and splitting a group into two sides
  (:func:`group_pairs`, :func:`split_sides`);
* which pairs a record already covers (:func:`covered_pairs`);
* what the lapse sweep does to an open record (:func:`decide_lapse`);
* which groups a run opens under its cap (:func:`select_under_cap`).

The I/O shell that gathers the inputs and applies the writes is
``particles.operations.contradiction_disclosure``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from particles.core.schema import (
    SCHEMA_VERSION,
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status

#: ``properties`` marker naming the record's origin; its value is :data:`CENSUS_ORIGIN`.
ORIGIN_KEY = "conflict:origin"
#: The origin value of a record the nightly census opened.
CENSUS_ORIGIN = "census"
#: ``{"a": [ids], "b": [ids]}``: every member, by side.
SIDES_KEY = "conflict:sides"
#: ``{member id: [corpus entry ids]}``: the SOURCE entries that stated each
#: member when the record was opened. The fingerprint a judged record's
#: coverage is checked against.
SOURCES_KEY = "conflict:sources"
#: On a replacement only: the id of the record it replaces.
REPLACES_KEY = "conflict:replaces"
#: ``[{"a", "b", "same_source", "reason", "reading"}, …]``: the confirmed pairs behind the
#: record. Sides alone cannot say which members were read against which, and a
#: regroup or the unsided fallback must reuse only pairs a second reading
#: actually confirmed.
PAIRS_KEY = "conflict:pairs"

#: On a record extraction opened: the instruction the second reading that
#: confirmed its pair ran under (:data:`READING_STANDING` or
#: :data:`READING_OBSERVED`). The per-record counterpart of the
#: ``reading`` a census record keeps per pair. A record without it was opened
#: without a confirming reading, and pass 3b reads it.
READING_KEY = "conflict:reading"

#: The author of every census record and of the events its lapse sweep writes.
DISCLOSURE_ACTOR = "memory-consolidate"

#: The second-reading instruction for two claims from standing sources (memory
#: notes, other notes kept current): two standing facts that disagree are a
#: contradiction whatever their dates. A pair recorded before
#: readings were named ran under it.
READING_STANDING = "standing/1"
#: The instruction for a pair with a claim from a time-stamped record (a
#: session transcript, another append-only log): a value observed at a later
#: time may be a change, and only the same moment or a fixed fact confirms
#:.
READING_OBSERVED = "observed/1"

_EXCERPT_CHARS = 120


@dataclass(frozen=True)
class ConfirmedPair:
    """A contradiction the second reading confirmed.

    ``a`` and ``b`` are in probe order, not date order: the record orders its
    representative pair by source date when it is built.
    """

    a: str
    b: str
    same_source: bool
    #: The second reading's reason.
    reason: str
    #: The instruction the confirming reading ran under (:data:`READING_STANDING`
    #: or :data:`READING_OBSERVED`). A pair whose instruction has since changed is
    #: read again before its record is trusted.
    reading: str = READING_STANDING

    @property
    def key(self) -> frozenset[str]:
        return frozenset((self.a, self.b))

    def to_payload(self) -> dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "same_source": self.same_source,
            "reason": self.reason,
            "reading": self.reading,
        }

    @classmethod
    def from_payload(cls, raw: Mapping[str, Any]) -> ConfirmedPair | None:
        a, b = raw.get("a"), raw.get("b")
        if not isinstance(a, str) or not isinstance(b, str) or a == b:
            return None
        return cls(
            a=a,
            b=b,
            same_source=bool(raw.get("same_source", False)),
            reason=str(raw.get("reason") or ""),
            reading=str(raw.get("reading") or READING_STANDING),
        )


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sides:
    """A disagreement split into two sides: every confirmed pair crosses them."""

    a: tuple[str, ...]
    b: tuple[str, ...]

    @property
    def members(self) -> tuple[str, ...]:
        return self.a + self.b

    def side_of(self, particle_id: str) -> str | None:
        if particle_id in self.a:
            return "a"
        if particle_id in self.b:
            return "b"
        return None

    def crossing_pairs(self) -> frozenset[frozenset[str]]:
        """Every pair with one member on each side: what the record covers while open."""
        return frozenset(frozenset((x, y)) for x in self.a for y in self.b)


def is_census_record(particle: Particle) -> bool:
    """Whether ``particle`` is an INCONSISTENCY record the nightly census opened."""
    return bool(particle.properties and particle.properties.get(ORIGIN_KEY) == CENSUS_ORIGIN)


def census_sides(record: Particle) -> Sides | None:
    """The record's sides, or ``None`` for a record that is not a readable census record."""
    if not is_census_record(record):
        return None
    raw = (record.properties or {}).get(SIDES_KEY)
    if not isinstance(raw, dict):
        return None
    a = [str(x) for x in raw.get("a") or () if isinstance(x, str)]
    b = [str(x) for x in raw.get("b") or () if isinstance(x, str)]
    if not a or not b:
        return None
    return Sides(a=tuple(a), b=tuple(b))


def census_sources(record: Particle) -> dict[str, frozenset[str]]:
    """The record's ``conflict:sources`` fingerprint, by member id."""
    raw = (record.properties or {}).get(SOURCES_KEY)
    if not isinstance(raw, dict):
        return {}
    return {
        str(pid): frozenset(str(e) for e in entries if isinstance(e, str))
        for pid, entries in raw.items()
        if isinstance(entries, list)
    }


def census_pairs(record: Particle) -> list[ConfirmedPair]:
    """The confirmed pairs a census record was opened from."""
    raw = (record.properties or {}).get(PAIRS_KEY)
    if not isinstance(raw, list):
        return []
    return [p for p in (ConfirmedPair.from_payload(r) for r in raw if isinstance(r, dict)) if p]


def reading_stamp(record: Particle) -> str | None:
    """The instruction a second reading confirmed an extract-time record under, if any."""
    value = (record.properties or {}).get(READING_KEY)
    return value if isinstance(value, str) and value else None


def census_replaces(record: Particle) -> str | None:
    """The id of the record this one replaced, when it is a replacement."""
    raw = (record.properties or {}).get(REPLACES_KEY)
    return raw if isinstance(raw, str) and raw else None


@dataclass(frozen=True)
class NoteLabel:
    """What the record's content names for one member: its note and that note's date."""

    name: str
    date: str


def build_census_record(
    *,
    sides: Sides,
    pairs: Sequence[ConfirmedPair],
    members: Mapping[str, Particle],
    labels: Mapping[str, NoteLabel],
    sources: Mapping[str, Iterable[str]],
    trigger_entry_id: str,
    trigger_snapshot_id: str | None,
    trigger_ref_type: ProvenanceRefType = ProvenanceRefType.SOURCE,
    replaces: str | None = None,
) -> Particle:
    """Construct a census ``INCONSISTENCY`` record.

    ``sides.a[0]`` and ``sides.b[0]`` are the representative pair: the first
    two PARTICLE refs, which review and the trust cascade read as A and B.
    The trigger ref follows, then every further member as a PARTICLE ref, so
    ``get_inconsistency_backrefs`` flags each member with this record's id.
    The trigger ref is ``SOURCE``-typed unless B has no corpus provenance (an
    agent assertion), when the caller passes ``PARTICLE`` and B's id, as the
    rung 3 builder does for a derived candidate. The second reading's reason
    shown is that of the pair joining the representative claims, else the
    first pair's.

    The field choices follow the rung 3 record: confidence is the weakest
    member's, the nature ``EPISTEMIC``, the subjects those of A (else B).
    Pure; the caller validates the ``None → INCONSISTENCY`` transition and
    persists the result.
    """
    rep_a, rep_b = members[sides.a[0]], members[sides.b[0]]
    rep_key = frozenset((rep_a.id, rep_b.id))
    read = next((p for p in pairs if p.key == rep_key), pairs[0] if pairs else None)
    reason = read.reason if read is not None else ""
    if read is not None and read.a in sides.b:
        # The reading names its claims A and B in the order it read them; the
        # record orders its sides by date, so say which is which.
        reason = f"(the reading's A is Side B's claim) {reason}"

    def line(side: Sequence[str]) -> str:
        parts = []
        for pid in side:
            label = labels.get(pid, NoteLabel("unknown", "unknown"))
            excerpt = members[pid].content[:_EXCERPT_CHARS]
            parts.append(f"{pid} — {excerpt} ({label.name}, {label.date})")
        return "; ".join(parts)

    content = (
        "INCONSISTENCY: two sources disagree (nightly check; a second reading confirmed it).\n"
        f"Side A: {line(sides.a)}\n"
        f"Side B: {line(sides.b)}\n"
        f"Second reading: {reason or 'no reason recorded'}"
    )
    extras = [pid for pid in sides.members if pid not in (rep_a.id, rep_b.id)]
    provenance = [
        ProvenanceRef(
            type=ProvenanceRefType.PARTICLE, corpus_entry_id=rep_a.id, snapshot_id=rep_a.id
        ),
        ProvenanceRef(
            type=ProvenanceRefType.PARTICLE, corpus_entry_id=rep_b.id, snapshot_id=rep_b.id
        ),
        ProvenanceRef(
            type=trigger_ref_type,
            corpus_entry_id=trigger_entry_id,
            snapshot_id=trigger_snapshot_id,
        ),
        *(
            ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=pid, snapshot_id=pid)
            for pid in extras
        ),
    ]
    properties: dict[str, Any] = {
        ORIGIN_KEY: CENSUS_ORIGIN,
        SIDES_KEY: {"a": list(sides.a), "b": list(sides.b)},
        SOURCES_KEY: {pid: sorted(set(sources.get(pid, ()))) for pid in sides.members},
        PAIRS_KEY: [p.to_payload() for p in pairs],
    }
    if replaces is not None:
        properties[REPLACES_KEY] = replaces
    return Particle(
        content=content,
        confidence=Confidence(
            value=min(members[pid].confidence.value for pid in sides.members),
            calibration_source=CalibrationSource.EXTRACTOR_DIRECT,
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=provenance,
        asserted_by=DISCLOSURE_ACTOR,
        status=Status.INCONSISTENCY,
        subject_ids=list(rep_a.subject_ids or rep_b.subject_ids),
        properties=properties,
        schema_version=SCHEMA_VERSION,
    )


# ---------------------------------------------------------------------------
# Grouping and sides
# ---------------------------------------------------------------------------


def group_pairs(pairs: Sequence[ConfirmedPair]) -> list[list[ConfirmedPair]]:
    """Group pairs connected through a shared claim, in order of first appearance.

    The same union-find the headline counts with
    (``count_disagreements``); here it returns the groups themselves.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        parent[find(p.a)] = find(p.b)
    groups: dict[str, list[ConfirmedPair]] = {}
    for p in pairs:
        groups.setdefault(find(p.a), []).append(p)
    return list(groups.values())


def split_sides(edges: Sequence[tuple[str, str]], representative: tuple[str, str]) -> Sides | None:
    """Two-colour a group's pair graph so every pair crosses the sides.

    ``representative`` fixes the colouring: its first claim is on side a, its
    second on side b. Returns ``None`` when the graph cannot be two-coloured
    (an odd cycle: three claims that each contradict the other two), the
    ``unsided`` case, which falls back to one record per pair. Member order
    within a side is the order claims first appear in ``edges``, with the
    representative first.
    """
    adjacency: dict[str, list[str]] = {}
    order: list[str] = []
    for x, y in edges:
        for node in (x, y):
            if node not in adjacency:
                adjacency[node] = []
                order.append(node)
        adjacency[x].append(y)
        adjacency[y].append(x)
    rep_a, rep_b = representative
    colour: dict[str, int] = {rep_a: 0}
    stack = [rep_a]
    while stack:
        node = stack.pop()
        for other in adjacency.get(node, ()):
            want = 1 - colour[node]
            if other not in colour:
                colour[other] = want
                stack.append(other)
            elif colour[other] != want:
                return None
    if colour.get(rep_b) != 1 or len(colour) != len(adjacency):
        return None
    side_a = [rep_a] + [n for n in order if colour[n] == 0 and n != rep_a]
    side_b = [rep_b] + [n for n in order if colour[n] == 1 and n != rep_b]
    return Sides(a=tuple(side_a), b=tuple(side_b))


@dataclass(frozen=True)
class PlannedRecord:
    """One record to open: its sides and the confirmed pairs behind it."""

    sides: Sides
    pairs: tuple[ConfirmedPair, ...]


def plan_group(
    group: Sequence[ConfirmedPair], orient: Callable[[str, str], tuple[str, str]]
) -> tuple[list[PlannedRecord], bool]:
    """The records one disagreement group becomes, and whether it was unsided.

    ``orient(x, y)`` returns the pair as ``(A, B)``: the caller orders a pair
    by source date, older first, which needs the store. The group's first pair
    is the representative. A two-coloured group becomes one record; an
    unsided group falls back to one record per confirmed pair.
    """
    first = group[0]
    rep = orient(first.a, first.b)
    sides = split_sides([(p.a, p.b) for p in group], rep)
    if sides is not None:
        return [PlannedRecord(sides=sides, pairs=tuple(group))], False
    planned = []
    for p in group:
        a, b = orient(p.a, p.b)
        planned.append(PlannedRecord(sides=Sides(a=(a,), b=(b,)), pairs=(p,)))
    return planned, True


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


class CloseCause(StrEnum):
    """How a closed census record was closed (table)."""

    REVIEW = "review"
    """A review action: a ``REVIEW_RESOLVED`` event names the record."""
    CASCADE = "cascade"
    """The trust cascade: ``PROVENANCE_STALE``, with no event."""
    LAPSED = "lapsed"
    """The lapse sweep: a side has no member left."""
    REGROUPED = "regrouped"
    """The lapse sweep or a merge on growth: a replacement names the record."""
    WITHDRAWN = "withdrawn"
    """A re-reading under a revised instruction no longer confirms any of its
    pairs. Mechanical, like a lapse: nothing covers the pairs after."""

    @property
    def judgment(self) -> bool:
        return self in (CloseCause.REVIEW, CloseCause.CASCADE)


@dataclass(frozen=True)
class RecordState:
    """One census record as coverage reads it."""

    record_id: str
    sides: Sides
    open: bool
    #: ``None`` while open.
    cause: CloseCause | None = None
    fingerprint: Mapping[str, frozenset[str]] = field(default_factory=dict)


def covered_pairs(
    records: Iterable[RecordState], current_sources: Mapping[str, frozenset[str]]
) -> frozenset[frozenset[str]]:
    """The pairs no census probe or mint should touch again.

    A pair is covered when its claims sit on **opposite sides** of one record
    that is open, or closed by a judgment while neither claim has gained a
    stating entry absent from the record's fingerprint. Two claims on one
    side are never covered: nothing has checked them against each other. A
    mechanically closed record covers nothing.
    """
    covered: set[frozenset[str]] = set()
    for record in records:
        if record.open:
            covered |= record.sides.crossing_pairs()
            continue
        if record.cause is None or not record.cause.judgment:
            continue
        for pair in record.sides.crossing_pairs():
            if all(
                current_sources.get(pid, frozenset()) <= record.fingerprint.get(pid, frozenset())
                for pid in pair
            ):
                covered.add(pair)
    return frozenset(covered)


# ---------------------------------------------------------------------------
# Lapse
# ---------------------------------------------------------------------------


class MemberState(StrEnum):
    """What the lapse sweep found for one member of an open record."""

    LIVE = "live"
    """``ACTIVE`` and stated by a current source."""
    LAPSED = "lapsed"
    """``ACTIVE``, but no current source states it: still a belief the digest shows."""
    GONE = "gone"
    """No longer ``ACTIVE``: no recall surface shows it."""
    MERGED = "merged"
    """Folded into an exact twin by the auto-merge; the survivor takes its place."""


class LapseAction(StrEnum):
    KEEP = "keep"
    CLOSE = "close"
    """No confirmed pair has both claims left: the disagreement is gone."""
    REGROUP = "regroup"
    """Close and replace with records for the pairs that remain."""


@dataclass(frozen=True)
class LapseDecision:
    action: LapseAction
    #: The confirmed pairs a ``REGROUP`` replacement is built from, with every
    #: merged member already swapped for its survivor.
    remaining: tuple[ConfirmedPair, ...] = ()
    #: Members that dropped out, for the event.
    dropped: tuple[str, ...] = ()
    #: Pairs a re-reading no longer confirms, for the event.
    withdrawn: tuple[ConfirmedPair, ...] = ()
    #: How a ``CLOSE`` is recorded: ``WITHDRAWN`` when a re-reading emptied the
    #: record, else ``LAPSED``.
    cause: CloseCause = CloseCause.LAPSED


def decide_lapse(
    pairs: Sequence[ConfirmedPair],
    states: Mapping[str, MemberState],
    survivors: Mapping[str, str] | None = None,
    rereads: Mapping[frozenset[str], ConfirmedPair | None] | None = None,
) -> LapseDecision:
    """Decide what the lapse sweep does to one open census record.

    Works from the record's confirmed pairs. ``rereads`` holds this run's
    second readings of pairs whose instruction has changed since they were
    confirmed, by pair key: ``None`` for a pair the new reading
    rejected, which is withdrawn, else the pair as the new reading confirmed
    it. A pair survives when it is not withdrawn and both its claims are live,
    a merged claim counting as its survivor. Then:

    * no pair survives: the disagreement is gone, ``CLOSE``, as ``withdrawn``
      when a re-reading withdrew a pair and ``lapsed`` otherwise;
    * a member is still ``ACTIVE`` but lapsed, a member was merged, a live
      member has lost every confirmed partner, or a pair was re-read:
      ``REGROUP`` from the surviving pairs, so no flag stays on a belief no
      source states, the survivor of a merge is flagged in its place, no flag
      stays on a belief with no disagreement left behind it, and a replacement
      records the new reading, so the pair is not read again the next night;
    * otherwise (only members that left ``ACTIVE`` dropped out, and every live
      member keeps a partner): ``KEEP``. No reader sees the dropped member,
      and closing would only churn the id.

    A member with no recorded state reads as live.
    """
    survivors = survivors or {}

    def state(pid: str) -> MemberState:
        return states.get(pid, MemberState.LIVE)

    def mapped(pid: str) -> str | None:
        match state(pid):
            case MemberState.LIVE:
                return pid
            case MemberState.MERGED:
                return survivors.get(pid)
            case _:
                return None

    rereads = rereads or {}
    remaining: dict[frozenset[str], ConfirmedPair] = {}
    withdrawn: list[ConfirmedPair] = []
    reread = False
    for p in pairs:
        if p.key in rereads:
            fresh = rereads[p.key]
            if fresh is None:
                withdrawn.append(p)
                continue
            reread = True
            p = replace(p, reason=fresh.reason, reading=fresh.reading)
        a, b = mapped(p.a), mapped(p.b)
        if a is None or b is None or a == b:
            continue
        pair = replace(p, a=a, b=b)
        remaining.setdefault(pair.key, pair)
    members = {pid for p in pairs for pid in (p.a, p.b)}
    dropped = tuple(sorted(pid for pid in members if state(pid) is not MemberState.LIVE))
    if not remaining:
        return LapseDecision(
            action=LapseAction.CLOSE,
            dropped=dropped,
            withdrawn=tuple(withdrawn),
            cause=CloseCause.WITHDRAWN if withdrawn else CloseCause.LAPSED,
        )
    still_paired = {pid for p in remaining.values() for pid in (p.a, p.b)}
    orphaned = any(state(pid) is MemberState.LIVE and pid not in still_paired for pid in members)
    swapped = any(state(pid) in (MemberState.LAPSED, MemberState.MERGED) for pid in members)
    if not (orphaned or swapped or reread or withdrawn):
        return LapseDecision(action=LapseAction.KEEP, dropped=dropped)
    return LapseDecision(
        action=LapseAction.REGROUP,
        remaining=tuple(remaining.values()),
        dropped=dropped,
        withdrawn=tuple(withdrawn),
    )


# ---------------------------------------------------------------------------
# The cap
# ---------------------------------------------------------------------------


def select_under_cap(is_new: Sequence[bool], *, cap: int) -> tuple[list[int], list[int]]:
    """Which groups open this run, and which wait.

    ``is_new[i]`` describes the i-th group, oldest-confirmed first. A group
    that only regroups an existing record (``False``) is not a new disclosure
    and never counts against the cap. Returns ``(opened, waiting)`` as group
    indexes.
    """
    opened: list[int] = []
    waiting: list[int] = []
    budget = max(cap, 0)
    for index, new in enumerate(is_new):
        if not new:
            opened.append(index)
        elif budget > 0:
            opened.append(index)
            budget -= 1
        else:
            waiting.append(index)
    return opened, waiting
