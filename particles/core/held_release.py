# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Pure decisions of the held-claim release pass.

A rung-3 record holds the newer claim quarantined behind the stored one. When
an autonomous pass later retires the stored claim, nothing reconciles the
held claim against what remains. The ``release`` consolidation pass asks one
question of each such record: does anything ACTIVE still contradict the held
claim? This module holds every decision that needs no store, so each is
testable on plain values (D2):

* which records the pass examines (:func:`qualifies`), including a record
  whose partner the generation cascade retired for a newer generation of the
  held claim's own source (:func:`is_generation_hold`);
* whether a record already examined is due again (:func:`is_due`);
* what the probes and second readings say about one record
  (:func:`decide_release`).

The I/O shell is ``particles.operations.held_release``.
"""

from __future__ import annotations

from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from particles.core.conflict_review import is_quarantined, is_retired_value
from particles.core.contradiction_disclosure import is_census_record
from particles.core.observer_scope import PairPrecondition
from particles.core.schema import Particle, ProvenanceRefType
from particles.core.status import Status, StatusReason

#: The retirements an autonomous pass writes: rung 2.5 and its sweep,
#: the re-anchor pass, and the document sweep.
#: A claim a person retired is not among them, and a record
#: behind one stays with review.
AUTONOMOUS_RETIREMENTS: frozenset[StatusReason] = frozenset(
    {
        StatusReason.SUPERSEDED_BY_UPDATE,
        StatusReason.SUPERSEDED_BY_REANCHOR,
        StatusReason.DOCUMENT_SUPERSEDED,
    }
)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def is_generation_hold(
    a: Particle, b: Particle, *, current: Mapping[str, str], captured: Mapping[str, datetime]
) -> bool:
    """Whether A was retired for the current generation of B's own source.

    The pair the write path now settles by generation, written
    before that rule: B was held behind A, both read from one ``MUTABLE``
    entry, and the generation cascade then retired A ``RETRACTED_DEPENDENCY``
    because B's snapshot no longer stated it. Such a record never
    released, since the cascade is not one of :data:`AUTONOMOUS_RETIREMENTS`.

    Every ``SOURCE`` ref of both claims names that one entry, so A was stated
    by no other source; every snapshot A's refs name was captured before every
    snapshot B's refs name; and one of B's refs names the entry's current
    generation. ``current`` maps each ``MUTABLE`` entry to that snapshot. A
    hold whose entry has moved past B's snapshot stays out: a quarantined claim
    is never carried forward, so whether the newer text still states B is
    unknown, and a released B would be read as lapsed. ``captured`` maps
    snapshot ids to capture times; a ref it cannot date keeps the record out,
    the safe direction.
    """
    if a.status_reason is not StatusReason.RETRACTED_DEPENDENCY:
        return False
    a_refs = [r for r in a.provenance if r.type is ProvenanceRefType.SOURCE]
    b_refs = [r for r in b.provenance if r.type is ProvenanceRefType.SOURCE]
    entries = {r.corpus_entry_id for r in a_refs + b_refs}
    if not a_refs or not b_refs or len(entries) != 1:
        return False
    (entry_id,) = entries
    if entry_id is None or entry_id not in current:
        return False
    if not any(r.snapshot_id == current[entry_id] for r in b_refs):
        return False
    a_at = [captured.get(r.snapshot_id) if r.snapshot_id else None for r in a_refs]
    b_at = [captured.get(r.snapshot_id) if r.snapshot_id else None for r in b_refs]
    if any(at is None for at in a_at + b_at):
        return False
    return max(_aware(at) for at in a_at if at) < min(_aware(at) for at in b_at if at)


def qualifies(
    record: Particle, a: Particle | None, b: Particle | None, *, generation: bool = False
) -> bool:
    """Whether the release pass examines ``record``.

    An open write-time record: still ``INCONSISTENCY``, neither a census
    record nor a retired-value hold, with B quarantined and A retired by an
    autonomous pass. The two named exclusions are already implied by the side
    conditions; they are checked anyway, as the trust cascade checks its own,
    so that a later record shape cannot qualify by accident. ``generation``
    is :func:`is_generation_hold` for the pair, which the shell decides from
    the entry and snapshot rows: such a record is examined too, and its held
    claim is still checked against the whole store before it is released.
    """
    if record.status is not Status.INCONSISTENCY:
        return False
    if is_census_record(record) or is_retired_value(record):
        return False
    if a is None or b is None or not is_quarantined(b):
        return False
    if a.status is Status.ACTIVE:
        return False
    return a.status_reason in AUTONOMOUS_RETIREMENTS or generation


def is_due(prior_blocking: Sequence[str] | None, active_ids: Container[str]) -> bool:
    """Whether a qualifying record is examined this run.

    ``prior_blocking`` is the ``blocking_ids`` of the record's latest
    ``HELD_CLAIM_EXAMINED`` event, or ``None`` when it has none. A record never
    examined is due; one examined is due again only once a claim that blocked
    it has left ``ACTIVE``. There is no positional cursor.
    """
    if prior_blocking is None:
        return True
    return any(pid not in active_ids for pid in prior_blocking)


@dataclass(frozen=True)
class PairCheck:
    """What the write path's two checks said about one (existing, held) pair."""

    existing_id: str
    #: The contradiction probe: ``None`` when it could not complete.
    probe: bool | None
    #: The second reading of a probe-confirmed pair: ``None`` when
    #: it failed or did not run.
    reading: bool | None = None
    precondition: PairPrecondition = PairPrecondition.RECONCILE


class ReleaseAction(StrEnum):
    RELEASE = "release"
    """No confirmed contradiction remains: the held claim goes ACTIVE."""
    BLOCKED = "blocked"
    """At least one ACTIVE claim still contradicts it: nothing is written to
    any claim, and the examination is recorded."""
    INCOMPLETE = "incomplete"
    """A check that could have blocked did not complete: fail-closed, the
    record stays open and is asked again next run."""


@dataclass(frozen=True)
class ReleaseDecision:
    action: ReleaseAction
    #: The claims that still contradict the held claim (``BLOCKED`` only).
    blocking: tuple[str, ...] = ()
    #: Confirmed pairs the observer precondition declined: they do not block,
    #: and become ``CONTRADICTS`` edges once the claim is minted.
    declined: tuple[str, ...] = ()
    #: Pairs whose check did not complete, for the report.
    unverified: tuple[str, ...] = ()


def _confirmed(check: PairCheck) -> bool | None:
    """Whether both checks confirm the pair: ``None`` when one did not complete."""
    if check.probe is None:
        return None
    if not check.probe:
        return False
    return check.reading


def decide_release(checks: Sequence[PairCheck]) -> ReleaseDecision:
    """Decide one record from the checks on every pair.

    A pair contradicts the held claim only when the probe and the second
    reading both confirm it. A confirmed pair the precondition declines never
    blocks. Of the rest, any confirmed pair blocks; failing that, any pair
    whose check did not complete makes the decision ``INCOMPLETE``, because
    the action on a clean result is to release. The ladder's verdict on a
    blocking pair is never consulted: the pass does not retire a claim, even
    one the held claim would beat.
    """
    blocking: list[str] = []
    declined: list[str] = []
    unverified: list[str] = []
    for check in checks:
        verdict = _confirmed(check)
        declines = check.precondition is PairPrecondition.DECLINE
        if verdict is None:
            if not declines:
                unverified.append(check.existing_id)
        elif verdict:
            (declined if declines else blocking).append(check.existing_id)
    if blocking:
        return ReleaseDecision(
            ReleaseAction.BLOCKED,
            blocking=tuple(blocking),
            declined=tuple(declined),
            unverified=tuple(unverified),
        )
    if unverified:
        return ReleaseDecision(
            ReleaseAction.INCOMPLETE, declined=tuple(declined), unverified=tuple(unverified)
        )
    return ReleaseDecision(ReleaseAction.RELEASE, declined=tuple(declined))


def blocking_from_payload(payload: Mapping[str, object] | None) -> list[str] | None:
    """The ``blocking_ids`` a ``HELD_CLAIM_EXAMINED`` payload names, or ``None``."""
    if not payload:
        return None
    raw = payload.get("blocking_ids")
    if not isinstance(raw, list):
        return None
    return [str(pid) for pid in raw if isinstance(pid, str)]
