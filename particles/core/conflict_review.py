# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Pure INCONSISTENCY-resolution decisions shared by Review and the trust cascade.

An INCONSISTENCY wrapper names two conflicting claims, A and B. Two operations
close it: a human review (§9.6, :mod:`particles.operations.review`) and the
source trust cascade (Extension B, :mod:`particles.operations.cascade`). Both
decide the same things — which claim loses, what demoting it means given its
current status, whether a quarantined claim is promoted, and which corpus
entry a trust statement is keyed on — and those decisions are pure functions
of the wrapper, the two claims and (for the cascade) their trust ranks.

They live here so each rule exists once and is testable without a database
(D2). The operations keep the I/O half: they gather the wrapper and
the claims, call these functions, and apply the plan in their existing write
order. Same split as :mod:`particles.core.cascade_gate`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from particles.core.conflict_resolution import RETIRED_VALUE_KEY
from particles.core.schema import Particle, ProvenanceRefType, ResolutionAction
from particles.core.status import Status, StatusReason

_TERMINAL = (Status.RETRACTED, Status.SUPERSEDED)


class Demotion(Enum):
    """What demoting a conflict's losing claim takes, given its current status."""

    NONE = "none"
    """Nothing to write: the claim is missing (a pre-ADR-0117 dangling ref),
    already terminal, or stale for a reason other than a pending
    conflict."""

    REASON_FLIP = "reason_flip"
    """A quarantined claim: ``CONFLICT_PENDING → CONFLICT_RESOLVED``
    in place, with no status transition."""

    TRANSITION = "transition"
    """A live claim: transition to ``PROVENANCE_STALE`` / ``CONFLICT_RESOLVED``."""


@dataclass(frozen=True)
class AleatoryMark:
    """One BOTH_VALID claim to keep queryable with ``ALEATORY`` nature."""

    particle: Particle
    mint: bool
    """True for a quarantined claim, which is promoted to a new ACTIVE
    particle carrying the nature; False to update the nature in place."""


@dataclass(frozen=True)
class TrustJudgment:
    """The SourceTrustStatement a PREFER review writes, before it is minted."""

    source_entry_id: str
    """Corpus entry the statement is keyed on."""
    preferred_id: str
    """The preferred claim as stored. When the plan promotes it, the operation
    names the minted particle instead."""
    demoted_id: str | None
    """The other wrapper ref, which may dangle on a pre-ADR-0117 wrapper."""


@dataclass(frozen=True)
class ResolutionPlan:
    """Every write a review resolution makes, decided up front.

    The operation applies the fields in declaration order, then closes (or,
    for DEFER, re-sets) the wrapper, and only then runs the trust cascade
    when ``trust`` is set. The wrapper closes first so the cascade never
    re-processes the resolution just made.
    """

    demote: tuple[Particle, Demotion] | None
    promote: Particle | None
    aleatory: tuple[AleatoryMark, ...]
    trust: TrustJudgment | None
    """None: no trust statement is written, and no cascade runs."""
    close_wrapper: bool
    """False only for DEFER, the one action that leaves the wrapper open."""


@dataclass(frozen=True)
class CascadeVerdict:
    """A trust-cascade resolution of one wrapper."""

    winner: Particle
    loser: Particle
    loser_demotion: Demotion
    promote_winner: bool
    """True when the winner is quarantined: it is promoted to a new ACTIVE
    particle, as a PREFER_B review would."""


def is_quarantined(particle: Particle) -> bool:
    """True for a §6.6 quarantined conflict loser."""
    return (
        particle.status is Status.PROVENANCE_STALE
        and particle.status_reason is StatusReason.CONFLICT_PENDING
    )


def conflict_pair_ids(inconsistency: Particle) -> tuple[str | None, str | None]:
    """The ids of claims A and B named by an INCONSISTENCY wrapper.

    By convention the first PARTICLE provenance ref is A and the second is B;
    the referenced particle id travels in ``corpus_entry_id``. A missing ref
    reads as None.
    """
    refs = [
        r.corpus_entry_id for r in inconsistency.provenance if r.type is ProvenanceRefType.PARTICLE
    ]
    return (
        refs[0] if len(refs) > 0 else None,
        refs[1] if len(refs) > 1 else None,
    )


def is_retired_value(inconsistency: Particle) -> bool:
    """Whether the wrapper records a retired value rather than a source conflict.

    A retired-value record pairs a judgment-retired twin (A) with a held
    re-assertion (B). Resolving it rules on the earlier judgment, not on a
    source, so neither review nor the cascade writes a trust verdict for it.
    """
    return bool(inconsistency.properties and inconsistency.properties.get(RETIRED_VALUE_KEY))


def decide_demotion(loser: Particle | None) -> Demotion:
    """What demoting a conflict's losing claim takes.

    A missing loser (a pre-ADR-0117 wrapper's dangling ref) and a terminal one
    (a judgment-retired twin) need nothing: there is no legal
    transition out of a terminal state. A stale loser is already demoted; only
    a quarantined one has its reason flipped.
    """
    if loser is None or loser.status in _TERMINAL:
        return Demotion.NONE
    if loser.status is Status.PROVENANCE_STALE:
        if loser.status_reason is StatusReason.CONFLICT_PENDING:
            return Demotion.REASON_FLIP
        return Demotion.NONE
    return Demotion.TRANSITION


def _source_entry_id(particle: Particle) -> str | None:
    """The corpus entry of the claim's first SOURCE provenance ref."""
    ref = next((r for r in particle.provenance if r.type is ProvenanceRefType.SOURCE), None)
    return ref.corpus_entry_id if ref is not None else None


def trust_statement_source(preferred: Particle | None) -> str | None:
    """The corpus-entry id a PREFER judgment's trust statement is keyed on.

    Every consumer of a SourceTrustStatement matches its ``source_ref``
    against a corpus-entry id: the §6.4 layered cascade and the reviewer
    confirmation gate. The statement is therefore keyed on
    the corpus entry of the preferred claim's SOURCE provenance, never on the
    particle id, which no consumer could look up.

    Returns None, meaning no statement is written, when the preferred claim is
    unknown (a dangling ref), is not ACTIVE (a retired claim is no source
    recommendation), or has no SOURCE provenance (an agent-asserted or
    derived claim names no corpus entry).
    """
    if preferred is None or preferred.status is not Status.ACTIVE:
        return None
    return _source_entry_id(preferred)


def decide_resolution(
    action: ResolutionAction,
    inconsistency: Particle,
    particle_a: Particle | None,
    particle_b: Particle | None,
) -> ResolutionPlan:
    """Decide every write a review resolution makes (§9.6).

    Args:
        action: The reviewer's resolution.
        inconsistency: The INCONSISTENCY wrapper.
        particle_a: Claim A as stored, or None when its ref dangles.
        particle_b: Claim B as stored, or None when its ref dangles.

    Returns:
        The plan the operation applies.
    """
    a_id, b_id = conflict_pair_ids(inconsistency)
    retired_value = is_retired_value(inconsistency)

    match action:
        case ResolutionAction.PREFER_A:
            trust_source = trust_statement_source(particle_a)
            trust = (
                TrustJudgment(trust_source, particle_a.id, b_id)
                if trust_source is not None and particle_a is not None and not retired_value
                else None
            )
            return ResolutionPlan(
                demote=_demote(particle_b),
                promote=None,
                aleatory=(),
                trust=trust,
                close_wrapper=True,
            )
        case ResolutionAction.PREFER_B:
            # A quarantined B is minted as a new ACTIVE particle carrying its
            # provenance verbatim, so the statement is keyed on B's source as
            # if B were ACTIVE; any other B is judged as stored.
            promote = particle_b if particle_b is not None and is_quarantined(particle_b) else None
            trust_source = (
                _source_entry_id(promote)
                if promote is not None
                else trust_statement_source(particle_b)
            )
            trust = (
                TrustJudgment(trust_source, particle_b.id, a_id)
                if trust_source is not None and particle_b is not None and not retired_value
                else None
            )
            return ResolutionPlan(
                demote=_demote(particle_a),
                promote=promote,
                aleatory=(),
                trust=trust,
                close_wrapper=True,
            )
        case ResolutionAction.BOTH_VALID:
            marks = tuple(
                AleatoryMark(particle=p, mint=is_quarantined(p))
                for p in (particle_a, particle_b)
                if p is not None
            )
            return ResolutionPlan(
                demote=None, promote=None, aleatory=marks, trust=None, close_wrapper=True
            )
        case ResolutionAction.DEFER:
            return ResolutionPlan(
                demote=None, promote=None, aleatory=(), trust=None, close_wrapper=False
            )


def _demote(loser: Particle | None) -> tuple[Particle, Demotion] | None:
    return (loser, decide_demotion(loser)) if loser is not None else None


def cascade_pair(
    inconsistency: Particle, particle_a: Particle | None, particle_b: Particle | None
) -> tuple[Particle, Particle] | None:
    """The structural half of the cascade decision: is this wrapper rankable?

    The trust cascade leaves a wrapper for a person when either claim is
    missing (a pre-ADR-0117 dangling ref), when it records a retired value, or
    when A has since left the surface by a terminal transition: source trust
    cannot answer a question about an earlier judgment.

    The operation calls this before looking up trust ranks, so a skipped
    wrapper costs no rank lookups. :func:`decide_cascade` applies it again,
    so it is a complete decision on its own.

    Returns:
        ``(A, B)`` to rank, or None when the wrapper is left open.
    """
    if particle_a is None or particle_b is None:
        return None
    if is_retired_value(inconsistency) or particle_a.status in _TERMINAL:
        return None
    return particle_a, particle_b


def decide_cascade(
    inconsistency: Particle,
    particle_a: Particle | None,
    particle_b: Particle | None,
    rank_a: float | None,
    rank_b: float | None,
    *,
    differential_threshold: float,
) -> CascadeVerdict | None:
    """Decide whether the trust cascade resolves a wrapper, and for whom.

    The claim whose source ranks higher wins, provided both ranks are known
    and differ by at least ``differential_threshold``. On an exact tie that
    clears a zero threshold, B wins.

    Returns:
        The verdict, or None when the wrapper stays open for manual review.
    """
    pair = cascade_pair(inconsistency, particle_a, particle_b)
    if pair is None or rank_a is None or rank_b is None:
        return None
    diff = rank_a - rank_b
    if abs(diff) < differential_threshold:
        return None
    a, b = pair
    winner, loser = (a, b) if diff > 0 else (b, a)
    return CascadeVerdict(
        winner=winner,
        loser=loser,
        loser_demotion=decide_demotion(loser),
        promote_winner=is_quarantined(winner),
    )
