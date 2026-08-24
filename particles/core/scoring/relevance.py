# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Owner-relevance (aboutness) math for recall ranking.

Unlike its two siblings this axis is **implementation-defined**: the technical
specification pins the truth axis (§6.3) and the usefulness axis (§6.4), but
names no ``ω`` term. Nothing here is normative for a second implementation of
the standard; it is this SDK's own read-time lens, and the properties asserted
below are its design contract rather than a spec requirement.

The third read-time axis of the trust vocabulary, and the one that asks a
question the other two cannot:

- **truth** — *is it believable?* (:mod:`~particles.core.scoring.confidence`,
  §6.3, composed with trust and :mod:`~particles.core.scoring.decay`)
- **use** — *has it earned its place?* (:mod:`~particles.core.scoring.utility`,
  §6.4)
- **aboutness** — *is it about me?* (this module)

The three are orthogonal by construction, which is the point: a belief about the
viewer may never be acted on, and the most heavily-used belief on a store is
typically not about anyone. Utility credits *use*; this credits *aboutness of
the viewer*. They are separate addends with separate coefficients so that
neither can mask the other, and so the most-skeptical-wins lens composition
(§6.4) cannot clamp both with one ``min()``.

This module is **pure** (no I/O). The Engine half — resolving *who the viewer
is* against the Subject store and deciding which beliefs are about them — lives
in ``operations/query/owner_policy.py`` (pure math stays Client, the
store-reading composition stays Engine).

Why the term is **additive rather than multiplicative**, inheriting an earlier
decision verbatim: on a store where the uncalibrated cap ties the
whole head at one effective-confidence value, a multiplier on a near-constant
base is a near-constant rescale and separates nothing. An additive lift also
gives the same *absolute* promotion to a low-confidence belief as to a
high-confidence one, which is what a relevance signal wants — being about the
viewer says nothing about how true a claim is, so it must not be worth more to
claims that are already believed.

**Promotion-only** by construction (``ω ≥ 0`` and ``A ≥ 0`` ⇒ bonus ``≥ 0``): the
lens ranks the viewer's beliefs *up* and can never rank a domain claim *down*.
That is what makes "solve this at read time, not by dropping domain claims" a
structural property of the composition rather than a policy someone has to
remember to honour.

The bonus enters the **recall ranking score only** — the projection, the digest,
and (as a node-selection term, never as a rendered confidence) the graph view
. It never enters the semantic-search ``query`` path, which already
has ``QueryRequest.subject_id`` for the caller who wants the viewer's beliefs
specifically; the lens exists to fix the *unqueried* surfaces. It is never
folded into the stored ``confidence.value`` or the read-time
``effective_confidence`` (§6.3), and it is never stored. The resulting
``rank_score`` is an *ordering* key, not a probability — it may exceed ``1.0``.

**Locality of rank contribution**. ``ω`` is *static policy*: it
never varies with the store's composition. A cohort-normalised ``ω`` — scaling
the lift by the viewer cohort's share of the store so one value behaves the same
everywhere — is deliberately **not** implemented, because a belief's rank
contribution must depend only on that belief's own properties, on static policy,
and on explicit relations to *specific named* beliefs, never on an aggregate
statistic over the particle set. An aggregate would leave federated ranking with
no defensible denominator, invalidate every cached score on every deposit, and
make "why is this belief ranked here" unanswerable from the belief plus the
config. The adaptivity is real and goes where this codebase already puts
adaptivity: **fitted offline over aggregates, applied per-item as a static
scalar** — exactly the extraction-time calibration pattern, and what
:func:`sweep_owner_rank_lift` below exists to support.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from particles.core.scoring.utility import RankLiftBand, SweepRow, _band

__all__ = [
    "OwnerHeadOutcome",
    "OwnerRankLiftSweep",
    "OwnerSweepPoint",
    "owner_rank_bonus",
    "sweep_owner_rank_lift",
]


def owner_rank_bonus(is_owner_relevant: bool, omega: float) -> float:
    """Additive recall rank-lift for one belief's owner-relevance.

    The ``ω · A(p)`` term of
    ``rank_score = effective_confidence + λ·ln(1+R) + ω·A(p)``, where ``A(p)``
    is the owner-aboutness indicator — ``1`` when the belief is about the
    viewer, ``0`` otherwise.

    Deliberately a **flat step** rather than a graded score in v1: tier 1
    aboutness is a set-membership test — the viewer either is
    among the belief's subjects or is not — and there is no defensible way to
    grade that without additional evidence. The consequence is that ``ω``
    behaves as a *threshold* over the whole viewer cohort rather than a graded
    lift, which is why it is calibrated against the cohort's **share of the
    rendered head** rather than against a single belief's rank.

    Args:
        is_owner_relevant: ``A(p)`` — whether the belief is about the viewer.
        omega: ``ω``, the configured owner rank-lift
            (``owner_lens.rank_lift``). Non-negative; ``0.0`` (the shipped
            default) makes the lens inert.

    Returns:
        ``omega`` when the belief is owner-relevant and ``omega`` is positive,
        else ``0.0``. Never negative, so the lens can only promote.
    """
    if not is_owner_relevant or omega <= 0.0:
        return 0.0
    return omega


# ---------------------------------------------------------------------------
# Calibration sweep
# ---------------------------------------------------------------------------
#
# ``ω`` is calibrated the same way ``λ`` is: swept, not fitted. But
# the acceptance criteria differ, because the two terms have different *shapes*.
#
# ``λ·ln(1+R)`` is graded per belief, so raising it re-orders a cohort
# internally and the failure mode is one runaway belief taking the head — which
# is why the ``λ`` sweep bounds the largest duplicate cluster.
#
# ``ω·A`` is a **flat step over a whole cohort**. Every belief about the viewer
# gets exactly the same lift, so ``ω`` behaves as a *threshold*: below it nothing
# moves, above it the entire cohort jumps together. On a store where the viewer
# is 5 % of a 20k-belief corpus, that is ~1000 beliefs arriving at a 60-line head
# at once. So the quantity to calibrate against is the cohort's **share of the
# head**, not any single belief's rank — which is what this sweep reports.
#
# Three criteria (mirrored). The third is the one that matters: two
# promotion-only terms on orthogonal axes compete for the same finite head, so
# the term added second must not silently undo the first.


@dataclass(frozen=True)
class OwnerHeadOutcome:
    """What one ``ω`` does to one surface's rendered head of ``head_size``.

    Attributes:
        head_size: The surface's rendered ``N``.
        owner_in_head: Viewer-relevant beliefs occupying head slots.
        min_owner_in_head: How many the head must hold to pass (criterion 1 —
            the lens must actually surface the viewer).
        max_owner_share: The largest fraction of the head the viewer cohort may
            occupy (criterion 2 — it must not *take* the head).
        target_ranks: ``(particle_id, rank)`` for beliefs that must **stay** in
            the head, ranked over the whole population (1-based; ``0`` = absent
            from the scored set). Criterion 3 — typically the utility targets
            , checked for non-regression under the new term.
        baseline_targets_in_head: the subset of ``target_ranks`` ids that were
            already inside this head at ``ω = 0``. Criterion 3 is evaluated
            **against this baseline**, not against absolute head membership: the
            criterion is *"aboutness must not push it out"*, which says nothing
            about a target that was already out for unrelated reasons (a grown
            store, a re-calibrated ``λ``). Conflating the two would blame this
            lens for a pre-existing regression — and, worse, would hide that
            regression behind an empty band.
    """

    head_size: int
    owner_in_head: int
    min_owner_in_head: int
    max_owner_share: float
    target_ranks: tuple[tuple[str, int], ...]
    baseline_targets_in_head: frozenset[str] = frozenset()

    @property
    def owner_share(self) -> float:
        """Fraction of the head held by viewer-relevant beliefs."""
        return self.owner_in_head / self.head_size if self.head_size else 0.0

    @property
    def owner_present(self) -> bool:
        """Criterion 1 — the viewer reaches the head at all."""
        return self.owner_in_head >= self.min_owner_in_head

    @property
    def share_bounded(self) -> bool:
        """Criterion 2 — the viewer cohort does not take the head."""
        return self.owner_share <= self.max_owner_share

    @property
    def targets_in_head(self) -> bool:
        """Criterion 3 — no target that was in the head at ``ω = 0`` has left it.

        Non-regression, not membership: a target already outside the head at
        baseline is *not applicable* here, and is disclosed by the renderer
        instead of quietly emptying the band.
        """
        return all(
            0 < rank <= self.head_size
            for pid, rank in self.target_ranks
            if pid in self.baseline_targets_in_head
        )

    @property
    def targets_absent_at_baseline(self) -> tuple[str, ...]:
        """Named targets that were already outside this head at ``ω = 0``.

        Surfaced so a pre-existing regression is reported rather than absorbed
        (the "no silent caps" rule applied to a calibration tool).
        """
        return tuple(
            pid for pid, _rank in self.target_ranks if pid not in self.baseline_targets_in_head
        )

    @property
    def admissible(self) -> bool:
        """All three criteria hold for this surface at this ``ω``."""
        return self.owner_present and self.share_bounded and self.targets_in_head


@dataclass(frozen=True)
class OwnerSweepPoint:
    """One grid point: the outcome at this ``ω`` for every requested head size."""

    rank_lift: float
    heads: tuple[OwnerHeadOutcome, ...]

    def head(self, head_size: int) -> OwnerHeadOutcome | None:
        """The outcome for one surface, or ``None`` if it was not swept."""
        for outcome in self.heads:
            if outcome.head_size == head_size:
                return outcome
        return None


@dataclass(frozen=True)
class OwnerRankLiftSweep:
    """The full ``ω`` sweep: every grid point, plus the per-surface bands.

    Attributes:
        points: One entry per grid value.
        bands: ``(head_size, band)`` — the admissible ``ω`` range per surface.
        intersection: The band admissible on *every* swept surface — the one an
            operator can actually configure, since ``ω`` is store-wide.
        configured_rank_lift: ``owner_lens.rank_lift`` as configured.
        owner_population: Viewer-relevant beliefs in the scored set.
        scored: Beliefs scored.
    """

    points: tuple[OwnerSweepPoint, ...]
    bands: tuple[tuple[int, RankLiftBand], ...]
    intersection: RankLiftBand
    configured_rank_lift: float | None
    owner_population: int
    scored: int

    @property
    def configured_admissible(self) -> bool:
        """Whether the configured ``ω`` sits inside the intersection band."""
        if self.configured_rank_lift is None:
            return False
        return self.intersection.contains(self.configured_rank_lift)


def sweep_owner_rank_lift(
    rows: Sequence[SweepRow],
    *,
    grid: Sequence[float],
    head_sizes: Sequence[int],
    lambda_: float,
    target_ids: Sequence[str] = (),
    min_owner_in_head: int = 1,
    max_owner_share: float = 0.5,
    configured_rank_lift: float | None = None,
) -> OwnerRankLiftSweep:
    """Rank the store at every ``ω`` and reduce to per-surface admissible bands.

    Pure and store-free. Ranks by the **full** three-axis key
    ``effective_confidence + λ·ln(1 + R) + ω·A`` with the digest's
    ``(-score, id)`` tiebreak — the same ordering the projection and digest
    render — so the sweep measures the rendered artifact rather than a proxy.

    ``λ`` is held **fixed** at the store's configured value while ``ω`` varies.
    That is deliberate and is what makes criterion 3 meaningful: the question is
    not "what does aboutness do to a bare confidence ranking" but "what does it
    do to the head utility has already shaped".

    Args:
        rows: One entry per scored belief; ``owner_relevant`` supplies ``A(p)``.
        grid: The ``ω`` values to evaluate (:func:`rank_lift_grid` works here —
            the two lifts are on the same scale, both added to effective
            confidence).
        head_sizes: Each rendered surface's ``N``.
        lambda_: The configured ``utility.default.rank_lift``, held fixed.
        target_ids: Beliefs that must **stay** in the head — the utility targets
            . Criterion 3 is the non-regression check.
        min_owner_in_head: Criterion 1's floor.
        max_owner_share: Criterion 2's ceiling, as a fraction of head slots.
        configured_rank_lift: The store's configured ``ω``, carried through for
            the in/out-of-band flag.

    Returns:
        An :class:`OwnerRankLiftSweep`. With no rows, every band is empty.
    """
    sizes = tuple(dict.fromkeys(int(n) for n in head_sizes if n > 0))
    owner_population = sum(1 for r in rows if r.owner_relevant)
    if not rows or not sizes:
        return OwnerRankLiftSweep(
            points=(),
            bands=tuple((n, RankLiftBand(None, None, True)) for n in sizes),
            intersection=RankLiftBand(None, None, True),
            configured_rank_lift=configured_rank_lift,
            owner_population=owner_population,
            scored=len(rows),
        )

    wanted = tuple(dict.fromkeys(target_ids))
    # The utility term is constant across the sweep, so fold it into the base
    # once rather than recomputing ln(1+R) at every grid point.
    base = {
        r.particle_id: r.effective_confidence + lambda_ * math.log1p(max(r.reinforcement, 0.0))
        for r in rows
    }

    def _rank(omega: float) -> tuple[list[SweepRow], dict[str, int]]:
        ordered = sorted(
            rows,
            key=lambda r: (
                -(base[r.particle_id] + (omega if r.owner_relevant else 0.0)),
                r.particle_id,
            ),
        )
        return ordered, {r.particle_id: i + 1 for i, r in enumerate(ordered)}

    # Criterion 3's baseline: where the targets sit with aboutness switched off.
    # Computed at ω = 0 explicitly rather than read off the grid, so the answer
    # does not depend on the caller having included 0.0 in it.
    _baseline_ordered, baseline_position = _rank(0.0)
    baseline_in_head = {
        n: frozenset(pid for pid in wanted if 0 < baseline_position.get(pid, 0) <= n) for n in sizes
    }
    points: list[OwnerSweepPoint] = []

    for omega in grid:
        ordered, position = _rank(omega)
        target_ranks = tuple((pid, position.get(pid, 0)) for pid in wanted)
        heads = tuple(
            OwnerHeadOutcome(
                head_size=n,
                owner_in_head=sum(1 for r in ordered[:n] if r.owner_relevant),
                min_owner_in_head=min_owner_in_head,
                max_owner_share=max_owner_share,
                target_ranks=target_ranks,
                baseline_targets_in_head=baseline_in_head[n],
            )
            for n in sizes
        )
        points.append(OwnerSweepPoint(rank_lift=omega, heads=heads))

    bands = tuple(
        (n, _band(list(grid), [p.heads[i].admissible for p in points])) for i, n in enumerate(sizes)
    )
    intersection = _band(list(grid), [all(h.admissible for h in p.heads) for p in points])
    return OwnerRankLiftSweep(
        points=tuple(points),
        bands=bands,
        intersection=intersection,
        configured_rank_lift=configured_rank_lift,
        owner_population=owner_population,
        scored=len(rows),
    )
