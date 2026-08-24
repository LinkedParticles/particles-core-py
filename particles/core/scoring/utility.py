# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Usefulness (outcome-learning) utility math for projection ranking.

The ``utility_rules`` layer of §6.4, whose ``rank_score(p) =
effective_confidence(p) + λ · ln(1 + R(p))`` and four normative constraints (a)
through (d) this module implements.

Utility is the *reinforcement* axis of the trust vocabulary, distinct from the
truth axis (§6.3 confidence): it measures how *useful* a belief has proven
to reload, not how likely it is to be *true*. On agent-memory the two are
negatively correlated — a precise dated fact scores high on confidence but is
rarely load-bearing next session, while a soft behavioural guideline scores
lower yet governs behaviour every session.

This module is **pure** (no I/O). It owns two quantities, and the split between
them is the one §6.4 makes normative — store-local *evidence* against portable
*judgment*, the same separation the confidence math observes (§6.3) and the one
given to decay:

- :func:`reinforcement_score` — from the *utility evidence* (per-belief,
  store-local: the ages of the utility events; how they are mined is
  implementation-defined and explicitly not part of the standard), a
  recency-weighted, **channel-weighted** count. Recent, repeated use scores
  high; use that stops fades with ``half_life_uses_days`` — the "deprioritizes
  when it stops being reinforced" mechanic. Two channels feed it: the
  transcript miner (``w = 1``) and the explicit operator gesture
  (``w = utility.explicit_weight``), which carries more because it
  fires once where the miner fires per session.
- :func:`utility_rank_bonus` — from the *utility policy* (per-observer,
  portable: the lens ``utility_rules`` ``rank_lift``, §6.4), an
  **additive** term on the projection *ordering* score. **Promotion-only** by
  construction (``λ ≥ 0``, ``R ≥ 0`` ⇒ bonus ``≥ 0``): lack of use never demotes
  a correct-but-new belief; it only withholds the bonus — §6.4 constraint (b),
  cold-start neutrality.

This supersedes the earlier design (superseded, and
recorded as such in §6.4 constraint (d)): utility was a bounded, saturating
*multiplier* on effective confidence (``clamp(1 + (cap−1)(1−e^{−weight·R}),
floor, cap)``). Measured on a real store, that form failed its own thesis —
``e^{−weight·R}`` saturates at ``R≈6``, so thousands of beliefs pegged the cap
and the head reverted to ordering by base confidence, while the multiplicative
shape gave *less* absolute lift to exactly the low-confidence, high-use
guidelines the lens exists to promote. The additive form fixes both: equal
absolute lift regardless of base confidence, and ``ln(1+R)`` never saturates, so
count magnitude survives monotonically — the sublinear growth §6.4 constraint
(d) requires.

The bonus is added to effective confidence **only on the projection / digest
ranking path** — never the semantic-search retrieval rank (§6.4 constraint (c)),
and never the stored ``confidence.value`` or the read-time
``effective_confidence`` (§6.3). The resulting ``rank_score`` is an *ordering*
score, **not** a probability: it is not confined to ``[0, 1]`` and must not be
presented as a confidence or stored (§6.4 constraint (a)). The resolution of the
policy parameters and the store read of the event ages live in the Engine
(``operations/query/utility_policy.py`` / ``store/utility_store.py``); the math
lives here, in exactly one place.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime


def reinforcement_score(
    events: Sequence[tuple[datetime, float]],
    half_life_uses_days: float,
    now: datetime | None = None,
) -> float:
    """Recency-weighted utility evidence: ``Σ wᵢ · 0.5 ** (age_daysᵢ / half_life)``.

    Each utility event contributes a channel weight ``wᵢ`` that decays with its
    age, so a belief reinforced last session counts far more than one reinforced
    a year ago. Returns ``0.0`` for no events — cold-start neutrality, §6.4
    constraint (b).

    **Why the weight is per-event**. The two channels that produce
    evidence have structurally different *rates*, so equal per-event weight does
    not mean equal influence. The transcript miner emits one event per (belief,
    session) and accumulates tens of them unattended; an operator's explicit
    usefulness gesture fires **once**, and is rate-limited to one
    credit per belief per principal per day. At equal weight a deliberate
    gesture would buy ``λ·ln 2`` of rank-lift against the ``λ·ln(1+R)`` a
    well-used belief earns for free — a fraction of head entry — so the explicit
    channel would be unable to move the belief class it exists for. The weight
    is what makes the two channels commensurable.

    The weight is applied here, at read time, rather than stored on the event:
    the half-life already works that way (``R`` is derived from event ages at
    read time), which keeps the channel weight retunable — and
    sweepable by the calibration harness — without rewriting history.

    Args:
        events: ``(observed_at, weight)`` per utility event — when it was
            observed and what its channel is worth. Naive datetimes are treated
            as UTC. Non-positive weights contribute nothing.
        half_life_uses_days: Days for a single event's weight to halve — the
            reinforcement half-life (a lens judgment; §6.4). Must be > 0.
        now: Override for current time (used in tests).
    """
    if half_life_uses_days <= 0 or not events:
        return 0.0
    reference = now if now is not None else datetime.now(UTC)
    total = 0.0
    for ts, weight in events:
        if weight <= 0.0:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        age_days = (reference - ts).total_seconds() / 86400.0
        if age_days < 0:
            age_days = 0.0
        total += weight * math.pow(0.5, age_days / half_life_uses_days)
    return total


def utility_rank_bonus(reinforcement: float, lambda_: float) -> float:
    """Additive projection rank-lift: ``λ · ln(1 + R)`` (§6.4).

    The term added to effective confidence to form the projection *ordering*
    score, ``rank_score = effective_confidence + utility_rank_bonus(R, λ)``.

    Monotone and **non-saturating** in ``reinforcement``: a belief used 200×
    still outranks one used 24×, which outranks one used 6× — the count
    magnitude the superseded exponential discarded. Growth is
    logarithmic, so influence is bounded without a hard ceiling: doubling
    reinforcement adds only ``λ·ln 2 ≈ 0.69λ``, so no single popular belief can
    run away with the head — the sublinearity §6.4 constraint (d) requires, and
    the guarantee formerly bought with ``cap``.

    **Promotion-only** (§6.4 constraint (b), preserved): with ``λ ≥ 0`` and
    ``R ≥ 0``
    the result is ``≥ 0``, so utility can only lift a belief's rank, never lower
    it. A belief with no utility evidence (``R = 0``) gets exactly ``+0`` and
    keeps its base position — the cold-start neutrality that makes a store with
    no mined evidence rank byte-for-byte as before. This sign constraint
    subsumes the retired ``floor`` clamp.

    Args:
        reinforcement: The :func:`reinforcement_score` for the belief (≥ 0).
        lambda_: The lens ``rank_lift`` — how far usefulness may reorder the
            projection head (≥ 0; ``0`` disables the lift entirely). A single
            interpretable knob replacing ``weight`` / ``floor`` /
            ``cap`` triple.
    """
    if reinforcement <= 0.0 or lambda_ <= 0.0:
        return 0.0
    return lambda_ * math.log1p(reinforcement)


# ---------------------------------------------------------------------------
# Calibration sweep
# ---------------------------------------------------------------------------
#
# Auto-fitting ``λ`` was declined — measured on a real store, every
# candidate closed form either overshot the admissible band by 3–8×, returned
# zero (the uncalibrated cap flattens the head's confidence spread to exactly
# 0.0000, so a spread-keyed fit disables the feature it calibrates), or keyed on
# head diversity, which is a proxy for over-extraction rather than for
# utility policy. What ships instead is the *sweep* the operator runs: score the
# grid, report where each named target belief lands and how many head slots hold
# distinct content, and print the resulting admissible band.
#
# The one input a fit cannot manufacture stays with the operator — which beliefs
# *ought* to be in the head. That is a judgment about their own store, and the
# sweep is structured around supplying it rather than inferring it.

_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")
_WS_RE = re.compile(r"\s+")


def content_dedup_key(content: str) -> str:
    """Normalised content key for near-duplicate detection in a rendered head.

    Lowercases, replaces every non-alphanumeric run with a space, and collapses
    whitespace, so two beliefs differing only in punctuation, casing, or
    backtick placement share a key. Deliberately conservative — it matches
    *after* normalisation only, never fuzzily — so a head's distinct-key count
    is an **upper bound** on the genuinely distinct claims it holds, and the
    over-extraction it measures is under- rather than over-reported.
    """
    return _WS_RE.sub(" ", _NON_ALNUM_RE.sub(" ", content.lower())).strip()


def rank_lift_grid(maximum: float, steps: int) -> tuple[float, ...]:
    """The sweep's ``λ`` grid: ``0.0`` plus ``steps`` even values up to ``maximum``.

    ``0.0`` is always included — it is the baseline the whole sweep is read
    against (the pre-utility head — cold-start), and its row is what
    shows that a duplicate pair already sits inside a large head before utility
    touches the ranking at all.

    Args:
        maximum: Largest ``λ`` to evaluate (> 0).
        steps: Number of non-zero grid points (≥ 1). Values are rounded to 6
            decimal places so the report and any JSON round-trip agree.
    """
    if maximum <= 0.0 or steps < 1:
        return (0.0,)
    return (0.0, *(round(maximum * i / steps, 6) for i in range(1, steps + 1)))


@dataclass(frozen=True)
class SweepRow:
    """One belief's inputs to the sweep — the two scored maps, plus its content key.

    Attributes:
        particle_id: The belief's id (also the ranking tiebreaker, matching the
            digest's ``(-score, id)`` order so the sweep ranks what is rendered).
        effective_confidence: The truth-axis score with the utility bonus
            **off** (§6.3) — the base the lift is added to.
        reinforcement: The belief's :func:`reinforcement_score` under its
            resolved half-life.
        content_key: :func:`content_dedup_key` of the belief's content.
        owner_relevant: ``A(p)`` — whether the belief is about the viewer
            ; unused by the ``λ`` sweep and consumed by
            :func:`particles.core.scoring.relevance.sweep_owner_rank_lift`,
            which sweeps ``ω`` over these same rows.
    """

    particle_id: str
    effective_confidence: float
    reinforcement: float
    content_key: str
    owner_relevant: bool = False


@dataclass(frozen=True)
class HeadOutcome:
    """What one ``λ`` does to one surface's rendered head of ``head_size`` beliefs.

    Attributes:
        head_size: The surface's rendered ``N`` — a per-surface
            config knob, *not* a store property, which is why the band has to be
            reported per surface rather than once.
        distinct_contents: Distinct :func:`content_dedup_key` values in the head.
        largest_duplicate_cluster: Size of the biggest same-key group in the
            head — 1 when every slot is distinct.
        required_distinct: The distinct count this head must reach to pass.
        target_ranks: ``(particle_id, rank)`` pairs for the operator's named
            target beliefs, ranked over the whole population (1-based). A target
            absent from the scored set is reported as rank ``0``.
    """

    head_size: int
    distinct_contents: int
    largest_duplicate_cluster: int
    required_distinct: int
    target_ranks: tuple[tuple[str, int], ...]

    @property
    def targets_in_head(self) -> bool:
        """Every named target ranks inside this head (vacuously true with none)."""
        return all(0 < rank <= self.head_size for _pid, rank in self.target_ranks)

    @property
    def diverse(self) -> bool:
        """The head holds at least ``required_distinct`` distinct contents."""
        return self.distinct_contents >= self.required_distinct

    @property
    def admissible(self) -> bool:
        """Both acceptance criteria hold for this surface at this ``λ``."""
        return self.targets_in_head and self.diverse


@dataclass(frozen=True)
class SweepPoint:
    """One grid point: the outcome at this ``λ`` for every requested head size."""

    rank_lift: float
    heads: tuple[HeadOutcome, ...]

    def head(self, head_size: int) -> HeadOutcome | None:
        """The outcome for one surface, or ``None`` if it was not swept."""
        return next((h for h in self.heads if h.head_size == head_size), None)


@dataclass(frozen=True)
class RankLiftBand:
    """The admissible ``λ`` interval, as resolved by the grid.

    ``low``/``high`` are grid points, so the true edges lie within one grid step.
    ``contiguous`` is False when some interior grid point fails — a shape worth
    seeing rather than smoothing over, since the acceptance objective is **not
    monotone** in ``λ`` — a target's rank improves, then worsens
    again as duplicate clusters overtake it.
    """

    low: float | None
    high: float | None
    contiguous: bool

    @property
    def empty(self) -> bool:
        """No grid point satisfied both criteria for this surface."""
        return self.low is None

    def contains(self, value: float) -> bool:
        """``value`` lies within the band's closed interval."""
        if self.low is None or self.high is None:
            return False
        return self.low <= value <= self.high


@dataclass(frozen=True)
class RankLiftSweep:
    """The whole sweep: every grid point, the per-surface bands, and their overlap.

    Attributes:
        points: One :class:`SweepPoint` per grid ``λ``, in grid order.
        bands: ``(head_size, band)`` per swept surface, in the order requested.
        intersection: The band of ``λ`` values admissible for **every** surface.
            Empty when the surfaces cannot be satisfied at once — a real
            property of the store, not a failure of the sweep.
        configured_rank_lift: The store's configured ``λ``, for the in/out-of-band
            flag. ``None`` when the caller did not supply it.
        scored: How many beliefs were scored.
    """

    points: tuple[SweepPoint, ...]
    bands: tuple[tuple[int, RankLiftBand], ...]
    intersection: RankLiftBand
    configured_rank_lift: float | None
    scored: int

    @property
    def configured_admissible(self) -> bool:
        """The configured ``λ`` sits inside the all-surface intersection."""
        return self.configured_rank_lift is not None and self.intersection.contains(
            self.configured_rank_lift
        )


def _band(grid: Sequence[float], admissible: Sequence[bool]) -> RankLiftBand:
    """Reduce a per-grid-point admissibility mask to its interval + contiguity."""
    hits = [lam for lam, ok in zip(grid, admissible, strict=True) if ok]
    if not hits:
        return RankLiftBand(low=None, high=None, contiguous=True)
    low, high = min(hits), max(hits)
    interior = [ok for lam, ok in zip(grid, admissible, strict=True) if low <= lam <= high]
    return RankLiftBand(low=low, high=high, contiguous=all(interior))


def sweep_rank_lift(
    rows: Sequence[SweepRow],
    *,
    grid: Sequence[float],
    head_sizes: Sequence[int],
    target_ids: Sequence[str] = (),
    distinct_ratio: float = 0.95,
    configured_rank_lift: float | None = None,
) -> RankLiftSweep:
    """Rank the store at every ``λ`` on the grid and reduce to per-surface bands.

    Pure and store-free: the caller supplies the two scored maps (the
    "two-map recipe") as :class:`SweepRow` values, and this ranks them
    by ``effective_confidence + λ·ln(1 + R)`` with the digest's ``(-score, id)``
    tiebreak — the same ordering key the projection and digest render, so the
    sweep measures the rendered artifact rather than a proxy for it.

    Args:
        rows: One entry per scored belief.
        grid: The ``λ`` values to evaluate (see :func:`rank_lift_grid`).
        head_sizes: Each surface's rendered ``N``. Sizes larger than the
            population are evaluated against the whole population.
        target_ids: Beliefs the operator asserts *ought* to reach the head — the
            label a fit cannot manufacture. With none supplied,
            only the diversity criterion constrains the band.
        distinct_ratio: Fraction of head slots that must hold distinct content.
            Defaults to ``0.95`` rather than ``1.0`` because a strict
            all-distinct criterion is unsatisfiable at large ``N`` on a store
            with any over-extraction at all — on the dogfood store a duplicate
            pair sits inside the top-120 at ``λ = 0``.
        configured_rank_lift: The store's configured ``λ``, carried through for
            the in/out-of-band flag.

    Returns:
        A :class:`RankLiftSweep`. With no rows, every band is empty.
    """
    sizes = tuple(dict.fromkeys(int(n) for n in head_sizes if n > 0))
    if not rows or not sizes:
        return RankLiftSweep(
            points=(),
            bands=tuple((n, RankLiftBand(None, None, True)) for n in sizes),
            intersection=RankLiftBand(None, None, True),
            configured_rank_lift=configured_rank_lift,
            scored=len(rows),
        )

    wanted = tuple(dict.fromkeys(target_ids))
    bonus_unit = {r.particle_id: math.log1p(max(r.reinforcement, 0.0)) for r in rows}
    points: list[SweepPoint] = []

    for lam in grid:
        ordered = sorted(
            rows,
            key=lambda r: (
                -(r.effective_confidence + lam * bonus_unit[r.particle_id]),
                r.particle_id,
            ),
        )
        position = {r.particle_id: i + 1 for i, r in enumerate(ordered)}
        target_ranks = tuple((pid, position.get(pid, 0)) for pid in wanted)
        heads: list[HeadOutcome] = []
        for n in sizes:
            keys = Counter(r.content_key for r in ordered[:n])
            effective_n = min(n, len(ordered))
            heads.append(
                HeadOutcome(
                    head_size=n,
                    distinct_contents=len(keys),
                    largest_duplicate_cluster=max(keys.values()),
                    required_distinct=math.ceil(effective_n * distinct_ratio),
                    target_ranks=target_ranks,
                )
            )
        points.append(SweepPoint(rank_lift=lam, heads=tuple(heads)))

    bands = tuple(
        (n, _band(list(grid), [p.heads[i].admissible for p in points])) for i, n in enumerate(sizes)
    )
    intersection = _band(
        list(grid),
        [all(h.admissible for h in p.heads) for p in points],
    )
    return RankLiftSweep(
        points=tuple(points),
        bands=bands,
        intersection=intersection,
        configured_rank_lift=configured_rank_lift,
        scored=len(rows),
    )
