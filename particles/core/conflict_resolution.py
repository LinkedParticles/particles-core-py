# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Pure §6.4 conflict-resolution decision logic.

This module owns the *decision* half of the §6.4 ladder — the one the
specification declares normative under "Conflict resolution with source trust",
applied at §9.2 step 7 and yielding §6.6 status transitions. Given two
particles (an existing ACTIVE particle and a newly extracted candidate) and any
trust inputs already resolved by the caller, return a ``ConflictVerdict``
saying what should happen. It also owns the constructor that builds an
``INCONSISTENCY`` ``Particle`` from a conflicting pair.

It also owns the overrides around the verdict (:func:`decide_ladder`): the
tri-state probe, the observer precondition, and which rung 2.5 input applies.
The *effect* half of the ladder — DB writes, trust-rank lookups, embedding
similarity computation, the LLM contradiction-signal call — stays in
``particles/ingest/`` because it touches I/O: ``conflict_plan`` maps an outcome
to writes, and ``pipeline`` gathers the inputs and applies the plan. Core code
must remain pure (see ``particles/core/AGENTS.md``).

Ladder (normative, applied in order — §6.4, rung for rung):

  1. **ALEATORY exclusion** (§6.4 rung 1, lifted to the top) — if
     either particle has ``UncertaintyNature.ALEATORY`` the pair is irreducibly
     inconsistent: skip *both* the supersession prior and trust resolution and
     fall through to INCONSISTENCY. An irreducible disagreement is never retired
     by an editorial relation or a trust differential.
  1.5. **Document-supersession prior** — §6.4 rung 1.5 (cap. 2,
     re-ordered). The caller passes ``new_supersedes_existing`` /
     ``existing_supersedes_new``, resolved from the corpus supersession
     relation (a document's authored ``supersedes:`` edge, followed
     transitively). When exactly one direction holds **and** the
     modality-appropriate conflict signal confirms a
     replacement (``has_contradiction_signal``), the superseding document's claim
     wins: the loser is demoted PROVENANCE_STALE / DOCUMENT_SUPERSEDED and **no**
     INCONSISTENCY is surfaced. **This branch moves ABOVE the truth-apt
     gate and makes it modality-independent**: an authored "this
     document replaces that one" is an *editorial* fact that does not depend on
     either claim's truth-aptness, so it must reach a superseded
     ``CONSTITUTIVE`` definition that the truth engine cannot see. It sits
     *above* the adjudicability gate and the trust rung but *below* the ALEATORY
     exclusion (rung 1). Single-trust-order stores only in v1 (matching
     rung 2). The
     ``has_contradiction_signal`` flag is **reframed** on this path as a
     *replacement signal* — "does the superseding claim replace, not merely
     restate, the superseded one?" — and a ``False`` signal keeps both claims
     (the default-safe direction), preserving the
     never-blanket-demote invariant (cap. 2(c)) — the same
     demotion-only rule §6.4 states normatively.
  1.7. **Truth-apt gate** — §6.4 rung 1.7, kept *below* supersession.
     If either side is non-truth-apt, the **truth engine** (the
     contradiction probe, trust arbitration, INCONSISTENCY manufacture) has
     nothing to adjudicate; return CORROBORATES. This gate's *scope* is narrowed:
     it no longer blocks the editorial supersession prior above it,
     only the truth-engine rungs below.
     Beside it sits the **genericity guard** (:func:`is_generic_instance_pair`):
     a generic claim ("most mammals bear live young") and an
     instance claim ("the platypus lays eggs") are not an adjudicable pair, so
     the truth engine returns CORROBORATES for them as well. Two generics, or
     two instance claims, still reach the rungs below. The guard is its own
     function so the two gates stay separable.
  2. **Source trust check** (§6.4 rung 2, Extension B) — caller passes
     pre-resolved trust scores. When ``|score_new - score_existing| >=
     trust_differential_threshold``, the higher-trust side wins:
       - new wins  → ``SUPERSEDES`` (caller inserts ``new`` as ACTIVE and
         demotes ``existing`` to PROVENANCE_STALE / LOWER_TRUST_SOURCE).
       - existing wins → ``SUPERSEDED_BY_EXISTING`` (caller drops ``new``).
     This rung fires **only in a single-trust-order store**.
     When the caller passes ``single_trust_order=False`` — a multi-contributor
     / consensus store, with no global trust order — rung 2 is
     skipped and the pair falls through to rung 3: a contributor's claim is
     never silently dropped by another contributor's trust.
  3. **Default** → ``INCONSISTENT`` (§6.4 rung 3; caller persists the losing
     candidate quarantined — born ``PROVENANCE_STALE`` / ``CONFLICT_PENDING``
     per §9.2 step 7 — and writes the INCONSISTENCY particle
     produced by
     :func:`build_inconsistency_particle`).

Two extra verdicts are emitted by the pre-ladder gate the caller may apply:

  - ``CORROBORATES``: the pair is high-similarity but the caller's
    contradiction-signal probe came back negative (paraphrase / attribution
    wrapper). The two particles co-exist as ACTIVE. The pure decision
    function does not infer this on its own — the caller passes
    ``has_contradiction_signal=False`` after running its own gate.
  - ``NO_CONFLICT``: reserved for callers that want to use
    :func:`resolve_conflict` for below-similarity pairs and need a verdict
    that means "do nothing special".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from particles.core.generics import is_generic_claim
from particles.core.observer_scope import PairPrecondition
from particles.core.schema import (
    SCHEMA_VERSION,
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
    is_truth_apt,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status


class ConflictVerdict(StrEnum):
    """Outcome of the §6.4 ladder for a single (existing, new) pair."""

    CORROBORATES = "CORROBORATES"
    """High similarity but no contradiction signal — keep both ACTIVE."""

    SUPERSEDES = "SUPERSEDES"
    """Trust resolution: ``new`` wins; ``existing`` should be demoted."""

    SUPERSEDED_BY_EXISTING = "SUPERSEDED_BY_EXISTING"
    """Trust resolution: ``existing`` wins; ``new`` should be dropped."""

    DOCUMENT_SUPERSEDES = "DOCUMENT_SUPERSEDES"
    """Rung 1.5 of §6.4 (cap. 2, modality-independent):
    ``new``'s
    provenance document (transitively) supersedes ``existing``'s. Caller inserts
    ``new`` ACTIVE (or, in the cross-entry sweep, leaves it ACTIVE) and demotes
    ``existing`` to PROVENANCE_STALE / DOCUMENT_SUPERSEDED (winner ``new``).
    Emitted regardless of either claim's truth-aptness."""

    DOCUMENT_SUPERSEDED_BY_EXISTING = "DOCUMENT_SUPERSEDED_BY_EXISTING"
    """Rung 1.5 of §6.4 (cap. 2, modality-independent):
    ``existing``'s
    provenance document supersedes ``new``'s. ``existing`` stays ACTIVE; the
    caller stores ``new`` but demotes it to PROVENANCE_STALE / DOCUMENT_SUPERSEDED
    (the loser stays auditable — never a silent drop, under the §6.4
    demotion-only rule).
    Emitted regardless of either claim's truth-aptness."""

    UPDATE_SUPERSEDES = "UPDATE_SUPERSEDES"
    """Rung 2.5 of §6.4: a same-lineage update — ``new`` is the
    strictly newer of two claims trust cannot tell apart. Caller inserts ``new``
    ACTIVE (with ``supersedes`` naming ``existing``) and demotes ``existing`` to
    PROVENANCE_STALE / SUPERSEDED_BY_UPDATE. No INCONSISTENCY is queued."""

    UPDATE_SUPERSEDED_BY_EXISTING = "UPDATE_SUPERSEDED_BY_EXISTING"
    """Rung 2.5 mirror: ``existing`` is the newer claim (``new`` is an
    out-of-order older one, e.g. a backfilled transcript). Caller stores ``new``
    demoted to PROVENANCE_STALE / SUPERSEDED_BY_UPDATE; ``existing`` stays ACTIVE."""

    INCONSISTENT = "INCONSISTENT"
    """No clear winner — emit an INCONSISTENCY particle."""

    NO_CONFLICT = "NO_CONFLICT"
    """Pair is not in conflict at all (e.g. below caller's similarity floor)."""


def is_generic_instance_pair(a: Particle, b: Particle) -> bool:
    """Whether exactly one side of the pair is a generic claim.

    A quantified or generic claim ("most mammals bear live young", "Xs
    typically Y") and an instance claim ("the platypus lays eggs") are **not an
    adjudicable pair**. An exception does not falsify "most", so the truth
    engine must never pit them against each other: no contradiction probe
    verdict, no trust-differential supersession, no update supersession, no
    INCONSISTENCY manufactured from the pair, and no co-evidential suggestion.
    Two generics about one kind ("most X are Y" against "most X are not Y")
    remain adjudicable, and so do two instance claims; only the
    generic-against-instance pairing is excluded. A universal ("all mammals
    bear live young") is read the same way: source prose uses *all* loosely,
    and the substrate cannot tell a strict universal from a loose one, so a
    counterexample is the reader's to weigh, not the ladder's.

    Genericity is read from each side's text by the deterministic detector
    (:func:`particles.core.generics.generic_quantifier`), the one the
    abstraction pass uses to keep promoted text premise-scoped.
    No LLM call is made.
    The detector leans toward flagging, and here a false flag costs only a
    pair left unadjudicated with both claims ACTIVE.

    **Recorded no: the substrate never infers an instance property from a
    generic.** Default inheritance in the Cyc style (the platypus is a mammal,
    mammals bear live young, therefore the platypus bears live young) is the
    operation that turns a statistical claim about a group into an assertion
    about a member, and it stays out of the substrate. If it ever exists, it is
    a per-reference-class lens opt-in whose default is never. A sourced generic
    is otherwise stored neutrally like any claim, with its provenance, stance
    holder and observer scope. There is no ``EXCEPTION_TO`` relation and no
    generic representation in the schema; this guard is the whole of the rule.

    Kept separate from :func:`~particles.core.schema.is_truth_apt` on purpose:
    that gate reads a claim-level default (assertion modality), while this one
    reads a property of the pair. Each can change without the other.
    """
    return is_generic_claim(a.content) != is_generic_claim(b.content)


def resolve_conflict(
    existing: Particle,
    new: Particle,
    *,
    has_contradiction_signal: bool = True,
    new_supersedes_existing: bool = False,
    existing_supersedes_new: bool = False,
    trust_score_existing: float | None = None,
    trust_score_new: float | None = None,
    trust_differential_threshold: float = 0.15,
    single_trust_order: bool = True,
    update_order: int | None = None,
) -> ConflictVerdict:
    """Apply the §6.4 ladder and return the verdict for one (existing, new) pair.

    Pure function — no I/O. The caller is responsible for:

      - running the embedding-similarity probe and only invoking
        ``resolve_conflict`` for pairs that exceed the threshold,
      - running the contradiction-signal gate (attribution patterns + LLM
        confirmation) and passing the result as
        ``has_contradiction_signal``,
      - resolving trust scores via the Extension B layered lookup (or the
        URL baseline) and passing them as ``trust_score_*``,
      - performing every resulting DB write (insert / status update) and
        emitting the INCONSISTENCY particle built by
        :func:`build_inconsistency_particle`.

    Args:
        existing: Particle A — the currently ACTIVE particle in the store.
        new: Particle B — the candidate just emitted by the extractor.
        has_contradiction_signal: Result of the caller's contradiction-signal
            probe. ``False`` corroborates (no supersession demotion, no trust
            resolution, no INCONSISTENCY). ``True`` runs the full ladder. On the
            supersession branch (rung 1.5) this flag is **reframed** as a
            *replacement signal* — for a non-truth-apt pair it answers "does the
            superseding claim replace, not merely restate, the superseded one?" —
            and ``False`` keeps both claims (the default-safe direction).
        new_supersedes_existing: Rung 1.5 input — §6.4, cap. 2.
            ``True`` when
            ``new``'s provenance corpus entry (transitively) supersedes
            ``existing``'s — an authored editorial "this document replaces that
            one". The caller resolves it from the corpus supersession relation.
            This branch runs **above the adjudicability gate** and is
            **modality-independent**, so it retires a superseded ``CONSTITUTIVE``
            definition the truth engine would otherwise never see — but only when
            ``has_contradiction_signal`` (the replacement signal) is ``True`` and
            the pair is not ALEATORY.
        existing_supersedes_new: Rung 1.5 input — the mirror direction.
            ``True`` when ``existing``'s document supersedes ``new``'s. Both
            ``True`` (a supersession cycle) fires neither branch and falls
            through to the adjudicability gate / trust rung.
        trust_score_existing: Pre-resolved trust score for ``existing``.
            ``None`` skips the trust rung (the ALEATORY exclusion still applies).
        trust_score_new: Pre-resolved trust score for ``new``. Same treatment.
        trust_differential_threshold: Minimum absolute score gap that lets
            rung 2 auto-resolve. Defaults to 0.15 (matches
            ``config.trust.differential_threshold`` historical default); the
            caller should pass the live value from
            ``get_config().trust.differential_threshold``.
        single_trust_order: Whether the store has a single global trust order.
            ``True`` (default) is today's behavior — rung 2
            auto-supersede may fire. ``False`` is a multi-contributor /
            consensus store, which has no global trust order, so
            rung 2 is **skipped entirely** and a confirmed contradiction falls
            through to ``INCONSISTENT`` (both claims stay ACTIVE, ranked
            per-viewer at query time) — a contributor's claim is never dropped
            by another's trust. The caller passes
            ``get_config().reconciliation.store_mode == "single"``.
        update_order: Rung 2.5 input. ``+1`` when the pair is a
            same-lineage update and ``new`` is strictly newer, ``-1`` when
            ``existing`` is, ``None`` when rung 2.5 does not apply (the caller
            found the sides distinguishable to trust, not both
            extractor-asserted, undated, equal-dated, or the rung disabled).
            Deliberately **not** gated on ``single_trust_order``: a single
            lineage updating itself is not the cross-contributor arbitration
            ``multi`` stores forbid (the same-holder analogue).

    Returns:
        ConflictVerdict — see the enum docstrings for the caller's required
        follow-up action.
    """
    # Rung 1 (lifted to the top): ALEATORY exclusion. An irreducibly
    # aleatory pair is never retired by an editorial supersession relation nor by
    # a trust differential; it skips the supersession prior (rung 1.5) and the
    # trust rung (rung 2) and falls through to INCONSISTENCY. ALEATORY is an
    # ``uncertainty_nature``, orthogonal to modality — this is the ONE exclusion
    # that sits above the supersession prior.
    aleatory = (
        existing.uncertainty_nature == UncertaintyNature.ALEATORY
        or new.uncertainty_nature == UncertaintyNature.ALEATORY
    )

    # Rung 1.5 (cap. 2 — ABOVE the adjudicability gate,
    # modality-independent): document-supersession prior. An explicit, authored
    # "this document replaces that one" is an *editorial* fact — it does not
    # depend on either claim's truth-aptness, so it must reach a superseded
    # CONSTITUTIVE definition that the truth engine (rung 1.7 onward) cannot see.
    # It therefore runs ABOVE the adjudicability gate. ``has_contradiction_signal`` is
    # reframed here as a *replacement signal* ("does the superseding claim
    # replace, not merely restate, the superseded one?"); a False signal keeps
    # both claims (the default-safe direction), preserving the
    # never-blanket-demote invariant (cap. 2(c)). ALEATORY still wins above it
    # (``not aleatory``). Gated to single-trust-order stores in v1, matching
    # rung 2. A supersession *cycle* (both directions true) fires
    # neither branch and falls through.
    if single_trust_order and not aleatory and has_contradiction_signal:
        if new_supersedes_existing and not existing_supersedes_new:
            return ConflictVerdict.DOCUMENT_SUPERSEDES
        if existing_supersedes_new and not new_supersedes_existing:
            return ConflictVerdict.DOCUMENT_SUPERSEDED_BY_EXISTING

    # Rung 1.7, the adjudicability gate, kept BELOW supersession.
    # The write path arbitrates only pairs whose stored default is FALSIFIABLE on
    # both sides. If either side is not adjudicable by default (an opinion, a
    # feeling, or a document's constitutive rule), two sources' disagreement
    # cannot be settled by weighing them: co-exist, never contradict or
    # trust-supersede. Only the stored default is read here; a lens's reading is
    # read-time and never reaches this gate. The editorial supersession
    # prior already ran above; this gate governs only the rungs below it. Defense
    # in depth: the pipeline's intra-entry ``_find_conflict`` already declines to
    # pair non-truth-apt particles; the cross-entry supersession sweep
    # is the path that deliberately pairs them, and it reaches rung 1.5 above.
    if not (is_truth_apt(existing) and is_truth_apt(new)):
        return ConflictVerdict.CORROBORATES

    # The genericity guard, beside the adjudicability gate and governing the same
    # truth-engine rungs: a generic and an instance claim are not an
    # adjudicable pair, since an exception does not falsify "most". Co-exist,
    # never contradict or supersede. Pairs of two generics or two instance
    # claims pass through unchanged. The pair selectors (the pipeline's
    # ``_find_conflict`` and subject pool, ``enumerate_candidate_pairs``, the
    # update sweep's gather) already decline these pairs before a probe is
    # spent; this is defense in depth.
    if is_generic_instance_pair(existing, new):
        return ConflictVerdict.CORROBORATES

    # Contradiction-signal gate (pre-ladder, not a numbered rung): high
    # similarity is not enough on its own. If the caller's probe said the pair is
    # not a contradiction, treat it as corroboration and write both as ACTIVE.
    if not has_contradiction_signal:
        return ConflictVerdict.CORROBORATES

    # Rung 2: trust resolution (single-trust-order stores only).
    # In a multi-contributor / consensus store there is no global trust order
    # at all, so auto-supersede is suppressed and the pair falls
    # through to INCONSISTENT: disagreement is surfaced, never resolved away.
    if (
        single_trust_order
        and not aleatory
        and trust_score_existing is not None
        and trust_score_new is not None
    ):
        # Differential above threshold → winner takes all.
        diff = trust_score_new - trust_score_existing
        if abs(diff) >= trust_differential_threshold:
            if diff > 0:
                return ConflictVerdict.SUPERSEDES
            return ConflictVerdict.SUPERSEDED_BY_EXISTING

    # Rung 2.5: same-lineage update supersession. Trust cannot tell
    # the two sides apart (rung 2 did not fire), so the strictly newer claim
    # wins. The caller computes ``update_order`` only when the pair shares every
    # trust key but the entry, both are extractor-asserted, and both are
    # strictly ordered by source date. ALEATORY and non-truth-apt pairs never
    # reach here (they returned above).
    if not aleatory and update_order is not None:
        if update_order > 0:
            return ConflictVerdict.UPDATE_SUPERSEDES
        if update_order < 0:
            return ConflictVerdict.UPDATE_SUPERSEDED_BY_EXISTING

    # Rung 3: default — INCONSISTENCY particle.
    return ConflictVerdict.INCONSISTENT


@dataclass(frozen=True)
class RungInputs:
    """The gathered §6.4 rung inputs for one pair; the defaults are "no input".

    The caller resolves these only when :func:`needs_rung_inputs` says the
    ladder can consult them (a confirmed or fail-closed signal on a pair the
    observer precondition does not decline).
    """

    new_supersedes_existing: bool = False
    """Rung 1.5: ``new``'s document (transitively) supersedes ``existing``'s."""
    existing_supersedes_new: bool = False
    """Rung 1.5 mirror."""
    trust_score_new: float | None = None
    """Rung 2: the candidate's pre-resolved trust score."""
    trust_score_existing: float | None = None
    """Rung 2: the existing particle's pre-resolved trust score."""
    update_order: int | None = None
    """Rung 2.5: see :func:`resolve_conflict`'s ``update_order``."""


@dataclass(frozen=True)
class LadderOutcome:
    """What the §6.6 ladder decided for one pair, overrides included."""

    verdict: ConflictVerdict | None
    """The verdict; ``None`` when the observer precondition declined the pair:
    the candidate is written ``ACTIVE`` beside the existing claim."""
    record_divergence: bool
    """Declined and the signal is set: the pair is recorded as a divergence."""


class SlotVerdict(StrEnum):
    """The update probe's verdict on a confirmed contradiction.

    A confirmed contradiction is not yet an update. The update probe says
    whether the two claims fill one slot and, when they do, what kind of slot
    it is, because only one kind is replaced by a later value.
    """

    DIFFERENT = "different"
    """The claims fill different slots: not an update."""
    CHANGES = "changes"
    """One slot that holds one value at a time and changes over time (where
    someone lives, their employer, the price a service charges now, a count
    that grows): the later value replaces the earlier one, rung 2.5."""
    FIXED = "fixed"
    """One slot whose value, once true, stays true (who wrote a book, a page
    count, a release date, a product's price as stated at one time): two values
    are a contradiction neither date settles, so the pair goes to review at
    rung 3 and nothing is retired."""


def admits_update(slot: SlotVerdict | None) -> bool:
    """Whether rung 2.5 may act on a pair with this verdict.

    Only a slot that changes over time is updated by a later value. A missing
    verdict (the probe did not run or did not complete) and a fixed slot both
    keep rung 2.5 off the pair: every tie breaks toward keeping.
    """
    return slot is SlotVerdict.CHANGES


def offers_subject_pair(slot: SlotVerdict | None) -> bool:
    """Whether a confirmed subject-pool pair is offered to the ladder.

    The subject-keyed search exists to find updates, so a pair the update probe
    says fills different slots is not offered and both claims stay ACTIVE. A
    same-slot pair is offered whatever the slot's kind: one that changes over
    time reaches rung 2.5, and a fixed one, whose two values contradict,
    reaches rung 3. A pair the probe gave no verdict on is offered,
    as before the probe existed; the ladder then keeps rung 2.5 off it.
    """
    return slot is not SlotVerdict.DIFFERENT


class SweepAction(StrEnum):
    """What the update sweep does with one confirmed contradiction."""

    DEMOTE = "demote"
    """Rung 2.5: the older claim is retired ``SUPERSEDED_BY_UPDATE``."""
    REVIEW = "review"
    """Rung 3: an INCONSISTENCY record names both claims, which stay ACTIVE."""
    KEEP = "keep"
    """Nothing is written: the claims fill different slots, or no verdict."""


def sweep_action(slot: SlotVerdict | None) -> SweepAction:
    """The update sweep's action on a pair the contradiction probe confirmed.

    The sweep is a maintenance pass over claims already in recall, so its
    rung 3 is disclosure: the record is opened for review and no status
    changes, as at the nightly census. A pair the slot probe did
    not answer is kept unchanged.
    """
    if admits_update(slot):
        return SweepAction.DEMOTE
    if slot is SlotVerdict.FIXED:
        return SweepAction.REVIEW
    return SweepAction.KEEP


class UpdateOrderSource(StrEnum):
    """Which rung 2.5 input the caller gathers for a pair."""

    UPDATE = "update"
    """Extraction's same-lineage update order."""
    OWN_ASSERTION = "own_assertion"
    """The assertion pathway's own-assertion order."""


def ladder_signal(probe: bool | None, *, fail_closed: bool) -> bool:
    """The contradiction signal the ladder runs on, from a tri-state probe.

    ``True``/``False`` is a verdict. ``None`` means the probe could not
    complete: extraction stays fail-open (no signal), and the assertion pathway
    fails closed (a signal, which :func:`forces_inconsistent` then overrides to
    ``INCONSISTENT``).
    """
    if probe is None:
        return fail_closed
    return probe


def forces_inconsistent(
    probe: bool | None, *, fail_closed: bool, precondition: PairPrecondition
) -> bool:
    """Whether the verdict is ``INCONSISTENT`` whatever the rungs would say.

    Two overrides: an incomplete probe under ``fail_closed``, and
    a project's candidate contesting a global claim with a signal (
    ``REVIEW``). A ``DECLINE`` outranks both; :func:`decide_ladder` checks it
    first.
    """
    if probe is None and fail_closed:
        return True
    return precondition is PairPrecondition.REVIEW and ladder_signal(probe, fail_closed=fail_closed)


def needs_rung_inputs(
    probe: bool | None, *, fail_closed: bool, precondition: PairPrecondition
) -> bool:
    """Whether the caller should gather :class:`RungInputs` for the pair.

    Only a set signal reaches the rungs, and a declined pair reaches none. A
    forced ``INCONSISTENT`` still gathers: the domain read alongside the trust
    scores is the INCONSISTENCY record's ``domain_hint``.
    """
    if precondition is PairPrecondition.DECLINE:
        return False
    return ladder_signal(probe, fail_closed=fail_closed)


def update_order_source(
    *,
    allow_update: bool,
    allow_own_assertion: bool,
    forced: bool,
    slot: SlotVerdict | None,
) -> UpdateOrderSource | None:
    """Which rung 2.5 input applies, if any.

    ``allow_update`` is extraction's door with the config switch already folded
    in; ``allow_own_assertion`` the assertion pathway's narrower one.
    A forced ``INCONSISTENT`` consults neither.

    ``slot`` is the update probe's verdict on the pair. A contradiction alone
    is not an update, so neither door opens unless both claims fill one slot
    (the same attribute of the same subject) and the later one states a new
    value for it, and that slot holds one value at a time and
    changes over time. A fixed slot (an author, a page count) given two values
    is a contradiction no date settles, and falls through to rung 3 like a
    different-slot pair.
    """
    if forced or not admits_update(slot):
        return None
    if allow_update:
        return UpdateOrderSource.UPDATE
    if allow_own_assertion:
        return UpdateOrderSource.OWN_ASSERTION
    return None


def effective_single_trust_order(override: bool | None, store_mode: str) -> bool:
    """The trust regime the ladder runs under.

    A caller's explicit ``override`` wins; otherwise the store mode decides.
    """
    if override is not None:
        return override
    return store_mode == "single"


def decide_ladder(
    existing: Particle,
    new: Particle,
    *,
    probe: bool | None,
    fail_closed: bool,
    precondition: PairPrecondition,
    rung_inputs: RungInputs | None,
    single_trust_order: bool,
    trust_differential_threshold: float,
) -> LadderOutcome:
    """The §6.6 ladder for one pair, with every override applied in order.

    Pure: the caller gathers the probe, the precondition and (when
    :func:`needs_rung_inputs`) the rung inputs, then applies the outcome. The
    override order is:

      1. ``DECLINE``: no rung runs, whatever the probe said. The
         divergence is recorded when the signal is set, a fail-closed
         incomplete probe included.
      2. a forced ``INCONSISTENT`` (:func:`forces_inconsistent`): the rungs are
         skipped.

    Neither override manufactures anything for a generic-against-instance
    pair (:func:`is_generic_instance_pair`): a declined one records
    no divergence, and a forced one is ``CORROBORATES``.
      3. otherwise :func:`resolve_conflict` on the signal and the rung inputs.
    """
    signal = ladder_signal(probe, fail_closed=fail_closed)
    # A generic and an instance claim are not an adjudicable pair:
    # neither override may record a divergence or force an INCONSISTENCY for
    # it. A forced pair corroborates outright rather than reaching the rungs,
    # so a failed probe never feeds the supersession prior either.
    adjudicable = not is_generic_instance_pair(existing, new)
    if precondition is PairPrecondition.DECLINE:
        return LadderOutcome(verdict=None, record_divergence=signal and adjudicable)
    if forces_inconsistent(probe, fail_closed=fail_closed, precondition=precondition):
        verdict = ConflictVerdict.INCONSISTENT if adjudicable else ConflictVerdict.CORROBORATES
        return LadderOutcome(verdict=verdict, record_divergence=False)
    inputs = rung_inputs if rung_inputs is not None else RungInputs()
    verdict = resolve_conflict(
        existing,
        new,
        has_contradiction_signal=signal,
        new_supersedes_existing=inputs.new_supersedes_existing,
        existing_supersedes_new=inputs.existing_supersedes_new,
        trust_score_existing=inputs.trust_score_existing,
        trust_score_new=inputs.trust_score_new,
        trust_differential_threshold=trust_differential_threshold,
        single_trust_order=single_trust_order,
        update_order=inputs.update_order,
    )
    return LadderOutcome(verdict=verdict, record_divergence=False)


# the ``properties`` marker on an INCONSISTENCY record whose
# "Particle A" is a judgment-retired twin rather than a live claim. Its value
# is the retired twin's ``status_reason``. Review reads it to know that
# PREFER_A means "the retirement stands" and PREFER_B means "lift it" — and
# that neither is a statement about a *source*, so no trust statement is
# written and no cascade runs.
RETIRED_VALUE_KEY = "conflict:retired_value"


def build_inconsistency_particle(
    existing: Particle,
    new: Particle,
    *,
    corpus_entry_id: str,
    snapshot_id: str,
    asserted_by: str = "extract-pipeline",
    trigger_ref_type: ProvenanceRefType = ProvenanceRefType.SOURCE,
    retired_twin: bool = False,
) -> Particle:
    """Construct the INCONSISTENCY ``Particle`` for an unresolvable pair.

    Pure function — no I/O, no DB writes. The caller persists the returned
    particle via the store layer (typically with a ``domain_hint`` for the
    Extension B cascade).

    Field choices (all are normative — §6.4 rung 3 and §9.2 step 7):

      - ``content``: a fixed template summarising both claims, with the
        existing particle's ID quoted so the audit trail survives even if
        either provenance edge is later lost.
      - ``confidence.value``: the lower of the two inputs — an
        INCONSISTENCY is no more certain than its weakest constituent.
      - ``uncertainty_nature``: ``EPISTEMIC``. The INCONSISTENCY is itself a
        claim about the state of the corpus; it can be reduced by review.
      - ``provenance``: two ``PARTICLE`` refs pointing to the conflicting
        originals, followed by one trigger ref to the corpus entry that
        triggered the conflict. The cascade resolver (``operations/cascade``)
        reads the first two refs as particle A and B respectively, so order
        matters. The trigger ref is ``SOURCE``-typed by default;
        ``trigger_ref_type=PARTICLE`` keeps the ref type honest when the
        conflicting candidate has no corpus provenance at all (a derived
        particle, whose refs are all PARTICLE-typed premise links —
        the caller then passes a particle id as ``corpus_entry_id`` per the
        field-reuse convention).
      - ``subject_ids``: **inherited from the existing (ACTIVE) particle**.
        Bug fix from the prior pipeline implementation, which left this empty
        and broke subject-filtered queries that should have surfaced the
        INCONSISTENCY. The existing particle has subject IDs resolved at its
        original extraction time; the new candidate's IDs are usually
        equivalent (same claim, same subjects) but may be incomplete if
        resolution failed for the candidate. Picking ``existing`` gives the
        stable, already-vetted set. If the existing particle has no
        subject_ids (older row or pre-Subject-store extraction), fall back
        to the new particle's set.
      - ``status``: ``Status.INCONSISTENCY``. The caller still goes through
        ``validate_transition(None, Status.INCONSISTENCY)`` before insertion.
      - ``retired_twin``: ``existing`` is not a live claim but a
        particle retired by judgment whose exact twin ``new`` re-asserts. The
        headline says so, and :data:`RETIRED_VALUE_KEY` carries the twin's
        ``status_reason`` in ``properties`` so Review can tell the two kinds
        of record apart.
    """
    # Inherit subject_ids from the existing particle (Particle A). Fall back
    # to the new particle's set if existing has none.
    subject_ids = list(existing.subject_ids) if existing.subject_ids else list(new.subject_ids)

    properties: dict[str, Any] | None = None
    if retired_twin:
        reason = existing.status_reason.value if existing.status_reason else existing.status.value
        inc_content = (
            f"INCONSISTENCY: a candidate re-asserts a claim retired by judgment "
            f"({reason}).\n"
            f"Particle A (retired): {existing.id} — {existing.content[:120]}\n"
            f"Particle B (new, quarantined): {new.content[:120]}"
        )
        properties = {RETIRED_VALUE_KEY: reason}
    else:
        inc_content = (
            f"INCONSISTENCY: conflict between two claims.\n"
            f"Particle A: {existing.id} — {existing.content[:120]}\n"
            f"Particle B (new): {new.content[:120]}"
        )

    return Particle(
        content=inc_content,
        confidence=Confidence(
            value=min(existing.confidence.value, new.confidence.value),
            calibration_source=CalibrationSource.EXTRACTOR_DIRECT,
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            # Cascade convention: first two PARTICLE refs are A then B.
            # ``corpus_entry_id`` carries the particle UUID here — the field
            # name is legacy (review.py and cascade.py both read it this way).
            ProvenanceRef(
                type=ProvenanceRefType.PARTICLE,
                corpus_entry_id=existing.id,
                snapshot_id=existing.id,
            ),
            ProvenanceRef(
                type=ProvenanceRefType.PARTICLE,
                corpus_entry_id=new.id,
                snapshot_id=new.id,
            ),
            ProvenanceRef(
                type=trigger_ref_type,
                corpus_entry_id=corpus_entry_id,
                snapshot_id=snapshot_id or None,
            ),
        ],
        asserted_by=asserted_by,
        status=Status.INCONSISTENCY,
        subject_ids=subject_ids,
        properties=properties,
        schema_version=SCHEMA_VERSION,
    )
