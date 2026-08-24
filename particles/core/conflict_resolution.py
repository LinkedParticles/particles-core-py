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

The *effect* half of the ladder — DB writes, trust-rank lookups, embedding
similarity computation, the LLM contradiction-signal call — stays in
``particles/extraction/pipeline.py`` because it touches I/O. Core code must
remain pure (see ``particles/core/AGENTS.md``).

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
     *above* the truth-apt gate and the trust rung but *below* the ALEATORY
     exclusion (step 1). Single-trust-order stores only in v1 (matching
     rung 2). The
     ``has_contradiction_signal`` flag is **reframed** on this path as a
     *replacement signal* — "does the superseding claim replace, not merely
     restate, the superseded one?" — and a ``False`` signal keeps both claims
     (the default-safe direction), preserving the
     never-blanket-demote invariant (cap. 2(c)) — the same
     demotion-only rule §6.4 states normatively.
  1.7. **Truth-apt gate** — §6.4 rung 1.7, kept *below* supersession
     . If either side is non-truth-apt, the **truth engine** (the
     contradiction probe, trust arbitration, INCONSISTENCY manufacture) has
     nothing to adjudicate; return CORROBORATES. This gate's *scope* is narrowed
     : it no longer blocks the editorial supersession prior above it,
     only the truth-engine rungs below.
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

from enum import StrEnum

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

    INCONSISTENT = "INCONSISTENT"
    """No clear winner — emit an INCONSISTENCY particle."""

    NO_CONFLICT = "NO_CONFLICT"
    """Pair is not in conflict at all (e.g. below caller's similarity floor)."""


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
            supersession branch (step 1.5) this flag is **reframed** as a
            *replacement signal* — for a non-truth-apt pair it answers "does the
            superseding claim replace, not merely restate, the superseded one?" —
            and ``False`` keeps both claims (the default-safe direction).
        new_supersedes_existing: Step 1.5 input — §6.4, cap. 2.
            ``True`` when
            ``new``'s provenance corpus entry (transitively) supersedes
            ``existing``'s — an authored editorial "this document replaces that
            one". The caller resolves it from the corpus supersession relation.
            This branch runs **above the truth-apt gate** and is
            **modality-independent**, so it retires a superseded ``CONSTITUTIVE``
            definition the truth engine would otherwise never see — but only when
            ``has_contradiction_signal`` (the replacement signal) is ``True`` and
            the pair is not ALEATORY.
        existing_supersedes_new: Step 1.5 input — the mirror direction.
            ``True`` when ``existing``'s document supersedes ``new``'s. Both
            ``True`` (a supersession cycle) fires neither branch and falls
            through to the truth-apt gate / trust rung.
        trust_score_existing: Pre-resolved trust score for ``existing``.
            ``None`` skips the trust rung (the ALEATORY exclusion still applies).
        trust_score_new: Pre-resolved trust score for ``new``. Same treatment.
        trust_differential_threshold: Minimum absolute score gap that lets
            rung 2 auto-resolve. Defaults to 0.15 (matches
            ``config.trust.differential_threshold`` historical default); the
            caller should pass the live value from
            ``get_config().trust.differential_threshold``.
        single_trust_order: Whether the store has a single global trust order
            . ``True`` (default) is today's behavior — rung 2
            auto-supersede may fire. ``False`` is a multi-contributor /
            consensus store, which has no global trust order, so
            rung 2 is **skipped entirely** and a confirmed contradiction falls
            through to ``INCONSISTENT`` (both claims stay ACTIVE, ranked
            per-viewer at query time) — a contributor's claim is never dropped
            by another's trust. The caller passes
            ``get_config().reconciliation.store_mode == "single"``.

    Returns:
        ConflictVerdict — see the enum docstrings for the caller's required
        follow-up action.
    """
    # Step 1 (lifted to the top): ALEATORY exclusion. An irreducibly
    # aleatory pair is never retired by an editorial supersession relation nor by
    # a trust differential; it skips the supersession prior (step 1.5) and the
    # trust rung (step 3) and falls through to INCONSISTENCY. ALEATORY is an
    # ``uncertainty_nature``, orthogonal to modality — this is the ONE exclusion
    # that sits above the supersession prior.
    aleatory = (
        existing.uncertainty_nature == UncertaintyNature.ALEATORY
        or new.uncertainty_nature == UncertaintyNature.ALEATORY
    )

    # Step 1.5 (cap. 2 — ABOVE the truth-apt gate,
    # modality-independent): document-supersession prior. An explicit, authored
    # "this document replaces that one" is an *editorial* fact — it does not
    # depend on either claim's truth-aptness, so it must reach a superseded
    # CONSTITUTIVE definition that the truth engine (step 1.7 onward) cannot see.
    # It therefore runs ABOVE the truth-apt gate. ``has_contradiction_signal`` is
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

    # Step 1.7 (pre-ladder gate, kept BELOW supersession):
    # truth-semantics apply only to FALSIFIABLE particles. If either side is
    # non-truth-apt (an opinion, feeling, or a document's constitutive rule), the
    # truth engine has no shared truth to adjudicate — co-exist, never contradict
    # or trust-supersede. The editorial supersession prior already ran above;
    # this gate now governs only the truth-engine rungs below it. Defense in
    # depth: the pipeline's intra-entry ``_find_conflict`` already declines to
    # pair non-truth-apt particles; the cross-entry supersession sweep
    # is the path that deliberately pairs them, and it reaches step 1.5 above.
    if not (is_truth_apt(existing) and is_truth_apt(new)):
        return ConflictVerdict.CORROBORATES

    # Step 2 (pre-ladder gate): high similarity is not enough on its
    # own. If the caller's probe said the pair is not a contradiction, treat it
    # as corroboration and write both as ACTIVE.
    if not has_contradiction_signal:
        return ConflictVerdict.CORROBORATES

    # Step 3: trust resolution (single-trust-order stores only).
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

    # Step 4: default — INCONSISTENCY particle.
    return ConflictVerdict.INCONSISTENT


def build_inconsistency_particle(
    existing: Particle,
    new: Particle,
    *,
    corpus_entry_id: str,
    snapshot_id: str,
    asserted_by: str = "extract-pipeline",
    trigger_ref_type: ProvenanceRefType = ProvenanceRefType.SOURCE,
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
    """
    # Inherit subject_ids from the existing particle (Particle A). Fall back
    # to the new particle's set if existing has none.
    subject_ids = list(existing.subject_ids) if existing.subject_ids else list(new.subject_ids)

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
        schema_version=SCHEMA_VERSION,
    )
