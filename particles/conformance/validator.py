# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The extractor conformance validator.

``validate_extractor(extractor_id, ...)`` runs a registered extractor against
its fixtures in the fixture corpus and produces a :class:`ConformanceReport`.
The report classifies every field in the conformance :mod:`contract` as
PASS / WARN / FAIL based on tier-specific thresholds and diversity rules. A
diversity rule declared ADVISORY reports its violation in
``report.advisories`` without changing any field's verdict.

**Which fixtures are "its" fixtures.** A fixture is in an
extractor's set iff the production registry would *select* that extractor for
the fixture's ``source_type`` — the ladder, read back through
``registry.selects``. It is deliberately **not** ``extractor.accepts(...)``:
``GeneralExtractor.accepts()`` is unconditionally True because it is the fallback, so the accepts() predicate hands the fallback every fixture in
the corpus and reports its REQUIRED-field coverage over inputs production will
never route it. ``all_accepted=True`` restores the wide set for the deliberate
"what would the fallback do with a Wikibase blob?" run; the caller must not
persist the verdict from such a run (see ``cli/extractor.py``).

Phase 1: report-only. The validator never modifies the
particle store, never registers extractors, and never auto-discounts trust
weights. It exists so operators and CI can answer *"is this extractor
populating the schema fields it should?"* with a single command.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from particles.conformance.contract import (
    CONTRACT,
    DIVERSITY,
    ENUM_FIELDS,
    FIELD_EXEMPTIONS,
    field_value,
    is_populated,
)
from particles.conformance.fixtures import (
    DEFAULT_FIXTURE_DIR,
    Fixture,
    compute_corpus_hash,
    iter_fixtures,
)
from particles.conformance.types import (
    ConformanceReport,
    DiversityRule,
    DiversitySeverity,
    FieldContract,
    FieldStat,
    FieldTier,
)
from particles.core.schema import ExtractorRef, Particle
from particles.extraction.general import CandidateParticle, candidate_to_particle
from particles.extraction.property_keys import bare_properties_keys
from particles.extraction.registry import ExtractorPlugin, get_extractors, selects

log = logging.getLogger(__name__)


class ExtractorNotFoundError(ValueError):
    """Raised when ``validate_extractor`` is given an unknown extractor_id."""


def _bare_properties_keys(candidate: CandidateParticle) -> list[str]:
    """Return keys in ``candidate.properties`` that lack a ``:`` separator.

    The spec requires every ``Particle.properties`` key to use a
    ``prefix:LocalName`` shape so consumers can determine provenance and
    applicability from the key alone. The validator surfaces violations as
    warning-level quality notes; Phase 1 is report-only.

    Shares its predicate with the ``L-STR-12`` lint rule, which asks the same
    question of *persisted* particles — this call site only ever sees
    fresh extractor output over a fixture.
    """
    return bare_properties_keys(candidate.properties)


def _find_extractor(extractor_id: str) -> ExtractorPlugin:
    """Locate a registered extractor by ID. Raises ExtractorNotFoundError if absent."""
    for plugin in get_extractors():
        if extractor_id == plugin.EXTRACTOR_ID:
            return plugin
    known = sorted(p.EXTRACTOR_ID for p in get_extractors())
    raise ExtractorNotFoundError(
        f"Unknown extractor_id {extractor_id!r}. Registered extractors: {known}"
    )


def select_fixtures(
    extractor: ExtractorPlugin,
    fixtures: list[Fixture],
    *,
    all_accepted: bool = False,
) -> list[Fixture]:
    """Return the fixtures this extractor is measured on.

    By default a fixture qualifies iff the production registry would route its
    ``source_type`` to ``extractor`` — one predicate, ``registry.selects``,
    shared with the benchmark auto-filter and the extract pipeline, so
    a conformance report cannot silently disagree with production about whose
    contract it measures.

    ``all_accepted=True`` restores the pre-0231 ``accepts()`` set: every fixture
    the extractor *would* take if handed it. That run is a legitimate question
    and stays one flag away, but it is not the extractor's conformance score —
    for the fallback it is the whole corpus by construction.
    """
    if all_accepted:
        return [f for f in fixtures if extractor.accepts(f.source_type)]
    return [f for f in fixtures if selects(extractor, [f.source_type])]


def _selection_note(
    extractor: ExtractorPlugin,
    selected: list[Fixture],
    corpus: list[Fixture],
    *,
    all_accepted: bool,
) -> str:
    """One quality note recording which fixture set produced this report.

    Emitted in both modes and in both output formats: ``ConformanceReport``'s
    shape is unchanged (§6), so ``quality_notes`` is where a
    checked-in baseline JSON says whether it was captured from the routed set
    or a widened one. A baseline that cannot answer that is not comparable.
    """
    if all_accepted:
        unrouted = [f for f in selected if not selects(extractor, [f.source_type])]
        return (
            f"Fixture selection: --all-accepted — {len(selected)} of "
            f"{len(corpus)} corpus fixtures accepted, of which {len(unrouted)} are NOT "
            f"routed to {extractor.EXTRACTOR_ID} in production. Coverage rates below "
            "are not this extractor's conformance score, and the verdict is "
            "not persisted from a widened run."
        )
    return (
        f"Fixture selection: routing precedence — {len(selected)} of "
        f"{len(corpus)} corpus fixtures route to {extractor.EXTRACTOR_ID}."
    )


async def _run_extractor_over_fixtures(
    extractor: ExtractorPlugin,
    fixtures: list[Fixture],
) -> tuple[list[Particle], list[str]]:
    """Run the extractor over the given fixtures; return particles + notes.

    ``fixtures`` is already narrowed by :func:`select_fixtures` — this function
    runs everything it is handed. Candidates are converted into Particle
    instances via the extractor-agnostic ``candidate_to_particle`` helper so
    that schema-field population reflects what the *real* pipeline would
    persist, not just what the extractor emitted as a CandidateParticle.
    """
    particles: list[Particle] = []
    notes: list[str] = []

    for fixture in fixtures:
        try:
            result = await extractor.extract(fixture.snapshot, fixture.content)
        except Exception as exc:  # pragma: no cover — fixture authoring error
            notes.append(f"Fixture {fixture.fixture_id}: extract() raised {exc!r}")
            continue
        ext_ref = ExtractorRef(name=extractor.EXTRACTOR_ID, version=extractor.EXTRACTOR_VERSION)
        for idx, candidate in enumerate(result.candidates):
            for bad_key in _bare_properties_keys(candidate):
                notes.append(
                    f"Fixture {fixture.fixture_id}: properties key "
                    f"{bad_key!r} on candidate {idx} lacks ':' separator "
                    f"(expected prefix:LocalName)"
                )
            # Map candidate.subjects → subject_ids so the conformance report
            # reflects whether the extractor *knew* which subjects the claim
            # is about. The real pipeline resolves these names to Subject
            # UUIDs; for a fixture run we use the names themselves as
            # synthetic IDs — what matters is presence, not identity.
            particles.append(
                candidate_to_particle(
                    candidate,
                    corpus_entry_id="conformance-fixture-entry",
                    snapshot_id=fixture.snapshot.snapshot_id,
                    asserted_by=extractor.EXTRACTOR_ID,
                    extractor_ref=ext_ref,
                    subject_ids=list(candidate.subjects),
                )
            )
        notes.extend(f"Fixture {fixture.fixture_id}: {n}" for n in result.quality_notes)

    return particles, notes


def _derive_provider_model(particles: list[Particle]) -> tuple[str | None, list[str]]:
    """Return the pairing that produced these particles, plus any anomaly notes.

    Per the pairing rule, the value is read off the run's own particles, not
    from ambient config: ``_call_llm`` stamps ``candidate.provider_model`` from
    the one resolved provider that served the call, and
    ``candidate_to_particle`` threads it onto the ``Particle``. So the pairing
    is already here, and reporting it needs neither an LLM call of our own nor
    a wider ``CompletionProvider`` port (that would be *served*-model
    question, which this deliberately does not answer — the stamp is what was
    *requested*).

    ``None`` when nothing carries a pairing: a deterministic extractor made no
    completion call. Reading config here instead would be the tempting
    one-liner and would be wrong for reason — ``get_provider``
    reads live config on every call, so a value read at report scope is not
    guaranteed to be what served the run.

    More than one distinct pairing is an anomaly (§2.4): reachable only via a
    config reload between calls inside one process, which no CLI invocation
    does. It is recorded rather than hidden — sorted and joined — and flagged
    in a note, because such a report is a capture error, not a baseline.
    """
    observed = sorted(
        {p.extraction_provider_model for p in particles if p.extraction_provider_model}
    )
    if not observed:
        return None, []
    if len(observed) == 1:
        return observed[0], []
    joined = ", ".join(observed)
    return joined, [
        f"Mixed extraction pairings in one run: {joined}. The provider resolved "
        "differently between calls (a config reload mid-run); this report is not "
        "a valid baseline capture."
    ]


def _compute_field_stat(
    contract_entry: FieldContract,
    particles: list[Particle],
    diversity_rules: dict[str, DiversityRule],
    recommended_threshold: float,
) -> FieldStat:
    """Compute the populated/distinct/passes_threshold for one contract field."""
    # a field may declare which particles its rate applies to. The
    # measured denominator is what remains — for `subject_ids`, the particles
    # techspec §9 says should carry a subject. Every other field measures over
    # the whole run, exactly as before.
    exempt = FIELD_EXEMPTIONS.get(contract_entry.field)
    measured = [p for p in particles if not exempt(p)] if exempt else particles
    excluded_count = len(particles) - len(measured)

    total = len(measured)
    if total == 0:
        # Unevaluated, not 100 %. Reached two ways — the run produced nothing,
        # or the exclusion removed everything — and neither is evidence about
        # the extractor. `has_evaluable_failure` keeps both out of
        # the trust cap for the same reason.
        return FieldStat(
            field=contract_entry.field,
            tier=contract_entry.tier,
            populated_count=0,
            total_count=0,
            rate=0.0,
            distinct_values=0,
            passes_threshold=contract_entry.tier == FieldTier.OPTIONAL,
            failure_reason=(
                (
                    f"All {excluded_count} particle(s) are exempt from this field "
                    "(techspec §9); cannot evaluate"
                    if excluded_count
                    else "No particles produced; cannot evaluate"
                )
                if contract_entry.tier != FieldTier.OPTIONAL
                else None
            ),
            excluded_count=excluded_count,
        )

    populated_count = 0
    value_counts: dict[str, int] = {}
    is_enum = contract_entry.field in ENUM_FIELDS

    for p in measured:
        value = field_value(p, contract_entry.field)
        if is_populated(value):
            populated_count += 1
        if is_enum and value is not None:
            key = str(value)
            value_counts[key] = value_counts.get(key, 0) + 1

    rate = populated_count / total
    distinct_values = len(value_counts)

    # Tier-specific pass/fail
    passes = True
    failure_reason: str | None = None

    if contract_entry.tier == FieldTier.REQUIRED and rate < 1.0:
        passes = False
        failure_reason = (
            f"Required field populated on {populated_count}/{total} particles "
            f"({rate:.0%}); expected 100%"
        )
    elif contract_entry.tier == FieldTier.RECOMMENDED and rate < recommended_threshold:
        passes = False
        failure_reason = (
            f"Recommended field populated on {populated_count}/{total} particles "
            f"({rate:.0%}); threshold is {recommended_threshold:.0%}"
        )
    # OPTIONAL never fails

    # Diversity rule overlay (applies to REQUIRED and RECOMMENDED only).
    # a violated rule flips the verdict only at FAIL severity. At
    # ADVISORY it is recorded in advisory_reason and passes_threshold is left
    # alone, so the violation reports without adjudicating.
    advisory_reason: str | None = None
    if (
        contract_entry.tier != FieldTier.OPTIONAL
        and contract_entry.field in diversity_rules
        and passes
    ):
        rule = diversity_rules[contract_entry.field]
        if distinct_values < rule.min_distinct_values:
            reason = (
                f"Diversity rule violated: {distinct_values} distinct value(s) "
                f"observed; rule requires {rule.min_distinct_values}. "
                f"Rationale: {rule.rationale}"
            )
            if rule.severity is DiversitySeverity.FAIL:
                passes = False
                failure_reason = reason
            else:
                advisory_reason = reason

    return FieldStat(
        field=contract_entry.field,
        tier=contract_entry.tier,
        populated_count=populated_count,
        total_count=total,
        rate=rate,
        distinct_values=distinct_values,
        passes_threshold=passes,
        failure_reason=failure_reason,
        value_counts=value_counts,
        advisory_reason=advisory_reason,
        excluded_count=excluded_count,
    )


async def validate_extractor(
    extractor_id: str,
    *,
    fixture_dir: Path | None = None,
    recommended_threshold: float = 0.8,
    all_accepted: bool = False,
) -> ConformanceReport:
    """Run the named extractor against its fixtures and return a conformance report.

    Args:
        extractor_id: The ``EXTRACTOR_ID`` constant of a registered extractor.
        fixture_dir: Where to load fixtures from. Defaults to
            ``tests/conformance/fixtures/`` relative to the working directory.
        recommended_threshold: Minimum rate for RECOMMENDED fields, in [0, 1].
        all_accepted: Score every fixture the extractor ``accepts()`` instead of
            only the ones the registry routes to it. The widened
            run is report-only by construction — a caller must not persist the conformance verdict from it.

    Returns:
        A :class:`ConformanceReport` with one :class:`FieldStat` per contract
        entry, plus ``failures`` (REQUIRED with rate<1.0 OR diversity violated)
        and ``warnings`` (RECOMMENDED with rate<threshold) subsets.
        ``quality_notes[0]`` always records which fixture set was scored, and
        ``extraction_provider_model`` the pairing that produced the particles
        (``None`` for a deterministic extractor).

    Raises:
        ExtractorNotFoundError: if ``extractor_id`` does not match any
            currently-registered extractor.
    """
    extractor = _find_extractor(extractor_id)

    corpus_dir = fixture_dir if fixture_dir is not None else DEFAULT_FIXTURE_DIR
    corpus_hash = compute_corpus_hash(corpus_dir)
    corpus = list(iter_fixtures(corpus_dir))
    fixtures = select_fixtures(extractor, corpus, all_accepted=all_accepted)

    particles, fixture_notes = await _run_extractor_over_fixtures(extractor, fixtures)

    quality_notes = [_selection_note(extractor, fixtures, corpus, all_accepted=all_accepted)]
    if not fixtures:
        # "Unknown", not "failed" — has_evaluable_failure() keeps a fixture-less
        # extractor out of the trust cap, and this note is what tells a
        # reader the all-REQUIRED-at-0 % report below is a data-availability gap.
        quality_notes.append(
            f"No corpus fixture is scored for {extractor.EXTRACTOR_ID}; "
            "report covers zero particles"
        )
    quality_notes.extend(fixture_notes)

    provider_model, pairing_notes = _derive_provider_model(particles)
    quality_notes.extend(pairing_notes)

    diversity_by_field = {rule.field: rule for rule in DIVERSITY}
    stats: list[FieldStat] = [
        _compute_field_stat(entry, particles, diversity_by_field, recommended_threshold)
        for entry in CONTRACT
    ]

    failures = [s for s in stats if s.tier == FieldTier.REQUIRED and not s.passes_threshold]
    warnings = [s for s in stats if s.tier == FieldTier.RECOMMENDED and not s.passes_threshold]
    advisories = [s for s in stats if s.advisory_reason is not None]

    return ConformanceReport(
        extractor_id=extractor.EXTRACTOR_ID,
        extractor_version=extractor.EXTRACTOR_VERSION,
        fixture_count=len(fixtures),
        particle_count=len(particles),
        fields=stats,
        failures=failures,
        warnings=warnings,
        generated_at=datetime.now(UTC),
        fixture_corpus_hash=corpus_hash,
        recommended_threshold=recommended_threshold,
        extraction_provider_model=provider_model,
        quality_notes=quality_notes,
        advisories=advisories,
    )


def has_evaluable_failure(report: ConformanceReport) -> bool:
    """True iff the report shows a *genuinely evaluable* REQUIRED failure.

    The trigger for the read-side trust cap. A report with **zero
    particles** (no fixture matched the extractor) is *unknown*, not failed —
    every REQUIRED field "fails" at 0 % coverage there (``_compute_field_stat``
    returns ``passes=False`` with reason "No particles produced"), and that
    data-availability gap must never clamp trust (it would clobber every
    fixture-less extractor — the exact false-negative Phase 2 was deferred
    to avoid). Returns ``True`` only when fixtures actually produced particles
    **and** a REQUIRED field genuinely fell short (low population or a violated
    diversity rule). "Unknown" (no particles) and "passed" both return ``False``.
    """
    return report.particle_count > 0 and bool(report.failures)
