"""Datatypes for the extractor conformance validator.

These types describe the conformance *contract* (what fields are expected of
each extractor) and the validator's *report* (what an actual extractor run
produced). The contract lives in :mod:`particles.conformance.contract`; the
runtime that produces a report lives in :mod:`particles.conformance.validator`.

§"What the validator is *not*", conformance is *completeness*,
not *correctness* — the validator answers *"did the extractor populate this
field?"*, not *"did the extractor populate it with the right value?"*.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class FieldTier(StrEnum):
    """Conformance tier for a particle field.

    REQUIRED   — must be populated on 100 % of outputs; failure exits non-zero.
    RECOMMENDED — should be populated above ``recommended_threshold`` (default 0.8).
    OPTIONAL   — informational only; population is not expected.
    """

    REQUIRED = "REQUIRED"
    RECOMMENDED = "RECOMMENDED"
    OPTIONAL = "OPTIONAL"


@dataclass(frozen=True)
class FieldContract:
    """One row of the conformance contract.

    ``field`` may be a dotted path (``"confidence.calibration_source"``) or
    include a list-indexing token (``"provenance[].snapshot_id"``); see
    :func:`particles.conformance.contract.field_value` for the resolver.
    """

    field: str
    tier: FieldTier
    rationale: str


class DiversitySeverity(StrEnum):
    """What a violated :class:`DiversityRule` costs the extractor.

    FAIL     — the violation joins ``ConformanceReport.failures``; the field's
               tier verdict flips and the run is non-conformant.
    ADVISORY — the violation joins ``ConformanceReport.advisories``; it is
               reported and checked into the baseline but never adjudicates.

    The severity must be stated explicitly on every rule —
    there is deliberately no default, so a future rule cannot inherit the
    ``uncertainty_nature`` answer by omission.
    """

    FAIL = "FAIL"
    ADVISORY = "ADVISORY"


@dataclass(frozen=True)
class DiversityRule:
    """A constraint that a field must take at least N distinct values.

    Applied on top of any tier rule. The motivating case is
    ``uncertainty_nature``: the field is REQUIRED so every particle has *some*
    value, but an extractor that always emits ``EPISTEMIC`` is technically
    populating it while giving queries no signal.

    ``severity`` decides whether a violation adjudicates or merely
    reports. It is required: the ``uncertainty_nature`` rule is ADVISORY
    because its outcome tracks the *class* of source an extractor reads rather
    than the extractor's completeness, but that is a finding about that field,
    not a default for the mechanism.
    """

    field: str
    min_distinct_values: int
    severity: DiversitySeverity
    rationale: str


@dataclass
class FieldStat:
    """Per-field statistics from one validator run.

    ``populated_count`` and ``rate`` describe non-null/non-default presence.
    ``distinct_values`` is filled in for enum-typed fields and matters for
    :class:`DiversityRule`; for non-enum fields it stays at zero.

    ``value_counts`` is the per-value histogram behind
    ``distinct_values``, populated for enum-typed fields and empty otherwise.
    A bare distinct count cannot say whether a passing diversity result held by
    one particle or by half of them, and on a non-deterministic LLM-backed
    extractor that margin is the whole signal.

    ``advisory_reason`` carries a violated ADVISORY-severity diversity rule.
    It never affects ``passes_threshold`` — that is what makes it advisory.

    ``excluded_count`` is how many of the run's particles this
    field's measurement did **not** apply to — the techspec §9 zero-subject
    populations for ``subject_ids``, zero for every other field. ``total_count``
    is the measured denominator *after* the exclusion, so ``populated_count +
    excluded_count`` need not equal the run's ``particle_count``. Two reports
    are comparable only when their exclusion rules match, which is one more
    reason the contract file is the versioned thing a report is read against.
    """

    field: str
    tier: FieldTier
    populated_count: int
    total_count: int
    rate: float
    distinct_values: int
    passes_threshold: bool
    failure_reason: str | None = None
    value_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    advisory_reason: str | None = None
    excluded_count: int = 0


@dataclass
class ConformanceReport:
    """The full output of one ``validate_extractor()`` invocation.

    ``failures`` lists every REQUIRED field with ``rate < 1.0`` or any field
    that violated a ``FAIL``-severity diversity rule. ``warnings`` lists every
    RECOMMENDED field with ``rate < threshold``. ``advisories`` lists every
    field that violated an ``ADVISORY``-severity diversity rule —
    reported, never adjudicating. All three lists are subsets of ``fields``;
    they are surfaced separately so renderers can highlight them cheaply.

    ``fixture_corpus_hash`` is the SHA-256 of the fixture manifest the run
    consumed. Two reports for the same extractor are only comparable if
    their hashes match; this prevents silent drift when the fixture set
    changes.
    """

    extractor_id: str
    extractor_version: str
    fixture_count: int
    particle_count: int
    fields: list[FieldStat]
    failures: list[FieldStat]
    warnings: list[FieldStat]
    generated_at: datetime
    fixture_corpus_hash: str
    recommended_threshold: float = 0.8
    # the "<provider>:<model>" pairing that produced the particles
    # this report scored — the disclosure key, the same string
    # stamps on each particle, aggregated up from them.
    #
    # ``None`` when no scored particle carries one, which at *report* scope
    # means the extractor made no completion call: a report covers only
    # particles minted during its own run, so other branch
    # ("the particle predates the field") is unreachable here. That is what
    # makes the null a usable deterministic-vs-LLM-derived discriminator —
    # and why the value must never be derived from *stored* particles, which
    # would reimport the ambiguity. Reaching the network is not the test:
    # ``wikidata-extractor`` fetches labels and is still a deterministic
    # parse, so it reports ``None``.
    extraction_provider_model: str | None = None
    quality_notes: list[str] = field(default_factory=list)
    advisories: list[FieldStat] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True iff no REQUIRED failures and no FAIL-severity diversity violations.

        Advisories never count: a violated ADVISORY rule is an
        observation about which class of source the extractor reads, not a
        completeness deficiency.
        """
        return not self.failures

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    @property
    def advisory_count(self) -> int:
        return len(self.advisories)
