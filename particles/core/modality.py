# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The adjudicability default as a stamped record, and the lens that reads it.

``assertion_modality`` is the extraction-time *default* for whether a claim may
be arbitrated against another (the amendment). This module is the pure
half of the three pieces that make that default regenerable and overridable:

- **The stamp.** :class:`ModalityStamp` names the classifier that last wrote a
  claim's value, as an extraction component identity (``<component>@<digest>``),
  plus the model that ran it. :func:`resolve_stamp` reads a stored row's stamp,
  falling back to the minting snapshot's component record when the row carries
  none (extraction never duplicates its stamp), and :func:`stamp_state` judges
  it current, stale, operator, or unclassified.
- **The lens reading.** :func:`effective_modality` composes the adopted lenses'
  ``modality_rules`` over the stored default for one observer, and
  :func:`pair_adjudicable` composes two readings. Both are read-time only: the
  write path reads the stored value through ``is_truth_apt`` and never this.

Pure: plain values in, plain values out. No store, no config, no I/O.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from particles.core.schema import AssertionModality, TrustLensModalityRule

#: The classifier identity an operator verdict carries. Never stale, and never
#: overwritten by regeneration.
OPERATOR_CLASSIFIER = "operator"

#: A value no classifier wrote: an operator or agent assertion supplied it, the
#: extraction had classification off, or the extractor is structured and its
#: claims are adjudicable by construction.
UNCLASSIFIED = "unclassified"

#: A classifying extractor's value from before any record named its rule. It
#: equals no current identity, so it reads stale: nothing recorded which rule
#: produced it.
LEGACY_MODALITY_CLASSIFIER = "legacy-extraction"


class StampState(StrEnum):
    """How a resolved stamp stands against the classifiers in force today."""

    CURRENT = "current"
    STALE = "stale"
    OPERATOR = "operator"
    UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class ModalityStamp:
    """Who last wrote a claim's modality, with what model, and when."""

    classifier: str
    model: str | None = None
    classified_at: datetime | None = None


def classifier_identity(component: str, digest: str) -> str:
    """The identity of a classifier defined by one extraction component."""
    return f"{component}@{digest}"


def resolve_stamp(
    *,
    stored_classifier: str | None,
    stored_model: str | None,
    stored_at: datetime | None,
    extractor_name: str | None,
    provider_model: str | None,
    snapshot_components: Mapping[str, str] | None,
    modality_components: Collection[str],
    classifying_extractors: Collection[str],
) -> ModalityStamp:
    """Resolve one row's stamp (the table).

    Args:
        stored_classifier: the row's ``modality_classifier`` column; set only by
            the regeneration pass and the operator verdict.
        stored_model: the row's ``modality_classifier_model`` column.
        stored_at: the row's ``modality_classified_at`` column.
        extractor_name: ``extractor_ref.name``; ``None`` for an assertion.
        provider_model: the particle's ``extraction_provider_model``.
        snapshot_components: the minting snapshot's exercised components
            (name to digest), or ``None`` when the snapshot has no record.
        modality_components: the component names that carry a modality rule.
        classifying_extractors: the extractors that classify modality at all.
    """
    if stored_classifier is not None:
        return ModalityStamp(stored_classifier, stored_model, stored_at)
    if extractor_name is None:
        return ModalityStamp(UNCLASSIFIED)
    if snapshot_components is not None:
        for name in sorted(modality_components):
            digest = snapshot_components.get(name)
            if digest is not None:
                return ModalityStamp(classifier_identity(name, digest), provider_model)
        return ModalityStamp(UNCLASSIFIED)
    if extractor_name in classifying_extractors:
        return ModalityStamp(LEGACY_MODALITY_CLASSIFIER, provider_model)
    return ModalityStamp(UNCLASSIFIED)


def stamp_state(stamp: ModalityStamp, current: Collection[str]) -> StampState:
    """Judge a resolved stamp against the classifier identities in force.

    The model does not enter the judgment: a provider swap is the
    re-extraction scope, and making every row stale on one would bury the rule
    edits this report exists to show.
    """
    if stamp.classifier == OPERATOR_CLASSIFIER:
        return StampState.OPERATOR
    if stamp.classifier == UNCLASSIFIED:
        return StampState.UNCLASSIFIED
    if stamp.classifier in current:
        return StampState.CURRENT
    return StampState.STALE


# ---------------------------------------------------------------------------
# The lens reading
# ---------------------------------------------------------------------------

#: Rule scopes from most to least specific. Within one lens, the most specific
#: scope its rules match at decides that lens's reading.
SCOPE_ORDER: tuple[str, ...] = ("particle", "subject", "url_pattern", "source_type")

#: Enum declaration order, used to break a tie among non-adjudicable readings.
_MODALITY_ORDER: dict[AssertionModality, int] = {m: i for i, m in enumerate(AssertionModality)}

#: The basis of a reading taken from the stored value with no rule applied.
BASIS_STORED = "stored"
#: The basis of a reading pinned by an operator verdict.
BASIS_OPERATOR = "operator"


def lens_basis(lens_name: str) -> str:
    """The basis label of a reading an adopted lens decided."""
    return f"lens:{lens_name}"


@dataclass(frozen=True)
class ModalityFacts:
    """What the lens composition needs to know about one claim."""

    particle_id: str
    stored: AssertionModality
    operator_pinned: bool = False
    subject_ids: frozenset[str] = frozenset()
    source_types: frozenset[str] = frozenset()
    source_uris: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModalityReading:
    """One observer's reading of a claim's adjudicability default."""

    modality: AssertionModality
    basis: str = BASIS_STORED

    @property
    def adjudicable(self) -> bool:
        """Whether this reading lets the claim be arbitrated against another."""
        return self.modality == AssertionModality.FALSIFIABLE


@dataclass(frozen=True)
class LensModalityRule:
    """An adopted lens's modality rule, tagged with the lens that carries it."""

    lens: str
    rule: TrustLensModalityRule


def _matches(rule: TrustLensModalityRule, facts: ModalityFacts) -> bool:
    if rule.when is not None and rule.when != facts.stored:
        return False
    match rule.scope:
        case "particle":
            return rule.pattern == facts.particle_id
        case "subject":
            return rule.pattern in facts.subject_ids
        case "source_type":
            return rule.pattern in facts.source_types
        case "url_pattern":
            try:
                compiled = re.compile(rule.pattern)
            except re.error:
                return False
            return any(compiled.search(uri) for uri in facts.source_uris)
    return False  # pragma: no cover — the Literal admits no other scope


def _lens_reading(rules: Sequence[LensModalityRule]) -> LensModalityRule:
    """One lens's own reading: its most specific matching rule, abstention breaking ties.

    ``rules`` are one lens's matching rules, non-empty.
    """
    for scope in SCOPE_ORDER:
        at_scope = [r for r in rules if r.rule.scope == scope]
        if at_scope:
            return min(
                at_scope,
                key=lambda r: (
                    r.rule.modality == AssertionModality.FALSIFIABLE,
                    _MODALITY_ORDER[r.rule.modality],
                ),
            )
    raise ValueError("a lens reading needs at least one matching rule")  # pragma: no cover


def effective_modality(facts: ModalityFacts, rules: Iterable[LensModalityRule]) -> ModalityReading:
    """Compose the adopted lenses' rules over one claim's stored default.

    1. An operator verdict wins: no rule overrides it.
    2. Within each lens, its most specific matching rule decides that lens's
       reading (particle, then subject, then URL pattern, then source type),
       so a lens can carve an exception out of its own broader rule.
    3. Across lenses, abstention wins: if any lens reads the claim as not
       adjudicable, it is not, whatever scope another lens's grant was made
       at; ties among non-adjudicable modalities break by enum order, then
       lens name. Adopting a lens can therefore only withdraw adjudication.
    4. Silence leaves the stored default.
    """
    if facts.operator_pinned:
        return ModalityReading(facts.stored, BASIS_OPERATOR)
    by_lens: dict[str, list[LensModalityRule]] = {}
    for r in rules:
        if _matches(r.rule, facts):
            by_lens.setdefault(r.lens, []).append(r)
    if not by_lens:
        return ModalityReading(facts.stored)
    readings = [_lens_reading(lens_rules) for lens_rules in by_lens.values()]
    withheld = [r for r in readings if r.rule.modality != AssertionModality.FALSIFIABLE]
    if withheld:
        decider = min(withheld, key=lambda r: (_MODALITY_ORDER[r.rule.modality], r.lens))
    else:
        decider = min(readings, key=lambda r: r.lens)
    return ModalityReading(decider.rule.modality, lens_basis(decider.lens))


def pair_adjudicable(a: ModalityReading, b: ModalityReading) -> bool:
    """Whether one observer may read a pair as arbitrable: both sides adjudicable."""
    return a.adjudicable and b.adjudicable


def diverges(reading: ModalityReading, stored: AssertionModality) -> bool:
    """Whether a reading's adjudicability differs from the stored default's."""
    return reading.adjudicable != (stored == AssertionModality.FALSIFIABLE)


def rules_from_lenses(
    lenses: Sequence[tuple[str, Sequence[TrustLensModalityRule]]],
) -> list[LensModalityRule]:
    """Flatten ``(lens name, rules)`` pairs into tagged rules."""
    return [LensModalityRule(name, rule) for name, rules in lenses for rule in rules]
