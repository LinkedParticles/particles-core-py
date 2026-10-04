# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The vocabulary report: what a store's claims say about its own ontology.

Pure: no I/O. The Engine gathers plain rows (each claim with the class of the
subject it is about, each subject's class and external links, the operator
event counts) and these functions fold them into a
:class:`~particles.core.schema.VocabularyReport`. The report is a derived view
computed per call and never stored.

Predicates are grouped by the canonical form predicate profiles compare
(:func:`particles.core.predicate_profile.normalise_predicate`), so
``moved to`` and ``moves to`` are one row with both surface forms listed. A
``URI`` predicate is already a vocabulary term and is its own canonical form.
A predicate's alignment is read from the adopted vocabulary documents
(:func:`alignments_from_documents`), which key their terms by the
same canonical form.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from decimal import Decimal

from particles.core.claims import normalize_term, parse_bound
from particles.core.predicate_profile import normalise_predicate
from particles.core.schema import (
    ClaimTerm,
    ExternalRef,
    LinkBand,
    NamespaceAlignment,
    ObjectShape,
    StructuredClaim,
    TermKind,
    VocabularyClassCount,
    VocabularyPredicate,
    VocabularyReport,
    VocabularySurfaceForm,
)
from particles.core.vocabulary import VocabularyDocument

__all__ = [
    "NO_SUBJECT",
    "UNCLASSED",
    "alignments_from_documents",
    "build_vocabulary_report",
    "canonical_form",
    "class_namespace",
    "link_band",
    "object_shape",
    "predicate_rows",
]

#: The class label of a resolved subject that carries no class.
UNCLASSED = "(unclassed)"
#: The class label of a claim whose subject term resolved to no Subject.
NO_SUBJECT = "(unresolved)"

#: The sentinel for a link no particle content could score.
_UNSCORED_SENTINEL = 0.5
_CURIE_PREFIX = re.compile(r"^([A-Za-z][\w.-]*):(?!//)")


def object_shape(term: ClaimTerm) -> ObjectShape:
    """The form of one object term.

    A typed numeric or date literal that parses keeps its type (the
    rule); an untyped literal with no language tag is read by its lexical
    form, so ``"1987"`` is numeric and ``"2026-10-01"`` dated. Anything else
    is text.
    """
    if term.kind is TermKind.URI:
        return ObjectShape.URI
    if term.kind is TermKind.TOKEN:
        return ObjectShape.TOKEN
    value: Decimal | datetime | str | None
    if term.datatype is not None:
        value = normalize_term(term)
    elif term.language is None:
        value = parse_bound(term.value.strip())
    else:
        value = None
    if isinstance(value, Decimal):
        return ObjectShape.NUMERIC
    if isinstance(value, datetime):
        return ObjectShape.DATED
    return ObjectShape.TEXT


def link_band(confidence: float, suppress_threshold: float) -> LinkBand:
    """The band of one external link's confidence."""
    if confidence >= 1.0:
        return LinkBand.ASSERTED
    if confidence == _UNSCORED_SENTINEL:
        return LinkBand.UNSCORED
    if confidence < suppress_threshold:
        return LinkBand.SUPPRESSED
    return LinkBand.SCORED


def canonical_form(predicate: ClaimTerm) -> str:
    """The canonical predicate a surface form groups under.

    A ``URI`` predicate is kept as spelled: it is already a vocabulary term,
    and the normaliser's lowercasing and stemming would corrupt an IRI.
    """
    if predicate.kind is TermKind.URI:
        return predicate.value
    return normalise_predicate(predicate.value)


def class_namespace(subject_class: str) -> str:
    """The namespace a class is minted in: a CURIE prefix, or an IRI's base.

    ``artifact:file`` is in ``artifact``; ``http://ex.org/onto#Coin`` in
    ``http://ex.org/onto#``. A bare name has no namespace and is returned as
    ``(none)``.
    """
    match = _CURIE_PREFIX.match(subject_class)
    if match:
        return match.group(1)
    for sep in ("#", "/"):
        if "://" in subject_class and sep in subject_class.split("://", 1)[1]:
            return subject_class[: subject_class.rindex(sep) + 1]
    return "(none)"


def alignments_from_documents(
    documents: Sequence[VocabularyDocument],
) -> tuple[dict[str, list[str]], str | None]:
    """What the adopted documents say each canonical form is, and which documents said it.

    A form a document's term covers (its own form, or a confirmed alias) gets
    the term itself, as ``prefix:localName on <class>``, then each outward
    alignment of that term, as ``<target> (<match>) on <class>``. A term is
    keyed by form on a class, and a report row lists every class a
    predicate attaches to, so the class is named on each entry. The source is
    ``None`` when no document is adopted.
    """
    if not documents:
        return {}, None
    entries: dict[str, list[str]] = defaultdict(list)
    for doc in documents:
        for term in doc.terms:
            cls = term.subject_class
            found = [f"{doc.prefix}:{term.local_name} on {cls}"]
            found += [f"{a.target} ({a.match.value}) on {cls}" for a in term.alignments]
            for form in sorted(term.forms()):
                entries[form].extend(e for e in found if e not in entries[form])
    source = ", ".join(f"{doc.name} v{doc.version}" for doc in documents)
    return dict(entries), source


def _ranked(counts: Counter[str]) -> list[VocabularyClassCount]:
    return [
        VocabularyClassCount(subject_class=name, count=n)
        for name, n in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _shape_counts(counts: Counter[ObjectShape]) -> dict[ObjectShape, int]:
    return {shape: counts.get(shape, 0) for shape in ObjectShape}


def predicate_rows(
    claims: Iterable[tuple[StructuredClaim, str]],
    alignments: Mapping[str, Sequence[str]] | None = None,
) -> list[VocabularyPredicate]:
    """One row per canonical predicate over ``(claim, subject class label)`` pairs.

    The label is the class of the subject the claim is about, or
    :data:`UNCLASSED` / :data:`NO_SUBJECT`. Each claim counts once in each of
    its row's columns, so every row's surface forms, object shapes and
    subject classes each sum to its claim count. Ordered by claim count
    descending, then canonical form.
    """
    alignments = alignments or {}
    forms: dict[str, Counter[tuple[str, TermKind]]] = defaultdict(Counter)
    shapes: dict[str, Counter[ObjectShape]] = defaultdict(Counter)
    classes: dict[str, Counter[str]] = defaultdict(Counter)
    for claim, class_label in claims:
        key = canonical_form(claim.predicate)
        forms[key][(claim.predicate.value, claim.predicate.kind)] += 1
        shapes[key][object_shape(claim.object)] += 1
        classes[key][class_label] += 1

    rows: list[VocabularyPredicate] = []
    for key, form_counts in forms.items():
        surface = [
            VocabularySurfaceForm(value=value, kind=kind, claim_count=n)
            for (value, kind), n in sorted(
                form_counts.items(), key=lambda item: (-item[1], item[0][0])
            )
        ]
        rows.append(
            VocabularyPredicate(
                canonical=key,
                label=surface[0].value,
                claim_count=sum(form_counts.values()),
                surface_forms=surface,
                object_shapes=_shape_counts(shapes[key]),
                subject_classes=_ranked(classes[key]),
                alignments=list(alignments.get(key, ())),
            )
        )
    rows.sort(key=lambda row: (-row.claim_count, row.canonical))
    return rows


def _subject_alignment(
    subjects: Iterable[tuple[str | None, Sequence[ExternalRef]]],
    suppress_threshold: float,
) -> tuple[int, int, list[NamespaceAlignment], Counter[str]]:
    total = 0
    aligned = 0
    best: dict[str, Counter[LinkBand]] = defaultdict(Counter)
    class_counts: Counter[str] = Counter()
    band_order = list(LinkBand)
    for subject_class, refs in subjects:
        total += 1
        if subject_class:
            class_counts[subject_class] += 1
        if refs:
            aligned += 1
        per_namespace: dict[str, float] = {}
        for ref in refs:
            per_namespace[ref.namespace] = max(
                per_namespace.get(ref.namespace, 0.0), ref.confidence
            )
        for namespace, confidence in per_namespace.items():
            best[namespace][link_band(confidence, suppress_threshold)] += 1
    by_namespace = [
        NamespaceAlignment(
            namespace=namespace,
            subjects=sum(bands.values()),
            bands={band: bands.get(band, 0) for band in band_order},
        )
        for namespace, bands in best.items()
    ]
    by_namespace.sort(key=lambda row: (-row.subjects, row.namespace))
    return total, aligned, by_namespace, class_counts


def build_vocabulary_report(
    claims: Sequence[tuple[StructuredClaim, str]],
    subjects: Iterable[tuple[str | None, Sequence[ExternalRef]]],
    *,
    structured_claims_total: int,
    event_counts: Mapping[str, int],
    decision_types: Sequence[str],
    suppress_threshold: float,
    alignments: Mapping[str, Sequence[str]] | None = None,
    alignment_source: str | None = None,
    as_of: datetime | None = None,
) -> VocabularyReport:
    """Fold the gathered rows into the report.

    ``claims`` is the candidate set the request selected (observer applied by
    the caller); ``subjects`` is every Subject in the store as ``(class,
    external links)``; ``event_counts`` is the operator log by event type, of
    which ``decision_types`` are reported, zero included so an absent kind of
    ruling reads as none rather than as missing.
    """
    rows = predicate_rows(claims, alignments)
    total, aligned, by_namespace, class_counts = _subject_alignment(subjects, suppress_threshold)
    class_namespaces: Counter[str] = Counter()
    for name, n in class_counts.items():
        class_namespaces[class_namespace(name)] += n
    kinds = Counter(claim.object.kind for claim, _ in claims)
    shapes = Counter(object_shape(claim.object) for claim, _ in claims)
    return VocabularyReport(
        subjects_total=total,
        subjects_aligned=aligned,
        aligned_by_namespace=by_namespace,
        link_suppress_threshold=suppress_threshold,
        subjects_classed=sum(class_counts.values()),
        classed_by_class=_ranked(class_counts),
        classed_by_namespace=_ranked(class_namespaces),
        structured_claims_total=structured_claims_total,
        claims_in_view=len(claims),
        object_kinds={kind: kinds.get(kind, 0) for kind in TermKind},
        object_shapes=_shape_counts(shapes),
        predicates_distinct=len({(c.predicate.value, c.predicate.kind) for c, _ in claims}),
        predicates_canonical=len(rows),
        modelling_decisions={t: int(event_counts.get(t, 0)) for t in decision_types},
        alignment_source=alignment_source,
        as_of=as_of,
        predicates=rows,
    )
