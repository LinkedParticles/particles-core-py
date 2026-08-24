# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Structural claim filters — pure term normalization + claim matching.

``StructuredClaim`` and ``ClaimTerm`` are the §6.2 derived-annotation fields;
the filter grammar and normalization rules below are this SDK's own read
surface and are not part of the standard, so the numbered section marks in this
module cite the governing decision record and not the
specification.

The Client-layer seam of the structural-claim read surface: a
pure,
store-free, I/O-free normalizer for :class:`~particles.core.schema.ClaimTerm`
object values, and the matcher that applies the filter grammar to
one
:class:`~particles.core.schema.StructuredClaim`. The Engine composition
(scan, aggregates, coverage footer) lives in ``operations/query/structural.py``;
the deferred deterministic contradiction pre-pass reuses this
module's equality semantics rather than growing a parallel second
implementation.

Normalization is read-time only — nothing here is ever stored: no sidecar
column, so nothing can drift against the JSON it summarizes.
No unit
parsing: ``"3 grams"`` as an untyped literal does not compare to ``3.0``; the
honest behaviour is the §2.2 disclosure (``NOT_COMPARABLE``), not a guess.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from particles.core.schema import ClaimTerm, StructuredClaim

_XSD_IRI_PREFIX = "http://www.w3.org/2001/XMLSchema#"
_XSD_CURIE_PREFIX = "xsd:"

# Exactly the datatypes named — the binding list, not a floor.
# Adding e.g. xsd:long or xsd:gYear is a semantics change, not a bug fix.
_NUMERIC_XSD_LOCAL = frozenset({"decimal", "integer", "float", "double"})
_TEMPORAL_XSD_LOCAL = frozenset({"date", "dateTime"})


def _xsd_local_name(datatype: str | None) -> str | None:
    """The XSD local name of a datatype IRI/CURIE, else ``None``.

    Extractors stamp both spellings — Numista writes the ``xsd:`` CURIE, the
    Wikidata extractor the full ``http://www.w3.org/2001/XMLSchema#`` IRI —
    and both name the same datatype, so the normalizer accepts either. (This
    is *not* the §2.2 predicate prefix expansion the ADR forbids: a datatype
    is machinery this module interprets, not a vocabulary term the user
    filters on.)
    """
    if datatype is None:
        return None
    if datatype.startswith(_XSD_IRI_PREFIX):
        return datatype[len(_XSD_IRI_PREFIX) :]
    if datatype.startswith(_XSD_CURIE_PREFIX):
        return datatype[len(_XSD_CURIE_PREFIX) :]
    return None


def _parse_decimal(lexical: str) -> Decimal | None:
    try:
        value = Decimal(lexical)
    except (InvalidOperation, ValueError):
        return None
    # Decimal accepts "NaN" / "Infinity"; neither is a comparable quantity
    # (NaN comparisons raise), so both fall back to the lexical string.
    return value if value.is_finite() else None


def _parse_datetime(lexical: str) -> datetime | None:
    try:
        value = datetime.fromisoformat(lexical)
    except (TypeError, ValueError):
        return None
    # A naive xsd:date / xsd:dateTime is assumed UTC (the store-wide
    # convention), so a typed comparison never mixes naive and aware.
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def normalize_term(term: ClaimTerm) -> Decimal | datetime | str:
    """Normalize one claim term's value at read time.

    Numeric xsd datatypes (``xsd:decimal`` / ``integer`` / ``float`` /
    ``double``) parse to :class:`~decimal.Decimal`; ``xsd:date`` /
    ``xsd:dateTime`` to an aware :class:`~datetime.datetime` (naive values
    assumed UTC); everything else — including a typed literal whose lexical
    form does not actually parse — falls back to the lexical string, which
    compares only by ``eq`` and ``contains``.
    """
    local = _xsd_local_name(term.datatype)
    if local in _NUMERIC_XSD_LOCAL:
        parsed_num = _parse_decimal(term.value)
        if parsed_num is not None:
            return parsed_num
    elif local in _TEMPORAL_XSD_LOCAL:
        parsed_dt = _parse_datetime(term.value)
        if parsed_dt is not None:
            return parsed_dt
    return term.value


def parse_bound(value: str) -> Decimal | datetime | None:
    """Parse a user-supplied comparison bound (``--object-eq/gt/lt`` value).

    A number parses to :class:`~decimal.Decimal`, an ISO-8601 date/datetime to
    an aware :class:`~datetime.datetime`; anything else returns ``None`` — for
    ``gt`` / ``lt`` that is a caller error (the bound itself must be
    comparable), for ``eq`` it means the lexical fallback.
    """
    parsed_num = _parse_decimal(value)
    if parsed_num is not None:
        return parsed_num
    return _parse_datetime(value)


class ClaimMatch(StrEnum):
    """Outcome of matching one claim against a filter set."""

    MATCHED = "MATCHED"
    UNMATCHED = "UNMATCHED"
    #: The claim sits in the filtered slice but its object would not normalize
    #: to a type comparable with the gt/lt bound — counted and disclosed in
    #: the result footer, never silently dropped (§2.2).
    NOT_COMPARABLE = "NOT_COMPARABLE"


@dataclass(frozen=True)
class ClaimFilters:
    """The §2.2 flag grammar as one immutable filter set.

    All conditions intersect (§2.4 — filters only ever narrow). ``predicate``
    is an exact string compared case-insensitively: a CURIE and its expanded
    IRI are different strings in v1 (no prefix expansion, no alias table).
    """

    predicate: str | None = None
    object_eq: str | None = None
    object_gt: str | None = None
    object_lt: str | None = None
    object_contains: str | None = None

    def __bool__(self) -> bool:
        return any(
            v is not None
            for v in (
                self.predicate,
                self.object_eq,
                self.object_gt,
                self.object_lt,
                self.object_contains,
            )
        )


def _eq_matches(term: ClaimTerm, raw: str) -> bool:
    """§2.3 equality: normalized when both sides normalize to the same type,
    else case-insensitive lexical."""
    normalized = normalize_term(term)
    bound = parse_bound(raw)
    if isinstance(normalized, Decimal) and isinstance(bound, Decimal):
        return normalized == bound
    if isinstance(normalized, datetime) and isinstance(bound, datetime):
        return normalized == bound
    return term.value.casefold() == raw.casefold()


def match_claim(claim: StructuredClaim, filters: ClaimFilters) -> ClaimMatch:
    """Apply one §2.2 filter set to one claim.

    Conditions intersect; the string conditions (predicate / contains / eq)
    are evaluated before the typed comparisons so a claim outside the filtered
    slice is ``UNMATCHED``, and only a claim that *would* be in the slice but
    whose object cannot be compared against a gt/lt bound is
    ``NOT_COMPARABLE`` (the disclosed exclusion).

    Raises:
        ValueError: If a gt/lt bound itself does not parse to a comparable
            type — a caller error the surfaces validate before dispatch.
    """
    if filters.predicate is not None and claim.predicate.value.casefold() != (
        filters.predicate.casefold()
    ):
        return ClaimMatch.UNMATCHED
    if filters.object_contains is not None and (
        filters.object_contains.casefold() not in claim.object.value.casefold()
    ):
        return ClaimMatch.UNMATCHED
    if filters.object_eq is not None and not _eq_matches(claim.object, filters.object_eq):
        return ClaimMatch.UNMATCHED

    for raw_bound, is_gt in ((filters.object_gt, True), (filters.object_lt, False)):
        if raw_bound is None:
            continue
        bound = parse_bound(raw_bound)
        if bound is None:
            raise ValueError(
                f"comparison bound {raw_bound!r} is neither a number nor an ISO-8601 date/datetime"
            )
        normalized = normalize_term(claim.object)
        if type(normalized) is not type(bound) or isinstance(normalized, str):
            return ClaimMatch.NOT_COMPARABLE
        # mypy narrows both operands through the same-type check above.
        if is_gt and not normalized > bound:  # type: ignore[operator]
            return ClaimMatch.UNMATCHED
        if not is_gt and not normalized < bound:  # type: ignore[operator]
            return ClaimMatch.UNMATCHED
    return ClaimMatch.MATCHED


def predicate_vocabulary(claims: Iterable[StructuredClaim]) -> list[tuple[str, str, int]]:
    """Distinct predicate terms with kind and claim count (§2.2 ``--predicates``).

    Terms are distinct **as stored** (exact string + kind) — the listing shows
    the heterogeneous vocabulary as it is, so the strict exact-string filter
    has a discovery surface. Ordered by claim count descending, then value.
    """
    counts = Counter((claim.predicate.value, claim.predicate.kind.value) for claim in claims)
    return sorted(
        ((value, kind, n) for (value, kind), n in counts.items()),
        key=lambda item: (-item[2], item[0]),
    )
