# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""World time on a claim: the window predicate and the prose-date default.

The pure half of the valid-time axis. A claim carries ``valid_from`` (when it
begins to hold) and ``valid_kind`` (a moment or a span) beside the
``valid_until`` it already had; this module answers, with no I/O, whether a
claim's interval sits in a requested window and under which reading, and
reads a default ``valid_from`` off the absolute dates the extractor already
writes into claim prose.

It is a membership predicate, never a weight: nothing here touches
``effective_confidence`` or the rank formula. The Engine half, which loads
the candidates, closes a retired claim's interval and lets a restatement stand
in for its original, is ``particles.operations.query.window``.

How a value was set rides on ``properties`` under the ``extraction:`` prefix
the validity crumbs use: ``valid_from_basis`` (which path set it),
``valid_from_cue`` (the verbatim cue), ``valid_from_confidence`` (the
extractor's self-assessed certainty), ``valid_from_granularity`` (for a
coarse anchor) and ``valid_from_rule`` (the digest of the cue rules a prose
default came from, which is what makes it stale).
"""

from __future__ import annotations

import calendar
import hashlib
import inspect
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Literal

from particles.core.schema import TimeWindow, ValidKind, WindowReading

VALID_FROM_BASIS_KEY = "extraction:valid_from_basis"
VALID_FROM_CUE_KEY = "extraction:valid_from_cue"
VALID_FROM_CONFIDENCE_KEY = "extraction:valid_from_confidence"
VALID_FROM_GRANULARITY_KEY = "extraction:valid_from_granularity"
VALID_FROM_RULE_KEY = "extraction:valid_from_rule"

#: The keys a validity record writes; cleared together when one is replaced.
VALID_FROM_PROPERTY_KEYS: tuple[str, ...] = (
    VALID_FROM_BASIS_KEY,
    VALID_FROM_CUE_KEY,
    VALID_FROM_CONFIDENCE_KEY,
    VALID_FROM_GRANULARITY_KEY,
    VALID_FROM_RULE_KEY,
)


class ValidFromBasis(StrEnum):
    """Which path set a claim's ``valid_from``."""

    STATED = "stated"
    """The extractor, from a cue in the source text."""
    PROSE = "prose"
    """The deterministic default, from an absolute date in the claim's prose."""
    AS_OF = "as-of"
    """A prose ``as of <date>`` cue: known to hold at the date, start unknown."""
    COARSE = "coarse"
    """A prose ``in <year>`` / ``in <month> <year>`` cue: the unit is the span."""
    OPERATOR = "operator"
    """Set by an operator or an authorized agent."""


#: Bases a regeneration pass may overwrite. ``stated`` and ``operator`` are
#: never touched by the prose default.
REGENERABLE_BASES: frozenset[str] = frozenset(
    {ValidFromBasis.PROSE, ValidFromBasis.AS_OF, ValidFromBasis.COARSE}
)

Granularity = Literal["day", "month", "year"]
CloseBasis = Literal["valid_until", "successor", "retired_at"]


def ensure_utc(dt: datetime) -> datetime:
    """Assume UTC for a naive timestamp (the store's existing normalization)."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


# ---------------------------------------------------------------------------
# The interval and the predicate.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidityFacts:
    """What the predicate needs to know about one claim."""

    valid_from: datetime | None
    valid_kind: ValidKind | None
    valid_until: datetime | None
    basis: str | None = None
    granularity: Granularity | None = None
    #: The close the Engine resolved for a retired STATE, or
    #: None for an ACTIVE claim or one the ladder could not date.
    closed_at: datetime | None = None


def coarse_end(start: datetime, granularity: Granularity | None) -> datetime:
    """The last instant of the unit a coarse anchor names."""
    start = ensure_utc(start)
    if granularity == "year":
        return start.replace(month=12, day=31, hour=23, minute=59, second=59)
    if granularity == "month":
        last = calendar.monthrange(start.year, start.month)[1]
        return start.replace(day=last, hour=23, minute=59, second=59)
    return start.replace(hour=23, minute=59, second=59)


def interval(facts: ValidityFacts) -> tuple[datetime, datetime | None] | None:
    """The world-time interval a claim is about; None when it is undated.

    An EVENT is the point ``[valid_from, valid_from]``; a coarse EVENT ("in
    1999") happened somewhere in its unit, so its interval is the unit. A
    STATE runs from its start until the earliest known close: its own
    ``valid_until``, the close the Engine resolved, or open; a coarse STATE
    spans its unit. An unkinded claim has a start and no interval: the caller
    treats it as ``BEGAN``-only.
    """
    if facts.valid_from is None:
        return None
    start = ensure_utc(facts.valid_from)
    if facts.valid_kind is ValidKind.EVENT:
        if facts.basis == ValidFromBasis.COARSE:
            return start, coarse_end(start, facts.granularity)
        return start, start
    if facts.valid_kind is None:
        return None
    ends = [ensure_utc(e) for e in (facts.valid_until, facts.closed_at) if e is not None]
    if facts.basis == ValidFromBasis.COARSE:
        ends.append(coarse_end(start, facts.granularity))
    return start, (min(ends) if ends else None)


def _window_bounds(window: TimeWindow) -> tuple[datetime | None, datetime | None]:
    start = ensure_utc(window.start) if window.start is not None else None
    end = ensure_utc(window.end) if window.end is not None else None
    return start, end


def _intersects(
    start: datetime, end: datetime | None, w_start: datetime | None, w_end: datetime | None
) -> bool:
    if w_end is not None and start > w_end:
        return False
    return not (end is not None and w_start is not None and end < w_start)


def read_window(facts: ValidityFacts, window: TimeWindow) -> WindowReading | None:
    """Under which reading, if any, a claim sits in ``window``.

    ``BEGAN``: ``valid_from`` in the window, any kind, except a basis-``as-of``
    claim whose start is unknown and a ``coarse`` anchor whose day is not
    known. ``IN_FORCE``: a STATE whose interval intersects the window and
    started before it. ``HELD_AT`` is the same for the ``as-of`` basis;
    ``WITHIN`` is a coarse anchor (an EVENT somewhere in a month or a year,
    or a STATE spanning one) whose unit intersects the window. An unkinded
    claim matches ``BEGAN`` only. The point form (``--on``) admits an EVENT
    on the date (a coarse one when the date lies in its unit) and a STATE
    whose interval covers it.
    """
    if facts.valid_from is None:
        return None
    start = ensure_utc(facts.valid_from)
    w_start, w_end = _window_bounds(window)
    unknown_start = facts.basis == ValidFromBasis.AS_OF
    coarse = facts.basis == ValidFromBasis.COARSE

    if window.is_point:
        at = w_start
        assert at is not None  # is_point guarantees both ends
        if facts.valid_kind is ValidKind.EVENT and not coarse:
            return WindowReading.BEGAN if start.date() == at.date() else None
        span = interval(facts)
        if span is None:
            return None
        s, e = span
        if s <= at and (e is None or at <= e):
            if facts.basis == ValidFromBasis.AS_OF:
                return WindowReading.HELD_AT
            if facts.basis == ValidFromBasis.COARSE:
                return WindowReading.WITHIN
            return WindowReading.BEGAN if s.date() == at.date() else WindowReading.IN_FORCE
        return None

    in_window = (w_start is None or start >= w_start) and (w_end is None or start <= w_end)
    if in_window and not unknown_start and not coarse:
        return WindowReading.BEGAN
    if facts.valid_kind is not ValidKind.STATE and not coarse:
        return None
    span = interval(facts)
    if span is None:
        return None
    s, e = span
    if not _intersects(s, e, w_start, w_end):
        return None
    if facts.basis == ValidFromBasis.AS_OF:
        return WindowReading.HELD_AT
    if facts.basis == ValidFromBasis.COARSE:
        return WindowReading.WITHIN
    return WindowReading.IN_FORCE


def resolve_close(
    *,
    valid_until: datetime | None,
    successor_valid_from: datetime | None,
    retired_at: datetime | None,
) -> tuple[datetime, CloseBasis] | None:
    """The earliest-of close of a retired STATE, with its basis.

    Its own ``valid_until``, its successor's ``valid_from`` (when that has a
    stated, prose or operator basis; the caller filters), and its retirement
    instant from the ladder. None when none is known: the caller
    excludes the claim fail-closed and discloses it.
    """
    candidates: list[tuple[datetime, CloseBasis]] = []
    if valid_until is not None:
        candidates.append((ensure_utc(valid_until), "valid_until"))
    if successor_valid_from is not None:
        candidates.append((ensure_utc(successor_valid_from), "successor"))
    if retired_at is not None:
        candidates.append((ensure_utc(retired_at), "retired_at"))
    if not candidates:
        return None
    return min(candidates, key=lambda c: c[0])


# ---------------------------------------------------------------------------
# The prose-date default.
# ---------------------------------------------------------------------------

_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_ISO = r"(?P<iso>\d{4}-\d{2}-\d{2})"
_MONTH_NAMES = "|".join(name for name in calendar.month_name if name)
_MONTH_YEAR = rf"(?P<mon>{_MONTH_NAMES})\s+(?P<my>\d{{4}})"
#: The long form, "August 31, 2023" and "31 August 2023".
_LONG_DATE = (
    rf"(?:(?P<lmon>{_MONTH_NAMES})\s+(?P<lday>\d{{1,2}}),?\s+(?P<lyear>\d{{4}})"
    rf"|(?P<lday2>\d{{1,2}})\s+(?P<lmon2>{_MONTH_NAMES}),?\s+(?P<lyear2>\d{{4}}))"
)
_YEAR = r"(?P<year>(?:19|20)\d{2})"
#: A date that could anchor a cue: ISO, the long form, or a month and year.
_DATE = rf"(?:{_ISO}|{_LONG_DATE}|{_MONTH_YEAR})"

#: Every absolute date mention the parser understands, used to count anchors:
#: a claim with more than one is skipped, since the rule cannot tell which one
#: anchors the claim.
_ANY_DATE = re.compile(rf"\b(?:{_ISO}|{_LONG_DATE}|{_MONTH_YEAR}|(?<![\d-]){_YEAR}(?![\d-]))\b")

#: The recognised cues. Each fixes the kind and the basis. The ``on`` cue
#: allows a weekday between it and the date ("on Thursday, 2026-10-15"). A cue
#: that anchors on the *end* of a period ("in the week before <date>") is
#: deliberately absent (§ Revision notes).
_CUES: tuple[tuple[re.Pattern[str], ValidKind, ValidFromBasis], ...] = (
    (
        re.compile(rf"\bon\s+(?:[A-Z][a-z]+day,?\s+)?{_DATE}", re.IGNORECASE),
        ValidKind.EVENT,
        ValidFromBasis.PROSE,
    ),
    (
        re.compile(
            rf"\b(?:since|from|starting(?:\s+on)?|beginning(?:\s+on)?)\s+{_DATE}",
            re.IGNORECASE,
        ),
        ValidKind.STATE,
        ValidFromBasis.PROSE,
    ),
    (
        re.compile(rf"\bas\s+of\s+{_DATE}", re.IGNORECASE),
        ValidKind.STATE,
        ValidFromBasis.AS_OF,
    ),
    # A coarse anchor dates an act somewhere in the unit ("published in
    # 1999", "filed in June 2025"): an EVENT whose interval is the unit, by
    # the owner's ruling on the 2026-10-13 sample. A span within a unit
    # ("six months in Powai in 2019") is read the same way; the extractor,
    # not this rule, is where the kind is judged.
    (
        re.compile(rf"\bin\s+(?:{_MONTH_YEAR}|{_YEAR})(?![\d-])", re.IGNORECASE),
        ValidKind.EVENT,
        ValidFromBasis.COARSE,
    ),
)


@dataclass(frozen=True)
class ProseAnchor:
    """A default ``valid_from`` read off a claim's prose, with its crumbs."""

    valid_from: datetime
    valid_kind: ValidKind
    basis: ValidFromBasis
    cue: str
    granularity: Granularity


@dataclass(frozen=True)
class ProseSkip:
    """Why the prose default set nothing on a claim."""

    reason: Literal["no_date", "several_dates", "no_cue"]


def _date_from_match(m: re.Match[str]) -> tuple[datetime, Granularity] | None:
    groups = m.groupdict()
    if groups.get("iso"):
        try:
            return datetime.strptime(groups["iso"], "%Y-%m-%d").replace(tzinfo=UTC), "day"
        except ValueError:
            return None
    long_form = (
        (groups.get("lmon"), groups.get("lday"), groups.get("lyear"))
        if groups.get("lmon")
        else (groups.get("lmon2"), groups.get("lday2"), groups.get("lyear2"))
    )
    if all(long_form):
        mon, day, year = long_form
        assert mon is not None and day is not None and year is not None
        try:
            return datetime(int(year), _MONTHS[mon.lower()], int(day), tzinfo=UTC), "day"
        except ValueError:
            return None
    if groups.get("mon") and groups.get("my"):
        month = _MONTHS[groups["mon"].lower()]
        return datetime(int(groups["my"]), month, 1, tzinfo=UTC), "month"
    if groups.get("year"):
        return datetime(int(groups["year"]), 1, 1, tzinfo=UTC), "year"
    return None


def anchor_from_prose(content: str) -> ProseAnchor | ProseSkip:
    """Read the default ``valid_from`` off a claim's prose.

    Sets a value only when exactly one anchoring date is present and it sits
    behind a recognised cue; every other case is a named skip. The date must
    be absolute (the rules resolve relative time before the
    claim is stored), so nothing here reads "yesterday".
    """
    dates = [m for m in _ANY_DATE.finditer(content) if _date_from_match(m) is not None]
    if not dates:
        return ProseSkip("no_date")
    if len({m.group(0) for m in dates}) > 1:
        return ProseSkip("several_dates")
    for pattern, kind, basis in _CUES:
        m = pattern.search(content)
        if m is None:
            continue
        parsed = _date_from_match(m)
        if parsed is None:
            continue
        when, granularity = parsed
        return ProseAnchor(
            valid_from=when, valid_kind=kind, basis=basis, cue=m.group(0), granularity=granularity
        )
    return ProseSkip("no_cue")


def prose_rule_digest() -> str:
    """The digest of the cue rules; stamped on each prose default.

    A default is stale when this differs from the digest it carries. The
    prose never changes, so this is the only thing that can.
    """
    hasher = hashlib.sha256()
    for pattern, kind, basis in _CUES:
        hasher.update(f"{pattern.pattern}|{kind.value}|{basis.value}".encode())
        hasher.update(b"\x1f")
    hasher.update(_ANY_DATE.pattern.encode())
    hasher.update(b"\x1f")
    hasher.update(inspect.getsource(anchor_from_prose).encode())
    return hasher.hexdigest()[:12]


def window_for_day(day: datetime) -> TimeWindow:
    """The ``--on`` form: a point window at the start of ``day`` (UTC)."""
    at = ensure_utc(day).replace(hour=0, minute=0, second=0, microsecond=0)
    return TimeWindow(start=at, end=at)


def window_end_of_day(end: datetime) -> datetime:
    """An inclusive end date read as the last instant of that day (UTC)."""
    end = ensure_utc(end)
    if end.hour or end.minute or end.second or end.microsecond:
        return end
    return end + timedelta(days=1) - timedelta(microseconds=1)
