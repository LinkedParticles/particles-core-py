# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Day marks in a speaker-turn transcript, store-free.

A distilled session transcript runs as ``## User`` / ``## Assistant`` turns.
When a turn falls on a later day than the turn before it, its heading carries
that day: ``## User · 2026-09-14`` (a space, U+00B7, a space, the ISO date).
The turns of the first day carry no mark, since the snapshot's own date
already names that day, so a transcript whose turns all fall on one day reads
exactly as it did before marks existed.

Four things live here, so the writer (the distiller) and the readers (the
extractor and the append-only delta read) agree on one syntax:

* :func:`format_turn_heading` writes a heading, marked or plain;
* :func:`escape_mark_lines` escapes a body line shaped like a mark with a
  leading backslash, so a pasted transcript can never date a session;
* :func:`mark_days` and :func:`day_at` say which day is in force at a point
  of a text;
* :func:`reduce_marks` turns a marked text back into its unmarked form, with
  the offset map between the two, so an unmarked snapshot and a marked one of
  the same transcript compare as the same text.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from datetime import date

#: Between the speaker and the date in a mark heading.
MARK_SEPARATOR = " · "

_SPEAKERS = "User|Assistant"

# The ISO date in ASCII digits only. A ``str`` pattern's digit class also
# matches other scripts' digits, which the byte patterns below never do, so a
# line the escape caught would keep its backslash through ``reduce_marks``.
_DATE = "[0-9]{4}-[0-9]{2}-[0-9]{2}"
# A rendered turn begins with its heading and a blank line.
TURN_HEADER = re.compile(rf"## (?:{_SPEAKERS})(?:{MARK_SEPARATOR}{_DATE})?\n\n")
# A mark is a whole line and nothing else; an escaped line starts with a
# backslash and so never matches.
_MARK_RE = re.compile(rf"^## (?:{_SPEAKERS}){MARK_SEPARATOR}({_DATE})$", re.M)
# A body line shaped like a mark, escaped or not: escaping adds one backslash.
_MARK_SHAPED_RE = re.compile(rf"^(\\*## (?:{_SPEAKERS}){MARK_SEPARATOR}{_DATE})$", re.M)

_SEP_BYTES = MARK_SEPARATOR.encode("utf-8")
_DATE_BYTES = _DATE.encode("ascii")
# The same two shapes over raw bytes: group 1 is what reduction removes.
_MARK_BYTES_RE = re.compile(
    rb"^## (?:User|Assistant)(" + re.escape(_SEP_BYTES) + _DATE_BYTES + rb")$", re.M
)
_ESCAPED_BYTES_RE = re.compile(
    rb"^(\\)\\*## (?:User|Assistant)" + re.escape(_SEP_BYTES) + _DATE_BYTES + rb"$", re.M
)


def format_turn_heading(speaker: str, day: date | None = None) -> str:
    """A turn heading: ``## <speaker>``, with ``· <day>`` when ``day`` is given."""
    if day is None:
        return f"## {speaker}"
    return f"## {speaker}{MARK_SEPARATOR}{day.isoformat()}"


def escape_mark_lines(text: str) -> str:
    """``text`` with one more leading backslash on every line shaped like a mark.

    A turn's own words can contain such a line (a pasted transcript, a session
    quoting the record that defines marks). Unescaped, it would read as the
    distiller's mark and date every later act of the session.
    """
    if MARK_SEPARATOR not in text:
        return text
    return _MARK_SHAPED_RE.sub(r"\\\1", text)


def unescape_mark_lines(text: str) -> str:
    """Recover one turn body by undoing a single round of mark escaping."""
    return _MARK_SHAPED_RE.sub(lambda match: match[0].removeprefix("\\"), text)


def mark_days(text: str) -> list[tuple[int, date]]:
    """Each mark in ``text``: the offset its line starts at, and its day, in order."""
    found: list[tuple[int, date]] = []
    if MARK_SEPARATOR not in text:
        return found
    for match in _MARK_RE.finditer(text):
        try:
            found.append((match.start(), date.fromisoformat(match.group(1))))
        except ValueError:
            continue
    return found


def day_at(marks: list[tuple[int, date]], offset: int) -> date | None:
    """The day the last mark starting before ``offset`` names, or ``None`` before any mark."""
    index = bisect.bisect_left(marks, offset, key=lambda m: m[0])
    return marks[index - 1][1] if index else None


@dataclass(frozen=True)
class Reduction:
    """A text with its day marks reduced to plain headings, and the offset map.

    ``text`` is the unmarked form: every mark heading loses its separator and
    date, and every escaped mark-shaped line loses the one backslash the
    escape added. ``cuts`` are the removed spans of the original, as
    ``(start, length)`` in ascending order.
    """

    text: bytes
    cuts: tuple[tuple[int, int], ...]

    def to_reduced(self, raw: int) -> int:
        """The reduced offset of raw offset ``raw`` (a point inside a cut maps to its start)."""
        reduced = raw
        for start, length in self.cuts:
            if raw >= start + length:
                reduced -= length
            elif raw > start:
                reduced -= raw - start
                break
            else:
                break
        return reduced

    def to_raw(self, reduced: int) -> int:
        """The smallest raw offset whose reduced offset is ``reduced``.

        At the start of an escaped line that is the backslash's position, so a
        span cut there keeps the escape with its line.
        """
        shift = 0
        for start, length in self.cuts:
            if reduced <= start - shift:
                break
            shift += length
        return reduced + shift


def reduce_marks(content: bytes) -> Reduction:
    """``content`` with its marks reduced, and the map back (see :class:`Reduction`).

    The identity, with no cuts, on a text that carries no mark and no escape.
    """
    if _SEP_BYTES not in content:
        return Reduction(text=content, cuts=())
    cuts = sorted(
        [(m.start(1), m.end(1) - m.start(1)) for m in _MARK_BYTES_RE.finditer(content)]
        + [(m.start(1), 1) for m in _ESCAPED_BYTES_RE.finditer(content)]
    )
    if not cuts:
        return Reduction(text=content, cuts=())
    parts: list[bytes] = []
    position = 0
    for start, length in cuts:
        parts.append(content[position:start])
        position = start + length
    parts.append(content[position:])
    return Reduction(text=b"".join(parts), cuts=tuple(cuts))


def first_mark_offset(content: bytes) -> int | None:
    """The raw offset of the first mark line in ``content``, or ``None`` when it has none."""
    match = _MARK_BYTES_RE.search(content)
    return match.start() if match else None
