# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""How two ``schema_version`` stamps relate (D1).

A stamp is three non-negative integers, ``MAJOR.MINOR.PATCH``. Two stamps are
compatible when their majors are equal and nonzero; major ``0`` and a string
that does not parse are never compatible, which keeps the 0.3.x refusal.
No comparison reads PATCH.

Everything here is pure and string-in, string-out, so the guard, lint,
reindex, the interchange codec and the conformance validator all classify a
stamp the same way. Versions are ordered as integers, never as text:
``"1.10.0"`` sorts after ``"1.9.0"``.
"""

from __future__ import annotations

import re
from enum import StrEnum

_STAMP = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


class SchemaCompatibility(StrEnum):
    """Where a stored stamp sits against the reader's (D1, D2).

    The first three are compatible: the reader reads and writes the record.
    The last three are refused by the guard (D3), each with its own remedy.
    """

    EQUAL = "EQUAL"
    OLDER_MINOR = "OLDER_MINOR"
    NEWER_MINOR = "NEWER_MINOR"
    OLDER_MAJOR = "OLDER_MAJOR"
    NEWER_MAJOR = "NEWER_MAJOR"
    MALFORMED = "MALFORMED"

    @property
    def compatible(self) -> bool:
        """True when the reader may read and write a record at this stamp."""
        return self in _COMPATIBLE


_COMPATIBLE = frozenset(
    {
        SchemaCompatibility.EQUAL,
        SchemaCompatibility.OLDER_MINOR,
        SchemaCompatibility.NEWER_MINOR,
    }
)


def parse_schema_version(stamp: str) -> tuple[int, int, int] | None:
    """``(major, minor, patch)`` for a well-formed stamp, else ``None``."""
    match = _STAMP.match(stamp)
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def schema_compatibility(stored: str, current: str) -> SchemaCompatibility:
    """Classify ``stored`` against the reader's ``current`` stamp.

    Args:
        stored: The stamp on a record (a row, an interchange unit, a bundle).
        current: The reader's own stamp, normally ``SCHEMA_VERSION``.

    Raises:
        ValueError: If ``current`` is itself malformed or major ``0``; the
            reader's own stamp is a programming constant, never data.
    """
    mine = parse_schema_version(current)
    if mine is None or mine[0] == 0:
        raise ValueError(f"reader schema_version {current!r} is not a 1.x-or-later stamp")
    theirs = parse_schema_version(stored)
    if theirs is None:
        return SchemaCompatibility.MALFORMED
    if theirs[0] != mine[0]:
        return (
            SchemaCompatibility.OLDER_MAJOR
            if theirs[0] < mine[0]
            else SchemaCompatibility.NEWER_MAJOR
        )
    if theirs[1] < mine[1]:
        return SchemaCompatibility.OLDER_MINOR
    if theirs[1] > mine[1]:
        return SchemaCompatibility.NEWER_MINOR
    return SchemaCompatibility.EQUAL


def lower_schema_version(record: str, reader: str) -> str:
    """The stamp a record carries once ``reader`` has handled it (D6).

    The lower of the two on ``MAJOR.MINOR``: a reader older than the record
    cannot carry the fields it does not know, so the record becomes one of the
    reader's contract. On a tie the record keeps its own stamp, PATCH and all.
    Callers reject an incompatible pair before asking.
    """
    if schema_compatibility(record, reader) is SchemaCompatibility.NEWER_MINOR:
        return reader
    return record


def major_prefix_range(current: str) -> tuple[str, str]:
    """The half-open text range holding every stamp of ``current``'s major (D3).

    ``("1.", "1/")``: ``/`` is the code point after ``.``, so the range holds
    exactly the strings that begin ``1.``, and ``"10.0.0"`` falls outside it.
    Two range seeks around it ask "any row of another major?" without ordering
    full versions as text.
    """
    parsed = parse_schema_version(current)
    if parsed is None:
        raise ValueError(f"reader schema_version {current!r} is malformed")
    major = parsed[0]
    return f"{major}.", f"{major}/"


def split_by_compatibility(
    counts: dict[str, int], current: str
) -> dict[SchemaCompatibility, dict[str, int]]:
    """Group a ``{stamp: count}`` census by how each stamp relates to ``current``."""
    grouped: dict[SchemaCompatibility, dict[str, int]] = {}
    for stamp, count in counts.items():
        grouped.setdefault(schema_compatibility(stamp, current), {})[stamp] = count
    return grouped


def stamp_sort_key(stamp: str) -> tuple[int, tuple[int, int, int], str]:
    """Order stamps by value, malformed ones last and by text (for messages)."""
    parsed = parse_schema_version(stamp)
    if parsed is None:
        return 1, (0, 0, 0), stamp
    return 0, parsed, stamp
