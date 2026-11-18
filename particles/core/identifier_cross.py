# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Whether two claims name different identifiers.

Two claims **cross** when each names a digit-bearing identifier token the
other lacks: ``1.114.1`` against ``1.114.2``, PR 284 against PR 288, a run
that measured 0.75 against one that measured 0.88. Such a pair is almost never
one claim, which is what the duplicate question's ``identifier_cross`` rung
reads (:data:`particles.core.ladders.IDENTIFIER_CROSS`). A pair where only one
side names an identifier is **one-sided**, not crossed: a restatement may
leave a detail out.

Tokens are the exact-identifier addend's
(:func:`particles.core.scoring.exact_token.identifier_tokens`), so
``PR #447`` and ``pull/447`` both yield ``447``. Before comparing, the
formatting that tokenisation does not see through is removed: a ``v`` before
a digit (``v1.70.0``), a thousands separator (``1,024``), a unit after a number
(``1M`` against ``1 million``), an all-zero fraction after a single decimal
point (``$5.00`` against ``$5``; ``3.10`` keeps its zero) and leading zeros on
a whole number (``issue 0042`` against ``issue 42``). The length floor applies to
the normalised token. Two commit ids where one is a prefix of the other agree.

Pure: no I/O, no logging.
"""

from __future__ import annotations

import re
from enum import StrEnum

from particles.core.scoring.exact_token import identifier_tokens

__all__ = [
    "MIN_LENGTH",
    "Relation",
    "crossed",
    "identifier_set",
    "pair_relation",
    "relation",
]

#: The shortest normalised token the rung counts. Part of the rung's version:
#: a different floor is a different rung version.
MIN_LENGTH = 3

#: A thousands separator between digit groups (``7,311``).
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
#: A hex run long enough to be a commit id: digits and letters a-f only, with
#: at least one letter, so a long plain number is never read as one.
_HEX = re.compile(r"^(?=[0-9]*[a-f])[0-9a-f]{7,40}$")
#: A number followed by a unit (``1m``, ``1x``, ``10s``, ``16gb``).
_UNIT = re.compile(r"^(\d+(?:\.\d+)?)(k|m|b|x|s|ms|h|kb|mb|gb|tb)$")
#: A decimal with one point and an all-zero fraction (``5.00``, ``1.0``).
_TRAILING_ZEROS = re.compile(r"^(\d+)\.0+$")


class Relation(StrEnum):
    """How two claims' identifier sets relate."""

    #: The sets are equal (both may be empty).
    EQUAL = "equal"
    #: One set strictly contains the other: one claim names an identifier the
    #: other leaves out.
    ONE_SIDED = "one_sided"
    #: Each side names a token the other lacks.
    CROSSED = "crossed"


def relation(a: frozenset[str], b: frozenset[str]) -> Relation:
    """Classify a pair by its two token sets."""
    if a == b:
        return Relation.EQUAL
    if a <= b or b <= a:
        return Relation.ONE_SIDED
    return Relation.CROSSED


def _norm_token(token: str) -> str:
    """One token with its formatting removed.

    A three-part version (``1.10.0``) is not a decimal and keeps its zeros.
    """
    if len(token) > 1 and token[0] == "v" and token[1].isdigit():
        token = token[1:]
    if unit := _UNIT.match(token):
        token = unit.group(1)
    if zeros := _TRAILING_ZEROS.match(token):
        token = zeros.group(1)
    if token.isdigit():
        token = token.lstrip("0") or "0"
    return token


def identifier_set(text: str, min_length: int = MIN_LENGTH) -> frozenset[str]:
    """A claim's identifier tokens, normalised, at least ``min_length`` long.

    ``$1.00`` and ``1M`` both read as ``1``, so a floor of 2 or more drops
    them together.
    """
    normalised = (_norm_token(t) for t in identifier_tokens(_THOUSANDS.sub("", text), 1))
    return frozenset(t for t in normalised if len(t) >= min_length)


def _commit_prefixes(a: frozenset[str], b: frozenset[str]) -> bool:
    """Whether every token private to either side is a commit-id prefix of one on the other."""
    only_a, only_b = a - b, b - a

    def covered(xs: frozenset[str], ys: frozenset[str]) -> bool:
        return all(
            _HEX.match(x)
            and any(_HEX.match(y) and (y.startswith(x) or x.startswith(y)) for y in ys)
            for x in xs
        )

    return bool(only_a and only_b) and covered(only_a, only_b) and covered(only_b, only_a)


def pair_relation(text_a: str, text_b: str, min_length: int = MIN_LENGTH) -> Relation:
    """How two claims' identifier sets relate; agreeing commit ids count as equal."""
    a, b = identifier_set(text_a, min_length), identifier_set(text_b, min_length)
    rel = relation(a, b)
    if rel is Relation.CROSSED and _commit_prefixes(a, b):
        return Relation.EQUAL
    return rel


def crossed(text_a: str, text_b: str, min_length: int = MIN_LENGTH) -> bool:
    """Whether each claim names an identifier token the other lacks."""
    return pair_relation(text_a, text_b, min_length) is Relation.CROSSED
