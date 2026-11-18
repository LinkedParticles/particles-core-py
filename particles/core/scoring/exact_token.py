# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Exact identifier-token match for rank scoring.

A sentence encoder places "RFC 9110" beside "RFC 9112": the digits that tell
two records apart carry almost no weight in the vector. This module supplies
the lexical half of relevance for questions that name something by number. The
query op adds ``μ · X(p, q)`` to its rank score, where ``X`` is the share of the
question's identifier tokens that occur as tokens of the belief's content
(technical specification §9.3 step 4).

**Tokens.** Text is case-folded by Unicode full case folding, with no
normalisation form. A token is a maximal run of letters (L*) and numbers (Nd,
Nl, No), extended across a ``.`` only between two decimal digits, so
``1.234.2`` is one token and ``REQ-0246``, ``REQ 0246`` and ``#0246`` each
yield ``0246``. A token *contains a digit* when it holds a character whose
Numeric_Type is Decimal or Digit (``str.isdigit``). A question's **identifier
tokens** are its tokens that contain a digit and are at least ``min_length``
characters long.

**Locality**. ``X`` reads the belief's own text and the question,
and no count over the store, so it relates a belief to the question the way the
cosine does. A rarity weighting (a token's share of the store) is an aggregate
over the particle set and is ruled out, which is why a common number such as a
year lifts every belief that states it.

This module is pure (no I/O), Client layer. The query op composes
it with the configured weight (``query.exact_token_weight``, default 0, so the
addend is inert until an operator sets it).
"""

from __future__ import annotations

import re

#: A run of letters and numbers (``[^\W_]`` is a word character that is not
#: ``_``), joined across a ``.`` only between digits.
_TOKEN = re.compile(r"[^\W_]+(?:(?<=\d)\.\d+)*")


def identifier_tokens(text: str, min_length: int) -> frozenset[str]:
    """The tokens of ``text`` that contain a digit and are ``min_length`` or longer.

    Case-folded. A question with none leaves every belief's ``X`` at 0, so the
    addend is inert on it.
    """
    return frozenset(
        t
        for t in _TOKEN.findall(text.casefold())
        if len(t) >= min_length and any(c.isdigit() for c in t)
    )


def token_share(content: str, tokens: frozenset[str]) -> float:
    """``X(p, q)``: the share of ``tokens`` that occur as tokens of ``content``.

    From 0 to 1; 0 when ``tokens`` is empty. A substring search for each token
    runs first, and the content is tokenised only on a hit, since most
    candidates contain none of the question's tokens.
    """
    if not tokens:
        return 0.0
    folded = content.casefold()
    if not any(t in folded for t in tokens):
        return 0.0
    own = set(_TOKEN.findall(folded))
    return sum(t in own for t in tokens) / len(tokens)
