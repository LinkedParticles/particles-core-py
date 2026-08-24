# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Claim-granularity soft-gate predicate.

The rule it proxies is the standard's, not this SDK's: §6.2 defines ``content``
as *a single falsifiable natural-language assertion* and defers the granularity
guidance to the whitepaper's §3.4, and §9.4's ``COMPOUND_ASSERTION`` lint
(L-STR-10) reports the beliefs that breach it.

A particle should express one atomic, separately-falsifiable claim. These are
pure, deterministic helpers — no LLM, no I/O, no config read — shared by the
MCP write-surface assert-time gate (``particles/mcp/tools/write.py``) and the
``COMPOUND_ASSERTION`` lint detector
(``particles/operations/lint/assertion_quality.py``) so the two can never
drift. Thresholds are passed in by the caller (read from
``mcp.write.max_assertion_chars`` / ``max_assertion_sentences``); the predicate
itself is config-free.

This is a **size proxy**, not a semantic granularity check: it catches the
compound/multi-paragraph blob but not a single short sentence that still packs
two claims. The principled fix is deferred.
"""

from __future__ import annotations

import re

# A sentence boundary is one-or-more terminators followed by whitespace or EOL.
# Deterministic and conservative: abbreviations ("U.S.") over-count, which is
# acceptable for a generous soft-gate default.
_SENTENCE_BOUNDARY = re.compile(r"[.!?]+(?:\s|$)")


def count_sentences(content: str) -> int:
    """Best-effort count of sentences in ``content`` (deterministic regex)."""
    return len([s for s in _SENTENCE_BOUNDARY.split(content.strip()) if s.strip()])


def granularity_violation(content: str, *, max_chars: int, max_sentences: int) -> str | None:
    """Return a human-readable reason ``content`` breaches the soft-gate, else ``None``.

    A particle should be one atomic, separately-falsifiable claim (§6.2).
    Either threshold ``<= 0`` disables that check (the "off" sentinel).
    """
    n_chars = len(content)
    if max_chars > 0 and n_chars > max_chars:
        return (
            f"content is {n_chars} chars (max {max_chars}) — a particle should be "
            "one atomic, separately-falsifiable claim (whitepaper §3.4)"
        )
    if max_sentences > 0:
        n_sentences = count_sentences(content)
        if n_sentences > max_sentences:
            return (
                f"content spans {n_sentences} sentences (max {max_sentences}) — a "
                "particle should be one atomic, separately-falsifiable claim "
                "(whitepaper §3.4)"
            )
    return None
