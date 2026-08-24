# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Claim-equivalence math — the §6.10 graded, observer-relative equivalence
lens.

``effective_equivalence`` is the query-time "how much are these the same claim"
lens over an observer-neutral substrate — a CO_EVIDENTIAL edge's link
confidence within a store (§6.10), or a computed similarity across
stores. It is the claim-identity sibling of ``compute_effective_confidence``
(§6.3): an observer-neutral value read through a per-observer trust term at
query time, never stored.

**MVP (this implementation): the lens is the identity** — ``effective_equivalence``
returns the substrate. ``observer_trust`` is a reserved per-observer hook (the
claim-relation analog of ``compute_effective_confidence``'s ``source_trust_rank``);
its exact formula and the per-observer trust-policy surface are deferred, along
with cross-lingual detection and the ``language`` field.
"""

from __future__ import annotations


def effective_equivalence(link_confidence: float, *, observer_trust: float | None = None) -> float:
    """Observer-relative claim-equivalence strength in ``[0, 1]`` (§6.10).

    Args:
        link_confidence: The observer-neutral substrate — a CO_EVIDENTIAL edge's
            confidence, or a computed cross-store similarity.
        observer_trust: Reserved per-observer term. ``None`` (the MVP default)
            yields the identity lens; when supplied it currently discounts
            multiplicatively, but the formula is provisional.

    Returns:
        The equivalence strength, clamped to ``[0, 1]``.
    """
    if observer_trust is None:
        return link_confidence
    return max(0.0, min(1.0, link_confidence * observer_trust))
