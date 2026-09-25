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

from collections.abc import Iterable

from particles.core.schema import ParticleRelation


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


def co_evidential_components(
    edges: Iterable[ParticleRelation], min_confidence: float
) -> dict[str, frozenset[str]]:
    """Group ids into CO_EVIDENTIAL components from the whole edge list at once.

    The in-memory equivalent of running ``get_co_evidential_group`` per particle,
    honouring the same ``effective_equivalence >= min_confidence`` gate, so a
    caller reads the relation table once and decides "already linked" without a
    session (D2). Ids with no qualifying edge are simply
    absent: their component is the singleton the caller supplies.

    Args:
        edges: CO_EVIDENTIAL edges, e.g. ``get_all_relations(session, CO_EVIDENTIAL)``.
        min_confidence: Edges whose effective equivalence falls below this are
            not traversed.

    Returns:
        Every id that sits on a qualifying edge, mapped to its whole component
        (itself included).
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        if effective_equivalence(e.confidence) < min_confidence:
            continue
        a, b = find(e.particle_a), find(e.particle_b)
        if a != b:
            parent[a] = b

    groups: dict[str, set[str]] = {}
    for node in list(parent):
        groups.setdefault(find(node), set()).add(node)
    return {node: frozenset(members) for members in groups.values() for node in members}
