# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The shape of an appended provenance observation (D1).

Provenance is append-only: a re-observation of a claim is a new ref, never an
edit of an old one. This module holds the pure rule for what that ref looks
like, so it is testable without a store; the write stays with
``particle_store.append_provenance_ref``, the one provenance writer.
"""

from __future__ import annotations

from particles.core.schema import Particle, ProvenanceRef, ProvenanceRefType


def reobservation_ref(particle: Particle, entry_id: str, snapshot_id: str) -> ProvenanceRef:
    """Return the ref recording that ``snapshot_id`` re-observed ``particle``.

    The ref copies the particle's latest SOURCE ref to ``entry_id`` (its
    location and chunk hash are the carried chunk's) and names the
    re-observing snapshot. A particle with no SOURCE ref to the entry gets a
    bare SOURCE ref naming the entry and snapshot.
    """
    prior = next(
        (
            r
            for r in reversed(particle.provenance)
            if r.type is ProvenanceRefType.SOURCE and r.corpus_entry_id == entry_id
        ),
        None,
    )
    if prior is not None:
        return prior.model_copy(update={"snapshot_id": snapshot_id})
    return ProvenanceRef(
        type=ProvenanceRefType.SOURCE, corpus_entry_id=entry_id, snapshot_id=snapshot_id
    )
