# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The carry-forward re-observation ref rule — pure, no store."""

from __future__ import annotations

from particles.core.provenance import reobservation_ref
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)

ENTRY = "entry-1"
NEW_SNAPSHOT = "snap-new"


def _source(entry: str, snapshot: str, location: str | None = None) -> ProvenanceRef:
    return ProvenanceRef(
        type=ProvenanceRefType.SOURCE,
        corpus_entry_id=entry,
        snapshot_id=snapshot,
        location=location,
        chunk_hash=f"hash-{location}" if location else None,
    )


def _particle(*refs: ProvenanceRef) -> Particle:
    return Particle(
        content="a claim",
        confidence=Confidence(value=0.9),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=list(refs),
        asserted_by="general-extractor",
    )


def test_copies_the_latest_ref_to_the_entry_with_the_new_snapshot() -> None:
    prior = _source(ENTRY, "snap-old", location="p3")
    particle = _particle(prior)

    ref = reobservation_ref(particle, ENTRY, NEW_SNAPSHOT)

    assert ref == prior.model_copy(update={"snapshot_id": NEW_SNAPSHOT})
    assert ref.location == "p3"
    assert ref.chunk_hash == "hash-p3"
    # The particle's own ref is not edited: provenance is append-only (D1).
    assert particle.provenance[0].snapshot_id == "snap-old"


def test_no_prior_source_ref_to_the_entry_gives_a_bare_ref() -> None:
    particle = _particle(
        _source("entry-other", "snap-x", location="p1"),
        # A PARTICLE ref carrying the entry id is not a SOURCE observation of it.
        ProvenanceRef(type=ProvenanceRefType.PARTICLE, corpus_entry_id=ENTRY),
    )

    ref = reobservation_ref(particle, ENTRY, NEW_SNAPSHOT)

    assert ref == ProvenanceRef(
        type=ProvenanceRefType.SOURCE, corpus_entry_id=ENTRY, snapshot_id=NEW_SNAPSHOT
    )


def test_the_latest_of_several_refs_to_the_entry_wins() -> None:
    particle = _particle(
        _source(ENTRY, "snap-1", location="p1"),
        _source(ENTRY, "snap-2", location="p2"),
        _source("entry-other", "snap-3", location="p9"),
    )

    ref = reobservation_ref(particle, ENTRY, NEW_SNAPSHOT)

    assert ref.corpus_entry_id == ENTRY
    assert ref.location == "p2"
    assert ref.chunk_hash == "hash-p2"
    assert ref.snapshot_id == NEW_SNAPSHOT
