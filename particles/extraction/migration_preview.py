# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Migration preview: what a mapping would produce, before anything is written.

A migration extractor turns another store's export into candidate particles
(see ``mcp_memory.py``). The mapping from a foreign schema is the part most
likely to be wrong on a first release, and the ``import <incumbent>`` verbs run
on a user's real data, so each offers a ``--dry-run`` that answers one question
first: *what would this import put in my store?*

This module is the format-independent half of that answer. A migration
extractor parses its export and runs **its own mapping code**, then hands the
resulting :class:`~particles.extraction.general.ExtractionResult` to
:func:`build_preview`, which counts and samples it. The preview never
re-implements a mapper: a report computed by a parallel code path could agree
with itself while disagreeing with the real import.

Store-free by construction. Everything here is a pure function of the export
bytes and static config, which is what lets a dry run promise that nothing was
deposited or written, and what bounds the numbers it can report: it sees
neither the Subjects a store already holds nor the claims already in it, so its
counts are what the export *contributes*, before an existing Subject re-attaches
or an identical claim dedups. A preview that did consult a store (a throwaway
one, say) can populate the same report shape; nothing here assumes the bytes
are the only possible input.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from particles.extraction.general import CandidateParticle, ExtractionResult


@dataclass(frozen=True)
class PreviewSample:
    """One candidate, shown the way the import would record it."""

    location: str | None
    subjects: list[str]
    content: str


@dataclass(frozen=True)
class MigrationPreview:
    """What one export would become, computed without a store.

    Attributes:
        source_type: The per-incumbent source type the export deposits under.
        extractor_id: The migration extractor that produced the mapping.
        extractor_version: That extractor's version.
        records: Record counts in the *export's own* vocabulary (for the
            reference memory server: entities, observations, relations).
        subjects: Distinct subject names the candidates carry, which is the
            number of Subjects the import resolves (existing ones re-attach).
        particles: Candidates the mapper produced.
        single_subject_particles: Candidates about one Subject (properties).
        multi_subject_particles: Candidates about several Subjects (edges).
        confidence_values: The distinct confidence values the candidates carry.
        calibration_sources: The distinct calibration sources they declare.
        entities_without_records: Entities the export declares that produced no
            particle of their own.
        entities_lost: The subset of those no other candidate names either, so
            no Subject is created and the entity does not survive the import.
        unmapped_fields: ``"<record kind>.<field>"`` to the number of records
            carrying a field the mapping does not place.
        dropped: Every note the parser and mapper raised: skipped lines,
            discarded values, unplaced fields. Preserved in the deposited
            export, absent from the particles.
        sample: A spread of candidates across the export.
        store_consulted: Always ``False`` for a preview built from bytes alone.
    """

    source_type: str
    extractor_id: str
    extractor_version: str
    records: dict[str, int]
    subjects: int
    particles: int
    single_subject_particles: int
    multi_subject_particles: int
    confidence_values: list[float]
    calibration_sources: list[str]
    entities_without_records: list[str] = field(default_factory=list)
    entities_lost: list[str] = field(default_factory=list)
    unmapped_fields: dict[str, int] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)
    sample: list[PreviewSample] = field(default_factory=list)
    store_consulted: bool = False

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable form, for ``--json`` output."""
        return asdict(self)


def _spread(candidates: Sequence[CandidateParticle], size: int) -> list[CandidateParticle]:
    """Pick ``size`` candidates at an even stride, so a sample spans the export.

    The head of a candidate list is one entity's records; a stride reaches the
    later record kinds too. Deterministic, so two runs over one export agree.
    """
    if size <= 0 or not candidates:
        return []
    if size >= len(candidates):
        return list(candidates)
    return [candidates[i * len(candidates) // size] for i in range(size)]


def build_preview(
    *,
    source_type: str,
    extractor_id: str,
    extractor_version: str,
    result: ExtractionResult,
    records: Mapping[str, int],
    declared_entities: Sequence[str] = (),
    entities_without_records: Sequence[str] = (),
    unmapped_fields: Mapping[str, int] | None = None,
    sample_size: int = 5,
) -> MigrationPreview:
    """Summarise a migration extractor's own output into a :class:`MigrationPreview`.

    Args:
        source_type: The export's source type string.
        extractor_id: The producing extractor's id.
        extractor_version: The producing extractor's version.
        result: What the extractor's mapping code returned for this export.
        records: Record counts in the export's own vocabulary.
        declared_entities: Names the export declares as entities, when the
            format has such a notion. Used only to find the ones that vanish.
        entities_without_records: Declared entities that produced no particle
            of their own (the caller knows this; candidates cannot show it).
        unmapped_fields: Per-field counts of data the mapping does not place.
        sample_size: How many candidates to include as a sample.
    """
    candidates = result.candidates
    named: set[str] = {name for c in candidates for name in c.subjects}
    # "Named" the way the subject resolver will decide it: case-insensitively.
    # A relation that spells an entity ``acme`` re-attaches to ``Acme``, so an
    # exact-match test here would report a loss the import does not incur.
    named_keys = {name.lower() for name in named}
    lost = [name for name in dict.fromkeys(declared_entities) if name.lower() not in named_keys]
    return MigrationPreview(
        source_type=source_type,
        extractor_id=extractor_id,
        extractor_version=extractor_version,
        records=dict(records),
        subjects=len(named),
        particles=len(candidates),
        single_subject_particles=sum(1 for c in candidates if len(c.subjects) == 1),
        multi_subject_particles=sum(1 for c in candidates if len(c.subjects) > 1),
        confidence_values=sorted({c.confidence_value for c in candidates}),
        calibration_sources=sorted(
            {str(c.calibration_source.value) for c in candidates if c.calibration_source}
        ),
        entities_without_records=list(entities_without_records),
        entities_lost=lost,
        unmapped_fields=dict(unmapped_fields or {}),
        dropped=list(result.quality_notes),
        sample=[
            PreviewSample(
                location=c.provenance_location, subjects=list(c.subjects), content=c.content
            )
            for c in _spread(candidates, sample_size)
        ],
    )
