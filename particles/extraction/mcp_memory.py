# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Reference memory-server migration: the on-disk format and its store vocabulary.

Two things live here, and they are the same knowledge seen from two sides:

**The export format.** ``@modelcontextprotocol/server-memory`` v0.6.3 persists
its knowledge graph as JSONL — one JSON object per line, each carrying a
``type`` discriminator the server strips on load::

    {"type": "entity",   "name": ..., "entityType": ..., "observations": [...]}
    {"type": "relation", "from": ..., "to": ..., "relationType": ...}

:func:`parse_memory_jsonl` reads that file; :class:`McpMemoryExtractor` turns it
into candidate particles. It is a **structured extractor** in the
sense — the Wikidata / Nomisma / Numista family, no prose stage and no LLM —
because the records are already claim-granular. Running them through the
general extractor would paraphrase claims that are already atomic, spend budget
per record, and introduce hallucination risk into content that carried none.

**The store vocabulary.** The tag encoding that represents a reference graph
inside the store is defined here rather than in the façade, so migrated records
and façade-written records are *the same records*. That is what makes "bring
your history with you" true: after ``particles import mcp-memory``, the
migrated graph is visible through ``particles memory serve`` — the same
``read_graph``, the same ``search_nodes`` — with no second encoding to keep in
sync. ``particles.mcp.memory_compat.graph`` imports these names (Surface →
Client, the allowed direction) and adds the store-reading
projection on top.

The **second-hand attribution rule** is what this extractor
exists to obey, so it is worth stating where the code implements it:

* provenance points at the deposited export snapshot with the record's line
  number as ``location`` — never at a synthesised reference into the incumbent
  store, which was never fetched and cannot be re-verified (§4a/§4b);
* the migrating operator is recorded as a ``ContributorRef`` with the
  ``importer`` role (§4c);
* every particle carries ``CalibrationSource.IMPORTED`` and the configured
  import floor — the reference format has no scores to inherit, and a format
  that does must still not map them onto ``confidence.value`` (§5);
* the entity's name doubles as its ``ExternalRef`` in the ``mcp-memory``
  namespace, which is what lets a second export re-attach to the same Subject
  instead of forking the graph (§7b).

Live-authority resolution is skipped for this source type
(``subjects.skip_live_authorities_source_types``), so a multi-thousand-entity
import is offline, fast, and cannot rewrite a ``canonical_name`` the migrating
user never chose (§7a).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, unquote

from particles.config import get_config
from particles.core.schema import (
    ApplicabilityClause,
    ContributorRef,
    ExternalRef,
    Snapshot,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.extraction.general import CandidateParticle, ExtractionResult
from particles.extraction.migration_preview import MigrationPreview, build_preview

log = logging.getLogger(__name__)

EXTRACTOR_ID = "mcp-memory-extractor"
EXTRACTOR_VERSION = "0.1.1"

#: The per-incumbent source type. One string keys four existing
#: levers: extractor applicability, the extractor trust weight, a
#: ``SourceTrustStatement`` with ``SourceRefType.SOURCE_TYPE``, and the
#: skip-live-authorities set — so trust in a migration lives in revisable
#: operator policy rather than baked into immutable confidence (§6).
SOURCE_TYPE = "MCP_MEMORY_EXPORT"

#: Below the LLM extractors' 0.70. A faithful mechanical translation introduces
#: no hallucination risk, but it is translating claims this store never saw
#: made and cannot check — second-hand by construction (§5/§6).
DEFAULT_TRUST_WEIGHT = 0.60

#: The namespace an entity's reference-server identity is recorded under. The
#: reference model has no entity ids — ``name`` *is* the identity — so the name
#: is the ref id, which is exactly what makes re-import idempotent (§7b).
EXTERNAL_REF_NAMESPACE = "mcp-memory"

APPLICABILITY = [
    ApplicabilityClause(
        keyword="MUST",
        domain_uri="urn:particles:domain:agent-memory",
        domain_label="agent memory",
        source_types=[SOURCE_TYPE],
    )
]

# ---------------------------------------------------------------------------
# Store vocabulary — shared with the façade (see module docstring)
# ---------------------------------------------------------------------------

#: Marks a particle as a reference *observation* (one Subject, one claim).
OBSERVATION_TAG = "memory-compat:observation"
#: Marks a particle as a reference *relation* (two Subjects, a directed edge).
RELATION_TAG = "memory-compat:relation"
#: Marks a Subject as deleted through the façade.
TOMBSTONE_TAG = "memory-compat:deleted"
#: Marks a particle as arriving by migration rather than by an agent write
#:. Distinct from ``CalibrationSource.IMPORTED`` on purpose: the
#: calibration source says *how the number was produced*, this says *which run
#: put the record here*, so an operator can find or retract one import.
IMPORT_TAG = "migration:mcp-memory"

_REL_TYPE_PREFIX = "memory-compat:rel="
_REL_FROM_PREFIX = "memory-compat:from="
_REL_TO_PREFIX = "memory-compat:to="


def _enc(value: str) -> str:
    """Percent-encode a reference string so it survives as a single tag token."""
    return quote(value, safe="")


def _dec(value: str) -> str:
    """Inverse of :func:`_enc`."""
    return unquote(value)


def _tag_value(tags: list[str] | None, prefix: str) -> str | None:
    for tag in tags or ():
        if tag.startswith(prefix):
            return _dec(tag[len(prefix) :])
    return None


def relation_tags(from_name: str, to_name: str, relation_type: str) -> list[str]:
    """Build the reserved tag set that carries a relation triple losslessly."""
    return [
        RELATION_TAG,
        f"{_REL_TYPE_PREFIX}{_enc(relation_type)}",
        f"{_REL_FROM_PREFIX}{_enc(from_name)}",
        f"{_REL_TO_PREFIX}{_enc(to_name)}",
    ]


def relation_from_particle(particle: Any) -> dict[str, str] | None:
    """Recover a reference relation dict from a relation particle.

    Returns ``None`` when the particle is not a well-formed relation (a tag was
    dropped or the particle predates the encoding), so a damaged record
    degrades to omission rather than a malformed edge.
    """
    tags = particle.tags
    rel = _tag_value(tags, _REL_TYPE_PREFIX)
    src = _tag_value(tags, _REL_FROM_PREFIX)
    dst = _tag_value(tags, _REL_TO_PREFIX)
    if rel is None or src is None or dst is None:
        log.warning("Skipping malformed reference relation particle %s", particle.id)
        return None
    return {"from": src, "to": dst, "relationType": rel}


def relation_content(from_name: str, to_name: str, relation_type: str) -> str:
    """The particle's human-readable content — the active voice the reference asks for."""
    return f"{from_name} {relation_type} {to_name}"


def is_observation(particle: Any) -> bool:
    return OBSERVATION_TAG in (particle.tags or ())


def is_relation(particle: Any) -> bool:
    return RELATION_TAG in (particle.tags or ())


def is_tombstone(particle: Any) -> bool:
    return TOMBSTONE_TAG in (particle.tags or ())


def entity_type_of(subject: Any) -> str:
    """Reference entities always carry a string ``entityType``; ours may be None."""
    return subject.subject_class or ""


# ---------------------------------------------------------------------------
# The export format
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntity:
    """One ``{"type": "entity", …}`` record, with the line it came from."""

    name: str
    entity_type: str
    observations: list[str]
    line: int


@dataclass
class MemoryRelation:
    """One ``{"type": "relation", …}`` record, with the line it came from."""

    from_name: str
    to_name: str
    relation_type: str
    line: int


@dataclass
class ParsedGraph:
    """The result of reading a ``memory.jsonl``.

    ``notes`` carries every line the parser declined, with its number and why.
    Nothing is silently dropped: a malformed line is a fact about the user's
    export they are entitled to see, and the verbatim bytes remain in the
    corpus for a later mapper (§3/§4e).
    """

    entities: list[MemoryEntity] = field(default_factory=list)
    relations: list[MemoryRelation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: ``"<record kind>.<field>"`` to the number of records carrying a field the
    #: reference format does not define (a fork's ``createdAt``, say). The
    #: mapping has no honest place for it, so it stays in the blob and out of
    #: the particle, and is counted rather than passed over.
    unmapped_fields: dict[str, int] = field(default_factory=dict)


#: The fields the reference format defines, per record kind. Anything else on a
#: record is unplaced, not guessed at.
_KNOWN_FIELDS: dict[str, frozenset[str]] = {
    "entity": frozenset({"type", "name", "entityType", "observations"}),
    "relation": frozenset({"type", "from", "to", "relationType"}),
}


def _count_unmapped(graph: ParsedGraph, kind: str, record: dict[str, Any]) -> None:
    for key in sorted(record.keys() - _KNOWN_FIELDS[kind]):
        label = f"{kind}.{key}"
        graph.unmapped_fields[label] = graph.unmapped_fields.get(label, 0) + 1


def parse_memory_jsonl(content: bytes) -> ParsedGraph:
    """Parse a reference-server ``memory.jsonl`` blob.

    Tolerant by design — a single bad line must not cost the user their whole
    migration — but never silent: each skip lands in ``ParsedGraph.notes`` and
    surfaces as an extraction quality note.
    """
    graph = ParsedGraph()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return ParsedGraph(notes=["Export is not valid UTF-8; nothing imported."])

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            graph.notes.append(f"Line {number}: not valid JSON ({exc.msg}); skipped.")
            continue
        if not isinstance(record, dict):
            graph.notes.append(f"Line {number}: expected a JSON object; skipped.")
            continue

        kind = record.get("type")
        if kind == "entity":
            name = record.get("name")
            if not isinstance(name, str) or not name.strip():
                graph.notes.append(f"Line {number}: entity has no usable name; skipped.")
                continue
            raw_observations = record.get("observations") or []
            observations = [o for o in raw_observations if isinstance(o, str) and o.strip()]
            dropped = len(raw_observations) - len(observations)
            if dropped:
                graph.notes.append(
                    f"Line {number}: dropped {dropped} non-string / empty observation(s) "
                    f"from entity {name!r}."
                )
            entity_type = record.get("entityType")
            _count_unmapped(graph, "entity", record)
            graph.entities.append(
                MemoryEntity(
                    name=name,
                    entity_type=entity_type if isinstance(entity_type, str) else "",
                    observations=observations,
                    line=number,
                )
            )
        elif kind == "relation":
            src, dst, rel = record.get("from"), record.get("to"), record.get("relationType")
            if not all(isinstance(v, str) and v.strip() for v in (src, dst, rel)):
                graph.notes.append(
                    f"Line {number}: relation is missing an endpoint or type; skipped."
                )
                continue
            _count_unmapped(graph, "relation", record)
            graph.relations.append(
                MemoryRelation(
                    from_name=str(src), to_name=str(dst), relation_type=str(rel), line=number
                )
            )
        else:
            graph.notes.append(f"Line {number}: unknown record type {kind!r}; skipped.")

    return graph


#: How many dropped entity names a single note spells out before summarising the
#: rest. A note is read in a terminal; the full list is always recoverable from
#: the deposited export, which is held verbatim.
_NAMED_ENTITY_LIMIT = 10


@dataclass
class EmptyEntityReport:
    """Where an export's observation-less entities end up.

    An extractor emits candidates, a candidate is a particle, and a Subject
    exists only because some candidate named it. So an entity with no
    observations has two possible fates, and they are not the same loss:

    * ``endpoint_only`` — some relation names it, so the relation candidate
      brings the Subject into being (with its ``entityType``). It migrates.
    * ``dropped`` — nothing names it. It does **not** migrate, by decision
      rather than by oversight: minting a bare Subject here would manufacture a
      phantom subject that ``particles lint`` flags on the very next command.
      The accepted cost is disclosed wherever the export is first read.
    """

    dropped: list[str] = field(default_factory=list)
    endpoint_only: list[str] = field(default_factory=list)


def _key(name: str) -> str:
    """Names compare the way the subject resolver compares them: case-insensitively."""
    return name.lower()


def empty_entity_report(graph: ParsedGraph) -> EmptyEntityReport:
    """Split the export's observation-less entities by whether a relation rescues them."""
    endpoints = {_key(r.from_name) for r in graph.relations} | {
        _key(r.to_name) for r in graph.relations
    }
    report = EmptyEntityReport()
    for entity in graph.entities:
        if entity.observations:
            continue
        if _key(entity.name) in endpoints:
            report.endpoint_only.append(entity.name)
        else:
            report.dropped.append(entity.name)
    return report


def empty_entity_notes(report: EmptyEntityReport) -> list[str]:
    """Render an :class:`EmptyEntityReport` as disclosure lines.

    Shared by the extractor's quality notes and the ``import mcp-memory`` verb,
    so the count a user is shown at the door is the count extraction reports.
    """
    notes: list[str] = []
    if report.dropped:
        named = ", ".join(repr(n) for n in report.dropped[:_NAMED_ENTITY_LIMIT])
        more = len(report.dropped) - _NAMED_ENTITY_LIMIT
        if more > 0:
            named += f", and {more} more"
        notes.append(
            f"{len(report.dropped)} entity/entities carry no observations and appear in no "
            f"relation, and will not migrate: {named}. A Particles store holds a name only "
            "through something believed about it."
        )
    # ``endpoint_only`` raises no note: those entities migrate whole, and a
    # note is a disclosure of loss, not a progress report.
    return notes


def _import_floor() -> float:
    """The one confidence every migrated record gets (read)."""
    return get_config().migration.import_confidence


class McpMemoryExtractor:
    """Turns a deposited ``memory.jsonl`` into candidate particles.

    Deterministic and offline: no LLM call, no network, no store access. The
    Engine half — subject resolution, §6.6 reconciliation, the write — is the
    ordinary pipeline's, which is the point of §1's "no new pipeline".
    """

    EXTRACTOR_ID: str = EXTRACTOR_ID
    EXTRACTOR_VERSION: str = EXTRACTOR_VERSION
    DEFAULT_TRUST_WEIGHT: float = DEFAULT_TRUST_WEIGHT
    APPLICABILITY = APPLICABILITY

    def accepts(self, source_type: str) -> bool:
        return source_type == SOURCE_TYPE

    async def extract(
        self,
        snapshot: Snapshot,
        content: bytes,
        **kwargs: object,
    ) -> ExtractionResult:
        # The pipeline passes the corpus entry's depositor;
        # the fallback covers a direct call in a test or a tool.
        actor = str(kwargs.get("deposited_by") or "operator")
        return map_graph(parse_memory_jsonl(content), deposited_by=actor)


def map_graph(graph: ParsedGraph, *, deposited_by: str = "operator") -> ExtractionResult:
    """Map a parsed export onto candidate particles: the whole of the mapping.

    Synchronous and store-free, and the **only** implementation of the mapping:
    :meth:`McpMemoryExtractor.extract` and :func:`preview_memory_export` both
    call it, so a dry run reports what the real import produces rather than
    what a second code path believes it would.
    """
    if not graph.entities and not graph.relations:
        return ExtractionResult(
            quality_notes=graph.notes or ["No entity or relation records found in export."]
        )

    floor = _import_floor()
    contributors = [
        ContributorRef(id=_contributor_id(deposited_by), role="importer", at=datetime.now(UTC))
    ]
    candidates: list[CandidateParticle] = []

    for entity in graph.entities:
        ref = ExternalRef(
            namespace=EXTERNAL_REF_NAMESPACE,
            id=entity.name,
            confidence=1.0,
        )
        classes = {entity.name: entity.entity_type} if entity.entity_type else {}
        for index, observation in enumerate(entity.observations):
            candidates.append(
                CandidateParticle(
                    content=observation,
                    confidence_value=floor,
                    uncertainty_nature=UncertaintyNature.EPISTEMIC,
                    subjects=[entity.name],
                    subject_classes=classes,
                    external_refs={entity.name: ref},
                    tags=[IMPORT_TAG, OBSERVATION_TAG],
                    contributors=contributors,
                    calibration_source=CalibrationSource.IMPORTED,
                    provenance_location=f"line {entity.line} observation {index}",
                )
            )

    # An endpoint's entityType rides the relation too. For an entity with
    # observations this repeats what its own candidates already said; for an
    # observation-less one it is the only candidate that names it, and without
    # this the entity migrated with its type silently blanked.
    entity_types = {_key(e.name): e.entity_type for e in graph.entities if e.entity_type}

    for relation in graph.relations:
        endpoint_classes = {
            name: entity_types[_key(name)]
            for name in (relation.from_name, relation.to_name)
            if _key(name) in entity_types
        }
        candidates.append(
            CandidateParticle(
                content=relation_content(
                    relation.from_name, relation.to_name, relation.relation_type
                ),
                confidence_value=floor,
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                subjects=[relation.from_name, relation.to_name],
                subject_classes=endpoint_classes,
                external_refs={
                    relation.from_name: ExternalRef(
                        namespace=EXTERNAL_REF_NAMESPACE, id=relation.from_name, confidence=1.0
                    ),
                    relation.to_name: ExternalRef(
                        namespace=EXTERNAL_REF_NAMESPACE, id=relation.to_name, confidence=1.0
                    ),
                },
                tags=[
                    IMPORT_TAG,
                    *relation_tags(relation.from_name, relation.to_name, relation.relation_type),
                ],
                contributors=contributors,
                calibration_source=CalibrationSource.IMPORTED,
                provenance_location=f"line {relation.line}",
            )
        )

    notes = [*graph.notes, *empty_entity_notes(empty_entity_report(graph))]
    if graph.unmapped_fields:
        fields = ", ".join(f"{name} ({count})" for name, count in graph.unmapped_fields.items())
        notes.append(
            f"Field(s) the mapping does not place were left in the export and out of the "
            f"particles: {fields}."
        )
    return ExtractionResult(candidates=candidates, quality_notes=notes)


def preview_memory_export(
    content: bytes, *, deposited_by: str = "operator", sample_size: int = 5
) -> MigrationPreview:
    """Report what importing this export would produce, without a store.

    The ``import mcp-memory --dry-run`` report. Parses the export and runs
    :func:`map_graph` (the mapping the real import runs), then counts the
    result: nothing is deposited, no Subject is resolved, no particle is
    written. The counts are therefore what the export contributes; on a store
    that already holds some of it, existing Subjects re-attach and identical
    claims dedup, so the real import can only write *fewer* records than this.

    An entity with no observations produces no particle. It survives only if a
    relation names it (the relation candidate carries its entity type);
    otherwise it is lost, and ``entities_lost`` names it, agreeing with
    :func:`empty_entity_report` by construction of the same name match.
    """
    graph = parse_memory_jsonl(content)
    return build_preview(
        source_type=SOURCE_TYPE,
        extractor_id=EXTRACTOR_ID,
        extractor_version=EXTRACTOR_VERSION,
        result=map_graph(graph, deposited_by=deposited_by),
        records={
            "entities": len(graph.entities),
            "observations": sum(len(e.observations) for e in graph.entities),
            "relations": len(graph.relations),
        },
        declared_entities=[e.name for e in graph.entities],
        entities_without_records=[e.name for e in graph.entities if not e.observations],
        unmapped_fields=graph.unmapped_fields,
        sample_size=sample_size,
    )


def _contributor_id(actor: str) -> str:
    """Normalize an operator id into the ``platform:identifier`` shape (§6.5).

    An actor that already names its platform is left alone; a bare name is
    scoped to this import path so it cannot collide with an identity minted
    elsewhere.
    """
    return actor if ":" in actor else f"import:{actor}"
