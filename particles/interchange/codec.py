# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Particle interchange codec.

Pure (no I/O) translation between a :class:`Particle` and a self-contained
JSON-LD *interchange unit* — the federation-exchange and store-export wire
format. Three load-bearing rules:

- **Substrate only (§2), as restated.** A unit carries the
  immutable stored substrate and never carries *per-observer or per-query*
  derived quantities (effective / calibrated confidence) — those are recomputed
  on import, and have no stable value to serialize anyway. (They are not model
  fields, so they cannot leak.) A **stamped** derived annotation — the structured claim — does travel, *with its stamp*, so the importer can see what
  produced it and when; dropping it would make a store round-trip lose an
  annotation that cost an LLM call per particle. The embedding still does not
  travel: it is large, model-specific, and locally recomputable from ``content``
  at zero marginal cost. Round-trip preserves the substrate exactly.
- **Cross-store identity (§3).** Subjects travel by *external reference* (the join key), not by store-local UUID. The source particle/subject
  UUIDs ride along as origin metadata (``sourceParticleId`` /
  ``sourceSubjectId``), never as the target's identity — ``from_unit`` mints a
  fresh particle id; import resolves subjects by external ref + claim identity.
- **Canonical serialization (§4 / §8).** JSON-LD with camelCase terms aligned
  to ``artifacts/schemas/context.jsonld``, decoupling the wire format from the
  SDK's snake_case. YAML-LD (deferred) is the same data model.

This module is store-free: ``to_unit`` takes the particle's subjects from the
caller (which resolved them from a session); the store-aware export/import
wrappers live above it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from particles.core.schema import (
    AssertionModality,
    CanonicalForm,
    ClaimTerm,
    Confidence,
    ContributorRef,
    ExternalRef,
    ExtractorRef,
    Particle,
    ParticleType,
    ProvenanceRef,
    ProvenanceRefType,
    StructuredClaim,
    Subject,
    TermKind,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason
from particles.extraction.polarity import POLARITY_KEY
from particles.extraction.scope import SCOPE_ACTION_KEY, SCOPE_KEY

#: Interchange container version, independent of Core ``schema_version`` (§7).
FORMAT_VERSION = "1.0"

#: Pre-ADR-0234 bare ``properties`` spelling -> the current prefixed key.
#: Read on import only; nothing writes the legacy form. Keyed off the constants
#: so a further rename keeps the map honest.
_LEGACY_PROPERTIES_KEYS = {
    "polarity": POLARITY_KEY,
    "scope": SCOPE_KEY,
    "scope_action": SCOPE_ACTION_KEY,
}

#: Published JSON-LD context the units reference (term -> IRI mapping).
CONTEXT_URL = "https://linkedparticles.org/schemas/context.jsonld"


@dataclass
class SubjectRef:
    """A particle's reference to one subject, portable across stores (§3).

    ``external_refs`` is the cross-store join key; ``canonical_name`` / ``aliases``
    let a QID-less subject import as a new bare-local subject. ``source_subject_id``
    is origin metadata only.
    """

    external_refs: list[ExternalRef] = field(default_factory=list)
    canonical_name: str | None = None
    aliases: list[str] = field(default_factory=list)
    subject_class: str | None = None
    source_subject_id: str | None = None


@dataclass
class ParsedUnit:
    """The result of decoding a unit: a substrate-only particle (no subject_ids
    yet — import assigns them after resolving ``subjects``) plus origin metadata.
    """

    particle: Particle
    subjects: list[SubjectRef] = field(default_factory=list)
    source_particle_id: str | None = None


def _ext_ref_to_json(ref: ExternalRef) -> dict[str, Any]:
    out: dict[str, Any] = {"namespace": ref.namespace, "externalId": ref.id}
    if ref.uri is not None:
        out["uri"] = ref.uri
    if ref.confidence != 1.0:
        out["confidence"] = ref.confidence
    return out


def _ext_ref_from_json(obj: dict[str, Any]) -> ExternalRef:
    return ExternalRef(
        namespace=obj["namespace"],
        id=obj["externalId"],
        uri=obj.get("uri"),
        confidence=obj.get("confidence", 1.0),
    )


#: Fields accepted on a serialized contributor object (security review F32).
#: ``ContributorRef`` itself lives in ``core/schema`` and is not interchange-
#: specific, so the strict-extras gate is applied at this untrusted-dict splat
#: boundary instead of widening the Core model's config: an interchange unit is
#: external input, and a contributor object carrying unknown keys is a malformed
#: bundle that must fail loud rather than silently truncate (Pydantic v2's
#: default ``extra="ignore"``).
_CONTRIBUTOR_FIELDS = frozenset(ContributorRef.model_fields)


def _contributor_from_json(obj: dict[str, Any]) -> ContributorRef:
    """Construct a ``ContributorRef`` from an interchange dict, rejecting extras.

    Fail-closed against a hostile/malformed contributor object (security review
    F32): unknown keys raise :class:`ValueError` here rather than being silently
    dropped by Pydantic's default ``extra="ignore"``.
    """
    extra = set(obj) - _CONTRIBUTOR_FIELDS
    if extra:
        raise ValueError(
            "interchange contributor object carries unexpected key(s) "
            f"{sorted(extra)}; refusing to decode (security review F32)."
        )
    return ContributorRef(**obj)


def _provenance_to_json(ref: ProvenanceRef) -> dict[str, Any]:
    out: dict[str, Any] = {"refType": ref.type.value, "corpusEntryId": ref.corpus_entry_id}
    if ref.snapshot_id is not None:
        out["snapshotId"] = ref.snapshot_id
    if ref.location is not None:
        out["location"] = ref.location
    if ref.chunk_hash is not None:
        out["chunkHash"] = ref.chunk_hash
    return out


def _provenance_from_json(obj: dict[str, Any]) -> ProvenanceRef:
    return ProvenanceRef(
        type=ProvenanceRefType(obj["refType"]),
        corpus_entry_id=obj["corpusEntryId"],
        snapshot_id=obj.get("snapshotId"),
        location=obj.get("location"),
        chunk_hash=obj.get("chunkHash"),
    )


def _claim_term_to_json(term: ClaimTerm) -> dict[str, Any]:
    out: dict[str, Any] = {"termKind": term.kind.value, "termValue": term.value}
    if term.datatype is not None:
        out["datatype"] = term.datatype
    if term.language is not None:
        out["language"] = term.language
    return out


def _claim_term_from_json(obj: dict[str, Any]) -> ClaimTerm:
    return ClaimTerm(
        kind=TermKind(obj["termKind"]),
        value=obj["termValue"],
        datatype=obj.get("datatype"),
        language=obj.get("language"),
    )


def _structured_claim_to_json(claim: StructuredClaim) -> dict[str, Any]:
    """Encode the annotation, stamp included.

    The stamp is what makes emitting a *derived* quantity sound (§2.8): an
    importer can see exactly which structurizer produced the triple and when,
    instead of inheriting an anonymous assertion. Per-observer derived
    quantities (effective / calibrated confidence) remain un-emitted — they
    have no stable value to serialize.
    """
    out: dict[str, Any] = {
        "subject": _claim_term_to_json(claim.subject),
        "predicate": _claim_term_to_json(claim.predicate),
        "object": _claim_term_to_json(claim.object),
        "structurizerId": claim.structurizer_id,
        "structurizerVersion": claim.structurizer_version,
        "generatedAt": claim.generated_at.isoformat(),
    }
    # The store-local Subject UUID is deliberately NOT emitted: cross-store
    # identity travels by external reference (§3), so a foreign UUID would be
    # meaningless — or worse, coincidentally valid — in the importing store.
    # Import re-binds it from the resolved subjects.
    return out


def _extractor_ref_from_json(obj: dict[str, Any]) -> ExtractorRef:
    """Construct an ``ExtractorRef`` from an interchange dict.

    Raises rather than coercing, unlike the store's read path: a malformed unit
    on the wire is an import the operator can reject and re-request, whereas a
    malformed stored row is history the operator cannot re-request. Different
    remedy, different strictness.
    """
    return ExtractorRef(name=obj["extractorName"], version=obj["extractorVersion"])


def _structured_claim_from_json(obj: dict[str, Any]) -> StructuredClaim:
    return StructuredClaim(
        subject=_claim_term_from_json(obj["subject"]),
        predicate=_claim_term_from_json(obj["predicate"]),
        object=_claim_term_from_json(obj["object"]),
        structurizer_id=obj["structurizerId"],
        structurizer_version=obj["structurizerVersion"],
        generated_at=datetime.fromisoformat(obj["generatedAt"]),
    )


def to_unit(particle: Particle, subjects: dict[str, Subject]) -> dict[str, Any]:
    """Encode a particle (+ its subjects) to a JSON-LD interchange unit.

    ``subjects`` maps subject_id -> Subject for the particle's ``subject_ids``;
    each becomes a portable :class:`SubjectRef`. Only the canonical substrate is
    emitted (§2). Optional fields are omitted when unset to keep units compact.
    """
    unit: dict[str, Any] = {
        "@context": CONTEXT_URL,
        "@type": "Particle",
        "formatVersion": FORMAT_VERSION,
        "schemaVersion": particle.schema_version,
        "sourceParticleId": particle.id,
        "particleType": particle.particle_type.value,
        "assertionModality": particle.assertion_modality.value,
        "canonicalForm": particle.canonical_form.value,
        "content": particle.content,
        "confidenceValue": particle.confidence.value,
        "calibrationSource": particle.confidence.calibration_source.value,
        "uncertaintyNature": particle.uncertainty_nature.value,
        "assertedBy": particle.asserted_by,
        "assertedAt": particle.asserted_at.isoformat(),
        "status": particle.status.value,
        "provenance": [_provenance_to_json(r) for r in particle.provenance],
        "subjects": [_subject_ref_to_json(particle, sid, subjects) for sid in particle.subject_ids],
    }

    # Optional substrate fields — emit only when set.
    conf = particle.confidence
    if conf.variance is not None:
        unit["variance"] = conf.variance
    if conf.calibration_method is not None:
        unit["calibrationMethod"] = conf.calibration_method
    if conf.calibration_ref is not None:
        unit["calibrationRef"] = conf.calibration_ref
    if particle.status_reason is not None:
        unit["statusReason"] = particle.status_reason.value
    if particle.uncertainty_kind is not None:
        unit["uncertaintyKind"] = particle.uncertainty_kind
    if particle.valid_until is not None:
        unit["validUntil"] = particle.valid_until.isoformat()
    if particle.supersedes is not None:
        unit["supersedes"] = particle.supersedes
    if particle.basis is not None:
        unit["basis"] = particle.basis
    # a node with mapped sub-terms, not the opaque ``@json`` payload
    # it was through 1.109.x. The keys are spelled ``extractorName`` /
    # ``extractorVersion`` rather than bare ``name`` / ``version`` for the same
    # reason ``termKind`` / ``termValue`` were spelled out: a bare ``name``
    # term in a shared context collides with every other structure that has
    # one. Renamed straight, with no ``FORMAT_VERSION`` bump and no
    # dual-spelling reader (owner decision at sign-off) — the repo is
    # unpublished, so no external producer emits the old spelling.
    if particle.extractor_ref is not None:
        unit["extractorRef"] = {
            "extractorName": particle.extractor_ref.name,
            "extractorVersion": particle.extractor_ref.version,
        }
    # immutable substrate — the pairing that produced the claim is
    # historical fact, not a derived or per-observer quantity, so it crosses
    # the boundary. Without it an imported particle's model is permanently
    # unknowable to the receiving store, which is exactly the gap the field
    # exists to close.
    if particle.extraction_provider_model is not None:
        unit["extractionProviderModel"] = particle.extraction_provider_model
    if particle.sequence_context is not None:
        unit["sequenceContext"] = particle.sequence_context
    if particle.tags is not None:
        unit["tags"] = particle.tags
    if particle.context_fingerprint is not None:
        unit["contextFingerprint"] = particle.context_fingerprint
    if particle.properties is not None:
        unit["properties"] = particle.properties
    if particle.contributors is not None:
        # Attribution travels as data so "who said this" survives a store copy
        #. model_dump(mode="json") renders ``at`` as an ISO string.
        unit["contributors"] = [c.model_dump(mode="json") for c in particle.contributors]
    if particle.structured_claim is not None:
        # a STAMPED derived annotation travels with its stamp.
        # Dropping it would make a store-bundle round-trip silently lose an
        # annotation that cost an LLM call per particle to produce.
        unit["structuredClaim"] = _structured_claim_to_json(particle.structured_claim)

    return unit


def _subject_ref_to_json(
    particle: Particle, subject_id: str, subjects: dict[str, Subject]
) -> dict[str, Any]:
    subj = subjects.get(subject_id)
    if subj is None:
        # Subject not provided by the caller — carry the origin id only so the
        # reference is not silently dropped; import treats it as unresolved.
        return {"sourceSubjectId": subject_id}
    out: dict[str, Any] = {
        "sourceSubjectId": subj.id,
        "canonicalName": subj.canonical_name,
        "externalRefs": [_ext_ref_to_json(r) for r in subj.external_ids],
    }
    if subj.aliases:
        out["aliases"] = subj.aliases
    if subj.subject_class is not None:
        out["subjectClass"] = subj.subject_class
    return out


def _subject_ref_from_json(obj: dict[str, Any]) -> SubjectRef:
    return SubjectRef(
        external_refs=[_ext_ref_from_json(r) for r in obj.get("externalRefs", [])],
        canonical_name=obj.get("canonicalName"),
        aliases=obj.get("aliases", []),
        subject_class=obj.get("subjectClass"),
        source_subject_id=obj.get("sourceSubjectId"),
    )


def _normalize_legacy_properties(
    properties: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Rewrite pre-ADR-0234 bare ``properties`` keys to their prefixed spelling.

    Alembic 035 fixes the keys in a *store*; it cannot reach a JSONL file
    exported before the rename. Both import paths — the fingerprint-merge
    :func:`particles.interchange.store.import_units` and the id-preserving
    :func:`particles.interchange.store.restore_store_bundle` — decode through
    this function, so an old bundle lands with the current spelling. Without
    it, re-importing one would silently un-hide its ``DECLINED`` /
    ``HYPOTHETICAL`` / ``DOCUMENT_META`` particles: the visibility predicates
    read only the prefixed key.

    A unit already carrying the prefixed key wins — a bundle holding both
    spellings is malformed, and the current one is the one the predicates mean.
    """
    if not properties:
        return properties
    if not _LEGACY_PROPERTIES_KEYS.keys() & properties.keys():
        return properties
    renamed = dict(properties)
    for legacy, current in _LEGACY_PROPERTIES_KEYS.items():
        if legacy in renamed:
            value = renamed.pop(legacy)
            renamed.setdefault(current, value)
    return renamed


def from_unit(unit: dict[str, Any]) -> ParsedUnit:
    """Decode an interchange unit to a substrate-only particle + subject refs.

    The returned particle has a freshly minted id and **no** ``subject_ids``
    (import assigns them after resolving the subject refs); the source UUID is
    returned separately as origin metadata (§3).
    """
    confidence = Confidence(
        value=unit["confidenceValue"],
        variance=unit.get("variance"),
        calibration_source=CalibrationSource(unit["calibrationSource"]),
        calibration_method=unit.get("calibrationMethod"),
        calibration_ref=unit.get("calibrationRef"),
    )

    valid_until_raw = unit.get("validUntil")
    status_reason_raw = unit.get("statusReason")

    particle = Particle(
        content=unit["content"],
        confidence=confidence,
        uncertainty_nature=UncertaintyNature(unit["uncertaintyNature"]),
        provenance=[_provenance_from_json(r) for r in unit.get("provenance", [])],
        asserted_by=unit["assertedBy"],
        asserted_at=datetime.fromisoformat(unit["assertedAt"]),
        status=Status(unit["status"]),
        status_reason=StatusReason(status_reason_raw) if status_reason_raw is not None else None,
        schema_version=unit.get("schemaVersion", particle_schema_default()),
        particle_type=ParticleType(unit.get("particleType", ParticleType.CLAIM.value)),
        assertion_modality=AssertionModality(
            unit.get("assertionModality", AssertionModality.FALSIFIABLE.value)
        ),
        canonical_form=CanonicalForm(unit.get("canonicalForm", CanonicalForm.PROSE.value)),
        structured_claim=(
            _structured_claim_from_json(unit["structuredClaim"])
            if "structuredClaim" in unit
            else None
        ),
        uncertainty_kind=unit.get("uncertaintyKind"),
        valid_until=datetime.fromisoformat(valid_until_raw) if valid_until_raw else None,
        supersedes=unit.get("supersedes"),
        basis=unit.get("basis"),
        extractor_ref=(
            _extractor_ref_from_json(unit["extractorRef"]) if "extractorRef" in unit else None
        ),
        extraction_provider_model=unit.get("extractionProviderModel"),
        sequence_context=unit.get("sequenceContext"),
        tags=unit.get("tags"),
        context_fingerprint=unit.get("contextFingerprint"),
        properties=_normalize_legacy_properties(unit.get("properties")),
        contributors=(
            [_contributor_from_json(c) for c in unit["contributors"]]
            if "contributors" in unit
            else None
        ),
    )

    return ParsedUnit(
        particle=particle,
        subjects=[_subject_ref_from_json(s) for s in unit.get("subjects", [])],
        source_particle_id=unit.get("sourceParticleId"),
    )


def particle_schema_default() -> str:
    from particles.core.schema import SCHEMA_VERSION

    return SCHEMA_VERSION


def subject_to_unit(subject: Subject) -> dict[str, Any]:
    """Encode a standalone subject to a JSON-LD unit for store-export bundles.

    Subjects travel with their external refs (the cross-store join key) so a
    QID-less subject still imports as a bare-local node. The source UUID is
    origin metadata only.
    """
    unit: dict[str, Any] = {
        "@context": CONTEXT_URL,
        "@type": "Subject",
        "formatVersion": FORMAT_VERSION,
        "sourceSubjectId": subject.id,
        "canonicalName": subject.canonical_name,
        "assertedBy": subject.asserted_by,
        "externalRefs": [_ext_ref_to_json(r) for r in subject.external_ids],
    }
    if subject.description is not None:
        unit["description"] = subject.description
    if subject.aliases:
        unit["aliases"] = subject.aliases
    if subject.subject_class is not None:
        unit["subjectClass"] = subject.subject_class
    return unit


def subject_from_unit(unit: dict[str, Any]) -> Subject:
    """Decode a subject unit to a Subject with a freshly minted id (§3)."""
    return Subject(
        canonical_name=unit["canonicalName"],
        description=unit.get("description"),
        aliases=unit.get("aliases", []),
        external_ids=[_ext_ref_from_json(r) for r in unit.get("externalRefs", [])],
        subject_class=unit.get("subjectClass"),
        asserted_by=unit.get("assertedBy", "interchange-import"),
    )
