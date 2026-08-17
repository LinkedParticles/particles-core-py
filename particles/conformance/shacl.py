"""SHACL conformance validation using pyshacl (§C.5, §10.1).

Validates JSON-LD serialised records against the five normative SHACL shapes:
  ParticleShape, SubjectShape, CorpusSnapshotShape, ProvenanceChainShape,
  TrustStatementShape (SubjectShape added in 0.52.1
  § Amendment).

Shapes are loaded from artifacts/schemas/shacl/*.ttl relative to the repo root.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from particles.conformance._resources import schemas_dir

log = logging.getLogger(__name__)

_SHAPES_DIR = schemas_dir() / "shacl"
_CONTEXT_FILE = schemas_dir() / "context.jsonld"

_XSD = "http://www.w3.org/2001/XMLSchema#"

_SHAPE_FILES = {
    "ParticleShape": _SHAPES_DIR / "ParticleShape.ttl",
    "SubjectShape": _SHAPES_DIR / "SubjectShape.ttl",
    "CorpusSnapshotShape": _SHAPES_DIR / "CorpusSnapshotShape.ttl",
    "ProvenanceChainShape": _SHAPES_DIR / "ProvenanceChainShape.ttl",
    "TrustStatementShape": _SHAPES_DIR / "TrustStatementShape.ttl",
}


class ValidationResult:
    def __init__(self, conforms: bool, report_text: str, violations: list[str]) -> None:
        self.conforms = conforms
        self.report_text = report_text
        self.violations = violations

    def __repr__(self) -> str:
        return f"ValidationResult(conforms={self.conforms}, violations={len(self.violations)})"


def validate_particle(particle_jsonld: dict[str, Any]) -> ValidationResult:
    """Validate a particle JSON-LD document against ParticleShape."""
    return _validate(particle_jsonld, "ParticleShape")


def validate_subject(subject_jsonld: dict[str, Any]) -> ValidationResult:
    """Validate a Subject JSON-LD document against SubjectShape."""
    return _validate(subject_jsonld, "SubjectShape")


def validate_snapshot(snapshot_jsonld: dict[str, Any]) -> ValidationResult:
    """Validate a snapshot JSON-LD document against CorpusSnapshotShape."""
    return _validate(snapshot_jsonld, "CorpusSnapshotShape")


def validate_trust_statement(stmt_jsonld: dict[str, Any]) -> ValidationResult:
    """Validate a SourceTrustStatement JSON-LD document against TrustStatementShape."""
    return _validate(stmt_jsonld, "TrustStatementShape")


def _validate(doc: dict[str, Any], shape_name: str) -> ValidationResult:
    shape_path = _SHAPE_FILES.get(shape_name)
    if shape_path is None or not shape_path.exists():
        log.warning("SHACL shape file not found: %s", shape_path)
        return ValidationResult(
            conforms=True,
            report_text=f"Shape {shape_name} not found; validation skipped",
            violations=[],
        )

    try:
        import pyshacl  # type: ignore[import-untyped,unused-ignore]
        import rdflib

        # Use Graph (not Dataset) for the data side: our particle documents are
        # flat JSON-LD with no @graph array, so a single default graph is enough.
        # Dataset triggers rdflib's own DeprecationWarnings for default_context
        # and identifier in current rdflib releases.
        data_graph = rdflib.Graph()
        data_graph.parse(data=json.dumps(doc), format="json-ld")
        shapes_graph = rdflib.Graph().parse(str(shape_path), format="turtle")

        conforms, report_graph, report_text = pyshacl.validate(
            data_graph,
            shacl_graph=shapes_graph,
            abort_on_first=False,
            allow_infos=True,
            meta_shacl=False,
            advanced=False,
            js=False,
            debug=False,
        )
        violations: list[str] = []
        if not conforms:
            for _s, _p, o in report_graph.triples(
                (None, rdflib.URIRef("http://www.w3.org/ns/shacl#resultMessage"), None)
            ):
                violations.append(str(o))
        return ValidationResult(
            conforms=conforms, report_text=str(report_text), violations=violations
        )

    except ImportError:
        log.warning("pyshacl not installed; SHACL validation skipped")
        return ValidationResult(conforms=True, report_text="pyshacl not available", violations=[])
    except Exception as exc:
        log.error("SHACL validation error: %s", exc)
        return ValidationResult(conforms=False, report_text=str(exc), violations=[str(exc)])


def _typed(value: Any, datatype: str) -> dict[str, Any]:
    """A JSON-LD value object carrying an explicit datatype IRI.

    Every key these serializers emit is written in its ``particles:``-prefixed
    form, which resolves through the context's *prefix* map but deliberately
    bypasses its *term* definitions — and the term definitions are where
    ``context.jsonld`` puts its ``@type`` coercions. So a bare Python value
    lands as whatever datatype the JSON type infers (a float becomes
    ``xsd:double``, a string becomes plain), which is not what the shapes
    constrain. Typing the literal here is the only form that survives that.

    The prefixed keys are not incidental: the context aliases several fields
    onto *different* IRIs than the shapes target (``assertedBy`` and
    ``uncertaintyNature`` alias into ``psum:``, ``subjectIds`` into
    ``particles:subject``), so switching to aliases would silently move the
    triples out from under ``ParticleShape``. It also keeps the serializers
    working when ``context.jsonld`` is absent and ``_load_context()`` falls
    back to a bare prefix map.
    """
    return {"@value": value, "@type": datatype}


def _term_to_jsonld(term: object) -> dict[str, Any]:
    """Serialise one ClaimTerm of a structured claim (ClaimTermShape)."""
    from particles.core.schema import ClaimTerm as _ClaimTerm

    t: _ClaimTerm = term  # type: ignore[assignment]
    node: dict[str, Any] = {
        "particles:termKind": t.kind.value,
        "particles:termValue": t.value,
    }
    if t.datatype is not None:
        node["particles:datatype"] = t.datatype
    if t.language is not None:
        node["particles:language"] = t.language
    return node


def particle_to_jsonld(particle: object) -> dict[str, Any]:
    """Serialise a Particle to a minimal JSON-LD document for SHACL validation.

    Emits every path ``ParticleShape`` constrains, so a validation run
    exercises the whole shape rather than a corner of it. Optional fields are
    omitted when unset — each carries no ``sh:minCount`` precisely because
    absence is legal.
    """
    from particles.core.schema import Particle as _Particle

    p: _Particle = particle  # type: ignore[assignment]
    doc: dict[str, Any] = {
        "@context": _load_context(),
        "@type": "particles:Particle",
        "particles:id": p.id,
        "particles:content": p.content,
        "particles:confidenceValue": _typed(p.confidence.value, f"{_XSD}float"),
        "particles:uncertaintyNature": p.uncertainty_nature.value,
        "particles:assertedBy": p.asserted_by,
        "particles:assertedAt": _typed(p.asserted_at.isoformat(), f"{_XSD}dateTime"),
        "particles:status": p.status.value,
        "particles:schemaVersion": p.schema_version,
        # Non-Optional on the model (each has a default), so always emitted.
        "particles:particleType": p.particle_type.value,
        "particles:assertionModality": p.assertion_modality.value,
        "particles:canonicalForm": p.canonical_form.value,
    }
    if p.status_reason is not None:
        doc["particles:statusReason"] = p.status_reason.value
    if p.extraction_provider_model is not None:
        doc["particles:extractionProviderModel"] = p.extraction_provider_model
    if p.subject_ids:
        # Plain array, not the context's @list container: the shape validates
        # each subject reference directly, and RDF list cells are not one.
        doc["particles:subject"] = list(p.subject_ids)
    # emit the ref as a node so ExtractorRefShape actually fires.
    # Omitted when absent — a direct assertion has no extractor (§9.1a), and
    # the shape carries no sh:minCount precisely so that validates.
    if p.extractor_ref is not None:
        doc["particles:extractorRef"] = {
            "particles:extractorName": p.extractor_ref.name,
            "particles:extractorVersion": p.extractor_ref.version,
        }
    if p.contributors:
        doc["particles:contributors"] = [
            {
                "particles:id": c.id,
                "particles:role": c.role,
                "particles:at": _typed(c.at.isoformat(), f"{_XSD}dateTime"),
            }
            for c in p.contributors
        ]
    if p.structured_claim is not None:
        sc = p.structured_claim
        claim: dict[str, Any] = {
            "rdf:subject": _term_to_jsonld(sc.subject),
            "rdf:predicate": _term_to_jsonld(sc.predicate),
            "rdf:object": _term_to_jsonld(sc.object),
            "particles:structurizerId": sc.structurizer_id,
            "particles:structurizerVersion": sc.structurizer_version,
            "particles:generatedAt": _typed(sc.generated_at.isoformat(), f"{_XSD}dateTime"),
        }
        if sc.subject_id is not None:
            claim["particles:subjectId"] = sc.subject_id
        doc["particles:structuredClaim"] = claim
    return doc


def subject_to_jsonld(subject: object) -> dict[str, Any]:
    """Serialise a Subject to a minimal JSON-LD document for SHACL validation."""
    from particles.core.schema import Subject as _Subject

    s: _Subject = subject  # type: ignore[assignment]
    doc: dict[str, Any] = {
        "@context": _load_context(),
        "@type": "particles:Subject",
        "particles:id": s.id,
        "particles:canonicalName": s.canonical_name,
        "particles:assertedBy": s.asserted_by,
        # Typed literal so SubjectShape's xsd:dateTime constraint is meaningful.
        "particles:createdAt": _typed(s.created_at.isoformat(), f"{_XSD}dateTime"),
    }
    if s.subject_class is not None:
        doc["particles:subjectClass"] = s.subject_class
    return doc


def _load_context() -> dict[str, Any]:
    """The JSON-LD context *mapping* — the value that goes under ``@context``.

    Returns the file's inner ``@context`` object, not the file itself: callers
    write ``{"@context": _load_context()}``, and returning the whole document
    produced a double-wrapped ``{"@context": {"@context": {…}}}``. rdflib
    happens to unwrap that, so the prefixes still resolved and nothing here
    ever depended on the extra layer — but it is not what the JSON-LD grammar
    says, and no other parser owes us that tolerance.
    """
    if _CONTEXT_FILE.exists():
        loaded: dict[str, Any] = json.loads(_CONTEXT_FILE.read_text())
        ctx = loaded.get("@context")
        if isinstance(ctx, dict):
            return ctx
        log.warning("context.jsonld has no @context object; using fallback prefixes")
    # Fallback prefix map: only what the serializers' own keys need.
    return {
        "particles": "https://linkedparticles.org/vocab#",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    }
