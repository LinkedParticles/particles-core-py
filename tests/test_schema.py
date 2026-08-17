"""Tests for particles/core/schema.py — Pydantic model validation."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from particles.core.schema import (
    SCHEMA_VERSION,
    AssertionModality,
    Confidence,
    CorpusEntry,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    QueryRequest,
    SourceType,
    UncertaintyNature,
    is_truth_apt,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status


def make_particle(**kwargs: object) -> Particle:
    defaults: dict[str, object] = {
        "content": "The Earth orbits the Sun.",
        "confidence": Confidence(value=0.95, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        "uncertainty_nature": UncertaintyNature.EPISTEMIC,
        "asserted_by": "test-agent",
    }
    defaults.update(kwargs)
    return Particle(**defaults)  # type: ignore[arg-type]


class TestConfidence:
    def test_value_bounds(self) -> None:
        with pytest.raises(ValidationError):
            Confidence(value=1.5)
        with pytest.raises(ValidationError):
            Confidence(value=-0.1)

    def test_valid_confidence(self) -> None:
        c = Confidence(value=0.75, calibration_source=CalibrationSource.CALIBRATED_BENCHMARK)
        assert c.value == 0.75
        assert c.calibration_source == CalibrationSource.CALIBRATED_BENCHMARK

    def test_agent_asserted_calibration_source(self) -> None:
        # the honest label for an uncalibrated agent self-report.
        c = Confidence(value=0.8, calibration_source=CalibrationSource.AGENT_ASSERTED)
        assert c.calibration_source == CalibrationSource.AGENT_ASSERTED
        assert c.value == 0.8

    def test_confidence_is_frozen(self) -> None:
        c = Confidence(value=0.5)
        with pytest.raises(ValidationError):
            c.value = 0.9  # type: ignore[misc]


class TestParticle:
    def test_default_schema_version(self) -> None:
        p = make_particle()
        assert p.schema_version == SCHEMA_VERSION

    def test_default_status_active(self) -> None:
        p = make_particle()
        assert p.status == Status.ACTIVE

    def test_id_generated(self) -> None:
        p1 = make_particle()
        p2 = make_particle()
        assert p1.id != p2.id

    def test_extension_fields_accepted(self) -> None:
        p = make_particle(tags=["foo", "bar"], context_fingerprint="abc123")
        assert p.tags == ["foo", "bar"]
        assert p.context_fingerprint == "abc123"

    def test_empty_content_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_particle(content="")

    def test_provenance_serialisation(self) -> None:
        p = make_particle(
            provenance=[
                ProvenanceRef(
                    type=ProvenanceRefType.SOURCE,
                    corpus_entry_id="entry-1",
                    snapshot_id="snap-1",
                )
            ]
        )
        d = p.model_dump()
        assert d["provenance"][0]["corpus_entry_id"] == "entry-1"


class TestAssertionModality:
    """truth-aptness axis + is_truth_apt predicate."""

    def test_default_is_falsifiable(self) -> None:
        p = make_particle()
        assert p.assertion_modality == AssertionModality.FALSIFIABLE
        assert is_truth_apt(p) is True

    def test_non_falsifiable_not_truth_apt(self) -> None:
        for modality in (
            AssertionModality.EVALUATIVE,
            AssertionModality.EXPERIENTIAL,
            AssertionModality.CONSTITUTIVE,
        ):
            p = make_particle(assertion_modality=modality)
            assert is_truth_apt(p) is False

    def test_roundtrip(self) -> None:
        p = make_particle(assertion_modality=AssertionModality.EXPERIENTIAL)
        restored = Particle.model_validate(p.model_dump())
        assert restored.assertion_modality == AssertionModality.EXPERIENTIAL

    def test_old_data_deserialises_to_falsifiable(self) -> None:
        # A particle serialized before this field existed has no
        # assertion_modality key; it must default to FALSIFIABLE (schema
        # freeze holds — old stores load unchanged).
        payload = make_particle().model_dump()
        del payload["assertion_modality"]
        restored = Particle.model_validate(payload)
        assert restored.assertion_modality == AssertionModality.FALSIFIABLE


class TestParticleProperties:
    """structured ontology properties field."""

    def test_properties_none_by_default(self) -> None:
        p = make_particle()
        assert p.properties is None

    def test_properties_accepted(self) -> None:
        p = make_particle(properties={"nmo:hasWeight": 0.75, "nmo:hasMaterial": "Aluminium"})
        assert p.properties["nmo:hasWeight"] == 0.75

    def test_properties_round_trips_json(self) -> None:
        import json

        props = {"nmo:hasWeight": 0.75, "nuds:references": ["KM# 1", "N# 8562"]}
        p = make_particle(properties=props)
        data = json.loads(p.model_dump_json())
        assert data["properties"]["nmo:hasWeight"] == 0.75
        assert data["properties"]["nuds:references"] == ["KM# 1", "N# 8562"]


class TestSubjectClass:
    """subject_class field."""

    def test_subject_class_none_by_default(self) -> None:
        from particles.core.schema import Subject

        s = Subject(canonical_name="Test", asserted_by="test")
        assert s.subject_class is None

    def test_subject_class_accepted(self) -> None:
        from particles.core.schema import Subject

        s = Subject(canonical_name="Aluminium", asserted_by="test", subject_class="nmo:Material")
        assert s.subject_class == "nmo:Material"

    def test_subject_class_round_trips_json(self) -> None:
        import json

        from particles.core.schema import Subject

        s = Subject(
            canonical_name="1 Pfennig", asserted_by="test", subject_class="nmo:NumismaticObject"
        )
        data = json.loads(s.model_dump_json())
        assert data["subject_class"] == "nmo:NumismaticObject"


class TestCorpusEntry:
    def test_basic(self) -> None:
        entry = CorpusEntry(source_type=SourceType.PDF, deposited_by="test")
        assert entry.deposited_by == "test"
        assert entry.entry_id  # auto-generated UUID

    def test_snapshots_default_empty(self) -> None:
        entry = CorpusEntry(source_type=SourceType.WEB_PAGE, deposited_by="test")
        assert entry.snapshots == []


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=50)
def test_confidence_roundtrip(v: float) -> None:
    """confidence.value survives round-trip encode/decode with <0.01 absolute error."""
    import json

    c = Confidence(value=v)
    data = json.loads(c.model_dump_json())
    assert abs(data["value"] - v) < 0.01


class TestQueryRequestQuestionBounds:
    """QueryRequest.question is length-bounded (security review F6)."""

    def test_empty_question_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(question="")

    def test_overlong_question_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(question="x" * 8193)

    def test_question_at_max_length_accepted(self) -> None:
        req = QueryRequest(question="x" * 8192)
        assert len(req.question) == 8192
