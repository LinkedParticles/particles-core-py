# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Normative-artifact test for the interchange-envelope JSON Schema.

`artifacts/schemas/interchange.schema.json` is the normative JSON Schema for the
particle-interchange / store-export unit — the self-contained JSON-LD object the
codec emits. This mirrors the other artifact-schema guards
(`test_context_schema_sync.py`, `test_trust_lens.py::test_schema_artifact_matches_model`):
every unit the shipped codec (`particles/interchange/codec.py`) produces MUST
validate against the committed schema, so the artifact can never silently drift
behind the codec's envelope shape. `additionalProperties: false` throughout makes
the schema a precise mirror — a new emitted key that the schema does not cover
fails these tests until the schema is updated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped]
import pytest

from particles.core.schema import (
    AssertionModality,
    Confidence,
    ExternalRef,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    Subject,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.interchange import to_unit
from particles.interchange.codec import subject_to_unit

_SCHEMA_PATH = Path("artifacts/schemas/interchange.schema.json")


def _load_schema() -> dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text())


def _full_particle() -> Particle:
    """A particle exercising every optional substrate field the codec emits."""
    return Particle(
        content="Water is H2O.",
        confidence=Confidence(
            value=0.91,
            variance=0.01,
            calibration_source=CalibrationSource.CALIBRATED_BENCHMARK,
            calibration_method="temperature_scaling",
            calibration_ref="cal-1",
        ),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE,
                corpus_entry_id="e1",
                snapshot_id="s1",
                location="p3",
                chunk_hash="abc",
            )
        ],
        asserted_by="general-extractor",
        extractor_ref={"name": "general-extractor", "version": "0.6.0"},
        extraction_provider_model="anthropic:claude-opus-4-8",
        tags=["physics/chemistry"],
        properties={"nmo:hasWeight": 0.75},
        context_fingerprint="fp123",
        sequence_context=["intro", "body"],
        subject_ids=["sid-water", "sid-bare"],
        assertion_modality=AssertionModality.FALSIFIABLE,
    )


def _subjects() -> dict[str, Subject]:
    return {
        "sid-water": Subject(
            id="sid-water",
            canonical_name="Water",
            asserted_by="t",
            aliases=["H2O"],
            external_ids=[
                ExternalRef(
                    namespace="wikidata", id="Q283", uri="http://www.wikidata.org/entity/Q283"
                )
            ],
        ),
        "sid-bare": Subject(id="sid-bare", canonical_name="Local Thing", asserted_by="t"),
    }


def test_schema_is_a_valid_json_schema() -> None:
    """The committed artifact is itself a well-formed Draft 2020-12 schema."""
    jsonschema.Draft202012Validator.check_schema(_load_schema())


def test_full_particle_unit_validates() -> None:
    """A codec unit carrying every optional field validates against the schema."""
    unit = to_unit(_full_particle(), _subjects())
    errors = sorted(
        jsonschema.Draft202012Validator(_load_schema()).iter_errors(unit),
        key=lambda e: e.path,
    )
    assert not errors, "\n".join(f"{list(e.path)}: {e.message}" for e in errors)


def test_minimal_particle_unit_validates() -> None:
    """A particle with only required substrate still emits a schema-valid unit."""
    minimal = Particle(
        content="A minimal claim.",
        confidence=Confidence(value=0.5, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[ProvenanceRef(type=ProvenanceRefType.AGENT, corpus_entry_id="e1")],
        asserted_by="agent",
        subject_ids=["sid-1"],
    )
    unit = to_unit(minimal, {})
    jsonschema.Draft202012Validator(_load_schema()).validate(unit)


def test_subject_unit_validates() -> None:
    """A standalone subject unit (store-export bundle member) validates."""
    unit = subject_to_unit(_subjects()["sid-water"])
    jsonschema.Draft202012Validator(_load_schema()).validate(unit)


def test_unknown_top_level_key_is_rejected() -> None:
    """`additionalProperties: false` makes the schema a precise mirror: an
    unrecognised key fails, which is what forces the artifact to track the codec."""
    unit = to_unit(_full_particle(), _subjects())
    unit["surpriseKey"] = "not in the codec"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(_load_schema()).validate(unit)


def test_wrong_type_discriminator_fails() -> None:
    """A unit whose @type is neither Particle nor Subject matches neither branch."""
    unit = to_unit(_full_particle(), _subjects())
    unit["@type"] = "Widget"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(_load_schema()).validate(unit)
