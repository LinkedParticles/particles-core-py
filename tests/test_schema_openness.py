# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The published particle schemas stay open, and the stamp stays three numbers.

Two properties of the normative artifacts are invariants rather than
accidents (D7):

- ``particle.schema.json`` and the SHACL shapes are **open**: no
  ``additionalProperties: false`` at the root or in any ``$defs`` definition,
  and no ``sh:closed true``. The openness is what lets a 1.0 validator accept
  a 1.1 record, whose added optional fields it does not declare.
  (``interchange.schema.json`` is deliberately not covered: its
  ``ParticleUnit`` stays closed to catch a producer's misspelled key.)
- the ``schema_version`` / ``schemaVersion`` pattern requires
  ``MAJOR.MINOR.PATCH``, the only form the compatibility rule parses (D1).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from particles.core._resources import schemas_dir

_PARTICLE_SCHEMA = schemas_dir() / "particle.schema.json"
_SHACL_DIR = schemas_dir() / "shacl"
_PARTICLE_SHAPE = _SHACL_DIR / "ParticleShape.ttl"
_SH = "http://www.w3.org/ns/shacl#"


def _particle_schema() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(_PARTICLE_SCHEMA.read_text(encoding="utf-8"))
    return loaded


def _shape_files() -> list[Path]:
    return sorted(_SHACL_DIR.glob("*.ttl"))


def test_particle_schema_root_is_open() -> None:
    assert _particle_schema().get("additionalProperties") is not False


@pytest.mark.parametrize("name", sorted(_particle_schema().get("$defs", {})))
def test_particle_schema_definitions_are_open(name: str) -> None:
    definition = _particle_schema()["$defs"][name]
    assert definition.get("additionalProperties") is not False, (
        f"$defs/{name} sets additionalProperties: false; a 1.x validator would then "
        "reject a newer-minor record's added optional fields (D7)."
    )


def test_there_are_shapes_to_check() -> None:
    assert _shape_files(), f"no SHACL shapes under {_SHACL_DIR}"


@pytest.mark.parametrize("shape", _shape_files(), ids=lambda p: p.name)
def test_no_shacl_shape_is_closed(shape: Path) -> None:
    rdflib = pytest.importorskip("rdflib")

    graph = rdflib.Graph().parse(str(shape), format="turtle")
    closed = [
        subject
        for subject, _p, value in graph.triples((None, rdflib.URIRef(f"{_SH}closed"), None))
        if str(value).strip().lower() == "true"
    ]
    assert not closed, f"{shape.name} declares sh:closed true on {closed} (D7)"


def test_no_shacl_shape_is_closed_textually() -> None:
    """The same check over the text, so it holds without rdflib too."""
    for shape in _shape_files():
        text = shape.read_text(encoding="utf-8")
        uncommented = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
        assert not re.search(r"sh:closed\s+true", uncommented), shape.name


_WELL_FORMED = ["1.0.0", "1.2.3", "10.20.30", "0.3.0"]
_MALFORMED = ["1.2", "1", "1.2.3.4", "1.2.3-rc1", "v1.2.3", "1.x.0", "", " 1.2.3"]


def _assert_pattern_requires_three_numbers(pattern: str, where: str) -> None:
    compiled = re.compile(pattern)
    for stamp in _WELL_FORMED:
        assert compiled.search(stamp), f"{where} pattern {pattern!r} rejects {stamp!r}"
    for stamp in _MALFORMED:
        assert not compiled.search(stamp), f"{where} pattern {pattern!r} accepts {stamp!r}"


def test_particle_schema_version_pattern_is_major_minor_patch() -> None:
    prop = _particle_schema()["properties"]["schema_version"]
    assert "pattern" in prop, "particle.schema.json schema_version has no pattern"
    _assert_pattern_requires_three_numbers(prop["pattern"], "particle.schema.json")


def test_particle_shape_schema_version_pattern_is_major_minor_patch() -> None:
    rdflib = pytest.importorskip("rdflib")

    graph = rdflib.Graph().parse(str(_PARTICLE_SHAPE), format="turtle")
    path = rdflib.URIRef(f"{_SH}path")
    pattern = rdflib.URIRef(f"{_SH}pattern")
    patterns = [
        str(value)
        for node, _p, target in graph.triples((None, path, None))
        if str(target).endswith("schemaVersion")
        for _n, _q, value in graph.triples((node, pattern, None))
    ]
    assert patterns, "ParticleShape.ttl constrains schemaVersion with no sh:pattern"
    for found in patterns:
        _assert_pattern_requires_three_numbers(found, "ParticleShape.ttl")
