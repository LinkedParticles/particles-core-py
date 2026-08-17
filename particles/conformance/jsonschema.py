"""JSON Schema validation for particles (§C.5).

Validates particles against artifacts/schemas/particle.schema.json.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from particles.conformance._resources import schemas_dir

log = logging.getLogger(__name__)

_SCHEMA_FILE = schemas_dir() / "particle.schema.json"


def _validate_against(doc: dict[str, Any], ref: str | None) -> list[str]:
    """Validate ``doc`` against the top-level schema (``ref=None``) or a ``$defs``
    subschema (e.g. ``ref="#/$defs/Subject"``). Returns error messages, or ``[]``
    when valid / when the schema file or jsonschema lib is missing."""
    if not _SCHEMA_FILE.exists():
        log.warning("JSON Schema file not found: %s; validation skipped", _SCHEMA_FILE)
        return []
    try:
        import jsonschema  # type: ignore[import-untyped]

        schema = json.loads(_SCHEMA_FILE.read_text())
        # ref=None → the top-level (Particle) schema; otherwise wrap the subschema
        # ref with the shared $defs so internal refs (e.g. ExternalRef) resolve.
        target = schema if ref is None else {"$ref": ref, "$defs": schema.get("$defs", {})}
        validator = jsonschema.Draft7Validator(target)
        return [e.message for e in validator.iter_errors(doc)]
    except ImportError:
        log.warning("jsonschema not installed; JSON Schema validation skipped")
        return []
    except Exception as exc:
        log.error("JSON Schema validation error: %s", exc)
        return [str(exc)]


def validate_particle_dict(particle_dict: dict[str, Any]) -> list[str]:
    """Return a list of validation error messages, or an empty list if valid."""
    return _validate_against(particle_dict, None)


def validate_subject_dict(subject_dict: dict[str, Any]) -> list[str]:
    """Validate a Subject dict against the ``$defs/Subject`` schema."""
    return _validate_against(subject_dict, "#/$defs/Subject")


def validate_particle(particle: object) -> list[str]:
    """Validate a Particle model instance against the JSON Schema."""
    from particles.core.schema import Particle as _Particle

    p: _Particle = particle  # type: ignore[assignment]
    d = json.loads(p.model_dump_json())
    return validate_particle_dict(d)


def validate_subject(subject: object) -> list[str]:
    """Validate a Subject model instance against the JSON Schema."""
    from particles.core.schema import Subject as _Subject

    s: _Subject = subject  # type: ignore[assignment]
    d = json.loads(s.model_dump_json())
    return validate_subject_dict(d)
