# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""JSON Schema validation for particles (§C.5).

Validates particles against artifacts/schemas/particle.schema.json, and
interchange units against artifacts/schemas/interchange.schema.json.

A record is judged against the contract this SDK knows (D7). The
record's own stamp decides how:

- the same or an older minor of the SDK's major is validated as it stands
  (an older minor passes by construction, since the schema only grows);
- a **newer** minor is validated with the keys the schema does not declare
  removed first, and the judgement notes that fields added after the SDK's
  minor were not checked. That is an informational note, never a failure;
- an incompatible stamp (another major, major ``0``, or a malformed stamp) is
  reported as not judgeable against this schema, which is a failure.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from particles.conformance._resources import schemas_dir
from particles.core import schema as _core_schema
from particles.core.schema_version import SchemaCompatibility, schema_compatibility

log = logging.getLogger(__name__)

_SCHEMA_FILE = schemas_dir() / "particle.schema.json"
_INTERCHANGE_SCHEMA_FILE = schemas_dir() / "interchange.schema.json"

#: The interchange definition a unit is judged against, by its ``@type``.
_INTERCHANGE_DEFS = {"Particle": "ParticleUnit", "Subject": "SubjectUnit"}


@dataclass
class RecordJudgement:
    """How one record stands against the schema this SDK bundles (D7).

    Attributes:
        schema_version: The record's own stamp, or ``None`` when it carries
            none (a subject unit, or a record missing the field, which the
            schema's ``required`` then reports).
        compatibility: How that stamp relates to the SDK's ``SCHEMA_VERSION``;
            ``None`` when there is no stamp to classify.
        errors: Schema violations, or the not-judgeable reason for an
            incompatible stamp. Empty means the record passes.
        unchecked_keys: Keys of a newer-minor record the schema does not
            declare, removed before validation, as dotted paths (``[]`` marks
            a list element).
        notes: Informational remarks that do not fail the record.
    """

    schema_version: str | None
    compatibility: SchemaCompatibility | None
    errors: list[str] = field(default_factory=list)
    unchecked_keys: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def judgeable(self) -> bool:
        """False when the stamp is incompatible: this schema cannot judge it."""
        return self.compatibility is None or self.compatibility.compatible

    @property
    def passed(self) -> bool:
        """True when the record is judgeable and the schema reports no error."""
        return self.judgeable and not self.errors


def sdk_schema_version() -> str:
    """The SDK's ``SCHEMA_VERSION``, read at call time so tests can patch it."""
    return _core_schema.SCHEMA_VERSION


def classify_record_version(stamp: object) -> SchemaCompatibility:
    """How a record's stamp relates to the bundled schema's (D1).

    A value that is not a string is ``MALFORMED``, like a string that does not
    parse.
    """
    if not isinstance(stamp, str):
        return SchemaCompatibility.MALFORMED
    return schema_compatibility(stamp, sdk_schema_version())


def not_judgeable_message(stamp: object, compatibility: SchemaCompatibility) -> str:
    """The failure an incompatible record reports instead of a verdict (D7)."""
    return (
        f"schema version {stamp!r} is incompatible with this SDK's schema "
        f"{sdk_schema_version()} ({compatibility.value}); the record is not "
        "judgeable against this schema."
    )


def newer_minor_note(stamp: str, unchecked: list[str]) -> str:
    """The note a newer-minor record carries: what this schema did not check (D7)."""
    keys = ", ".join(unchecked) if unchecked else "none present"
    return (
        f"record schema version {stamp} is a newer minor than this SDK's schema "
        f"{sdk_schema_version()}: fields added after {sdk_schema_version()} were not "
        f"checked (undeclared keys removed before validation: {keys})."
    )


def _resolve_ref(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    """Follow local ``#/...`` ``$ref`` pointers until a concrete subschema."""
    seen: set[str] = set()
    while isinstance(schema.get("$ref"), str) and schema["$ref"].startswith("#/"):
        ref = schema["$ref"]
        if ref in seen:
            break
        seen.add(ref)
        target: Any = root
        for part in ref[2:].split("/"):
            target = target.get(part, {}) if isinstance(target, dict) else {}
        schema = target if isinstance(target, dict) else {}
    return schema


def _object_branches(
    schema: dict[str, Any], root: dict[str, Any], key: str
) -> list[dict[str, Any]]:
    """The ``key`` combinator branches (``allOf`` / ``oneOf`` / ``anyOf``) that
    describe an object, with their refs resolved."""
    branches = schema.get(key)
    if not isinstance(branches, list):
        return []
    resolved = [_resolve_ref(b, root) for b in branches if isinstance(b, dict)]
    return [b for b in resolved if "properties" in b or b.get("type") == "object"]


def strip_undeclared(
    instance: Any, schema: dict[str, Any], root: dict[str, Any], path: str = ""
) -> tuple[Any, list[str]]:
    """Remove the keys ``schema`` does not declare, at every level it describes.

    Walks the schema beside the instance: an object whose subschema lists
    ``properties`` keeps those keys (and any ``patternProperties`` match) and
    loses the rest; list items follow ``items``; ``allOf`` branches all apply,
    and a ``oneOf`` / ``anyOf`` applies only when exactly one branch describes
    an object (the ``[{"$ref": …}, {"type": "null"}]`` nullable idiom). The
    instance is not mutated.

    Returns:
        The stripped copy, and the removed keys as sorted dotted paths.
    """
    schema = _resolve_ref(schema, root)
    removed: list[str] = []
    if isinstance(instance, dict):
        props = schema.get("properties")
        if isinstance(props, dict):
            patterns = schema.get("patternProperties")
            pattern_keys = list(patterns) if isinstance(patterns, dict) else []
            kept: dict[str, Any] = {}
            for key, value in instance.items():
                child = f"{path}.{key}" if path else key
                if key in props:
                    sub = props[key]
                    if isinstance(sub, dict):
                        value, inner = strip_undeclared(value, sub, root, child)
                        removed.extend(inner)
                    kept[key] = value
                elif any(re.search(p, key) for p in pattern_keys):
                    kept[key] = value
                else:
                    removed.append(child)
            instance = kept
        branches = _object_branches(schema, root, "allOf")
        for combinator in ("oneOf", "anyOf"):
            candidates = _object_branches(schema, root, combinator)
            if len(candidates) == 1:
                branches.extend(candidates)
        for branch in branches:
            instance, inner = strip_undeclared(instance, branch, root, path)
            removed.extend(inner)
    elif isinstance(instance, list):
        items = schema.get("items")
        if isinstance(items, dict):
            stripped: list[Any] = []
            for item in instance:
                item, inner = strip_undeclared(item, items, root, f"{path}[]")
                stripped.append(item)
                removed.extend(inner)
            instance = stripped
    # Sorted, and a key removed from several list elements is reported once.
    return instance, sorted(set(removed))


def _schema_errors(
    doc: Any, schema_file: Path, target: dict[str, Any] | None, *, draft7: bool
) -> list[str]:
    """Validate ``doc``; ``target`` replaces the root schema when given.

    Returns ``[]`` when valid, or when the schema file or the jsonschema
    library is missing (the check is skipped, with a warning).
    """
    if not schema_file.exists():
        log.warning("JSON Schema file not found: %s; validation skipped", schema_file)
        return []
    try:
        import jsonschema  # type: ignore[import-untyped]

        schema = json.loads(schema_file.read_text())
        effective = schema if target is None else target
        cls = jsonschema.Draft7Validator if draft7 else jsonschema.Draft202012Validator
        validator = cls(effective)
        return [e.message for e in validator.iter_errors(doc)]
    except ImportError:
        log.warning("jsonschema not installed; JSON Schema validation skipped")
        return []
    except Exception as exc:
        log.error("JSON Schema validation error: %s", exc)
        return [str(exc)]


def _load_schema(schema_file: Path) -> dict[str, Any] | None:
    if not schema_file.exists():
        return None
    loaded = json.loads(schema_file.read_text())
    return loaded if isinstance(loaded, dict) else None


def _judge(
    doc: dict[str, Any],
    *,
    version_key: str,
    schema_file: Path,
    target: dict[str, Any] | None,
    draft7: bool,
) -> RecordJudgement:
    """The shared D7 rule over one schema file (see the module docstring)."""
    if version_key not in doc:
        return RecordJudgement(
            schema_version=None,
            compatibility=None,
            errors=_schema_errors(doc, schema_file, target, draft7=draft7),
        )
    stamp = doc[version_key]
    compat = classify_record_version(stamp)
    shown = stamp if isinstance(stamp, str) else repr(stamp)
    if not compat.compatible:
        return RecordJudgement(
            schema_version=shown,
            compatibility=compat,
            errors=[not_judgeable_message(stamp, compat)],
        )
    if compat is not SchemaCompatibility.NEWER_MINOR:
        return RecordJudgement(
            schema_version=shown,
            compatibility=compat,
            errors=_schema_errors(doc, schema_file, target, draft7=draft7),
        )
    root = _load_schema(schema_file)
    unchecked: list[str] = []
    if root is not None:
        doc, unchecked = strip_undeclared(doc, target or root, root)
    return RecordJudgement(
        schema_version=shown,
        compatibility=compat,
        errors=_schema_errors(doc, schema_file, target, draft7=draft7),
        unchecked_keys=unchecked,
        notes=[newer_minor_note(shown, unchecked)],
    )


def judge_particle_dict(particle_dict: dict[str, Any]) -> RecordJudgement:
    """Judge a particle record against ``particle.schema.json`` (D7)."""
    return _judge(
        particle_dict,
        version_key="schema_version",
        schema_file=_SCHEMA_FILE,
        target=None,
        draft7=True,
    )


def judge_interchange_unit(unit: dict[str, Any]) -> RecordJudgement:
    """Judge an interchange unit against ``interchange.schema.json`` (D7).

    The unit is validated against the definition its ``@type`` names
    (``ParticleUnit`` or ``SubjectUnit``), or against the whole schema when
    the type is missing or unknown. ``ParticleUnit`` stays closed, so an
    undeclared key fails a unit of the SDK's own or an older minor (a
    misspelled key is a producer bug); a newer-minor unit has its undeclared
    keys removed and reported instead, as specification §10.1 requires.
    """
    target: dict[str, Any] | None = None
    definition = _INTERCHANGE_DEFS.get(str(unit.get("@type")))
    root = _load_schema(_INTERCHANGE_SCHEMA_FILE)
    if definition is not None and root is not None:
        target = {
            "$schema": root.get("$schema"),
            "$ref": f"#/$defs/{definition}",
            "$defs": root.get("$defs", {}),
        }
    return _judge(
        unit,
        version_key="schemaVersion",
        schema_file=_INTERCHANGE_SCHEMA_FILE,
        target=target,
        draft7=False,
    )


def _validate_against(doc: dict[str, Any], ref: str | None) -> list[str]:
    """Validate ``doc`` against the top-level schema (``ref=None``) or a ``$defs``
    subschema (e.g. ``ref="#/$defs/Subject"``). Returns error messages, or ``[]``
    when valid / when the schema file or jsonschema lib is missing."""
    target: dict[str, Any] | None = None
    if ref is not None:
        # Wrap the subschema ref with the shared $defs so internal refs (e.g.
        # ExternalRef) resolve.
        root = _load_schema(_SCHEMA_FILE) or {}
        target = {"$ref": ref, "$defs": root.get("$defs", {})}
    return _schema_errors(doc, _SCHEMA_FILE, target, draft7=True)


def validate_particle_dict(particle_dict: dict[str, Any]) -> list[str]:
    """Return a list of validation error messages, or an empty list if valid.

    Applies the D7 rule (:func:`judge_particle_dict`): a record of an
    incompatible ``schema_version`` fails as not judgeable, and a newer-minor
    record is checked on the fields this schema declares. Call
    :func:`judge_particle_dict` for the notes as well.
    """
    return judge_particle_dict(particle_dict).errors


def validate_interchange_unit(unit: dict[str, Any]) -> list[str]:
    """Error messages for an interchange unit (:func:`judge_interchange_unit`)."""
    return judge_interchange_unit(unit).errors


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
