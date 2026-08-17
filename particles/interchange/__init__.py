"""Particle interchange + store-export format.

Standard-tier wire format for federation exchange and store portability. The
codec is pure (no I/O); store-aware export/import wrappers layer on top.

Public surface:
    to_unit, from_unit          particle <-> JSON-LD interchange unit (codec)
    SubjectRef, ParsedUnit      codec value types
    write_jsonl, read_jsonl     the canonical JSON Lines container
    write_yaml_ld, read_yaml_ld the human-editable YAML-LD container
    FORMAT_VERSION, CONTEXT_URL

The store-aware half — ``export_particles``, ``export_active``,
``import_units``, ``export_store_bundle``, ``import_store_bundle``,
``restore_store_bundle``, ``ImportSummary``, ``RestoreSummary``,
``RestoreError`` — is Engine-layer and is imported from
:mod:`particles.interchange.store`, not from this package root. It ships in a
different distribution (D4), so re-exporting it here would make the
store-free Client distribution unimportable on its own.
"""

from __future__ import annotations

from pkgutil import extend_path

# Straddling package: `codec` / `jsonl` / `yaml_ld` ship in
# `linkedparticles-core`, which owns this file, while `store` ships in
# `linkedparticles`. See the note in `particles/__init__.py` (D1).
# Placement matters — these lines must follow the `__future__` import, which
# the language requires to be the first statement in the module.
__path__ = extend_path(__path__, __name__)

from .codec import (  # noqa: E402
    CONTEXT_URL,
    FORMAT_VERSION,
    ParsedUnit,
    SubjectRef,
    from_unit,
    to_unit,
)
from .jsonl import read_jsonl, write_jsonl  # noqa: E402
from .yaml_ld import read_yaml_ld, write_yaml_ld  # noqa: E402

__all__ = [
    "to_unit",
    "from_unit",
    "SubjectRef",
    "ParsedUnit",
    "write_jsonl",
    "read_jsonl",
    "write_yaml_ld",
    "read_yaml_ld",
    "FORMAT_VERSION",
    "CONTEXT_URL",
]
