"""YAML-LD container for interchange units.

The human-editable single-document sibling of the canonical JSONL container
(``jsonl.py``). Same data model, same JSON-LD ``@context`` (``codec.py``) — only
the concrete syntax differs: one YAML document holding a sequence of the same
unit dicts, rather than one JSON object per line. YAML-LD **MUST round-trip** to
the canonical JSON-LD: ``read_yaml_ld(write_yaml_ld(units))``
reproduces the units exactly, so a store bundle written as YAML and re-imported
is byte-equivalent at the document-model level to the JSONL form.

This module is store-free (Client layer): it depends only on ``yaml``
and stdlib, exactly like ``jsonl.py``. It is a serialization container, not a
codec — it never touches a :class:`Particle`; ``codec.to_unit`` / ``from_unit``
own that translation and are shared verbatim with the JSONL path.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import yaml

# Import-bomb guardrail (security review F22), mirroring ``jsonl.py``. A crafted
# bundle could otherwise amplify memory/CPU at the import boundary: the whole
# member is buffered and parsed unbounded. These caps fail closed so an
# oversized member is refused. Kept identical to the JSONL caps so the container
# a bundle happens to use never changes its import ceiling.
_MAX_BUNDLE_BYTES = 256 * 1024 * 1024  # 256 MiB per member
_MAX_UNITS = 500_000  # per member


def write_yaml_ld(units: Iterable[dict[str, Any]]) -> str:
    """Serialize units to a single YAML-LD document (a top-level sequence).

    ``sort_keys=False`` preserves the codec's field order (``@context`` /
    ``@type`` first) so the YAML reads like the JSON-LD it mirrors and stays
    human-editable; ``allow_unicode=True`` mirrors ``jsonl``'s ``ensure_ascii=
    False`` so non-ASCII content is emitted verbatim rather than escaped.
    """
    return yaml.safe_dump(
        list(units),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


def read_yaml_ld(text: str) -> list[dict[str, Any]]:
    """Parse a YAML-LD document into its list of units.

    Accepts the single-document sequence :func:`write_yaml_ld` produces; an
    empty (or whitespace-only) document reads as no units. Fail-closed against
    import bombs (security review F22), matching :func:`particles.interchange.
    jsonl.read_jsonl`: refuses a member larger than ``_MAX_BUNDLE_BYTES`` before
    parsing, and a document carrying more than ``_MAX_UNITS`` units. A document
    whose top level is not a sequence of mappings is a malformed member and
    raises :class:`ValueError` rather than being silently coerced.
    """
    byte_len = len(text.encode("utf-8"))
    if byte_len > _MAX_BUNDLE_BYTES:
        raise ValueError(
            f"interchange bundle exceeds the {_MAX_BUNDLE_BYTES}-byte import cap "
            f"({byte_len} bytes); refusing to parse (security review F22)."
        )

    loaded = yaml.safe_load(text)
    if loaded is None:
        return []
    if not isinstance(loaded, list):
        raise ValueError(
            "interchange YAML-LD member must be a sequence of units at its top "
            f"level; got {type(loaded).__name__}. Refusing to decode."
        )
    if len(loaded) > _MAX_UNITS:
        raise ValueError(
            f"interchange bundle exceeds the {_MAX_UNITS}-unit import cap; "
            "refusing to parse (security review F22)."
        )
    units: list[dict[str, Any]] = []
    for item in loaded:
        if not isinstance(item, dict):
            raise ValueError(
                "interchange YAML-LD member must be a sequence of mapping units; "
                f"found a {type(item).__name__} entry. Refusing to decode."
            )
        units.append(item)
    return units
