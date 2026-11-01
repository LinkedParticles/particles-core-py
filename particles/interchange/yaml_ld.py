# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

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

from particles.core import bounded_yaml

# Import-bomb guardrail (security review F22), mirroring ``jsonl.py``. A crafted
# bundle could otherwise amplify memory/CPU at the import boundary: the whole
# member is buffered and parsed unbounded. These caps fail closed so an
# oversized member is refused. Kept identical to the JSONL caps so the container
# a bundle happens to use never changes its import ceiling: the reader builds
# one unit at a time, so a member at the cap costs about what a JSONL member
# at the cap does (security review 2026-10-28-2, AA16).
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
    parsing, and a document carrying more than ``_MAX_UNITS`` units, counted
    while the member is read. Each unit is built as soon as it is composed
    (:func:`particles.core.bounded_yaml.load_sequence`), so the member's peak
    memory stays near the units it builds, as the JSONL reader's does, rather
    than an order of magnitude past them (security review 2026-10-28-2, AA16). Anchors and
    aliases are kept (the writer emits them for shared values), but each alias
    counts as the nodes it stands for against a budget linear in the member's
    text (:func:`particles.core.bounded_yaml.load`), so a few alias lines cannot
    stand for millions of nodes (security review 2026-10-26-1, W20), and the
    units they decode to are held to the same byte cap, measured as compact
    JSON (:func:`_check_decoded_size`; security review 2026-10-25-2, R14). A document
    whose top level is not a sequence of mappings is a malformed member and
    raises :class:`ValueError` rather than being silently coerced.
    """
    byte_len = len(text.encode("utf-8"))
    if byte_len > _MAX_BUNDLE_BYTES:
        raise ValueError(
            f"interchange bundle exceeds the {_MAX_BUNDLE_BYTES}-byte import cap "
            f"({byte_len} bytes); refusing to parse (security review F22)."
        )

    try:
        # Built one unit at a time, so the composer's nodes for the whole
        # member are never held at once and the unit cap refuses while the
        # member is read (security review 2026-10-28-2, AA16).
        loaded = bounded_yaml.load_sequence(text, max_items=_MAX_UNITS)
    except bounded_yaml.YAMLItemLimitError as exc:
        raise ValueError(
            f"interchange bundle exceeds the {_MAX_UNITS}-unit import cap; "
            "refusing to parse (security review F22)."
        ) from exc
    except bounded_yaml.YAMLExpansionError as exc:
        raise ValueError(
            "interchange YAML-LD member expands past its node budget through "
            "anchors and aliases; refusing to decode (security review "
            "2026-10-26-1, W20)."
        ) from exc
    except yaml.YAMLError as exc:
        # A malformed member is refused as one, never raised past the import
        # (security review 2026-10-28-1, G17).
        raise ValueError(
            f"interchange YAML-LD member is not valid YAML ({type(exc).__name__}); "
            "refusing to decode."
        ) from exc
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
    _check_decoded_size(loaded, _MAX_BUNDLE_BYTES)
    units: list[dict[str, Any]] = []
    for item in loaded:
        if not isinstance(item, dict):
            raise ValueError(
                "interchange YAML-LD member must be a sequence of mapping units; "
                f"found a {type(item).__name__} entry. Refusing to decode."
            )
        units.append(item)
    return units


def _check_decoded_size(value: object, cap: int) -> None:
    """Refuse a decoded document whose compact-JSON size passes ``cap`` (R14).

    ``yaml.safe_load`` resolves an alias to the object its anchor built, so the
    loaded value is small however many times an alias repeats it; the size
    appears when a unit is serialized (a particle's ``properties`` at insert,
    its ``basis`` on dump). The walk counts every occurrence, as serialization
    would, and stops as soon as the count passes ``cap``, so its own cost is
    bounded by the cap. A self-referencing alias passes the cap the same way.
    """
    total = 0
    stack: list[object] = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            total += len(item.encode("utf-8")) + 2
        elif isinstance(item, dict):
            total += 2 + 2 * len(item)
            for key, child in item.items():
                stack.append(key if isinstance(key, str) else str(key))
                stack.append(child)
        elif isinstance(item, list | tuple | set):
            total += 2 + len(item)
            stack.extend(item)
        else:
            total += len(str(item))
        if total > cap:
            raise ValueError(
                f"interchange YAML-LD member decodes past the {cap}-byte import cap "
                "through anchors and aliases; refusing to decode (security review "
                "2026-10-25-2, R14)."
            )
