# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The Client/Engine config-layer declaration stays honest (D3).

Both distributions ship one config model — `particles/config.py` rides the
Client dist and the Engine has none of its own — so a core-alone install
carries every section including the ones nothing there reads. That surface is
made *legible* rather than carved, via `CLIENT_SECTIONS` plus per-section tags
in the sample. These are the checks that keep the declaration
from becoming a stale artifact nobody trusts.

Four of the six run anywhere, including in a published Client tree — after D2
both the model and the sample ship there. The two that read the export script's
allowlists are upstream-only: those lists are the single source of truth for the
layer boundary and by design never ship, and duplicating them here to make the
checks portable would create a second definition of the boundary.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest
import yaml

from particles.config import CLIENT_SECTIONS, ParticlesConfig

from ._upstream import IS_UPSTREAM, upstream_only

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "config.yaml.sample"

#: `name:` at column 0, with or without an inline value, capturing any tag.
_SECTION_RE = re.compile(r"^([a-z_][a-z0-9_]*):(?P<rest>.*)$", re.MULTILINE)
_TAG_RE = re.compile(r"#\s*\[(client|engine)\]")


def _sample_tags() -> dict[str, str | None]:
    """Top-level section name → its declared layer tag (None when untagged)."""
    out: dict[str, str | None] = {}
    for m in _SECTION_RE.finditer(SAMPLE.read_text(encoding="utf-8")):
        tag = _TAG_RE.search(m.group("rest"))
        out[m.group(1)] = tag.group(1) if tag else None
    return out


def _sections_read_by(path: Path) -> set[str]:
    """Top-level config sections read in *path*.

    Catches both `get_config().section` and the `cfg = get_config()` alias form.
    An alias is dropped the moment its name is rebound to anything else, so a
    later `cfg = get_config().http` cannot smuggle `http`'s own field names in
    as if they were sections.
    """
    found: set[str] = set()
    aliases: set[str] = set()

    def is_get_config(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "get_config"
        )

    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    (aliases.add if is_get_config(node.value) else aliases.discard)(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                (aliases.add if is_get_config(node.value) else aliases.discard)(node.target.id)
        elif isinstance(node, ast.Attribute) and (
            is_get_config(node.value)
            or (isinstance(node.value, ast.Name) and node.value.id in aliases)
        ):
            found.add(node.attr)

    return found & set(ParticlesConfig.model_fields)


def test_declared_client_sections_are_real_sections() -> None:
    """Every name in the declaration is an actual `ParticlesConfig` section."""
    invented = CLIENT_SECTIONS - set(ParticlesConfig.model_fields)
    assert not invented, f"CLIENT_SECTIONS names non-sections: {sorted(invented)}"


def test_sample_covers_every_section_exactly_once() -> None:
    """The sample documents all 64 sections and invents none (D2)."""
    assert set(_sample_tags()) == set(ParticlesConfig.model_fields)


def test_sample_is_loadable_as_written() -> None:
    """`cp config.yaml.sample config.yaml` produces a config that validates.

    The sample's own header tells the operator to copy it verbatim, and D2 now
    ships it to a second distribution — so a section whose every key is
    commented out (parsing to `None` rather than an empty mapping) breaks
    startup for anyone who follows the documented path.
    """
    loaded = yaml.safe_load(SAMPLE.read_text(encoding="utf-8"))
    ParticlesConfig.model_validate(loaded)


def test_sample_tags_agree_with_the_declaration() -> None:
    """Each section is tagged, and the tag matches `CLIENT_SECTIONS`."""
    mismatched = {
        name: tag
        for name, tag in _sample_tags().items()
        if tag != ("client" if name in CLIENT_SECTIONS else "engine")
    }
    assert not mismatched, (
        f"config.yaml.sample tags disagree with CLIENT_SECTIONS for {sorted(mismatched)}. "
        f"Retag the section, or add/remove it in CLIENT_SECTIONS if its layer really changed."
    )


@upstream_only
def test_client_substrate_reads_only_declared_sections() -> None:
    """A Client-layer module may only read a section the declaration names.

    Subset, not equality, and deliberately so: an *undeclared read* is the
    failure that matters — it means a core-alone consumer is acting on a
    section the sample tells them is inert. A declared section that no longer
    has a reader is a mild documentation inaccuracy, caught by the next
    measurement rather than by churning the frozenset every time a read moves
    between modules.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        from publish_export import _CLIENT_SUBSTRATE  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    sources: list[Path] = []
    for entry in _CLIENT_SUBSTRATE:
        target = REPO_ROOT / entry
        if target.is_dir():
            sources.extend(sorted(target.rglob("*.py")))
        elif target.suffix == ".py":
            sources.append(target)
    assert sources, "the Client substrate expanded to no Python sources"

    undeclared: dict[str, str] = {}
    for source in sources:
        for section in _sections_read_by(source) - CLIENT_SECTIONS:
            undeclared.setdefault(section, str(source.relative_to(REPO_ROOT)))

    assert not undeclared, (
        f"Client-layer modules read undeclared config sections: {undeclared}. "
        f"Add each to CLIENT_SECTIONS in particles/config.py and retag it "
        f"[client] in config.yaml.sample — a core-alone install acts on it."
    )


@pytest.mark.skipif(not IS_UPSTREAM, reason="needs the export script's allowlist")
def test_sample_rides_both_code_targets() -> None:
    """D2: the sample ships with the model it documents, not only the Engine.

    It belongs to the root metadata both code repos carry, not to the Client
    substrate — that list is the module carve, and every entry in it must be
    excluded from the Engine tree so exactly one wheel owns each package path.
    The sample is in neither wheel, so riding both trees is correct for it.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        from publish_export import (  # type: ignore[import-not-found]
            _CLIENT_SUBSTRATE,
            _CODE_ROOT_METADATA,
        )
    finally:
        sys.path.pop(0)

    assert "config.yaml.sample" in _CODE_ROOT_METADATA
    assert "config.yaml.sample" not in _CLIENT_SUBSTRATE
