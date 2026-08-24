# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Locate the normative schema artifacts at runtime.

The JSON Schema, JSON-LD context, and SHACL shapes live in ``artifacts/schemas/``
at the repo root for source / editable use. A built wheel force-includes them at
``particles/_artifacts/schemas/`` (see the ``[tool.hatch.build.targets.wheel.force-include]``
mapping in ``pyproject.toml``) so a ``pip install`` carries them. Without that the
validators resolved to ``Path(__file__).parents[2]/artifacts/schemas`` — a path
that only exists in a source checkout — so for installed users conformance
validation silently no-opped and the JSON-LD context loader fell back to a stub.

Prefer the wheel-packaged copy; fall back to the source tree.

Lives in ``core`` rather than ``conformance`` (where it started) because two
Client subpackages now need it and ``extraction`` may not import ``conformance``
— that edge would close a new subpackage cycle against the ``acyclic_siblings``
contract, since ``conformance.validator`` already imports the extractor registry
. ``particles.conformance._resources`` re-exports it unchanged.
"""

from __future__ import annotations

from pathlib import Path


def schemas_dir() -> Path:
    """Return the directory holding the normative schema artifacts.

    Prefers the wheel-packaged ``particles/_artifacts/schemas``; falls back to the
    source-tree ``<repo-root>/artifacts/schemas`` for editable / source use. The
    caller still tolerates a missing file (the relevant validation layer skips with
    a warning), so a deliberately minimal fork that ships neither copy is unaffected.
    """
    packaged = Path(__file__).parent.parent / "_artifacts" / "schemas"
    if packaged.exists():
        return packaged
    return Path(__file__).parents[2] / "artifacts" / "schemas"
