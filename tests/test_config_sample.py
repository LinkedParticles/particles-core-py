# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""`config.yaml.sample` says what the code actually does.

The sample is hand-maintained beside `particles/config.py`, and the
structural checks in `test_config_client_sections.py` pass a sample whose
*values* have gone stale. Three such drifts shipped unnoticed:

- a default widened in the model (``subjects.skip_live_authorities_source_types``
  gained ``MCP_MEMORY_EXPORT`` in 1.137.0) and not in the sample;
- two ``extraction`` fields documented under ``extraction_scope``, where the
  model silently ignores unknown keys, so editing them there changed nothing;
- ``http.user_agent`` pinned to a 0.3.0 literal that the default derives from
  the installed version.

So: every key the sample sets must be a real field, and every value it sets must
equal the compiled-in default. A deliberately non-default example belongs in a
comment, where copying the sample verbatim cannot pick it up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from particles.config import ParticlesConfig

SAMPLE = Path(__file__).resolve().parents[1] / "config.yaml.sample"


def _drift(
    raw: dict[str, Any], loaded: BaseModel, default: BaseModel, path: str = ""
) -> tuple[list[str], list[str]]:
    """Walk the sample's keys: (keys the model does not have, values off-default).

    Recurses only into sub-model fields. A ``dict[str, Model]`` field (e.g.
    ``llm.providers``) is compared whole, since its keys are data, not fields.
    """
    unknown: list[str] = []
    changed: list[str] = []
    fields = type(default).model_fields
    for key, value in raw.items():
        dotted = f"{path}{key}"
        if key not in fields:
            unknown.append(dotted)
            continue
        got, want = getattr(loaded, key), getattr(default, key)
        if isinstance(want, BaseModel) and isinstance(value, dict):
            u, c = _drift(value, got, want, f"{dotted}.")
            unknown += u
            changed += c
            continue
        got_j = got.model_dump(mode="json") if isinstance(got, BaseModel) else got
        want_j = want.model_dump(mode="json") if isinstance(want, BaseModel) else want
        if got_j != want_j:
            changed.append(f"{dotted}: sample={got_j!r} default={want_j!r}")
    return unknown, changed


def _sample_drift() -> tuple[list[str], list[str]]:
    raw = yaml.safe_load(SAMPLE.read_text(encoding="utf-8"))
    return _drift(raw, ParticlesConfig.model_validate(raw), ParticlesConfig())


def test_sample_sets_only_real_fields() -> None:
    """A key the model does not define is silently ignored at load time."""
    unknown, _ = _sample_drift()
    assert not unknown, (
        f"config.yaml.sample sets keys the config model does not define: {unknown}. "
        "Move each under the section whose model owns it, or delete it."
    )


def test_sample_values_match_the_compiled_defaults() -> None:
    """Copying the sample verbatim must not change behaviour."""
    _, changed = _sample_drift()
    assert not changed, (
        "config.yaml.sample disagrees with the defaults in particles/config.py:\n  "
        + "\n  ".join(changed)
        + "\nUpdate the sample, or turn a deliberate example into a comment."
    )
