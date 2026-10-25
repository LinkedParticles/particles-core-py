# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""No code builds a ``config.yaml`` path except the loader.

The loader reads ``PARTICLES_CONFIG`` or the user file and never the working
directory. Code that built ``Path.cwd() / "config.yaml"`` itself (the
installer did, until) would read or write a file a cloned repository
controls, so every construction outside ``particles/config.py`` fails here,
in the shape of the egress chokepoint test.
"""

from __future__ import annotations

import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "particles"

#: A ``config.yaml`` path built by joining or by ``Path("…config.yaml")``.
_CONSTRUCTION = re.compile(r"""/\s*["']config\.yaml["']|Path\(\s*["'][^"']*config\.yaml["']\s*\)""")

#: The named exemptions, and the one criterion for adding to the list: the file
#: is written by the code that pins it with ``PARTICLES_CONFIG``, in a directory
#: that code created, and nothing reads it except through that pin. A new entry
#: is a reviewed edit to this list, never a pattern.
EXEMPT = {
    "config.py",  # the loader itself
    # Writes its pinned config into the run's work_dir and hands it to each
    # child process through PARTICLES_CONFIG.
    "benchmark/query_latency/runner.py",
    # Writes one into each case's run_dir, beside a private HOME, pinned the same way.
    "benchmark/update_gate/case.py",
}


def config_path_constructions(source: str) -> list[int]:
    """The 1-based line numbers of ``source`` that build a ``config.yaml`` path."""
    return [n for n, line in enumerate(source.splitlines(), 1) if _CONSTRUCTION.search(line)]


def test_only_the_loader_and_the_named_exemptions_build_a_config_path() -> None:
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE).as_posix()
        if rel in EXEMPT:
            continue
        for line in config_path_constructions(path.read_text(encoding="utf-8")):
            offenders.append(f"particles/{rel}:{line}")
    assert offenders == [], (
        "config.yaml path built outside the loader; "
        "ask particles.config for the path instead: " + ", ".join(offenders)
    )


def test_every_exemption_still_builds_one() -> None:
    """An exemption that no longer needs its entry is a dead entry; remove it.

    The two published distributions each carry part of the package, so an
    exempt file a tree does not carry is skipped rather than failed.
    """
    for rel in EXEMPT:
        if not (PACKAGE / rel).exists():
            continue
        source = (PACKAGE / rel).read_text(encoding="utf-8")
        assert config_path_constructions(source), f"particles/{rel} no longer needs its exemption"


def test_the_installer_shape_is_caught() -> None:
    assert config_path_constructions('    return Path.cwd() / "config.yaml"\n') == [1]
    assert config_path_constructions("cfg = Path('./config.yaml')\n") == [1]
    assert config_path_constructions('"""Edit config.yaml by hand."""\n') == []
