# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""A YAML-LD member's peak memory stays near the JSON Lines reader's.

``read_yaml_ld`` builds each unit as it is composed. Two shapes kept every
composed node alive anyway: a member whose units each carry an anchor, and a
member whose top level is a mapping (composed whole, then refused). Both now
peak within three times the JSON Lines reader per input byte (security review
2026-10-30, AB16).
"""

from __future__ import annotations

import contextlib
import json
import tracemalloc

import yaml

_UNIT = {
    "@type": "Particle",
    "id": "p-x",
    "content": "c" * 40,
    "tags": ["a", "b", "c"],
    "properties": {"k": "v", "n": 1},
}
# Peak per input byte does not depend on the unit count (measured 750 to 3000:
# the same ratio to the JSONL reader within 0.1), so a thousand units pin it at
# a third of the cost under ``tracemalloc``.
_N = 1000


def _units() -> list[dict[str, object]]:
    return [dict(_UNIT, id=f"p-{i}") for i in range(_N)]


def _peak_per_byte(text: str) -> float:
    from particles.interchange.yaml_ld import read_yaml_ld

    tracemalloc.start()
    try:
        with contextlib.suppress(ValueError):
            read_yaml_ld(text)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return peak / len(text.encode())


def _jsonl_peak_per_byte() -> float:
    from particles.interchange.jsonl import read_jsonl

    text = "\n".join(json.dumps(u) for u in _units())
    tracemalloc.start()
    try:
        read_jsonl(text)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return peak / len(text.encode())


def _indent(block: str) -> str:
    return "".join("  " + line + "\n" for line in block.splitlines())


def test_anchored_units_peak_stays_near_the_jsonl_reader() -> None:
    """A member whose every unit carries an anchor peaks near the JSONL reader per byte."""
    text = "".join(
        f"- &u{i}\n" + _indent(yaml.safe_dump(u, sort_keys=False, default_flow_style=False))
        for i, u in enumerate(_units())
    )
    assert _peak_per_byte(text) <= 3 * _jsonl_peak_per_byte()


def test_top_level_mapping_is_refused_before_it_is_composed_whole() -> None:
    """A member whose top level is a mapping is refused without composing every node first."""
    from particles.interchange.yaml_ld import read_yaml_ld

    body = yaml.safe_dump(_units(), sort_keys=False, default_flow_style=False)
    text = "root:\n" + _indent(body)
    try:
        read_yaml_ld(text)
    except ValueError as exc:
        assert "sequence of units" in str(exc)
    else:
        raise AssertionError("a top-level mapping was accepted")
    assert _peak_per_byte(text) <= 3 * _jsonl_peak_per_byte()


def test_plain_sequence_peak_stays_near_the_jsonl_reader() -> None:
    """The plain top-level sequence the writer emits meets the same bar."""
    text = yaml.safe_dump(_units(), sort_keys=False, default_flow_style=False)
    assert _peak_per_byte(text) <= 3 * _jsonl_peak_per_byte()
