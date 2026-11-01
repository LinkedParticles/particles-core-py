# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""JSON Lines container for interchange units.

One JSON-LD unit per line — the canonical, streamable particle-interchange
serialization. YAML-LD (a human-editable single-document form of the same data
model) is deferred.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

# Import-bomb guardrail (security review F22). A crafted bundle could otherwise
# amplify memory/CPU at the import boundary: the whole bundle is buffered and
# every line is ``json.loads``'d unbounded. These caps fail closed *while*
# reading so an oversized bundle is refused before it is fully parsed. Module
# constants for now; promote to config.py if runtime tuning is ever needed.
_MAX_BUNDLE_BYTES = 256 * 1024 * 1024  # 256 MiB per member
_MAX_UNITS = 500_000  # per member


#: Characters ``str.splitlines`` (and many JSON Lines readers) take as a line
#: break that ``json.dumps(ensure_ascii=False)`` writes raw inside a string.
_RAW_LINE_BREAKS = str.maketrans({"\u2028": "\\u2028", "\u2029": "\\u2029", "\x85": "\\u0085"})


def json_line(unit: Any) -> str:
    """``unit`` as one line of JSON Lines, with no character a reader splits on.

    A claim holding U+2028, U+2029 or U+0085 is written with those characters
    as JSON escapes, so the unit stays on its own line and the bundle reads
    back (security review 2026-10-30-1, AB35). They can only occur inside a
    string, where the escape is the same value.
    """
    return json.dumps(unit, ensure_ascii=False).translate(_RAW_LINE_BREAKS)


def write_jsonl(units: Iterable[dict[str, Any]]) -> str:
    """Serialize units to newline-delimited JSON (one unit per line)."""
    return "".join(json_line(unit) + "\n" for unit in units)


def read_jsonl(text: str) -> list[dict[str, Any]]:
    """Parse newline-delimited JSON units; blank lines are skipped.

    Fail-closed against import bombs (security review F22): refuses a member
    larger than ``_MAX_BUNDLE_BYTES`` or carrying more than ``_MAX_UNITS``
    non-blank lines, raising :class:`ValueError`. The byte cap is checked up
    front (the input is already a buffered ``str``, so this short-circuits the
    expensive ``json.loads`` per line); the unit cap is enforced incrementally
    so parsing stops at the limit rather than after building an oversized list.
    """
    byte_len = len(text.encode("utf-8"))
    if byte_len > _MAX_BUNDLE_BYTES:
        raise ValueError(
            f"interchange bundle exceeds the {_MAX_BUNDLE_BYTES}-byte import cap "
            f"({byte_len} bytes); refusing to parse (security review F22)."
        )

    units: list[dict[str, Any]] = []
    # Only a line feed ends a unit: a bundle an older writer wrote with a raw
    # U+2028 or U+0085 inside a string still reads back (AB35).
    for line in text.split("\n"):
        if not line.strip():
            continue
        if len(units) >= _MAX_UNITS:
            raise ValueError(
                f"interchange bundle exceeds the {_MAX_UNITS}-unit import cap; "
                "refusing to parse (security review F22)."
            )
        units.append(json.loads(line))
    return units
