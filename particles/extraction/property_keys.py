# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The `properties`-key shape, as one shared predicate.

The spec requires every ``Particle.properties`` key to use a
``prefix:LocalName`` shape, so a consumer can determine a key's provenance and
applicability from the key alone. Bare keys (``score``, ``polarity``) are
disallowed.

Until the rule was checked in exactly one place — inside a
``particles extractor conform`` run, over candidates an extractor produces from
a **fixture**. That never looks at a store. Which is why the bare ``polarity`` /
``scope`` keys survived from their introduction until the first
general-extractor baseline capture in 1.109.0: no
operator-facing surface read the store and asked the question, and three routes
reach the store without a conform run — an interchange import, a third-party
extractor, and a store predating a convention change.

The predicate lives here rather than in ``conformance`` so both the fresh-output
check (``conformance.validator``) and the persisted-particle check
(``operations.lint.coverage``, ``L-STR-12``) can share it. It sits alongside the
``scope`` / ``polarity`` contracts for the same reason they do: a small
dependency-free rule about the ``properties`` dict, shared by a producer and its
consumers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def bare_properties_keys(properties: Mapping[str, Any] | None) -> list[str]:
    """Return the keys of ``properties`` that lack a ``:`` separator.

    Order-preserving, so a report reads in the order the producer wrote them.
    An absent or empty dict has no violations.

    The check is deliberately the weakest one that catches the real failure —
    presence of a colon, not membership in the registry. An
    unregistered prefix is a governance question with a documented answer (add
    the row); a *bare* key is unattributable to any namespace at all, which is
    the thing a consumer cannot work around.
    """
    if not properties:
        return []
    return [key for key in properties if ":" not in key]
