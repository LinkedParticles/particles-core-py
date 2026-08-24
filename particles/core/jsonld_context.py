# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The published JSON-LD context, read as the one CURIE-prefix authority.

:class:`~particles.core.schema.TermKind` defines a ``URI`` term as "an absolute
IRI, **or a CURIE in a ``context.jsonld`` prefix**", and §6.8's prefix registry
requires every ``Particle.properties`` key to carry a prefix. Three call sites were
answering "is this prefix published?" from three independently hand-maintained
tuples, and all three disagreed with the artifact — ``extraction.structure``
claimed ``wd: wdt: nm: schema: rdfs: owl:``, ``extraction.numista._shared``
claimed ``schema:``, and the file published none of them. A predicate spelled
with an unpublished prefix is a ``URI`` term nothing can expand, which is the
failure the ``URI``-term rule exists to avoid (§6.8).

So the question is answered once, here, by reading the artifact.

**Absent artifact ⇒ no published prefixes.** A deliberately minimal fork that
ships neither copy of ``artifacts/schemas/`` gets an empty tuple and every CURIE
degrades to ``TOKEN`` — which is the honest reading of the rule, not a
degradation to paper over: with no published context, nothing *can* expand a
CURIE. A wheel force-includes the artifact (``core._resources``), so this branch
is a fork's choice rather than an install accident.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache

from particles.core._resources import schemas_dir

log = logging.getLogger(__name__)

#: A term maps a prefix only when its value is a plain IRI string ending in a
#: gen-delim (JSON-LD 1.1 §4.1.5 — a simple term definition is usable as a
#: prefix when its value ends in a gen-delim). Terms carrying an expanded
#: definition (``{"@id": …, "@type": …}``) and value terms (``"ALEATORY":
#: "psum:ALEATORY"``) are not prefixes and are skipped.
_GEN_DELIMS = ("#", "/", ":", "?", "[", "]", "@")


@lru_cache(maxsize=1)
def published_prefixes() -> tuple[str, ...]:
    """Return every CURIE prefix the published context can expand, ``"prefix:"``-shaped.

    Sorted, so callers get a stable order. Cached: the artifact is immutable at
    runtime and the readers are on per-candidate paths.

    Note the one prefix the artifact **cannot** carry: the §6.8
    prefix registry lists ``content:`` (generic content metadata —
    ``content:hasUrl``), but the term
    ``content`` is already bound to ``particles:content``, and a term has exactly
    one definition. Publishing it as a prefix would expand ``content:hasUrl`` to
    ``particles:contenthasUrl``. It stays unpublished, so a ``content:`` CURIE is
    a ``TOKEN`` — harmless today, since ``properties`` is an opaque ``@json``
    payload no code path expands.
    """
    path = schemas_dir() / "context.jsonld"
    try:
        ctx = json.loads(path.read_text())["@context"]
    except (OSError, ValueError, KeyError):
        log.warning("context.jsonld unreadable at %s; no CURIE prefix is published", path)
        return ()
    return tuple(
        sorted(
            f"{term}:"
            for term, value in ctx.items()
            if isinstance(value, str) and value.endswith(_GEN_DELIMS)
        )
    )


def is_published_curie(value: str) -> bool:
    """True when ``value`` is a CURIE whose prefix the published context expands."""
    return value.startswith(published_prefixes())
