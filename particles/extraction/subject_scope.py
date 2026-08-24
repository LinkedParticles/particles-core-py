# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Subject-scope contract — which claims are *expected* to carry a subject.

Two things live here, and the second is the load-bearing one:

1. ``extraction:subject_scope`` — the ``properties`` key an extractor sets to
   record that a claim's only available subject is the journal's **author**
   . It is a *record*, never a subject: it says "the subject of
   this claim is the author, and this store does not currently hold that
   Subject". The rule and its owner-resolved fork **O2** hold author
   resolution — byline → real-person Subject, *and/or* a ``journal:<handle>``
   Subject — behind sensitivity policy, because that step is what
   converts a latent dossier into a named one. This key does not cross that
   gate; it marks the population so that closing the gate later is a *backfill*
   rather than a re-extraction.

2. :func:`subject_expected` — techspec §9's zero-subject table, made
   executable. Both the ``L-STR-09`` lint rule and the conformance
   validator ask "should this particle have a subject?", and before
   they answered differently: §9 enumerates legitimate zero-subject cases and
   lint honoured three of them, while §14.5 measured a 100 % floor over every
   emitted particle with no exclusions at all. One predicate, one answer.

The ``extraction:`` prefix is rung 4 — a signal the extractor
**computes about a claim**, never one it reads from a source. This module sits
beside ``scope.py`` and ``polarity.py``, which solve the
identical shape for their own axes.

**The key must never be set to clear a conformance floor.** It asserts that a
claim's subject is the author; on a claim about the world that is a false
record, and :func:`subject_expected` makes false records load-bearing.
"""

from __future__ import annotations

from particles.core.schema import ParticleType
from particles.extraction.polarity import is_non_asserted
from particles.extraction.scope import is_excluded_document_meta

#: Key on the ``properties`` dict carrying the subject-scope record. Prefixed
#:; the ``extraction:`` namespace is registered in §5.
SUBJECT_SCOPE_KEY = "extraction:subject_scope"

#: The one value that marks a claim as author-scoped. Absence of the key means
#: the ordinary case — the claim is about the world, and its subjects are
#: whatever the extractor resolved.
SUBJECT_SCOPE_SELF = "SELF"


def is_self_scoped(properties: dict[str, object] | None) -> bool:
    """True when the claim's only available subject is the journal's author."""
    if not properties:
        return False
    return str(properties.get(SUBJECT_SCOPE_KEY, "")).upper() == SUBJECT_SCOPE_SELF


def subject_expected(particle_type: ParticleType, properties: dict[str, object] | None) -> bool:
    """Should this particle carry at least one subject? (techspec §9)

    False for the four populations the spec already treats as legitimately
    subjectless — a zero-subject particle here is the honest state, not a gap:

    * **Non-CLAIM types.** A REVIEW audit record carries no subjects by design;
      a NARRATIVE labels an entry rather than an entity.
    * **DOCUMENT_META claims** — scoped to the source document, not
      to a subject.
    * **Non-asserted claims** — a DECLINED or HYPOTHETICAL
      proposition is off the factual surface entirely.
    * **Author-scoped claims** — the subject exists but the store
      is deliberately not permitted to hold it.

    True otherwise, including for an extraction whose subject resolution simply
    produced nothing: that is the gap both `L-STR-09` and the conformance
    `subject_ids` floor exist to surface, and excluding it would be excluding
    the signal.
    """
    if particle_type != ParticleType.CLAIM:
        return False
    if is_excluded_document_meta(properties):
        return False
    if is_non_asserted(properties):
        return False
    return not is_self_scoped(properties)
