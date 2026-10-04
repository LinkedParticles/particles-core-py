# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The probe-verdict key: what a remembered pairwise probe answer is keyed on.

A pairwise LLM probe asks one question about two claims. Its answer is a
record (D1): a fact about two contents at one prompt version, not a
property of either particle and not a relation between them. This module is
the pure half of that record, shared by the store that keeps it and the
operations that consult it:

* :class:`ProbeKind` names which question was asked. Each kind has its own
  prompt and is keyed separately.
* :func:`prompt_hash` turns the text of a prompt, rendered with placeholders,
  into the version string a record is keyed on. Each probe exposes its own
  hash from the one function that builds its prompt, so the key cannot drift
  from the text.
* :func:`verdict_key` turns two claim contents into the ordered pair of
  content hashes. Symmetric questions sort the pair; the update-slot question
  is directional (earlier, later) and keeps the order it was asked in.
* :func:`remembered_clear` decides whether a pair's recorded answers already
  clear it, so no new call is needed.
* :func:`subject_link_key` keys the one kind that is not a pair of claims: the
  subject-link judge's pick among a name's Wikidata candidates.

An edit to either claim changes its content hash and misses the record, and a
prompt change misses too. Pure: no I/O, no logging.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from enum import StrEnum

from particles.core.duplicate_key import content_hash

__all__ = [
    "ProbeKind",
    "VerdictKey",
    "prompt_hash",
    "remembered_clear",
    "subject_link_key",
    "verdict_key",
]

#: ``(hash_a, hash_b)``: the two claims' content hashes, in key order.
VerdictKey = tuple[str, str]


class ProbeKind(StrEnum):
    """Which pairwise question a recorded verdict answers."""

    #: The census probe: can both claims be true at once.
    CONTRADICTION = "contradiction"
    #: The second, context-rich reading of a census flag.
    SECOND_READING = "second_reading"
    #: The §6.6 reconcile probe the backlog sweeps ask first.
    RECONCILE_CONTRADICTION = "reconcile_contradiction"
    #: The update-slot probe: does the later claim give the earlier one's slot a
    #: new value. Directional.
    UPDATE_SLOT = "update_slot"
    #: The subject-link judge: which of a name's Wikidata candidates the claim
    #: means, or none. Keyed on the name and claim, then the
    #: candidate set, by :func:`subject_link_key`; its answer is a QID, not a
    #: yes or no.
    SUBJECT_LINK = "subject_link"

    @property
    def symmetric(self) -> bool:
        """True when the question does not depend on which claim is asked first."""
        return self in _SYMMETRIC


_SYMMETRIC = frozenset(
    {ProbeKind.CONTRADICTION, ProbeKind.SECOND_READING, ProbeKind.RECONCILE_CONTRADICTION}
)


def prompt_hash(*parts: str) -> str:
    """A short, stable version string for a prompt rendered with placeholders.

    The caller passes every piece of trusted text the probe sends (system turn,
    user template) with the variable parts replaced by fixed placeholders. Any
    change to that text changes the hash.
    """
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def verdict_key(kind: ProbeKind, content_a: str, content_b: str) -> VerdictKey:
    """The ordered pair of content hashes ``kind``'s verdict on two claims is keyed on.

    For a directional kind, ``content_a`` is the claim the prompt presents
    first (the earlier claim of an update).
    """
    a, b = content_hash(content_a), content_hash(content_b)
    if kind.symmetric and b < a:
        a, b = b, a
    return (a, b)


def subject_link_key(
    name: str, claim: str, candidates: Sequence[tuple[str, str, str]]
) -> VerdictKey:
    """The key a subject-link verdict is recorded under.

    ``hash_a`` covers the name and the claim text together, and ``hash_b`` the
    candidate set: each candidate's ``(id, label, description)`` in search-rank
    order, since the judge is shown them in that order. A changed claim, a
    changed search response, or a reordered one misses the record.
    """
    hash_a = content_hash(f"{name}\x1f{claim}")
    rendered = "\x1e".join("\x1f".join(candidate) for candidate in candidates)
    hash_b = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    return (hash_a, hash_b)


def remembered_clear(first: bool | None, second: bool | None) -> bool:
    """Whether recorded answers already clear a two-stage pair.

    Both two-stage checks act only when both stages answer YES: the census
    reports a flag only once a second reading confirms it, and the update sweep
    demotes only when the contradiction probe and the slot probe agree. A
    recorded NO from either stage therefore clears the pair, whatever the other
    stage would say now. ``None`` means no answer is recorded. For a one-stage
    check, pass ``second=None``.
    """
    return first is False or second is False
