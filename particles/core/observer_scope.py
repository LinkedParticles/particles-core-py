# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Observer scope — the set algebra of "for which observer is this belief in view".

A project is an observer. A belief's observer scope is never stored on the
claim: it is derived at read time from *where the belief was observed* — the
``project:<key>`` tags on the corpus entries its sources name. This module is
the **pure** half of that derivation (no I/O, no store, no config): it
classifies one entry's tags, folds a belief's entries into a scope, takes the
meet for a derived belief, and answers the visibility predicate. The Engine
half that joins particles to their entries lives in
``particles.operations.query.observer_scope``.

It is a membership **predicate**, not a weight: nothing here touches
``effective_confidence`` or the rank formula (the separation, and the
locality rule, which a predicate over a belief's own provenance
satisfies by construction).

Not part of the standard: the tag spelling and the harness-tag convention are
SDK conventions, which is why this sits beside, not inside, the normative
``core/scoring`` surface.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from enum import StrEnum

#: Corpus-entry tag prefix carrying a project key. The key is opaque here; the
#: harness adapter that stamps it decides what it names.
PROJECT_TAG_PREFIX = "project:"


def project_tag(key: str) -> str:
    """The corpus-entry tag that records ``key``."""
    return f"{PROJECT_TAG_PREFIX}{key}"


def project_keys(tags: Iterable[str] | None) -> frozenset[str]:
    """The project keys among an entry's tags (empty keys are ignored)."""
    if not tags:
        return frozenset()
    return frozenset(
        tag[len(PROJECT_TAG_PREFIX) :]
        for tag in tags
        if tag.startswith(PROJECT_TAG_PREFIX) and len(tag) > len(PROJECT_TAG_PREFIX)
    )


class EntryScope(StrEnum):
    """What one corpus entry says about where its claims were observed."""

    KEYED = "keyed"
    """Carries one or more project keys."""
    GLOBAL = "global"
    """Keyless and not harness-harvested: a hand deposit, a web page, a
    user-level rule file. Its claims are in view for every observer."""
    UNATTRIBUTED = "unattributed"
    """Harness-harvested but keyless. The fail-closed class: a stamping gap
    must not make a harness's deposits visible everywhere by accident."""


def classify_entry(tags: Iterable[str] | None, harness_tags: Collection[str]) -> EntryScope:
    """Classify one corpus entry from its tags.

    ``harness_tags`` are the tags a harness adapter puts on everything it
    harvests (``observer_scope.harness_tags``). Keyed wins over both other
    classes: an entry with a key is attributed, whoever deposited it.
    """
    tag_list = list(tags or ())
    if project_keys(tag_list):
        return EntryScope.KEYED
    if any(tag in harness_tags for tag in tag_list):
        return EntryScope.UNATTRIBUTED
    return EntryScope.GLOBAL


@dataclass(frozen=True)
class BeliefScope:
    """One belief's observer scope: global, or a set of project keys.

    ``is_global`` with any keys is still global. Neither global nor keyed is
    in view store-wide and for no project observer, for one of two reasons:
    **unattributed** — a stamping gap, the source was never keyed — or
    **lapsed** — sources did state it, and none still does.
    """

    is_global: bool = False
    keys: frozenset[str] = frozenset()
    lapsed: bool = False
    """Some source once stated the belief and none currently does. Only
    meaningful when the scope is neither global nor keyed."""

    @property
    def unattributed(self) -> bool:
        return not self.is_global and not self.keys and not self.lapsed

    @property
    def in_view_for_no_project(self) -> bool:
        return not self.is_global and not self.keys


GLOBAL_SCOPE = BeliefScope(is_global=True)
UNATTRIBUTED_SCOPE = BeliefScope()
LAPSED_SCOPE = BeliefScope(lapsed=True)


def scope_of_entries(
    entries_tags: Iterable[Iterable[str] | None], harness_tags: Collection[str]
) -> BeliefScope:
    """Fold the entries attesting one belief into its scope.

    Global as soon as one attesting entry is global; otherwise the union of
    the keyed entries' keys. No attesting entries at all is unattributed —
    the caller decides what a sourceless belief means (a derived belief takes
    :func:`meet` over its premises instead).
    """
    keys: set[str] = set()
    for tags in entries_tags:
        tag_list = list(tags or ())
        match classify_entry(tag_list, harness_tags):
            case EntryScope.GLOBAL:
                return GLOBAL_SCOPE
            case EntryScope.KEYED:
                keys |= project_keys(tag_list)
            case EntryScope.UNATTRIBUTED:
                pass
    return BeliefScope(keys=frozenset(keys))


def meet(premises: Iterable[BeliefScope]) -> BeliefScope:
    """The scope of a belief derived from ``premises``: their intersection.

    Global only if every premise is global (a global premise is the universal
    set, so it never narrows the others). An abstraction over one project's
    beliefs therefore stays in that project, and a contradiction between two
    projects' beliefs has an empty meet — it reaches the store-wide review and
    no project session. No premises at all is unattributed; an empty meet
    over a lapsed premise is lapsed.
    """
    keys: frozenset[str] | None = None
    seen = False
    lapsed = False
    for premise in premises:
        seen = True
        if premise.is_global:
            continue
        lapsed = lapsed or premise.lapsed
        keys = premise.keys if keys is None else keys & premise.keys
    if not seen:
        return UNATTRIBUTED_SCOPE
    if keys is None:
        return GLOBAL_SCOPE
    # An empty meet over a lapsed premise is itself lapsed, not a stamping gap.
    return BeliefScope(keys=keys, lapsed=lapsed and not keys)


def visible(scope: BeliefScope, observer_project: str | None, *, widened: bool = False) -> bool:
    """The predicate: is a belief with ``scope`` in view for this observer?

    No observer project is the store-wide view, for which everything is in
    view — the neutrality rule that keeps every unscoped read byte-identical.
    ``widened`` is an operator's standing judgement that the belief applies
    everywhere; it lives in lens-side policy, never on the claim.
    """
    if observer_project is None:
        return True
    return widened or scope.is_global or observer_project in scope.keys


class PairPrecondition(StrEnum):
    """Whether a (candidate, existing) pair may be reconciled."""

    RECONCILE = "reconcile"
    """Every project observing the existing claim also observes the candidate:
    the §6.6 ladder runs exactly as it would without observers."""
    DECLINE = "decline"
    """Another project observes the existing claim: no rung runs, nothing is
    quarantined, both stay ``ACTIVE``; a confirmed contradiction is recorded."""
    REVIEW = "review"
    """The existing claim is global and the candidate a project's: rung 2.5 is
    skipped and a confirmed contradiction falls through to ``INCONSISTENT``, so
    a project never silently retires the operator's own claim."""


def pair_precondition(candidate: BeliefScope, existing: BeliefScope) -> PairPrecondition:
    """``scope(existing) ⊆ scope(candidate)``, with the carve-outs.

    A global candidate (an operator's own deposit) and an unattributed one (a
    stamping gap, which must not switch reconciliation off) pair as today. An
    existing claim that is unattributed or lapsed has no observer to protect,
    so the subset holds trivially.
    """
    if candidate.is_global or not candidate.keys:
        return PairPrecondition.RECONCILE
    if existing.is_global:
        return PairPrecondition.REVIEW
    if existing.keys <= candidate.keys:
        return PairPrecondition.RECONCILE
    return PairPrecondition.DECLINE
