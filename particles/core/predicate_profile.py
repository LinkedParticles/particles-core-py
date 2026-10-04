# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The predicate vocabulary's pure pieces: a canonical form, and a slot's kind.

Pure: no I/O. Letting reviewed predicate profiles answer the update rung's
slot question in the probe's place was proposed and declined after
measurement, since on two stores profiles found no pair to act on, and the
code that let a profile steer the rung was removed. What remains serves the
vocabulary documents a store keeps and exports:

* :func:`normalise_predicate`, the deterministic form a canonical predicate
  is keyed by;
* :class:`SlotKind` and :class:`PredicateRole`, what a reviewed profile
  records about a predicate;
* :func:`kind_from_source`, which reads a kind from the constraints an
  external vocabulary publishes (OWL, SHACL, Wikidata ``P2302``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "PredicateRole",
    "SlotKind",
    "SourceConstraints",
    "kind_from_source",
    "normalise_predicate",
]


class SlotKind(StrEnum):
    """What kind of slot a canonical predicate fills."""

    TIMELESS_SINGLE = "timeless_single"
    """One value for all time (a date of birth, who wrote a work). Two values
    are a contradiction no source date settles."""
    ONE_AT_A_TIME = "one_at_a_time"
    """One value at a time, changing over time (where someone lives). A later
    value may replace an earlier one; the slot probe still decides."""
    MANY_AT_ONCE = "many_at_once"
    """Several values at once (trips taken, co-authors). Two values are not a
    replacement."""


class PredicateRole(StrEnum):
    """What a member form says about its slot's value."""

    CURRENT = "current"
    """The form gives the slot's value (``move to``, ``lives in``)."""
    PAST = "past"
    """The form records a value the slot held before (``move from``)."""


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_ARTICLES = frozenset({"a", "an", "the"})
_BE = frozenset({"is", "are", "was", "were", "be", "been", "being", "am"})
_DO = frozenset({"does", "do", "did"})
_MODALS = frozenset({"can", "could", "will", "would", "shall", "should", "may", "might", "must"})
_HAVE = frozenset({"has", "have", "had", "having"})

#: Irregular forms → base verb. Applied before stemming, so an irregular past
#: and its base stem alike.
_IRREGULAR_PAIRS = """
    ran:run written:write wrote:write made:make built:build began:begin begun:begin
    brought:bring bought:buy caught:catch chose:choose chosen:choose came:come
    did:do done:do drew:draw drawn:draw drove:drive driven:drive found:find
    gave:give given:give went:go gone:go goes:go got:get gotten:get grew:grow
    grown:grow held:hold kept:keep knew:know known:know led:lead left:leave
    lost:lose meant:mean met:meet paid:pay said:say saw:see seen:see sent:send
    shown:show sold:sell spent:spend stood:stand taken:take took:take
    taught:teach told:tell thought:think understood:understand won:win wore:wear
    worn:wear fell:fall fallen:fall felt:feel fed:feed fought:fight forgot:forget
    forgotten:forget hid:hide hidden:hide rode:ride ridden:ride rose:rise
    risen:rise shot:shoot slept:sleep spoke:speak spoken:speak stole:steal
    stolen:steal struck:strike threw:throw thrown:throw woke:wake woken:wake
    broke:break broken:break has:have had:have having:have is:be are:be was:be
    were:be been:be am:be being:be does:do doing:do uses:use used:use using:use
    run:run put:put read:read set:set split:split cut:cut hit:hit let:let
    """
_IRREGULAR: Mapping[str, str] = dict(p.split(":", 1) for p in _IRREGULAR_PAIRS.split())
_PARTICIPLE_SUFFIXES = ("ed", "en")
_TOKEN = re.compile(r"[a-z0-9][a-z0-9'_./-]*")


def _strip_front(tokens: list[str]) -> list[str]:
    """Drop articles anywhere and auxiliaries / modals at the front.

    ``has`` / ``have`` / ``had`` is an auxiliary only before a participle
    (``has implemented``) or another auxiliary (``has been``); before a noun
    (``has status``) it is the main verb and stays.
    """
    toks = [t for t in tokens if t not in _ARTICLES]
    while len(toks) > 1:
        head, nxt = toks[0], toks[1]
        auxiliary = head in _BE or head in _DO or head in _MODALS
        perfect = head in _HAVE and (
            nxt in _BE or nxt.endswith(_PARTICIPLE_SUFFIXES) or nxt in _IRREGULAR
        )
        if not (auxiliary or perfect):
            break
        toks = toks[1:]
    return toks


def _undouble(stem: str) -> str:
    """``stopp`` → ``stop``, but not ``pass``, ``call`` or a stem left under three letters."""
    if len(stem) > 3 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
        return stem[:-1]
    return stem


def _stem(word: str) -> str:
    """An inflectional stem: every inflection of one verb maps to one string.

    Not a lemma: ``moved``, ``moves``, ``moving`` and ``move`` all become
    ``mov``. Equality is what the canonical predicate needs, and a stem gets
    it with no lexicon, where choosing a lemma (``cover`` but ``rename``)
    needs one.
    """
    word = _IRREGULAR.get(word, word)
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith(("sses", "ches", "shes", "xes", "zes")) or (
        word.endswith("ses") and len(word) > 4
    ):
        word = word[:-2]
    elif word.endswith("ied") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("ed") and len(word) > 3:
        word = _undouble(word[:-2])
    elif word.endswith("ing") and len(word) > 4:
        word = _undouble(word[:-3])
    elif word.endswith("s") and len(word) > 3 and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    if word.endswith("y") and len(word) > 2:
        word = word[:-1] + "i"
    while word.endswith("e") and len(word) > 2:
        word = word[:-1]
    return word


def normalise_predicate(value: str) -> str:
    """The form two predicates are compared by.

    Lowercased, articles dropped, leading auxiliaries and modals dropped, and
    the head verb reduced to its inflectional stem. Negation and prepositions
    are kept, so ``does not support`` stays apart from ``supports`` and
    ``moved to`` from ``moved from``. Deterministic and local to the one
    predicate: no other claim in the store changes its result.
    """
    toks = _strip_front(_TOKEN.findall(value.lower()))
    if not toks:
        return value.strip().lower()
    toks[0] = _stem(toks[0])
    return " ".join(toks)


# ---------------------------------------------------------------------------
# Reading the kind from the source
# ---------------------------------------------------------------------------

#: Wikidata ``P2302`` constraint items.
SINGLE_VALUE = "Q19474404"
SINGLE_BEST_VALUE = "Q52060874"
MULTI_VALUE = "Q21510857"
#: Qualifiers that date a value: start time, end time, point in time.
TIME_QUALIFIERS = frozenset({"P580", "P582", "P585"})


@dataclass(frozen=True)
class SourceConstraints:
    """What an aligned external property publishes about its cardinality.

    ``wikidata`` holds ``(constraint item, separator properties)`` for each
    ``P2302`` statement; the separators are its ``P4155`` qualifiers.
    """

    owl_functional: bool = False
    sh_max_count: int | None = None
    wikidata: tuple[tuple[str, frozenset[str]], ...] = ()


def kind_from_source(constraints: SourceConstraints) -> SlotKind | None:
    """The kind an external vocabulary states, or ``None`` when it states none.

    Read only where the source says values exclude one another or that
    several are expected. An allowed start or end time says a value can be
    dated, not that values exclude one another, so it yields nothing on its
    own. When sources disagree, the kind that keeps more wins: a timeless or
    many-at-once reading over a one-at-a-time one.
    """
    kinds: set[SlotKind] = set()
    if constraints.owl_functional or constraints.sh_max_count == 1:
        kinds.add(SlotKind.TIMELESS_SINGLE)
    for item, separators in constraints.wikidata:
        timed = bool(separators & TIME_QUALIFIERS)
        if item == SINGLE_VALUE:
            if timed:
                kinds.add(SlotKind.ONE_AT_A_TIME)
            elif not separators:
                kinds.add(SlotKind.TIMELESS_SINGLE)
        elif item == SINGLE_BEST_VALUE:
            kinds.add(SlotKind.ONE_AT_A_TIME if timed else SlotKind.TIMELESS_SINGLE)
        elif item == MULTI_VALUE:
            kinds.add(SlotKind.MANY_AT_ONCE)
    for keeping in (SlotKind.TIMELESS_SINGLE, SlotKind.MANY_AT_ONCE, SlotKind.ONE_AT_A_TIME):
        if keeping in kinds:
            return keeping
    return None
