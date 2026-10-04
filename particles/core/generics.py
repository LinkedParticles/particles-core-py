# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The deterministic generic-claim detector: does a claim range over a kind?

One pure detector, two readers:

- **Abstraction promotion** (addendum) rejects a candidate
  abstraction whose text ranges over a kind or a population rather than over
  its observed premise set. It calls :func:`population_quantifier`, which also
  flags an evidential clause standing in for a scope.
- **§6.6 and the lint candidate filters** exclude a pair in which
  exactly one side is a generic ("most mammals bear live young") and the other
  an instance claim ("the platypus lays eggs"): an exception does not falsify
  "most". They call :func:`generic_quantifier`, the kind-ranging half alone.
  An evidential clause says where an instance claim came from ("Alice uses vim,
  as the recorded sessions show"), not that it ranges over a kind, so it does
  not make the claim generic.

Both readers share every pattern below and the same order of tests, so the
detector is one rule read two ways, not two rules. English-only and
regex-only: no LLM call, no library outside the standard one (this is
``core/``).
"""

from __future__ import annotations

import re
from functools import lru_cache

#: Phrases that bind a quantifier to the observed premise set rather than to a
#: kind: "every backgrounded commit *in this store*", "the ten operators
#: *observed*", "*all three* deploys", "*these* parishioners". One anywhere in
#: the claim clears the pre-check, which then leaves the claim to the
#: entailment judge (the backstop for a marker used to dress up a generic).
#: ``those who`` is excluded: it names a kind ("most of those who attend").
_PREMISE_SCOPE_PATTERN = re.compile(
    r"""
      \b(?:observed|recorded|on\s+record|listed|cited|examined|sampled)\b
    | \bin\s+(?:this|the)\s+(?:store|corpus|record|records)\b
    | \bthese\b
    | \bthose\b(?!\s+who\b)
    | \bboth\b
    | \b(?:all|each|every\s+one|the|of\s+the)\s+
        (?:\d+|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

#: An evidential clause that cites the observations ("…, as these four records
#: show", "according to the recorded runs") without restricting the subject.
#: A marker inside one says where the claim came from, not what it ranges over,
#: so it never clears the pre-check; and a claim that leans on one with no
#: scope left outside it is rejected outright. The live judge reads such a
#: clause as a scope ("explicitly scoped to 'these four records'"), so this
#: shape is the pre-check's alone.
_EVIDENTIAL_CLAUSE_PATTERN = re.compile(
    r"""
      \b(?:as|which)\s+(?:these|those|the)\b[^,.;:]*?
        \b(?:show|shows|showed|shown|indicate|indicates|suggest|suggests
           |confirm|confirms|attest|attests|demonstrate|demonstrates)\b
    | \baccording\s+to\b[^,.;:]*
    """,
    re.IGNORECASE | re.VERBOSE,
)

#: A coordinated list of names carrying a floating quantifier ("Alice, Bob
#: and Carol *all* prefer vim") ranges over the named, observed members.
#: Case-sensitive on purpose: the capital is what marks a name.
_NAMED_SET_PATTERN = re.compile(r"(?:\band|&)\s+(?:[A-Z][\w'’-]*,?\s+){1,3}(?:all|each|both)\b")

#: Determiner quantifiers over a noun phrase: "most Christians", "all
#: mammals", "every operator", "everyone on the team". The negative
#: lookarounds drop the non-quantifying uses ("the most reliable", "at most",
#: "not at all", "after all", "each other", "most likely").
_DETERMINER_QUANTIFIER_PATTERN = re.compile(
    r"""
      (?<!\bthe\s)(?<!\bat\s)\bmost\b(?!\s+(?:likely|recent|recently)\b)
    | (?<!\bat\s)(?<!\bafter\s)(?<!\babove\s)(?<!\bfirst\sof\s)\ball\b
    | \bevery(?:one|body)?\b(?!\s+other\b)
    | \beach\b(?!\s+other\b)
    | \bthe\s+(?:vast\s+)?majority\s+of\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

#: Adverbial generic quantifiers. These quantify over a kind only when the
#: claim's subject is one ("Christians *generally* attend church"); over an
#: individual subject ("Alice *typically* uses vim") the pre-check leaves the
#: claim to the judge.
_GENERIC_ADVERB_PATTERN = re.compile(
    r"""
      \b(?:generally|typically|usually|normally|commonly|mostly|predominantly)\b
    | \bin\s+general\b
    | \bas\s+a\s+rule\b
    | \bby\s+and\s+large\b
    | \bon\s+the\s+whole\b
    | \btend(?:s)?\s+to\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

#: Words that open a determined noun phrase (an individual or a named set,
#: never a bare kind) — a subject starting with one is not a bare plural.
_DETERMINERS = frozenset(
    [
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "my",
        "our",
        "your",
        "their",
        "his",
        "her",
        "its",
        "some",
        "any",
        "no",
        "both",
        "either",
        "neither",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
    ]
)

#: Present-tense copulas, auxiliaries, and a few habitual verbs that close a
#: bare-plural generic subject ("mammals *bear* live young", "Xs *are* Y").
#: A generic whose verb is missing here falls to the entailment judge.
_GENERIC_VERBS = frozenset(
    [
        "are",
        "have",
        "do",
        "don't",
        "can",
        "cannot",
        "can't",
        "will",
        "won't",
        "must",
        "should",
        "tend",
        "need",
        "prefer",
        "bear",
        "lay",
        "live",
        "lack",
        "like",
        "use",
        "require",
        "believe",
        "hold",
        "favor",
        "favour",
        "support",
        "oppose",
        "vote",
        "attend",
        "eat",
        "own",
        "fail",
        "avoid",
        "rely",
    ]
)

#: Irregular plurals the ``-s`` heuristic cannot see.
_IRREGULAR_PLURALS = frozenset(
    ["people", "men", "women", "children", "folk", "mice", "geese", "cattle", "police"]
)

_FRONTED_ADVERBIAL_PATTERN = re.compile(
    r"^\s*(?:generally|typically|usually|normally|commonly|in\s+general|as\s+a\s+rule"
    r"|by\s+and\s+large|on\s+the\s+whole)\s*,?\s*",
    re.IGNORECASE,
)

_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z'’-]*")


def _plural_looking(word: str) -> bool:
    w = word.lower()
    if w in _IRREGULAR_PLURALS:
        return True
    return len(w) >= 4 and w.endswith("s") and not w.endswith(("ss", "us", "is"))


def _bare_plural_subject(claim: str) -> tuple[bool, str | None]:
    """Is the claim's subject a bare plural noun phrase (a kind)?

    Returns ``(is_kind, closing_word)``: the subject is the run of words
    before the first generic verb or generic adverb within the opening six
    words (after any fronted "Generally," adverbial), and it names a kind when
    it opens with no determiner, carries no possessive (an individual's
    things), and contains a plural-looking word. ``closing_word`` is the
    lower-cased word that ended the subject, or ``None`` when none did.
    """
    words = _WORD_PATTERN.findall(_FRONTED_ADVERBIAL_PATTERN.sub("", claim, count=1))
    subject: list[str] = []
    closing: str | None = None
    for word in words[:6]:
        lowered = word.lower()
        if lowered in _GENERIC_VERBS or _GENERIC_ADVERB_PATTERN.fullmatch(lowered):
            closing = lowered
            break
        subject.append(word)
    if closing is None or not subject:
        return False, closing
    if subject[0].lower() in _DETERMINERS:
        return False, closing
    if any("'" in w or "’" in w for w in subject):
        return False, closing
    return any(_plural_looking(w) for w in subject), closing


def _strip_evidential(claim: str) -> str:
    """The claim with every evidential clause blanked out."""
    return _EVIDENTIAL_CLAUSE_PATTERN.sub(" ", claim)


def _scope_binds(scoped_text: str) -> bool:
    """A premise-scope marker or a named set binds the claim to observed members."""
    return bool(
        _PREMISE_SCOPE_PATTERN.search(scoped_text) or _NAMED_SET_PATTERN.search(scoped_text)
    )


def _kind_shape(scoped_text: str) -> str | None:
    """The determiner quantifier or bare-plural generic in an unscoped claim."""
    determiner = _DETERMINER_QUANTIFIER_PATTERN.search(scoped_text)
    if determiner is not None:
        return determiner.group(0)
    is_kind, closing = _bare_plural_subject(scoped_text)
    if is_kind:
        adverb = _GENERIC_ADVERB_PATTERN.search(scoped_text)
        return adverb.group(0) if adverb is not None else f"bare plural subject + '{closing}'"
    return None


def generic_quantifier(claim: str) -> str | None:
    """The kind-ranging quantifier in a claim, or ``None`` for an instance claim.

    The first three shapes of :func:`population_quantifier`, under the same
    premise-scope exemption: a determiner quantifier over a noun phrase, an
    adverbial generic over a bare-plural subject, and a bare-plural subject
    closed by a present-tense generic verb. The fourth shape, an evidential
    clause standing in for a scope, is the abstraction pass's concern alone and
    is not read here: "Alice uses vim, according to the recorded sessions" is
    still a claim about Alice.

    The detector leans toward flagging, which for this reader means toward
    *not* adjudicating a pair. Two shapes are the known false flags: a name that
    only looks plural ("James typically uses vim"), and a quantifier inside an
    instance claim ("the release removed all deprecated flags"). Either costs a
    pair the ladder would have adjudicated and leaves both claims ACTIVE, the
    default-safe direction; nothing is retired or manufactured.

    Returns:
        The matched quantifier text, or ``None`` when the claim is not generic.
    """
    scoped_text = _strip_evidential(claim)
    if _scope_binds(scoped_text):
        return None
    return _kind_shape(scoped_text)


@lru_cache(maxsize=16384)
def is_generic_claim(claim: str) -> bool:
    """Whether a claim ranges over a kind (see :func:`generic_quantifier`).

    Cached: the pair filters ask about one stored claim once per pair it
    appears in, and the answer depends on the text alone.
    """
    return generic_quantifier(claim) is not None


def population_quantifier(claim: str) -> str | None:
    """The kind-ranging quantifier in a candidate abstraction, or ``None``.

    A promoted abstraction's text ranges over its observed premise set ("every
    backgrounded commit in this store", "the ten operators observed"), never
    over a kind or a population. Ten particles about ten members of a group
    support a claim about those ten; "most Xs are Y" is an inductive leap the
    engine would be making and asserting, which is where a system encodes a
    bias of its own rather than importing a source's. The entailment judge
    already rejects that leap, since a population generic is not entailed by
    the conjunction of its premises; this cheap deterministic pre-check
    rejects the plain cases first, so the judge call is not spent on them and
    is the backstop rather than the only line.

    Four shapes are flagged, each only when no premise-scope marker
    (``_PREMISE_SCOPE_PATTERN``) binds the claim to the observed set. A marker
    inside an evidential clause ("as these four records show") does not
    count: citing the observations is not restricting the subject to them.

    - a determiner quantifier over a noun phrase ("most Christians", "all
      mammals", "everyone on the team");
    - an adverbial generic over a bare-plural subject ("Christians generally
      attend", "engineers tend to prefer");
    - a bare-plural subject closed by a present-tense generic verb ("mammals
      bear live young", "Xs are Y");
    - an evidential clause standing in for a scope ("parishioners volunteer,
      as these four records show"), which the live judge accepts as scoped.

    Like the abstraction pass's time-anchor guard the detector is deliberately simple and leans
    toward rejection: a false rejection costs one un-promoted cluster, while a
    miss still meets the judge. A subject that only looks plural (a name
    ending in ``-s``) is the known false-rejection shape.

    Returns:
        The matched quantifier text for the run report, or ``None`` when the
        claim passes the pre-check.
    """
    evidential = _EVIDENTIAL_CLAUSE_PATTERN.search(claim)
    scoped_text = _strip_evidential(claim)
    if _scope_binds(scoped_text):
        return None
    shape = _kind_shape(scoped_text)
    if shape is not None:
        return shape
    return evidential.group(0) if evidential is not None else None
