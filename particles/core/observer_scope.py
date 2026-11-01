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

import unicodedata
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from enum import StrEnum

#: Corpus-entry tag prefix carrying a project key. The key is opaque here; the
#: harness adapter that stamps it decides what it names.
PROJECT_TAG_PREFIX = "project:"

#: The compiled-in tag that makes an entry *unattributed* whatever
#: ``observer_scope.harness_tags`` says: agent material that no
#: project key was stamped on. A code constant rather than config, so no
#: ``config.yaml`` can leave it out. A project key still outranks it, and it
#: stays on an entry that gains one, as the record that the entry began as an
#: agent write.
UNATTRIBUTED_MARKER = "scope:unattributed"


def project_tag(key: str) -> str:
    """The corpus-entry tag that records ``key``."""
    return f"{PROJECT_TAG_PREFIX}{key}"


#: Letters and symbols that render as nothing: the Hangul fillers, the braille
#: blank and the musical null notehead (security review 2026-10-27-1, G74;
#: 2026-10-28-1, G65).
_BLANK_GLYPHS = frozenset("\u115f\u1160\u3164\uffa0\u2800\U0001d159")

#: The categories of a character a reader can see: letters, numbers,
#: punctuation and symbols.
_VISIBLE_CATEGORIES = frozenset("LNPS")

#: Unicode's default-ignorable code points (DerivedCoreProperties.txt): a
#: renderer draws none of them, so a key holding one reads as the key without
#: it. The variation selectors and the Khmer inherent vowels are among them
#: (security review 2026-10-28-1, G65).
_DEFAULT_IGNORABLE = (
    (0x00AD, 0x00AD),
    (0x034F, 0x034F),
    (0x061C, 0x061C),
    (0x115F, 0x1160),
    (0x17B4, 0x17B5),
    (0x180B, 0x180F),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x206F),
    (0x3164, 0x3164),
    (0xFE00, 0xFE0F),
    (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0),
    (0xFFF0, 0xFFF8),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0000, 0xE0FFF),
)

#: Scripts that one name legitimately mixes: Japanese writes kanji beside both
#: kana, and Korean may write hanja beside hangul. The other entries name one
#: script under two spellings of its Unicode names (the Arabic-Indic digits are
#: Arabic's; the ideographic number zero is Han's).
_ONE_SCRIPT = {
    "HIRAGANA": "CJK",
    "KATAKANA": "CJK",
    "KATAKANA-HIRAGANA": "CJK",
    "HANGUL": "CJK",
    "BOPOMOFO": "CJK",
    "IDEOGRAPHIC": "CJK",
    "CJK": "CJK",
    "ARABIC-INDIC": "ARABIC",
    "EXTENDED ARABIC-INDIC": "ARABIC",
}

#: The words a character's Unicode name uses after its script's name
#: (``OLD ITALIC LETTER A``, ``DEVANAGARI DIGIT ZERO``), so the script is every
#: word before the first of them.
_AFTER_SCRIPT = frozenset(
    {
        "LETTER",
        "DIGIT",
        "NUMBER",
        "NUMERAL",
        "SYLLABLE",
        "SYLLABICS",
        "CHARACTER",
        "LIGATURE",
        "VOWEL",
        "CONSONANT",
        "INDEPENDENT",
        "SIGN",
        "CAPITAL",
        "SMALL",
        "FRACTION",
        "LOGOGRAM",
        "HIEROGLYPH",
    }
)


#: Categories that never belong in a key a reader can check: controls,
#: private-use and unassigned code points, and the line and paragraph
#: separators. Each renders as nothing, as a box, or as a break, so a key
#: holding one reads as the key without it (security review 2026-10-30-1,
#: AB32). A space separator other than U+0020 is refused beside them.
_HIDDEN_CATEGORIES = frozenset({"Cc", "Co", "Cn", "Zl", "Zp"})

#: Letters of the Latin script that Unicode's confusables data maps to an
#: ASCII letter or mark: a key spelt with one reads as the ASCII key within one
#: script, which the mixed-script rule cannot see (security review
#: 2026-10-30-1, AB32). A hand-kept list of the common ones, not the whole
#: table.
_LATIN_CONFUSABLES = frozenset(
    "\u0131\u0237\u0251\u0261\u0269\u026a\u028f\u01c0\u01c3\u02bc"
    "\u1d04\u1d0f\u1d1c\u1d20\u1d21\u1d22\ua731"
)

#: Non-ASCII punctuation marks and symbols that read as an ASCII one: the
#: quotation marks, primes, slashes, the minus sign, the ratio and the
#: Armenian and Hebrew stops. Every dash (category Pd) is refused beside them.
#: Refused in any key, not only beside ASCII characters, so a Unicode hyphen
#: in a key of another script is refused too (security review 2026-10-30-1,
#: AB32).
_ASCII_LOOKALIKE_MARKS = frozenset(
    "\u02b9\u02ba\u02c2\u02c3\u02c4\u02c6\u02c8\u02cb\u02d0\u02d7\u02dc"
    "\u0589\u05c3\u2018\u2019\u201a\u201b\u201c\u201d\u201e\u201f"
    "\u2032\u2033\u2035\u2039\u203a\u2044\u2212\u2215\u2216\u2217"
    "\u2223\u2236\u223c\u29f8\u29f9"
)


def _default_ignorable(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _DEFAULT_IGNORABLE)


def _script(ch: str) -> str | None:
    """A letter's or number's script, read from its Unicode name (``LATIN``, ``OLD ITALIC``).

    The script is the words of the name before the word that says what the
    character is (``LETTER``, ``DIGIT``, ``SIGN``), so ``OLD ITALIC`` and ``OLD
    PERSIAN`` are two scripts. Numbers count as letters do: a Devanagari digit
    is Devanagari's (security review 2026-10-28-2, AA24). The ASCII digits are
    every script's, and have none.
    """
    if unicodedata.category(ch)[0] not in "LN" or ("0" <= ch <= "9"):
        return None
    words = unicodedata.name(ch, "").split(" ")
    if not words[0]:
        return None
    taken: list[str] = []
    for word in words:
        if word in _AFTER_SCRIPT or word.startswith("IDEOGRAPH"):
            break
        taken.append(word)
    script = " ".join(taken) if taken and len(taken) < len(words) else words[0]
    if script.startswith("CJK"):
        script = "CJK"
    return _ONE_SCRIPT.get(script, script)


def names_a_project(key: str | None) -> bool:
    """Whether ``key`` has a visible character, so it can name a project.

    A key of only whitespace, separators or format characters (zero-width
    space, joiner and non-joiner, word joiner, byte-order mark, Mongolian vowel
    separator) names no project anyone can see or type: ``str.strip`` keeps the
    format characters, so a key made of them passed every blank check (security
    review 2026-10-26-1, E9). Neither does one of only combining marks,
    controls, private-use or unassigned code points, or the letters and symbols
    that render blank (security review 2026-10-27-1, G65, G74), so a key needs
    one letter, number, punctuation mark or symbol that is not a blank glyph.
    """
    if not key:
        return False
    return any(
        unicodedata.category(ch)[0] in _VISIBLE_CATEGORIES and ch not in _BLANK_GLYPHS for ch in key
    )


def hides_a_character(key: str) -> bool:
    """Whether ``key`` holds a character that makes it read as another project.

    A format character (zero-width space, joiner and non-joiner, word joiner,
    soft hyphen, byte-order mark) or a variation selector leaves a key looking
    exactly like another, so ``acme`` with one of them inside is a second
    project no reader can tell from the first (security review 2026-10-28-1,
    O46). So do the other default-ignorable code points and a blank glyph
    beside visible characters, a compatibility form that Unicode NFKC
    normalisation changes (a fullwidth letter; a decomposed accent is NFC's
    and :func:`canonical_project_key` composes it), and letters from more than
    one script, such as a Cyrillic letter among Latin ones (security review
    2026-10-28-1, G65). Digits count by their script, and a key that holds an
    ASCII character may hold no other punctuation or symbol, since a Unicode
    hyphen or minus sign reads as the ASCII one (security review 2026-10-28-2,
    AA24). No key may hold a control, private-use, unassigned or separator
    character, a dash or another mark that reads as an ASCII one, or a Latin
    letter that reads as an ASCII letter (security review 2026-10-30-1, AB32).
    """
    return lookalike_reason(key) is not None


def lookalike_reason(key: str) -> str | None:
    """Why :func:`hides_a_character` refuses ``key``, worded for a refusal, or None.

    So a refusal names the rule that fired rather than always an invisible
    character (security review 2026-10-28-2, AA26).
    """
    composed = unicodedata.normalize("NFC", key)
    if unicodedata.normalize("NFKC", composed) != composed:
        return "holds a compatibility form (a fullwidth or styled character) that reads as another"
    if any(ch.isascii() for ch in key) and any(
        not ch.isascii() and unicodedata.category(ch)[0] in "PS" for ch in key
    ):
        return "holds a non-ASCII punctuation mark or symbol beside ASCII characters"
    for ch in key:
        category = unicodedata.category(ch)
        if category in _HIDDEN_CATEGORIES or (category == "Zs" and ch != " "):
            return "holds a control, private-use, unassigned or separator character"
        if not ch.isascii() and (category == "Pd" or ch in _ASCII_LOOKALIKE_MARKS):
            return "holds a punctuation mark or symbol that reads as an ASCII one"
        if ch in _LATIN_CONFUSABLES:
            return "holds a letter that reads as an ASCII letter"
    scripts: set[str] = set()
    for ch in key:
        if (
            unicodedata.category(ch) == "Cf"
            or "\U000e0100" <= ch <= "\U000e01ef"
            or _default_ignorable(ch)
            or ch in _BLANK_GLYPHS
        ):
            return "holds an invisible character"
        script = _script(ch)
        if script is not None:
            scripts.add(script)
    if len(scripts) > 1:
        return f"mixes the {' and '.join(sorted(scripts))} scripts"
    return None


def canonical_project_key(key: str) -> str:
    """``key`` as a binding stores and compares it: stripped and in Unicode NFC.

    The composed and decomposed spellings of one name (``café`` typed, and as a
    macOS path returns it) are one project (security review 2026-10-28-1, O46).
    """
    return unicodedata.normalize("NFC", key.strip())


def project_keys(tags: Iterable[str] | None) -> frozenset[str]:
    """The project keys among an entry's tags, canonical (keys that name no project are ignored)."""
    if not tags:
        return frozenset()
    # Canonical on read, so a tag stored in another normalisation form (an
    # operator's deposit, an entry from before keys were canonicalised) is the
    # project its NFC spelling names (security review 2026-10-28-2, AA25).
    return frozenset(
        canonical_project_key(tag[len(PROJECT_TAG_PREFIX) :])
        for tag in tags
        if tag.startswith(PROJECT_TAG_PREFIX) and names_a_project(tag[len(PROJECT_TAG_PREFIX) :])
    )


class EntryScope(StrEnum):
    """What one corpus entry says about where its claims were observed."""

    KEYED = "keyed"
    """Carries one or more project keys."""
    GLOBAL = "global"
    """Keyless and not harness-harvested: a hand deposit, a web page, a
    user-level rule file. Its claims are in view for every observer."""
    UNATTRIBUTED = "unattributed"
    """Harness-harvested, or marked ``scope:unattributed``, but keyless. The
    fail-closed class: a stamping gap must not make a harness's or an agent's
    deposits visible everywhere by accident."""


def classify_entry(tags: Iterable[str] | None, harness_tags: Collection[str]) -> EntryScope:
    """Classify one corpus entry from its tags.

    ``harness_tags`` are the tags a harness adapter puts on everything it
    harvests (``observer_scope.harness_tags``). :data:`UNATTRIBUTED_MARKER`
    counts as one whatever that list says, and so does a ``project:`` tag
    whose key names no project. Keyed wins over both other classes: an entry
    with a key is attributed, whoever deposited it.
    """
    tag_list = list(tags or ())
    if project_keys(tag_list):
        return EntryScope.KEYED
    if UNATTRIBUTED_MARKER in tag_list or any(tag in harness_tags for tag in tag_list):
        return EntryScope.UNATTRIBUTED
    # A project tag whose key names no project (one an earlier release
    # accepted, before the rule tightened) still says the entry was written
    # for one project, so it fails closed rather than reading as global
    # (security review 2026-10-28-1, E21).
    if any(tag.startswith(PROJECT_TAG_PREFIX) for tag in tag_list):
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
    widened: bool = False
    """An operator's standing statement that this belief applies everywhere
    (``memory widen``). Read in the shared join, so the lens and the write
    gate see it alike; the other fields keep the belief's own
    reading. A widening belongs to one belief: it is not pooled by
    :func:`join` and not inherited by :func:`meet`."""

    @property
    def everywhere(self) -> bool:
        """In view for every observer: global by its sources, or widened."""
        return self.is_global or self.widened

    @property
    def unattributed(self) -> bool:
        return not self.everywhere and not self.keys and not self.lapsed

    @property
    def in_view_for_no_project(self) -> bool:
        return not self.everywhere and not self.keys


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


def join(members: Iterable[BeliefScope]) -> BeliefScope:
    """The scope a ``CO_EVIDENTIAL`` merge gives every sourced member: their union.

    The join pools the group's attesting entries, so the merged
    scope is global as soon as one member is, and otherwise the union of their
    keys. A union with no keys is in view for no project; it is reported as
    lapsed when some member was lapsed, which is a reading for display only,
    since the pooled entries decide between lapsed and unattributed. Applies to
    sourced beliefs: a derived belief keeps the meet of its premises.
    """
    keys: set[str] = set()
    lapsed = False
    for member in members:
        if member.is_global:
            return GLOBAL_SCOPE
        keys |= member.keys
        lapsed = lapsed or member.lapsed
    return BeliefScope(keys=frozenset(keys), lapsed=lapsed and not keys)


def visible(scope: BeliefScope, observer_project: str | None, *, widened: bool = False) -> bool:
    """The predicate: is a belief with ``scope`` in view for this observer?

    No observer project is the store-wide view, for which everything is in
    view — the neutrality rule that keeps every unscoped read byte-identical.
    ``widened`` is an operator's standing judgement that the belief applies
    everywhere; it lives in lens-side policy, never on the claim. A scope read
    through the join carries it as ``scope.widened``.
    """
    if observer_project is None:
        return True
    # Keys compare canonical on both sides (security review 2026-10-28-2, AA25).
    return widened or scope.everywhere or canonical_project_key(observer_project) in scope.keys


def share_an_observer(a: BeliefScope, b: BeliefScope) -> bool:
    """Whether some reader has both beliefs in view: the disclosure test.

    Not :func:`pair_precondition`'s subset test: reconciliation asks whether
    one side may retire the other, disclosure asks whether anyone would see
    both. A global belief is in view everywhere. An unattributed one is a
    stamping gap, which must not switch disclosure off, exactly as it does not
    switch reconciliation off. Otherwise the two key sets must meet.
    """
    if a.everywhere or b.everywhere:
        return True
    if a.unattributed or b.unattributed:
        return True
    return bool(a.keys & b.keys)


def same_scope(a: BeliefScope, b: BeliefScope) -> bool:
    """Whether two beliefs have exactly the same observers: the agent link test.

    A ``CO_EVIDENTIAL`` edge unions the two groups' sources, so it leaves both
    scopes unchanged only when they are already equal. Visibility is not
    enough: a project's belief linked to a global one becomes global. Lapsed
    and unattributed are distinct readings and only equal themselves. A
    widening is not compared: an edge does not pool it.
    """
    if a.is_global or b.is_global:
        return a.is_global and b.is_global
    if a.keys or b.keys:
        return a.keys == b.keys
    return a.lapsed == b.lapsed


def describe_scope(scope: BeliefScope) -> str:
    """A short operator-facing reading of a scope, for refusals and reports."""
    if scope.is_global:
        own = "global"
    elif scope.keys:
        own = "project " + ", ".join(sorted(scope.keys))
    else:
        own = "lapsed" if scope.lapsed else "unattributed"
    return f"{own}, widened to every project" if scope.widened else own


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
    so the subset holds trivially. A widened claim is in view everywhere, so a
    project's update against it is reviewed exactly as one against a global
    claim is.
    """
    if candidate.everywhere or not candidate.keys:
        return PairPrecondition.RECONCILE
    if existing.everywhere:
        return PairPrecondition.REVIEW
    if existing.keys <= candidate.keys:
        return PairPrecondition.RECONCILE
    return PairPrecondition.DECLINE
