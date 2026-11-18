# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Whether a transcript claim reports only its session's moment, and what that folds.

Three pure halves:

* **The stored reading.** A model's answer to the ``episode`` question is kept
  in a claim's ``properties`` as :data:`EPISODE_KEY` (the verdict) with
  :data:`EPISODE_RULE_KEY` (the stamp of the writer that produced it), read
  back by :func:`parse_reading`. Two writers exist: the extractor's
  ``prompt.episode`` component and the regeneration pass's ``regen.episode``
  prompt, each stamped ``<writer>@<digest>``, so admission can tell them
  apart.
* **The lexical shapes** the ``shape`` rung reads (:func:`shapes`), with the
  digest of the pattern set as that rung's version.
* **The fold arithmetic** (§4): which pair cards a settled
  reading would withhold, in which stratum and under which stamp
  combination (:func:`fold_entry`), and the shadow report a curation build
  or a consolidation run records (:func:`episode_shadow`). Nothing here
  withholds a card: stage 1 only counts.

Pure: no I/O, no logging.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import NamedTuple

from particles.core.rung_ladder import EpisodeVerdict, RungStamp, Settled

__all__ = [
    "EPISODE_KEY",
    "EPISODE_PROPERTY_KEYS",
    "EPISODE_MODEL_KEY",
    "EPISODE_RULE_KEY",
    "CardSides",
    "EXTRACTOR_WRITER",
    "OPERATOR_RULE",
    "REGEN_WRITER",
    "SHAPES_DIGEST",
    "EpisodeClaim",
    "EpisodeInput",
    "EpisodeReading",
    "EpisodeShadow",
    "FoldEntry",
    "Stratum",
    "WITHHELD_KINDS",
    "entry_key",
    "episode_shadow",
    "fold_entry",
    "fold_stamp",
    "parse_reading",
    "reading_of",
    "reading_verdict",
    "shapes",
    "stamp_for",
]

#: The stored verdict, ``session_bound`` or ``standing``.
EPISODE_KEY = "extraction:episode"
#: The stamp of the writer that set :data:`EPISODE_KEY`: ``<writer>@<digest>``,
#: or :data:`OPERATOR_RULE`.
EPISODE_RULE_KEY = "extraction:episode_rule"
#: The ``"<provider>:<model>"`` that served the reading. An extracted reading
#: names none and is read under the particle's own extraction model.
EPISODE_MODEL_KEY = "extraction:episode_model"
#: The keys a reading writes; cleared together when one is replaced.
EPISODE_PROPERTY_KEYS: tuple[str, ...] = (EPISODE_KEY, EPISODE_RULE_KEY, EPISODE_MODEL_KEY)

#: The extractor's writer name: the prompt component's stable name.
EXTRACTOR_WRITER = "prompt.episode"
#: The regeneration pass's writer name (``particles episode``).
REGEN_WRITER = "regen.episode"
#: The rule an operator's reading is written with; never regenerated.
OPERATOR_RULE = "operator"

#: The card kinds a fold can withhold: the pair questions. A
#: duplicate question is one card per connected group since, and a
#: ``duplicate_pair`` card survives only in a collection built before it.
WITHHELD_KINDS: frozenset[str] = frozenset(
    {"duplicate_group", "duplicate_pair", "inconsistency", "contradiction"}
)


def stamp_for(rule: str, model: str | None) -> RungStamp:
    """The stamp a reading by ``rule`` on ``model`` answers under.

    A model reading is a function of its prompt and the model that served it,
    so both are in the stamp: ``<writer>@<digest>~<provider>:<model>``. Two
    models under one prompt are two stamps, admitted apart. An operator's
    reading is ``operator@1`` whatever the model.
    """
    if rule == OPERATOR_RULE:
        return RungStamp(OPERATOR_RULE, "1")
    base = RungStamp.parse(rule)
    return RungStamp(base.name, f"{base.version}~{model}") if model else base


class EpisodeReading(NamedTuple):
    """One stored reading: the verdict, the writer's rule and the model that served it."""

    verdict: EpisodeVerdict
    rule: str
    model: str | None = None

    @property
    def stamp(self) -> RungStamp:
        """The rung stamp this reading answers under (:func:`stamp_for`)."""
        return stamp_for(self.rule, self.model)


def reading_verdict(raw: object) -> EpisodeVerdict | None:
    """A stored or replied verdict string as a verdict; ``None`` for anything else.

    ``ABSTAIN`` is never stored: a writer with no answer writes nothing.
    """
    if not isinstance(raw, str):
        return None
    value = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if value == EpisodeVerdict.SESSION_BOUND.value:
        return EpisodeVerdict.SESSION_BOUND
    if value == EpisodeVerdict.STANDING.value:
        return EpisodeVerdict.STANDING
    return None


def parse_reading(
    properties: Mapping[str, object] | None, *, fallback_model: str | None = None
) -> EpisodeReading | None:
    """The reading in a claim's ``properties``, or ``None`` when it has none or a bad one.

    ``fallback_model`` is the model a reading that names none was served by:
    the store passes the particle's extraction model for an extracted reading.
    """
    if not properties:
        return None
    verdict = reading_verdict(properties.get(EPISODE_KEY))
    rule = properties.get(EPISODE_RULE_KEY)
    if verdict is None or not isinstance(rule, str) or not rule:
        return None
    if rule == OPERATOR_RULE:
        return EpisodeReading(verdict, rule)
    try:
        RungStamp.parse(rule)
    except ValueError:
        return None
    model = properties.get(EPISODE_MODEL_KEY)
    served = model if isinstance(model, str) and model else fallback_model
    return EpisodeReading(verdict, rule, served)


def reading_of(
    properties: Mapping[str, object] | None, extraction_model: str | None
) -> EpisodeReading | None:
    """A stored claim's reading, an extracted one read under the claim's own extraction model."""
    rule = (properties or {}).get(EPISODE_RULE_KEY)
    extracted = isinstance(rule, str) and rule.startswith(f"{EXTRACTOR_WRITER}@")
    return parse_reading(properties, fallback_model=extraction_model if extracted else None)


# ---------------------------------------------------------------------------
# The lexical shapes (the ``shape`` rung, shadow)
# ---------------------------------------------------------------------------

_SHAPES: tuple[tuple[str, str, int], ...] = (
    ("pr_number", r"\b(PR|pull request)\s*#?\d+", re.I),
    ("version_change", r"\bfrom\s+v?\d+\.\d+\.\d+\S*\s+to\s+v?\d+\.\d+\.\d+", re.I),
    ("session_self", r"\b(the|this) (session|conversation|agent)\b", re.I),
    ("worktree_branch", r"\b(worktree|branch|checkout|HEAD)\b", 0),
    ("commit_sha", r"\b(commit\s+)?[0-9a-f]{7,40}\b", 0),
    (
        "now_state",
        r"\b(is now|are now|now (has|holds|reads|shows|points|runs|passes|operates)|currently)\b",
        re.I,
    ),
    (
        "count_or_result",
        r"\b\d[\d,]*\s+(tests?|PDRs?|files?|particles|beliefs|claims|cards|rows|records|"
        r"warnings|errors|commits)\b|\b(passe[sd]|failed|fails|is green|is red|succeeded)\b",
        re.I,
    ),
)
_COMPILED: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, flags)) for name, pattern, flags in _SHAPES
)

#: The ``shape`` rung's version: a hash of the pattern set, so an edit to any
#: pattern is a new stamp whose verdicts run in shadow again.
SHAPES_DIGEST = hashlib.sha256(
    "\n".join(f"{n}\t{p}\t{f}" for n, p, f in _SHAPES).encode()
).hexdigest()[:12]


def shapes(text: str) -> frozenset[str]:
    """The names of the session-report shapes ``text`` carries (Context)."""
    return frozenset(name for name, rx in _COMPILED if rx.search(text))


# ---------------------------------------------------------------------------
# The ladder's input
# ---------------------------------------------------------------------------


class EpisodeInput(NamedTuple):
    """What the episode ladder reads about one claim: plain values the shell gathered."""

    content: str
    credited: bool
    """The use ledger credits the claim with a use."""
    asserted: bool
    """An operator or agent asserted the claim directly (no extractor)."""
    valid_from_basis: str | None
    """The stored ``valid_from`` basis (``session``)."""
    reading: EpisodeReading | None


# ---------------------------------------------------------------------------
# The fold (§4)
# ---------------------------------------------------------------------------


class Stratum(StrEnum):
    """The two admission strata the measurement separated."""

    ONE_ENTRY = "one_entry"
    """Every side comes from one and the same source entry."""
    REST = "rest"
    """The sides come from two or more entries."""


@dataclass(frozen=True)
class EpisodeClaim:
    """One asked claim: its source entries and its settled ladder."""

    particle_id: str
    entries: frozenset[str]
    settled: Settled[EpisodeVerdict]


def fold_stamp(settled: Settled[EpisodeVerdict]) -> RungStamp | None:
    """The stamp under which a claim reads session-bound, or ``None`` when it does not.

    A reading decides when one exists: a ``SESSION_BOUND`` reading names its
    writer's stamp, and a ``STANDING`` one (or a credit) folds nothing. A claim
    with no reading falls back to the first shadow rung that read it
    session-bound, so the shadow report can say what those rungs would fold.
    """
    if settled.verdict is EpisodeVerdict.SESSION_BOUND:
        return settled.decided_by
    if settled.verdict is not EpisodeVerdict.ABSTAIN:
        return None
    for stamp, verdict in settled.shadow:
        if verdict is EpisodeVerdict.SESSION_BOUND:
            return stamp
    return None


def entry_key(stratum: Stratum, stamps: Iterable[str]) -> str:
    """The admission key ``curation.episode.admitted`` names: ``<stratum>:<stamp>[+<stamp>…]``.

    The combination is the set of distinct stamps, so a card whose members
    were all read under one stamp is ``<stratum>:<stamp>`` whatever its size.
    """
    return f"{stratum.value}:{'+'.join(sorted(set(stamps)))}"


@dataclass(frozen=True, order=True)
class FoldEntry:
    """Which admission entry a withheld card falls under: a stratum and a stamp combination."""

    stratum: Stratum
    stamps: tuple[str, ...]
    """The distinct stamps its sides were read under, sorted: the unordered
    combination."""

    @property
    def key(self) -> str:
        """The entry as config and reports name it."""
        return entry_key(self.stratum, self.stamps)


def fold_entry(sides: Sequence[EpisodeClaim]) -> FoldEntry | None:
    """The entry a card with these sides would be withheld under, or ``None`` if it stays.

    A card is withheld only when every side has exactly one source entry and
    reads session-bound. A claim with two or more entries never
    folds: it has no single session to group under. A ``duplicate_group``
    card is withheld whole or not at all, by the same rule over
    every member.
    """
    if len(sides) < 2:
        return None
    stamps: list[str] = []
    for side in sides:
        if len(side.entries) != 1:
            return None
        stamp = fold_stamp(side.settled)
        if stamp is None:
            return None
        stamps.append(str(stamp))
    entries = {e for side in sides for e in side.entries}
    stratum = Stratum.ONE_ENTRY if len(entries) == 1 else Stratum.REST
    return FoldEntry(stratum, tuple(sorted(set(stamps))))


@dataclass(frozen=True)
class CardSides:
    """One emitted card the fold is asked about: its kind and the claims it names."""

    kind: str
    particle_ids: tuple[str, ...]


@dataclass
class _EntryTally:
    by_kind: Counter[str] = field(default_factory=Counter)
    sessions: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class EpisodeShadow:
    """What the fold would do on one card list, in shadow (§6)."""

    asked: int
    """Claims the question was asked of (every source a session type)."""
    entries: Mapping[str, Mapping[str, object]]
    """Per admission entry: the cards it would withhold by kind, and the
    session cards it would raise."""
    withheld: int
    """Cards every entry together would withhold."""
    session_cards: int
    """Session cards every entry together would raise."""
    admitted_withheld: int
    """Cards the admitted entries would withhold."""
    admitted_session_cards: int
    """Session cards the admitted entries would raise."""

    @property
    def admitted_net(self) -> int:
        """Cards the admitted entries would save: withheld less raised (the stage 2 floor)."""
        return self.admitted_withheld - self.admitted_session_cards

    def as_dict(self) -> dict[str, object]:
        """A JSON-ready form for a build's diagnostics or a run record."""
        return {
            "asked": self.asked,
            "withheld": self.withheld,
            "session_cards": self.session_cards,
            "admitted_withheld": self.admitted_withheld,
            "admitted_session_cards": self.admitted_session_cards,
            "admitted_net": self.admitted_net,
            "entries": {k: dict(v) for k, v in self.entries.items()},
        }


def episode_shadow(
    cards: Iterable[CardSides],
    claims: Mapping[str, EpisodeClaim],
    admitted: Iterable[str] = (),
) -> EpisodeShadow:
    """Count what the fold would withhold among ``cards``, by entry.

    ``claims`` holds only the asked claims (every source a session type); a
    card with any side outside it stays. A withheld card's sides each join the
    session card of their one source entry, so the session cards an entry
    raises are the distinct entries its withheld sides name.
    """
    admitted_keys = frozenset(admitted)
    tallies: dict[str, _EntryTally] = {}
    all_sessions: set[str] = set()
    admitted_sessions: set[str] = set()
    withheld = admitted_withheld = 0
    for card in cards:
        if card.kind not in WITHHELD_KINDS:
            continue
        sides = [claims.get(pid) for pid in card.particle_ids]
        if any(s is None for s in sides):
            continue
        entry = fold_entry([s for s in sides if s is not None])
        if entry is None:
            continue
        names = {e for s in sides if s is not None for e in s.entries}
        tally = tallies.setdefault(entry.key, _EntryTally())
        tally.by_kind[card.kind] += 1
        tally.sessions |= names
        withheld += 1
        all_sessions |= names
        if entry.key in admitted_keys:
            admitted_withheld += 1
            admitted_sessions |= names
    entries: dict[str, Mapping[str, object]] = {
        key: {
            "withheld": dict(sorted(t.by_kind.items())),
            "session_cards": len(t.sessions),
            "admitted": key in admitted_keys,
        }
        for key, t in sorted(tallies.items())
    }
    return EpisodeShadow(
        asked=len(claims),
        entries=entries,
        withheld=withheld,
        session_cards=len(all_sessions),
        admitted_withheld=admitted_withheld,
        admitted_session_cards=len(admitted_sessions),
    )
