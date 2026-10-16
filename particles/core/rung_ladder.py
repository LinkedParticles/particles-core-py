# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The rung ladder: the cheapest reasoner that can make progress answers first.

The engine asks a few yes-or-no questions about a pair of claims or about one
candidate: is this pair a contradiction, does this update fill that slot, are
these one claim, is this candidate entailed by its premises. Each is a
**question kind**, and each kind has one **ladder** of **rungs** that can
answer it. This module is the pure half of that protocol:

* a per-kind verdict vocabulary that always contains ``ABSTAIN``
  (:class:`ContradictionVerdict`, :class:`UpdateVerdict`,
  :class:`DuplicateVerdict`, :class:`CandidateVerdict`);
* :class:`Rung`: ``applies``, ``decide``, an ordinal :class:`CostClass`, and a
  name-plus-version :class:`RungStamp`;
* **confirmation rungs**, which run only on a named provisional verdict of a
  cheaper rung and replace it;
* **shadow rungs**, whose verdict is recorded and never returned;
* :class:`Ladder` and its stepper :class:`LadderRun`: rungs in ascending cost
  class, the first verdict that is not ``ABSTAIN`` wins, stamped with its
  rung; ``ABSTAIN`` with no rung named when every rung abstains.

A rung that needs I/O (a store lookup, a fetch, a completion) is declared
here with ``decide=None`` and bound in the shell to an injected callable: the
shell asks :meth:`LadderRun.next_rung` which rung to run, runs it, and hands
the verdict back with :meth:`LadderRun.record`. The decision of which rung
runs, and which verdict stands, never leaves this module (D2). A
ladder whose rungs are all pure is settled in one call by :func:`settle`.

Availability is the shell's to report, rung by rung: a ``network`` rung is
unavailable when the egress policy refuses its endpoint, a
``model`` rung while the completion circuit breaker is open. An
unavailable rung abstains; it never fails the ladder.

Pure: no I/O, no logging.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Generic, TypeVar

__all__ = [
    "Availability",
    "ContradictionVerdict",
    "CostClass",
    "DuplicateVerdict",
    "CandidateVerdict",
    "Ladder",
    "LadderRun",
    "QuestionKind",
    "Rung",
    "RungStamp",
    "Settled",
    "UpdateVerdict",
    "annotation_is_current",
    "settle",
    "stamp_guarded",
]


class QuestionKind(StrEnum):
    """Which question a ladder answers. One ladder per kind."""

    CONTRADICTION = "contradiction"
    """Can both claims of a pair be true at once?"""
    UPDATE = "update"
    """Does the later claim give the earlier one's slot a new value?"""
    DUPLICATE = "duplicate"
    """Are the two claims one claim?"""
    ENTAILMENT = "entailment"
    """Is a candidate entailed by the conjunction of its premises?"""


class ContradictionVerdict(StrEnum):
    """The contradiction ladder's vocabulary."""

    YES = "yes"
    NO = "no"
    ABSTAIN = "abstain"


class UpdateVerdict(StrEnum):
    """The update ladder's vocabulary: the four-way slot verdict plus ``ABSTAIN``.

    The first three mirror :class:`particles.core.conflict_resolution.SlotVerdict`
    value for value, so the port keeps ``FIXED`` apart from ``DIFFERENT``.
    ``ABSTAIN`` is the old "no verdict", which every caller reads as keep.
    """

    DIFFERENT = "different"
    CHANGES = "changes"
    FIXED = "fixed"
    ABSTAIN = "abstain"


class DuplicateVerdict(StrEnum):
    """The duplicate ladder's vocabulary."""

    SAME = "same"
    DIFFERENT = "different"
    ABSTAIN = "abstain"


class CandidateVerdict(StrEnum):
    """The candidate ladder's vocabulary: the abstraction gate."""

    ENTAILED = "entailed"
    NOT_ENTAILED = "not_entailed"
    ABSTAIN = "abstain"


class CostClass(IntEnum):
    """What a rung touches, as an ordinal: the order is stable across machines.

    Dollar cost is a property of the provider a model rung resolves to and
    lives in ``config.llm``; a ladder never learns a price.
    """

    PURE = 0
    """Only the input's own fields."""
    LOCAL = 1
    """One store lookup or a cached index."""
    NETWORK = 2
    """A fetch the egress policy may refuse."""
    MODEL = 3
    """A completion, behind the circuit breaker."""


@dataclass(frozen=True, order=True)
class RungStamp:
    """Which reasoner, at which version, produced a verdict (the discipline)."""

    name: str
    version: str

    def __str__(self) -> str:
        return f"{self.name}@{self.version}"

    @classmethod
    def parse(cls, text: str) -> RungStamp:
        """The stamp :meth:`__str__` rendered. Raises ``ValueError`` on another shape."""
        name, sep, version = text.rpartition("@")
        if not sep or not name or not version:
            raise ValueError(f"not a rung stamp: {text!r}")
        return cls(name, version)


I = TypeVar("I")  # noqa: E741 - the conventional name for a ladder's input type
V = TypeVar("V", bound=StrEnum)


def _always(_: object) -> bool:
    return True


@dataclass(frozen=True)
class Rung(Generic[I, V]):
    """One reasoner on one ladder.

    ``decide`` is ``None`` for a rung that needs I/O; the shell binds it. A
    pure or local rung declared with ``decide`` can be settled here.
    """

    name: str
    version: str
    cost: CostClass
    decide: Callable[[I], V] | None = None
    """The verdict on one input. Abstention is always permitted and always free."""
    applies: Callable[[I], bool] = _always
    """Whether this rung can see this input at all; ``False`` reads as abstain."""
    confirms: frozenset[str] = frozenset()
    """Non-empty for a **confirmation rung**: the provisional verdicts (values)
    of a cheaper rung it runs on. Its verdict, ``ABSTAIN`` included, replaces
    the provisional one, and it never runs without one."""
    shadow: bool = False
    """A rung that has not met its bar: its verdict is recorded
    and never returned. A ``model`` rung is never shadow."""
    endpoint: tuple[str, int | None] | None = None
    """A ``network`` rung's destination, which the egress policy is asked about."""
    content_keyed: bool = True
    """The verdict is a function of the two claim texts and the rung's version,
    so the probe ledger may key it on their content hashes. ``False`` for a
    rung that reads a per-record field (the stored modality), whose verdict two
    records with one text need not share."""

    def __post_init__(self) -> None:
        if self.shadow and self.cost is CostClass.MODEL:
            raise ValueError(f"rung {self.name}: a model rung is never run in shadow")
        if self.shadow and self.confirms:
            raise ValueError(f"rung {self.name}: a confirmation rung cannot be shadow")
        if self.cost is CostClass.NETWORK and self.endpoint is None:
            raise ValueError(f"rung {self.name}: a network rung names its endpoint")

    @property
    def stamp(self) -> RungStamp:
        """The stamp every verdict of this rung carries."""
        return RungStamp(self.name, self.version)

    @property
    def confirming(self) -> bool:
        """True for a confirmation rung."""
        return bool(self.confirms)


def _allow_all(_host: str, _port: int | None) -> bool:
    return True


@dataclass(frozen=True)
class Availability:
    """What the shell reports about the world before each rung runs.

    The default is everything available, which is what a test or an all-pure
    ladder wants.
    """

    egress_allows: Callable[[str, int | None], bool] = _allow_all
    """The no-egress setting's verdict on a destination."""
    breaker_open: bool = False
    """The completion circuit breaker is tripped."""

    def admits(self, rung: Rung[I, V]) -> bool:
        """Whether ``rung`` may run at all under this availability."""
        if rung.cost is CostClass.NETWORK:
            assert rung.endpoint is not None  # checked at construction
            return self.egress_allows(*rung.endpoint)
        if rung.cost is CostClass.MODEL:
            return not self.breaker_open
        return True


@dataclass(frozen=True)
class Settled(Generic[V]):
    """A ladder's answer, and what every rung that ran said."""

    verdict: V
    decided_by: RungStamp | None
    """The rung whose verdict stands; ``None`` when every rung abstained."""
    trail: tuple[tuple[RungStamp, V], ...] = ()
    """Every non-shadow rung that ran, with its verdict, in run order."""
    shadow: tuple[tuple[RungStamp, V], ...] = ()
    """Every shadow rung that ran, with its verdict. Never returned."""


@dataclass(frozen=True)
class Ladder(Generic[I, V]):
    """The rungs that answer one question kind, in ascending cost class.

    The order within a class is the declaration order: a rung's docstring
    records the measured median wall time that placed it.
    The order is fixed and declared, never a market.
    """

    kind: QuestionKind
    abstain: V
    rungs: tuple[Rung[I, V], ...]

    def __post_init__(self) -> None:
        names = [r.name for r in self.rungs]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.kind} ladder: rung names repeat: {names}")
        for rung in self.rungs:
            if self.abstain.value in rung.confirms:
                raise ValueError(f"rung {rung.name}: a confirmation rung never runs on ABSTAIN")
        ordered = tuple(sorted(self.rungs, key=lambda r: r.cost))
        # A confirmation rung runs only on a provisional verdict, and the
        # stepper never revisits a rung it has passed, so a confirmer sorted
        # before a rung that could produce its verdict would be skipped
        # forever with no error. Every confirmation rung must therefore sort
        # after every rung that is not one.
        confirming_seen: Rung[I, V] | None = None
        for rung in ordered:
            if rung.confirming:
                confirming_seen = rung
            elif confirming_seen is not None:
                raise ValueError(
                    f"rung {confirming_seen.name}: a confirmation rung sorts after every "
                    f"non-confirming rung, but {rung.name} ({rung.cost.name.lower()}) follows it"
                )
        object.__setattr__(self, "rungs", ordered)

    def rung(self, name: str) -> Rung[I, V]:
        """The rung called ``name``."""
        for rung in self.rungs:
            if rung.name == name:
                return rung
        raise KeyError(name)

    @property
    def stamps(self) -> tuple[RungStamp, ...]:
        """Every rung's stamp, in run order."""
        return tuple(r.stamp for r in self.rungs)

    def start(self, item: I) -> LadderRun[I, V]:
        """A fresh run of this ladder on one input."""
        return LadderRun(self, item)


@dataclass
class LadderRun(Generic[I, V]):
    """One input's way up one ladder: a pure stepper the shell drives.

    The shell calls :meth:`next_rung` with the availability as it stands now,
    runs the returned rung (its own ``decide`` or the shell's binding), and
    hands the verdict back to :meth:`record`, until :meth:`next_rung` returns
    ``None``. :meth:`settled` then says what stands.
    """

    ladder: Ladder[I, V]
    item: I
    _position: int = 0
    _pending: Rung[I, V] | None = None
    _provisional: tuple[V, RungStamp] | None = None
    _trail: list[tuple[RungStamp, V]] = field(default_factory=list)
    _shadow: list[tuple[RungStamp, V]] = field(default_factory=list)

    def next_rung(self, availability: Availability | None = None) -> Rung[I, V] | None:
        """The next rung to run on this input, or ``None`` when the ladder is settled.

        Before a provisional verdict, every non-confirmation rung is offered in
        cost order and skipped (an abstention) when it does not apply or is
        unavailable. After one, only a confirmation rung that confirms it is
        offered. A confirmation rung that cannot run, because it is unavailable
        or because its ``applies`` is false for this input, is not returned:
        its abstention replaces the provisional verdict here, with the
        confirming rung named as the decider, and it joins no trail since it
        never ran. An unconfirmed provisional verdict never stands on its own.
        """
        if self._pending is not None:
            raise RuntimeError(f"rung {self._pending.name} was offered and never recorded")
        avail = availability or Availability()
        rungs = self.ladder.rungs
        while self._position < len(rungs):
            rung = rungs[self._position]
            self._position += 1
            if self._provisional is None:
                if rung.confirming:
                    continue
                if avail.admits(rung) and rung.applies(self.item):
                    self._pending = rung
                    return rung
                continue
            if self._provisional[0].value not in rung.confirms:
                continue
            if avail.admits(rung) and rung.applies(self.item):
                self._pending = rung
                return rung
            # The confirming rung cannot run (unavailable, or it does not
            # apply): it abstains, and its abstention replaces the provisional
            # verdict. It did not run, so it joins no trail.
            self._provisional = (self.ladder.abstain, rung.stamp)
        return None

    def record(self, rung: Rung[I, V], verdict: V) -> None:
        """Hand back the verdict of the rung :meth:`next_rung` offered."""
        if rung is not self._pending:
            raise RuntimeError(f"rung {rung.name} was not the rung offered")
        self._pending = None
        if rung.shadow:
            self._shadow.append((rung.stamp, verdict))
            return
        if rung.confirming:
            self._replace(rung, verdict)
            return
        self._trail.append((rung.stamp, verdict))
        if verdict != self.ladder.abstain:
            self._provisional = (verdict, rung.stamp)

    def _replace(self, rung: Rung[I, V], verdict: V) -> None:
        self._trail.append((rung.stamp, verdict))
        self._provisional = (verdict, rung.stamp)

    def settled(self) -> Settled[V]:
        """What stands: the provisional verdict and its rung, or ``ABSTAIN`` and no rung."""
        if self._pending is not None:
            raise RuntimeError(f"rung {self._pending.name} was offered and never recorded")
        trail, shadow = tuple(self._trail), tuple(self._shadow)
        if self._provisional is None:
            return Settled(self.ladder.abstain, None, trail, shadow)
        verdict, stamp = self._provisional
        return Settled(verdict, stamp, trail, shadow)


def settle(ladder: Ladder[I, V], item: I, availability: Availability | None = None) -> Settled[V]:
    """Run a ladder whose offered rungs all carry their own ``decide``.

    Raises ``ValueError`` when a rung that needs the shell's binding is offered.
    """
    run = ladder.start(item)
    while (rung := run.next_rung(availability)) is not None:
        if rung.decide is None:
            raise ValueError(f"rung {rung.name} needs the shell's binding")
        run.record(rung, rung.decide(item))
    return run.settled()


def annotation_is_current(recorded: str | None, current: str) -> bool:
    """Whether a derived annotation's stamp is the current generator's.

    A missing stamp is never current: a rung over a stale or unstamped
    annotation abstains rather than answer from it.
    """
    return recorded is not None and recorded == current


def stamp_guarded(
    rung: Rung[I, V],
    *,
    abstain: V,
    stamp_of: Callable[[I], str | None],
    current: str,
) -> Rung[I, V]:
    """``rung``, abstaining on any input whose annotation stamp is not ``current``.

    For a rung that reads a regenerable annotation (the structured triple, the
    modality record): a comparator over a stale triple is a
    confident wrong answer, so it says nothing instead.
    """
    decide = rung.decide
    if decide is None:
        raise ValueError(f"rung {rung.name}: guard the shell's binding instead")

    def guarded(item: I) -> V:
        if not annotation_is_current(stamp_of(item), current):
            return abstain
        return decide(item)

    return Rung(
        name=rung.name,
        version=rung.version,
        cost=rung.cost,
        decide=guarded,
        applies=rung.applies,
        confirms=rung.confirms,
        shadow=rung.shadow,
        endpoint=rung.endpoint,
        content_keyed=rung.content_keyed,
    )
