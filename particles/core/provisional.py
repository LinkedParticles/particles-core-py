# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Which Subject a name answers, by who is resolving it.

A Subject an agent principal mints for a name the store does not know is
**provisional**: it answers that principal's later resolutions of the name, and
no extraction, import or operator write, until an operator accepts it by id or
an independent resolution reaches the same external reference. A second agent
principal's use **shares** it among agents, and still not beyond them.

This module is the pure half: the resolution kinds, the mark, and the choice
between the Subjects that hold a name (D2). The store keeps the mark
in its own table, and the resolver performs the transitions the choice names.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TypedDict

from pydantic import BaseModel


class ResolutionKind(StrEnum):
    """Whose resolution of a name this is, which decides what it may see.

    Every caller of the resolver states its kind; nothing is inferred from a
    label.
    """

    #: An agent principal's resolution (an agent write, the memory-server
    #: facade, the extraction of an entry an agent route deposited). A mint is
    #: provisional to the principal.
    MINTING = "MINTING"
    #: An operator's write. Sees confirmed Subjects only; mints confirmed.
    OPERATOR = "OPERATOR"
    #: Every other resolution: extraction of an unstamped entry, an import, a
    #: maintenance pass. Sees confirmed Subjects only; mints confirmed.
    NON_MINTING = "NON_MINTING"


class ProvisionalMark(BaseModel):
    """A provisional Subject's mark: who minted it, and whether it is shared.

    ``shared_by`` is ``None`` while the Subject is **scoped** (it answers only
    ``principal``) and names the second agent principal once it is **shared**
    (it answers every agent principal's minting resolution).
    """

    model_config = {"frozen": True}

    principal: str
    minted_at: datetime
    shared_by: str | None = None
    shared_at: datetime | None = None

    @property
    def shared(self) -> bool:
        return self.shared_by is not None


@dataclass(frozen=True)
class NameCandidate:
    """A Subject that holds a name, in the order the name lookup ranks it.

    ``mark`` is ``None`` for a confirmed Subject.
    """

    subject_id: str
    mark: ProvisionalMark | None = None


@dataclass(frozen=True)
class LocalChoice:
    """What the local-index step resolves a name to.

    ``subject_id`` is ``None`` when no candidate answers this resolution and
    the cascade goes on to the authorities. ``share`` is set when the answer is
    another principal's scoped provisional Subject, which this resolution
    shares.
    """

    subject_id: str | None
    share: bool = False
    provisional: bool = False


NO_CHOICE = LocalChoice(subject_id=None)


def choose_local(
    candidates: Sequence[NameCandidate], *, principal: str | None, kind: ResolutionKind
) -> LocalChoice:
    """The Subject a name resolves to at the local-index step.

    ``candidates`` are the Subjects that hold the name, in lookup order
    (canonical names before aliases, earliest first). A confirmed candidate
    always wins, so an agent never shadows a confirmed Subject. Otherwise only a
    minting resolution sees a provisional one: its principal's own first, then
    a shared one, then another principal's scoped one, which it shares. Two
    principals are the same principal whatever key each wrote with.
    """
    for candidate in candidates:
        if candidate.mark is None:
            return LocalChoice(subject_id=candidate.subject_id)
    if kind is not ResolutionKind.MINTING or principal is None:
        return NO_CHOICE
    marked = [c for c in candidates if c.mark is not None]
    for candidate in marked:
        assert candidate.mark is not None
        if candidate.mark.principal == principal:
            return LocalChoice(subject_id=candidate.subject_id, provisional=True)
    for candidate in marked:
        assert candidate.mark is not None
        if candidate.mark.shared:
            return LocalChoice(subject_id=candidate.subject_id, provisional=True)
    for candidate in marked:
        return LocalChoice(subject_id=candidate.subject_id, share=True, provisional=True)
    return NO_CHOICE


def by_ref_choice(
    candidate: NameCandidate, *, principal: str | None, kind: ResolutionKind
) -> tuple[bool, bool]:
    """What reaching a Subject by its external reference does: ``(share, accept)``.

    The recognize and live passes match by external reference and see
    provisional Subjects too. A minting resolution treats the
    match as the local step does (another principal's scoped Subject is
    shared); any other resolution confirms it, since an independent reading
    arrived at the same referent.
    """
    mark = candidate.mark
    if mark is None:
        return False, False
    if kind is ResolutionKind.MINTING:
        if principal is None:
            return False, False
        return (not mark.shared and mark.principal != principal), False
    return False, True


def mints_provisional(kind: ResolutionKind, principal: str | None) -> bool:
    """Whether a Subject this resolution mints is provisional."""
    return kind is ResolutionKind.MINTING and principal is not None


class ResolutionArgs(TypedDict):
    """The resolver's ``principal`` and ``kind`` keyword arguments, as one value."""

    principal: str | None
    kind: ResolutionKind


def resolution_args(principal: str | None, *, operator: bool = False) -> ResolutionArgs:
    """Whose resolution a write's subject names get.

    ``operator`` gives an operator's resolution. Otherwise a write for an agent
    ``principal`` is a minting resolution, and one for no principal (an
    extraction of an unstamped entry, an import) is non-minting.
    """
    if operator:
        return {"principal": None, "kind": ResolutionKind.OPERATOR}
    if principal is None:
        return {"principal": None, "kind": ResolutionKind.NON_MINTING}
    return {"principal": principal, "kind": ResolutionKind.MINTING}
