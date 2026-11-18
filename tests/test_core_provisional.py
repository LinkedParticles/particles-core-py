# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The pure choice of which Subject a name answers, by who resolves it."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from particles.core.provisional import (
    NameCandidate,
    ProvisionalMark,
    ResolutionKind,
    by_ref_choice,
    choose_local,
    mints_provisional,
    resolution_args,
)

_T = datetime(2026, 11, 8, tzinfo=UTC)


def _scoped(principal: str) -> ProvisionalMark:
    return ProvisionalMark(principal=principal, minted_at=_T)


def _shared(principal: str, by: str) -> ProvisionalMark:
    return ProvisionalMark(principal=principal, minted_at=_T, shared_by=by, shared_at=_T)


CONFIRMED = NameCandidate("confirmed")
OWN = NameCandidate("own", _scoped("agent-p"))
SHARED = NameCandidate("shared", _shared("agent-q", "agent-r"))
OTHERS_SCOPED = NameCandidate("others", _scoped("agent-q"))

MINTING = ResolutionKind.MINTING
OPERATOR = ResolutionKind.OPERATOR
NON_MINTING = ResolutionKind.NON_MINTING


@pytest.mark.parametrize(
    ("candidate", "kind", "expected"),
    [
        # The table, one cell per case.
        (CONFIRMED, MINTING, ("confirmed", False)),
        (CONFIRMED, OPERATOR, ("confirmed", False)),
        (CONFIRMED, NON_MINTING, ("confirmed", False)),
        (OWN, MINTING, ("own", False)),
        (OWN, OPERATOR, (None, False)),
        (OWN, NON_MINTING, (None, False)),
        (SHARED, MINTING, ("shared", False)),
        (SHARED, OPERATOR, (None, False)),
        (SHARED, NON_MINTING, (None, False)),
        (OTHERS_SCOPED, MINTING, ("others", True)),
        (OTHERS_SCOPED, OPERATOR, (None, False)),
        (OTHERS_SCOPED, NON_MINTING, (None, False)),
    ],
)
def test_each_cell_of_the_resolution_table(
    candidate: NameCandidate, kind: ResolutionKind, expected: tuple[str | None, bool]
) -> None:
    choice = choose_local([candidate], principal="agent-p", kind=kind)
    assert (choice.subject_id, choice.share) == expected


def test_a_confirmed_candidate_wins_over_every_provisional_one() -> None:
    for kind in ResolutionKind:
        choice = choose_local(
            [OWN, SHARED, OTHERS_SCOPED, CONFIRMED], principal="agent-p", kind=kind
        )
        assert choice.subject_id == "confirmed"
        assert not choice.share and not choice.provisional


def test_a_minting_resolution_prefers_its_own_then_shared_then_another_principals() -> None:
    assert choose_local([OTHERS_SCOPED, SHARED, OWN], principal="agent-p", kind=MINTING) == (
        choose_local([OWN], principal="agent-p", kind=MINTING)
    )
    picked = choose_local([OTHERS_SCOPED, SHARED], principal="agent-p", kind=MINTING)
    assert (picked.subject_id, picked.share) == ("shared", False)


def test_no_candidate_answers_nothing() -> None:
    for kind in ResolutionKind:
        assert choose_local([], principal="agent-p", kind=kind).subject_id is None


def test_a_minting_resolution_with_no_principal_sees_confirmed_subjects_only() -> None:
    assert choose_local([OWN], principal=None, kind=MINTING).subject_id is None
    assert not mints_provisional(MINTING, None)


def test_reaching_a_provisional_subject_by_reference() -> None:
    # Another principal's scoped Subject is shared by a minting resolution.
    assert by_ref_choice(OTHERS_SCOPED, principal="agent-p", kind=MINTING) == (True, False)
    assert by_ref_choice(OWN, principal="agent-p", kind=MINTING) == (False, False)
    assert by_ref_choice(SHARED, principal="agent-p", kind=MINTING) == (False, False)
    # Any other resolution confirms it: an independent reading reached the referent.
    for kind in (OPERATOR, NON_MINTING):
        assert by_ref_choice(OTHERS_SCOPED, principal=None, kind=kind) == (False, True)
    assert by_ref_choice(CONFIRMED, principal=None, kind=NON_MINTING) == (False, False)
    # A minting call with no principal sees and changes nothing, as choose_local.
    assert by_ref_choice(OTHERS_SCOPED, principal=None, kind=MINTING) == (False, False)


def test_resolution_args() -> None:
    assert resolution_args("agent-p") == {"principal": "agent-p", "kind": MINTING}
    assert resolution_args(None) == {"principal": None, "kind": NON_MINTING}
    assert resolution_args("operator", operator=True) == {"principal": None, "kind": OPERATOR}
