# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The release pass's pure decisions: who qualifies, who is due, and the verdict."""

from __future__ import annotations

from typing import Any

import pytest

from particles.core.conflict_resolution import RETIRED_VALUE_KEY
from particles.core.contradiction_disclosure import CENSUS_ORIGIN, ORIGIN_KEY
from particles.core.held_release import (
    AUTONOMOUS_RETIREMENTS,
    PairCheck,
    ReleaseAction,
    blocking_from_payload,
    decide_release,
    is_due,
    qualifies,
)
from particles.core.observer_scope import PairPrecondition
from particles.core.schema import Confidence, Particle, UncertaintyNature
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason


def _p(
    status: Status = Status.ACTIVE,
    reason: StatusReason | None = None,
    properties: dict[str, Any] | None = None,
) -> Particle:
    return Particle(
        content="x",
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        provenance=[],
        asserted_by="test",
        status=status,
        status_reason=reason,
        properties=properties,
    )


HELD = _p(Status.PROVENANCE_STALE, StatusReason.CONFLICT_PENDING)
RECORD = _p(Status.INCONSISTENCY)


class TestQualifies:
    @pytest.mark.parametrize("reason", sorted(AUTONOMOUS_RETIREMENTS))
    def test_a_partner_an_autonomous_pass_retired_qualifies(self, reason: StatusReason) -> None:
        partner = _p(Status.PROVENANCE_STALE, reason)
        assert qualifies(RECORD, partner, HELD)

    @pytest.mark.parametrize(
        "reason",
        [
            StatusReason.EXPLICIT_RETRACTION,
            StatusReason.EXPLICIT_SUPERSESSION,
            StatusReason.CONFLICT_RESOLVED,
            StatusReason.RETRACTED_DEPENDENCY,
        ],
    )
    def test_a_partner_a_person_or_a_cascade_retired_does_not(self, reason: StatusReason) -> None:
        assert not qualifies(RECORD, _p(Status.SUPERSEDED, reason), HELD)

    def test_an_active_partner_does_not(self) -> None:
        assert not qualifies(RECORD, _p(), HELD)

    def test_a_closed_record_does_not(self) -> None:
        partner = _p(Status.PROVENANCE_STALE, StatusReason.SUPERSEDED_BY_UPDATE)
        assert not qualifies(_p(Status.RETRACTED, StatusReason.CONFLICT_RESOLVED), partner, HELD)

    def test_a_held_claim_no_longer_quarantined_does_not(self) -> None:
        partner = _p(Status.PROVENANCE_STALE, StatusReason.SUPERSEDED_BY_UPDATE)
        promoted = _p(Status.SUPERSEDED, StatusReason.CONFLICT_RESOLVED)
        assert not qualifies(RECORD, partner, promoted)

    def test_census_records_and_retired_value_holds_never_qualify(self) -> None:
        partner = _p(Status.PROVENANCE_STALE, StatusReason.SUPERSEDED_BY_UPDATE)
        census = _p(Status.INCONSISTENCY, properties={ORIGIN_KEY: CENSUS_ORIGIN})
        hold = _p(Status.INCONSISTENCY, properties={RETIRED_VALUE_KEY: True})
        assert not qualifies(census, partner, HELD)
        assert not qualifies(hold, partner, HELD)

    def test_a_missing_side_does_not(self) -> None:
        partner = _p(Status.PROVENANCE_STALE, StatusReason.SUPERSEDED_BY_UPDATE)
        assert not qualifies(RECORD, None, HELD)
        assert not qualifies(RECORD, partner, None)


class TestDue:
    def test_never_examined_is_due(self) -> None:
        assert is_due(None, set())

    def test_still_blocked_by_the_same_active_claims_is_not(self) -> None:
        assert not is_due(["c", "d"], {"c", "d", "e"})

    def test_due_again_once_a_blocker_left_active(self) -> None:
        assert is_due(["c", "d"], {"d"})

    def test_payload_reader(self) -> None:
        assert blocking_from_payload({"blocking_ids": ["c", 3, "d"]}) == ["c", "d"]
        assert blocking_from_payload({}) is None
        assert blocking_from_payload(None) is None


class TestDecide:
    def test_nothing_confirmed_releases(self) -> None:
        decision = decide_release(
            [PairCheck("c", probe=False), PairCheck("d", probe=True, reading=False)]
        )
        assert decision.action is ReleaseAction.RELEASE

    def test_no_pairs_releases(self) -> None:
        assert decide_release([]).action is ReleaseAction.RELEASE

    def test_a_pair_both_checks_confirm_blocks(self) -> None:
        decision = decide_release(
            [PairCheck("c", probe=True, reading=True), PairCheck("d", probe=False)]
        )
        assert decision.action is ReleaseAction.BLOCKED and decision.blocking == ("c",)

    def test_a_failed_probe_keeps_it_open(self) -> None:
        decision = decide_release([PairCheck("c", probe=None), PairCheck("d", probe=False)])
        assert decision.action is ReleaseAction.INCOMPLETE and decision.unverified == ("c",)

    def test_a_failed_reading_keeps_it_open(self) -> None:
        decision = decide_release([PairCheck("c", probe=True, reading=None)])
        assert decision.action is ReleaseAction.INCOMPLETE

    def test_a_confirmed_block_outranks_an_incomplete_check(self) -> None:
        decision = decide_release(
            [PairCheck("c", probe=None), PairCheck("d", probe=True, reading=True)]
        )
        assert decision.action is ReleaseAction.BLOCKED and decision.blocking == ("d",)
        assert decision.unverified == ("c",)

    def test_a_declined_pair_never_blocks_and_becomes_an_edge(self) -> None:
        declined = PairCheck("c", probe=True, reading=True, precondition=PairPrecondition.DECLINE)
        unread = PairCheck("d", probe=None, precondition=PairPrecondition.DECLINE)
        decision = decide_release([declined, unread])
        assert decision.action is ReleaseAction.RELEASE
        assert decision.declined == ("c",)
