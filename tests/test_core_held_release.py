# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The release pass's pure decisions: who qualifies, who is due, and the verdict."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
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
    is_generation_hold,
    qualifies,
)
from particles.core.observer_scope import PairPrecondition
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
    UncertaintyNature,
)
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


# ---------------------------------------------------------------------------
# a hold the generation cascade orphaned before
# ---------------------------------------------------------------------------


def _at(day: int) -> datetime:
    return datetime(2026, 10, day, tzinfo=UTC)


def _sourced(p: Particle, *refs: tuple[str, str | None]) -> Particle:
    return p.model_copy(
        update={
            "provenance": [
                ProvenanceRef(type=ProvenanceRefType.SOURCE, corpus_entry_id=e, snapshot_id=s)
                for e, s in refs
            ]
        }
    )


CAPTURED = {"s1": _at(1), "s2": _at(2), "t1": _at(3)}
CASCADED = _p(Status.PROVENANCE_STALE, StatusReason.RETRACTED_DEPENDENCY)


class TestGenerationHold:
    def _hold(self, a: Particle, b: Particle, current: dict[str, str] | None = None) -> bool:
        return is_generation_hold(
            a, b, current={"E": "s2"} if current is None else current, captured=CAPTURED
        )

    def test_an_older_generation_of_the_same_entry_is_one(self) -> None:
        assert self._hold(_sourced(CASCADED, ("E", "s1")), _sourced(HELD, ("E", "s2")))

    def test_it_qualifies_only_when_the_shell_says_so(self) -> None:
        assert qualifies(RECORD, CASCADED, HELD, generation=True)
        assert not qualifies(RECORD, CASCADED, HELD)

    def test_another_reason_is_not(self) -> None:
        a = _sourced(_p(Status.RETRACTED, StatusReason.EXPLICIT_RETRACTION), ("E", "s1"))
        assert not self._hold(a, _sourced(HELD, ("E", "s2")))

    def test_two_entries_are_not(self) -> None:
        assert not self._hold(
            _sourced(CASCADED, ("E", "s1"), ("F", "t1")), _sourced(HELD, ("E", "s2"))
        )
        assert not self._hold(_sourced(CASCADED, ("F", "t1")), _sourced(HELD, ("E", "s2")))

    def test_an_entry_with_no_current_generation_is_not(self) -> None:
        """Not MUTABLE, never extracted, or its current record is an absence."""
        a, b = _sourced(CASCADED, ("E", "s1")), _sourced(HELD, ("E", "s2"))
        assert not self._hold(a, b, current={})

    def test_an_entry_that_moved_past_the_held_claim_is_not(self) -> None:
        """A quarantined claim is never carried forward, so a release would read as lapsed."""
        a, b = _sourced(CASCADED, ("E", "s1")), _sourced(HELD, ("E", "s2"))
        assert not self._hold(a, b, current={"E": "s3"})

    def test_the_held_claim_must_be_the_newer_generation(self) -> None:
        assert not self._hold(
            _sourced(CASCADED, ("E", "s2")), _sourced(HELD, ("E", "s1")), current={"E": "s1"}
        )

    def test_an_undated_ref_keeps_it_out(self) -> None:
        assert not self._hold(_sourced(CASCADED, ("E", None)), _sourced(HELD, ("E", "s2")))
        assert not self._hold(_sourced(CASCADED, ("E", "s9")), _sourced(HELD, ("E", "s2")))

    def test_a_derived_partner_with_no_source_ref_is_not(self) -> None:
        """A cascaded A is often derived: its premise refs are PARTICLE-typed, never SOURCE."""
        derived = CASCADED.model_copy(
            update={
                "provenance": [
                    ProvenanceRef(
                        type=ProvenanceRefType.PARTICLE, corpus_entry_id="p0", snapshot_id="p0"
                    )
                ]
            }
        )
        assert not self._hold(derived, _sourced(HELD, ("E", "s2")))

    def test_two_claims_from_one_snapshot_are_not(self) -> None:
        """Neither is the newer generation, so the cascade did not retire A for B's source."""
        assert not self._hold(_sourced(CASCADED, ("E", "s2")), _sourced(HELD, ("E", "s2")))

    def test_naive_and_offset_capture_times_order_by_instant(self) -> None:
        """SQLite hands capture times back naive; they read as UTC beside an offset one."""
        plus5 = timezone(timedelta(hours=5))
        a, b = _sourced(CASCADED, ("E", "s1")), _sourced(HELD, ("E", "s2"))

        def hold(a_at: datetime, b_at: datetime) -> bool:
            return is_generation_hold(a, b, current={"E": "s2"}, captured={"s1": a_at, "s2": b_at})

        # 03:00+05:00 is 22:00 UTC the day before, so B is the older snapshot.
        assert not hold(datetime(2026, 10, 2), datetime(2026, 10, 2, 3, tzinfo=plus5))
        # 06:00+05:00 is 01:00 UTC, after a naive midnight.
        assert hold(datetime(2026, 10, 2), datetime(2026, 10, 2, 6, tzinfo=plus5))
        # An offset A against a naive B reads the same way round.
        assert hold(datetime(2026, 10, 2, 3, tzinfo=plus5), datetime(2026, 10, 1, 23))
        assert not hold(datetime(2026, 10, 2, 6, tzinfo=plus5), datetime(2026, 10, 2))
