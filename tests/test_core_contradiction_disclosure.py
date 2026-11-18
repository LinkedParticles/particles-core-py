# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Pure decisions of the nightly contradiction disclosure."""

from __future__ import annotations

from particles.core.contradiction_disclosure import (
    CENSUS_ORIGIN,
    ORIGIN_KEY,
    PAIRS_KEY,
    READING_KEY,
    READING_OBSERVED,
    READING_STANDING,
    REPLACES_KEY,
    SIDES_KEY,
    SOURCES_KEY,
    CloseCause,
    ConfirmedPair,
    LapseAction,
    MemberState,
    NoteLabel,
    RecordState,
    Sides,
    build_census_record,
    census_pairs,
    census_replaces,
    census_sides,
    census_sources,
    covered_pairs,
    decide_lapse,
    group_pairs,
    is_census_record,
    plan_group,
    reading_stamp,
    select_under_cap,
    split_sides,
)
from particles.core.observer_scope import (
    GLOBAL_SCOPE,
    LAPSED_SCOPE,
    UNATTRIBUTED_SCOPE,
    BeliefScope,
    share_an_observer,
)
from particles.core.schema import (
    Confidence,
    Particle,
    ProvenanceRefType,
    UncertaintyNature,
)
from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status


def _pair(a: str, b: str, reason: str = "they disagree", same: bool = False) -> ConfirmedPair:
    return ConfirmedPair(a=a, b=b, same_source=same, reason=reason)


def _claim(content: str, value: float = 0.9, subjects: list[str] | None = None) -> Particle:
    return Particle(
        content=content,
        confidence=Confidence(value=value, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="test",
        status=Status.ACTIVE,
        subject_ids=subjects or [],
    )


class TestGrouping:
    def test_pairs_sharing_a_claim_are_one_group(self) -> None:
        groups = group_pairs([_pair("x", "a1"), _pair("y", "z"), _pair("x", "a2")])
        assert [[(p.a, p.b) for p in g] for g in groups] == [
            [("x", "a1"), ("x", "a2")],
            [("y", "z")],
        ]

    def test_one_claim_against_two_phrasings_splits_into_sides(self) -> None:
        sides = split_sides([("b", "a1"), ("b", "a2")], ("a1", "b"))
        assert sides == Sides(a=("a1", "a2"), b=("b",))

    def test_one_claim_against_claims_in_two_notes_splits(self) -> None:
        sides = split_sides([("x", "n1"), ("x", "n2"), ("n2", "y")], ("x", "n1"))
        assert sides == Sides(a=("x", "y"), b=("n1", "n2"))

    def test_odd_cycle_is_unsided(self) -> None:
        assert split_sides([("a", "b"), ("b", "c"), ("c", "a")], ("a", "b")) is None

    def test_plan_group_falls_back_to_one_record_per_pair_when_unsided(self) -> None:
        group = [_pair("a", "b"), _pair("b", "c"), _pair("c", "a")]
        planned, unsided = plan_group(group, lambda x, y: (x, y))
        assert unsided
        assert [(p.sides.a, p.sides.b) for p in planned] == [
            (("a",), ("b",)),
            (("b",), ("c",)),
            (("c",), ("a",)),
        ]
        assert [p.pairs for p in planned] == [(g,) for g in group]

    def test_plan_group_orients_the_representative(self) -> None:
        planned, unsided = plan_group([_pair("new", "old")], lambda x, y: (y, x))
        assert not unsided
        assert planned[0].sides == Sides(a=("old",), b=("new",))


class TestCoverage:
    SIDES = Sides(a=("a1", "a2"), b=("b",))

    def test_open_record_covers_only_opposite_sides(self) -> None:
        covered = covered_pairs([RecordState("r", self.SIDES, open=True)], {})
        assert frozenset(("a1", "b")) in covered
        assert frozenset(("a2", "b")) in covered
        # Two claims on one side were never read against each other.
        assert frozenset(("a1", "a2")) not in covered

    def test_judged_record_covers_while_no_new_source_states_a_member(self) -> None:
        fingerprint = {"a1": frozenset({"e1"}), "a2": frozenset({"e1"}), "b": frozenset({"e2"})}
        record = RecordState(
            "r", self.SIDES, open=False, cause=CloseCause.REVIEW, fingerprint=fingerprint
        )
        same = {"a1": frozenset({"e1"}), "b": frozenset({"e2"}), "a2": frozenset({"e1"})}
        assert frozenset(("a1", "b")) in covered_pairs([record], same)
        grown = dict(same, b=frozenset({"e2", "e3"}))
        assert covered_pairs([record], grown) == frozenset()

    def test_cascade_close_is_a_judgment(self) -> None:
        record = RecordState(
            "r",
            Sides(a=("a",), b=("b",)),
            open=False,
            cause=CloseCause.CASCADE,
            fingerprint={"a": frozenset({"e1"}), "b": frozenset({"e2"})},
        )
        assert covered_pairs([record], {"a": frozenset({"e1"}), "b": frozenset({"e2"})})

    def test_mechanical_close_covers_nothing(self) -> None:
        for cause in (CloseCause.LAPSED, CloseCause.REGROUPED):
            record = RecordState("r", Sides(a=("a",), b=("b",)), open=False, cause=cause)
            assert covered_pairs([record], {}) == frozenset()

    def test_every_open_record_covers_its_own_pairs(self) -> None:
        records = [
            RecordState("r1", Sides(a=("a1",), b=("b1",)), open=True),
            RecordState("r2", Sides(a=("a2",), b=("b2",)), open=True),
        ]
        assert covered_pairs(records, {}) == {frozenset(("a1", "b1")), frozenset(("a2", "b2"))}

    def test_a_judged_record_after_a_mechanical_one_covers_members_with_no_sources(self) -> None:
        # Neither member appears in the current sources or the fingerprint: no
        # stating entry was gained, so the judgment still stands.
        records = [
            RecordState("m", Sides(a=("x",), b=("y",)), open=False, cause=CloseCause.LAPSED),
            RecordState("j", Sides(a=("a",), b=("b",)), open=False, cause=CloseCause.REVIEW),
        ]
        assert covered_pairs(records, {}) == {frozenset(("a", "b"))}


class TestLapse:
    PAIRS = [_pair("a1", "b"), _pair("a2", "b")]

    def test_all_live_keeps(self) -> None:
        assert decide_lapse(self.PAIRS, {}).action is LapseAction.KEEP

    def test_side_with_no_live_member_closes(self) -> None:
        decision = decide_lapse(self.PAIRS, {"b": MemberState.GONE})
        assert decision.action is LapseAction.CLOSE
        assert decision.dropped == ("b",)

    def test_lapsed_side_closes(self) -> None:
        decision = decide_lapse(self.PAIRS, {"b": MemberState.LAPSED})
        assert decision.action is LapseAction.CLOSE

    def test_gone_member_with_partners_intact_keeps(self) -> None:
        pairs = [_pair("a1", "b1"), _pair("a1", "b2")]
        decision = decide_lapse(pairs, {"b2": MemberState.GONE})
        assert decision.action is LapseAction.KEEP
        assert decision.dropped == ("b2",)

    def test_lapsed_but_active_member_regroups(self) -> None:
        decision = decide_lapse(self.PAIRS, {"a2": MemberState.LAPSED})
        assert decision.action is LapseAction.REGROUP
        assert [(p.a, p.b) for p in decision.remaining] == [("a1", "b")]
        assert decision.dropped == ("a2",)

    def test_merged_member_is_replaced_by_its_survivor(self) -> None:
        decision = decide_lapse(self.PAIRS, {"a2": MemberState.MERGED}, {"a2": "a2-survivor"})
        assert decision.action is LapseAction.REGROUP
        assert {(p.a, p.b) for p in decision.remaining} == {("a1", "b"), ("a2-survivor", "b")}

    def test_merge_never_closes_as_lapsed(self) -> None:
        decision = decide_lapse([_pair("a", "b")], {"a": MemberState.MERGED}, {"a": "a-survivor"})
        assert decision.action is LapseAction.REGROUP

    def test_live_member_left_without_a_partner_regroups(self) -> None:
        pairs = [_pair("a1", "b1"), _pair("a2", "b2")]
        decision = decide_lapse(pairs, {"b2": MemberState.GONE})
        assert decision.action is LapseAction.REGROUP
        assert [(p.a, p.b) for p in decision.remaining] == [("a1", "b1")]

    def test_a_dropped_first_pair_keeps_the_pairs_after_it(self) -> None:
        pairs = [_pair("a1", "b1"), _pair("a2", "b2")]
        decision = decide_lapse(pairs, {"b1": MemberState.GONE})
        assert decision.action is LapseAction.REGROUP
        assert [(p.a, p.b) for p in decision.remaining] == [("a2", "b2")]

    def test_a_merged_side_b_member_is_replaced_by_its_survivor(self) -> None:
        decision = decide_lapse([_pair("a", "b")], {"b": MemberState.MERGED}, {"b": "b-survivor"})
        assert decision.action is LapseAction.REGROUP
        assert [(p.a, p.b) for p in decision.remaining] == [("a", "b-survivor")]


class TestReread:
    """A re-reading under a revised instruction."""

    PAIRS = [_pair("a1", "b"), _pair("a2", "b")]

    def test_every_pair_withdrawn_closes_as_withdrawn(self) -> None:
        rereads = {p.key: None for p in self.PAIRS}
        decision = decide_lapse(self.PAIRS, {}, rereads=rereads)
        assert decision.action is LapseAction.CLOSE
        assert decision.cause is CloseCause.WITHDRAWN
        assert decision.withdrawn == tuple(self.PAIRS)
        assert decision.dropped == ()

    def test_lapse_without_a_rereading_still_closes_as_lapsed(self) -> None:
        decision = decide_lapse(self.PAIRS, {"b": MemberState.GONE})
        assert decision.cause is CloseCause.LAPSED

    def test_one_pair_withdrawn_regroups_from_the_rest(self) -> None:
        decision = decide_lapse(self.PAIRS, {}, rereads={self.PAIRS[1].key: None})
        assert decision.action is LapseAction.REGROUP
        assert [(p.a, p.b) for p in decision.remaining] == [("a1", "b")]
        assert decision.withdrawn == (self.PAIRS[1],)

    def test_a_reconfirmed_pair_regroups_with_the_new_reading(self) -> None:
        fresh = ConfirmedPair(
            a="a1", b="b", same_source=False, reason="a fixed fact", reading=READING_OBSERVED
        )
        decision = decide_lapse(self.PAIRS, {}, rereads={fresh.key: fresh})
        assert decision.action is LapseAction.REGROUP
        first = decision.remaining[0]
        assert (first.reason, first.reading) == ("a fixed fact", READING_OBSERVED)
        assert decision.remaining[1].reading == READING_STANDING

    def test_withdrawn_is_mechanical(self) -> None:
        assert not CloseCause.WITHDRAWN.judgment

    def test_reading_round_trips_and_defaults_for_older_records(self) -> None:
        pair = ConfirmedPair(a="x", b="y", same_source=False, reason="r", reading=READING_OBSERVED)
        assert ConfirmedPair.from_payload(pair.to_payload()) == pair
        legacy = {"a": "x", "b": "y", "same_source": False, "reason": "r"}
        restored = ConfirmedPair.from_payload(legacy)
        assert restored is not None and restored.reading == READING_STANDING


class TestCap:
    def test_regroups_never_count(self) -> None:
        opened, waiting = select_under_cap([True, False, True, True], cap=2)
        assert opened == [0, 1, 2]
        assert waiting == [3]

    def test_zero_cap_opens_only_regroups(self) -> None:
        assert select_under_cap([True, False], cap=0) == ([1], [0])


class TestRecord:
    def test_record_names_every_member_and_marks_its_origin(self) -> None:
        a1, a2, b = _claim("endpoint exists", 0.8, ["s1"]), _claim("it exists"), _claim("404", 0.6)
        sides = Sides(a=(a1.id, a2.id), b=(b.id,))
        pairs = [_pair(a1.id, b.id, "one says exists, one says not found"), _pair(a2.id, b.id)]
        record = build_census_record(
            sides=sides,
            pairs=pairs,
            members={p.id: p for p in (a1, a2, b)},
            labels={
                a1.id: NoteLabel("n1.md", "2026-09-18"),
                b.id: NoteLabel("n2.md", "2026-09-25"),
            },
            sources={a1.id: ["e1"], a2.id: ["e1"], b.id: ["e2"]},
            trigger_entry_id="e2",
            trigger_snapshot_id="s2",
            replaces="old",
        )
        assert record.status is Status.INCONSISTENCY
        refs = [(r.type, r.corpus_entry_id) for r in record.provenance]
        assert refs == [
            (ProvenanceRefType.PARTICLE, a1.id),
            (ProvenanceRefType.PARTICLE, b.id),
            (ProvenanceRefType.SOURCE, "e2"),
            (ProvenanceRefType.PARTICLE, a2.id),
        ]
        assert record.properties is not None
        assert record.properties[ORIGIN_KEY] == CENSUS_ORIGIN
        assert record.properties[SIDES_KEY] == {"a": [a1.id, a2.id], "b": [b.id]}
        assert record.properties[SOURCES_KEY][b.id] == ["e2"]
        assert record.properties[REPLACES_KEY] == "old"
        assert len(record.properties[PAIRS_KEY]) == 2
        # The weakest member's confidence; A's subjects.
        assert record.confidence.value == 0.6
        assert record.subject_ids == ["s1"]
        assert "Second reading: one says exists, one says not found" in record.content
        assert "(n2.md, 2026-09-25)" in record.content

        assert is_census_record(record)
        assert census_sides(record) == sides
        assert census_sources(record)[a1.id] == frozenset({"e1"})
        assert [(p.a, p.b) for p in census_pairs(record)] == [(a1.id, b.id), (a2.id, b.id)]
        assert census_replaces(record) == "old"

    def test_a_reading_in_the_other_order_says_which_is_which(self) -> None:
        old, new = _claim("200 everywhere"), _claim("404")
        record = build_census_record(
            sides=Sides(a=(old.id,), b=(new.id,)),
            pairs=[_pair(new.id, old.id, "A says 404, B says 200")],
            members={old.id: old, new.id: new},
            labels={},
            sources={},
            trigger_entry_id="e",
            trigger_snapshot_id=None,
        )
        assert (
            "Second reading: (the reading's A is Side B's claim) A says 404, B says 200"
            in record.content
        )

    def test_a_non_census_record_reads_as_none(self) -> None:
        plain = _claim("x")
        assert not is_census_record(plain)
        assert census_sides(plain) is None
        assert census_pairs(plain) == []

    def test_confirmed_pair_round_trips(self) -> None:
        pair = _pair("a", "b", "why", same=True)
        assert ConfirmedPair.from_payload(pair.to_payload()) == pair
        assert ConfirmedPair.from_payload({"a": "a", "b": "a"}) is None


class TestRecordEdges:
    """The record's reason, its trigger ref, and readers of a malformed payload."""

    @staticmethod
    def _record(
        sides: Sides, pairs: list[ConfirmedPair], *members: Particle, snapshot: str | None = None
    ) -> Particle:
        return build_census_record(
            sides=sides,
            pairs=pairs,
            members={p.id: p for p in members},
            labels={},
            sources={},
            trigger_entry_id="e2",
            trigger_snapshot_id=snapshot,
        )

    def test_the_representative_pairs_reason_wherever_it_is_listed(self) -> None:
        a1, a2, b = _claim("x"), _claim("x again"), _claim("not x")
        record = self._record(
            Sides(a=(a1.id, a2.id), b=(b.id,)),
            [_pair(a2.id, b.id, "the second phrasing"), _pair(a1.id, b.id, "the representative")],
            a1,
            a2,
            b,
        )
        assert record.content.endswith("Second reading: the representative")

    def test_with_no_representative_pair_the_first_pairs_reason(self) -> None:
        a, a2, b, b2 = _claim("x"), _claim("x again"), _claim("not x"), _claim("never x")
        record = self._record(
            Sides(a=(a.id, a2.id), b=(b.id, b2.id)),
            [_pair(a2.id, b2.id, "the first pair"), _pair(a.id, b2.id, "the second pair")],
            a,
            a2,
            b,
            b2,
        )
        assert record.content.endswith("Second reading: the first pair")

    def test_with_no_pairs_no_reason_is_recorded(self) -> None:
        a, b = _claim("x"), _claim("not x")
        record = self._record(Sides(a=(a.id,), b=(b.id,)), [], a, b)
        assert record.content.endswith("Second reading: no reason recorded")
        assert record.properties is not None
        assert record.properties[PAIRS_KEY] == []

    def test_the_trigger_ref_keeps_its_snapshot(self) -> None:
        a, b = _claim("x"), _claim("not x")
        record = self._record(Sides(a=(a.id,), b=(b.id,)), [_pair(a.id, b.id)], a, b, snapshot="s2")
        trigger = record.provenance[2]
        assert (trigger.type, trigger.corpus_entry_id, trigger.snapshot_id) == (
            ProvenanceRefType.SOURCE,
            "e2",
            "s2",
        )

    def test_a_pair_payload_needs_two_string_ids(self) -> None:
        assert ConfirmedPair.from_payload({"a": "x", "b": 5}) is None
        assert ConfirmedPair.from_payload({"a": 5, "b": "x"}) is None
        assert ConfirmedPair.from_payload({"a": "x"}) is None

    def test_a_pair_payload_without_same_source_is_cross_source(self) -> None:
        pair = ConfirmedPair.from_payload({"a": "x", "b": "y"})
        assert pair == ConfirmedPair(a="x", b="y", same_source=False, reason="")

    def test_a_census_record_with_an_empty_side_has_no_sides(self) -> None:
        def census(sides: object) -> Particle:
            return _claim("x").model_copy(
                update={"properties": {ORIGIN_KEY: CENSUS_ORIGIN, SIDES_KEY: sides}}
            )

        assert census_sides(census({"a": ["x"], "b": []})) is None
        assert census_sides(census({"a": [], "b": ["y"]})) is None
        assert census_sides(census({"a": ["x"], "b": ["y"]})) == Sides(a=("x",), b=("y",))

    def test_reading_stamp_and_replaces_read_only_a_non_empty_string(self) -> None:
        def record(key: str, value: object) -> Particle:
            return _claim("x").model_copy(update={"properties": {key: value}})

        for bad in ("", 5, None):
            assert reading_stamp(record(READING_KEY, bad)) is None
            assert census_replaces(record(REPLACES_KEY, bad)) is None
        assert reading_stamp(record(READING_KEY, READING_OBSERVED)) == READING_OBSERVED
        assert census_replaces(record(REPLACES_KEY, "old")) == "old"
        assert reading_stamp(_claim("x")) is None
        assert census_replaces(_claim("x")) is None


class TestShareAnObserver:
    def test_global_and_unattributed_share_with_anyone(self) -> None:
        alpha = BeliefScope(keys=frozenset({"alpha"}))
        assert share_an_observer(GLOBAL_SCOPE, alpha)
        assert share_an_observer(UNATTRIBUTED_SCOPE, alpha)

    def test_keyed_scopes_must_meet(self) -> None:
        alpha = BeliefScope(keys=frozenset({"alpha"}))
        both = BeliefScope(keys=frozenset({"alpha", "beta"}))
        beta = BeliefScope(keys=frozenset({"beta"}))
        assert share_an_observer(alpha, both)
        assert not share_an_observer(alpha, beta)

    def test_lapsed_shares_with_nobody_keyed(self) -> None:
        assert not share_an_observer(LAPSED_SCOPE, BeliefScope(keys=frozenset({"alpha"})))
