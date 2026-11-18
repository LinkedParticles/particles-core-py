# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The observer-scope set algebra — pure, so tested exhaustively."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from particles.core.observer_scope import (
    GLOBAL_SCOPE,
    LAPSED_SCOPE,
    REFERENCE_SCOPE,
    UNATTRIBUTED_SCOPE,
    BeliefScope,
    EntryScope,
    PairPrecondition,
    classify_entry,
    describe_scope,
    join,
    lookalike_reason,
    meet,
    pair_precondition,
    project_keys,
    project_tag,
    same_scope,
    scope_of_entries,
    standing,
    visible,
    with_reference_marker,
)

HARNESS = ("claude-code",)


def keyed(*keys: str) -> BeliefScope:
    return BeliefScope(keys=frozenset(keys))


class TestEntryClassification:
    def test_project_tag_round_trips(self) -> None:
        assert project_keys([project_tag("-a-repo"), "memory-file"]) == {"-a-repo"}

    def test_an_empty_key_is_not_a_key(self) -> None:
        assert project_keys(["project:"]) == frozenset()

    def test_a_key_makes_an_entry_keyed_whoever_deposited_it(self) -> None:
        assert classify_entry(["claude-code", "project:-a"], HARNESS) is EntryScope.KEYED
        assert classify_entry(["rule-file", "project:-a"], HARNESS) is EntryScope.KEYED

    def test_a_keyless_hand_deposit_is_global(self) -> None:
        assert classify_entry([], HARNESS) is EntryScope.GLOBAL
        assert classify_entry(None, HARNESS) is EntryScope.GLOBAL
        assert classify_entry(["rule-file"], HARNESS) is EntryScope.GLOBAL

    def test_a_keyless_harness_deposit_fails_closed(self) -> None:
        assert classify_entry(["claude-code", "audit"], HARNESS) is EntryScope.UNATTRIBUTED


class TestScopeOfEntries:
    def test_union_of_the_projects_a_belief_was_observed_in(self) -> None:
        scope = scope_of_entries([["claude-code", "project:-a"], ["project:-b"]], HARNESS)
        assert scope == keyed("-a", "-b")

    def test_one_global_source_makes_the_belief_global(self) -> None:
        scope = scope_of_entries([["claude-code", "project:-a"], ["web"]], HARNESS)
        assert scope.is_global

    def test_an_unattributed_source_neither_widens_nor_narrows(self) -> None:
        scope = scope_of_entries([["claude-code"], ["claude-code", "project:-a"]], HARNESS)
        assert scope == keyed("-a")

    def test_only_unattributed_sources(self) -> None:
        assert scope_of_entries([["claude-code"]], HARNESS).unattributed

    def test_no_sources_at_all_is_unattributed_not_global(self) -> None:
        assert scope_of_entries([], HARNESS) == UNATTRIBUTED_SCOPE


class TestMeet:
    def test_global_only_if_every_premise_is_global(self) -> None:
        assert meet([GLOBAL_SCOPE, GLOBAL_SCOPE]).is_global
        assert meet([GLOBAL_SCOPE, keyed("-a")]) == keyed("-a")

    def test_an_abstraction_over_one_project_stays_in_it(self) -> None:
        assert meet([keyed("-a"), keyed("-a", "-b")]) == keyed("-a")

    def test_a_contradiction_across_projects_reaches_no_project(self) -> None:
        assert meet([keyed("-a"), keyed("-b")]).unattributed

    def test_an_unattributed_premise_empties_the_meet(self) -> None:
        assert meet([keyed("-a"), UNATTRIBUTED_SCOPE]).unattributed

    def test_no_premises(self) -> None:
        assert meet([]).unattributed


class TestVisible:
    def test_no_observer_sees_everything(self) -> None:
        for scope in (GLOBAL_SCOPE, UNATTRIBUTED_SCOPE, keyed("-a")):
            assert visible(scope, None)

    def test_a_project_sees_global_and_its_own(self) -> None:
        assert visible(GLOBAL_SCOPE, "-a")
        assert visible(keyed("-a", "-b"), "-a")
        assert not visible(keyed("-b"), "-a")

    def test_unattributed_is_in_view_for_no_project(self) -> None:
        assert not visible(UNATTRIBUTED_SCOPE, "-a")

    def test_widening_overrides_scope(self) -> None:
        assert visible(keyed("-b"), "-a", widened=True)
        assert visible(UNATTRIBUTED_SCOPE, "-a", widened=True)


class TestUnattributedMarker:
    """The compiled-in marker: unattributed whatever the configured list says."""

    def test_the_marker_reads_unattributed_under_any_harness_list(self) -> None:
        from particles.core.observer_scope import UNATTRIBUTED_MARKER

        assert classify_entry([UNATTRIBUTED_MARKER], ()) is EntryScope.UNATTRIBUTED
        assert classify_entry([UNATTRIBUTED_MARKER], HARNESS) is EntryScope.UNATTRIBUTED

    def test_a_key_outranks_the_marker(self) -> None:
        from particles.core.observer_scope import UNATTRIBUTED_MARKER

        assert classify_entry([UNATTRIBUTED_MARKER, project_tag("-a")], ()) is EntryScope.KEYED


class TestWidenedScope:
    """A widening rides on the scope the join returns."""

    WIDE_B = BeliefScope(keys=frozenset({"-b"}), widened=True)

    def test_a_widened_belief_is_in_view_everywhere(self) -> None:
        assert visible(self.WIDE_B, "-a")
        assert self.WIDE_B.everywhere and not self.WIDE_B.is_global
        assert not BeliefScope(widened=True).unattributed

    def test_a_projects_update_against_a_widened_claim_is_reviewed(self) -> None:
        own = BeliefScope(keys=frozenset({"-b"}), widened=True)
        assert pair_precondition(keyed("-b"), own) is PairPrecondition.REVIEW
        assert pair_precondition(keyed("-b"), keyed("-b")) is PairPrecondition.RECONCILE
        assert pair_precondition(self.WIDE_B, keyed("-a")) is PairPrecondition.RECONCILE

    def test_a_widened_belief_shares_an_observer_with_anyone(self) -> None:
        from particles.core.observer_scope import share_an_observer

        assert share_an_observer(self.WIDE_B, keyed("-a"))
        assert not share_an_observer(keyed("-b"), keyed("-a"))

    def test_a_widening_is_neither_pooled_nor_inherited(self) -> None:
        assert join((self.WIDE_B, keyed("-a"))) == keyed("-a", "-b")
        assert meet((self.WIDE_B, keyed("-b"))) == keyed("-b")
        assert same_scope(self.WIDE_B, keyed("-b"))

    def test_the_description_names_the_widening(self) -> None:
        assert describe_scope(self.WIDE_B) == "project -b, widened to every project"


_keys = st.frozensets(st.sampled_from(["-a", "-b", "-c"]))
_scopes = st.builds(BeliefScope, is_global=st.booleans(), keys=_keys)


@given(st.lists(_scopes, min_size=1, max_size=4), st.sampled_from(["-a", "-b", "-c"]))
def test_a_derived_belief_is_never_in_view_where_a_premise_is_not(
    premises: list[BeliefScope], observer: str
) -> None:
    """The meet can only narrow: deriving a belief never leaks a premise's content."""
    if visible(meet(premises), observer):
        assert all(visible(premise, observer) for premise in premises)


class TestLapsed:
    """A lapsed scope is in view for no project, and is not a stamping gap."""

    def test_lapsed_is_not_unattributed(self) -> None:
        assert LAPSED_SCOPE.lapsed and not LAPSED_SCOPE.unattributed
        assert LAPSED_SCOPE.in_view_for_no_project and UNATTRIBUTED_SCOPE.in_view_for_no_project
        assert not visible(LAPSED_SCOPE, "-a")
        assert visible(LAPSED_SCOPE, None)

    def test_an_empty_meet_over_a_lapsed_premise_is_lapsed(self) -> None:
        assert meet([LAPSED_SCOPE, keyed("-a")]).lapsed
        assert not meet([UNATTRIBUTED_SCOPE, keyed("-a")]).lapsed

    def test_a_non_empty_meet_is_keyed_whatever_its_premises(self) -> None:
        assert meet([keyed("-a", "-b"), keyed("-a")]) == keyed("-a")
        assert meet([GLOBAL_SCOPE, LAPSED_SCOPE]) == LAPSED_SCOPE


def test_the_disclosure_line_names_lapsed_beliefs_apart_from_unattributed() -> None:
    from particles.core.schema import ObserverScopeNote
    from particles.render.markdown import observer_scope_line

    note = ObserverScopeNote(
        project="-a", engaged=True, total=10, in_scope=6, unattributed=1, lapsed=3
    )
    line = observer_scope_line(note)
    assert "1 unattributed" in line
    assert "3 lapsed" in line
    assert "lapsed" not in observer_scope_line(note.model_copy(update={"lapsed": 0}))


class TestPairPrecondition:
    """``scope(existing) ⊆ scope(candidate)`` and its carve-outs."""

    def test_own_update_reconciles(self) -> None:
        assert pair_precondition(keyed("-a"), keyed("-a")) is PairPrecondition.RECONCILE

    def test_another_projects_claim_is_declined(self) -> None:
        assert pair_precondition(keyed("-a"), keyed("-b")) is PairPrecondition.DECLINE

    def test_a_claim_both_projects_observe_is_declined_for_either(self) -> None:
        assert pair_precondition(keyed("-a"), keyed("-a", "-b")) is PairPrecondition.DECLINE

    def test_a_global_candidate_pairs_as_today(self) -> None:
        assert pair_precondition(GLOBAL_SCOPE, keyed("-b")) is PairPrecondition.RECONCILE

    def test_an_unattributed_side_pairs_as_today(self) -> None:
        assert pair_precondition(UNATTRIBUTED_SCOPE, keyed("-b")) is PairPrecondition.RECONCILE
        assert pair_precondition(keyed("-a"), UNATTRIBUTED_SCOPE) is PairPrecondition.RECONCILE

    def test_a_lapsed_claim_has_no_observer_to_protect(self) -> None:
        assert pair_precondition(keyed("-a"), LAPSED_SCOPE) is PairPrecondition.RECONCILE

    def test_a_global_claim_contested_by_a_project_goes_to_review(self) -> None:
        assert pair_precondition(keyed("-a"), GLOBAL_SCOPE) is PairPrecondition.REVIEW

    @given(
        st.frozensets(st.sampled_from(["-a", "-b", "-c"])),
        st.frozensets(st.sampled_from(["-a", "-b", "-c"])),
    )
    def test_a_keyed_pair_reconciles_iff_the_subset_holds(
        self, cand: frozenset[str], existing: frozenset[str]
    ) -> None:
        verdict = pair_precondition(keyed(*cand), keyed(*existing))
        if not cand:
            assert verdict is PairPrecondition.RECONCILE
        else:
            assert (verdict is PairPrecondition.RECONCILE) == (existing <= cand)


class TestJoin:
    """The scope a CO_EVIDENTIAL merge gives both sides (Atlas review finding 8)."""

    def test_the_union_of_two_projects(self) -> None:
        assert join([keyed("-a"), keyed("-b")]) == keyed("-a", "-b")

    def test_one_global_member_makes_the_group_global(self) -> None:
        assert join([keyed("-a"), GLOBAL_SCOPE]) == GLOBAL_SCOPE

    def test_a_member_in_view_for_no_project_adds_nothing(self) -> None:
        assert join([keyed("-a"), UNATTRIBUTED_SCOPE]) == keyed("-a")
        assert join([keyed("-a"), LAPSED_SCOPE]) == keyed("-a")

    def test_no_keys_stays_in_view_for_no_project(self) -> None:
        assert join([UNATTRIBUTED_SCOPE, LAPSED_SCOPE]).in_view_for_no_project
        assert join([LAPSED_SCOPE]).lapsed
        assert join([UNATTRIBUTED_SCOPE]).unattributed


class TestSameScope:
    """The agent link test: equal observers, not mutual visibility."""

    SCOPES = (
        GLOBAL_SCOPE,
        UNATTRIBUTED_SCOPE,
        LAPSED_SCOPE,
        keyed("a"),
        keyed("b"),
        keyed("a", "b"),
    )

    def test_each_scope_equals_only_itself(self) -> None:
        for i, x in enumerate(self.SCOPES):
            for j, y in enumerate(self.SCOPES):
                assert same_scope(x, y) == (i == j), (x, y)

    def test_a_global_scope_carrying_keys_is_still_global(self) -> None:
        assert same_scope(BeliefScope(is_global=True, keys=frozenset({"a"})), GLOBAL_SCOPE)

    def test_describe(self) -> None:
        assert [describe_scope(s) for s in self.SCOPES] == [
            "global",
            "unattributed",
            "lapsed",
            "project a",
            "project b",
            "project a, b",
        ]


class TestLookalikeReason:
    """Which rule refuses a project key a reader could mistake for another."""

    def test_a_non_ascii_mark_beside_ascii_is_refused(self) -> None:
        # U+00B7 MIDDLE DOT is neither a dash nor a listed lookalike mark, so
        # only the punctuation-beside-ASCII rule catches it.
        reason = lookalike_reason("a\u00b7b")
        assert reason is not None and "punctuation mark or symbol beside ASCII" in reason

    def test_a_format_character_outside_the_ignorable_ranges_is_invisible(self) -> None:
        # U+FFF9 INTERLINEAR ANNOTATION ANCHOR and U+0600 ARABIC NUMBER SIGN are
        # category Cf but not default-ignorable.
        for key in ("a\ufff9b", "a\u0600b"):
            assert lookalike_reason(key) == "holds an invisible character", key

    def test_a_cjk_ideograph_beside_latin_mixes_scripts(self) -> None:
        assert lookalike_reason("a\u4e2d") == "mixes the CJK and LATIN scripts"

    def test_the_ascii_digits_belong_to_no_script(self) -> None:
        # 0 and 9 are the bounds of the ASCII digit range.
        assert lookalike_reason("app09") is None


class TestReferenceScope:
    """Reference entries: global on question surfaces, not standing memory."""

    REF = ["scope:reference"]

    def test_a_marked_keyless_entry_is_reference(self) -> None:
        assert classify_entry(self.REF, HARNESS) is EntryScope.REFERENCE
        assert EntryScope.REFERENCE.everywhere and EntryScope.GLOBAL.everywhere
        assert not EntryScope.KEYED.everywhere

    def test_a_key_and_the_unattributed_class_outrank_the_marker(self) -> None:
        assert classify_entry([*self.REF, "project:-a"], HARNESS) is EntryScope.KEYED
        assert classify_entry([*self.REF, "claude-code"], HARNESS) is EntryScope.UNATTRIBUTED
        assert classify_entry([*self.REF, "scope:unattributed"], HARNESS) is EntryScope.UNATTRIBUTED

    def test_reference_reads_global_on_question_surfaces(self) -> None:
        scope = scope_of_entries([self.REF], HARNESS)
        assert scope == REFERENCE_SCOPE
        assert scope.is_global and scope.everywhere
        assert visible(scope, "-a") and visible(scope, None)
        assert pair_precondition(keyed("-a"), scope) is PairPrecondition.REVIEW

    def test_a_global_entry_beside_a_reference_one_makes_it_standing(self) -> None:
        assert scope_of_entries([self.REF, []], HARNESS) == GLOBAL_SCOPE

    def test_a_keyed_entry_beside_a_reference_one_keeps_its_key(self) -> None:
        scope = scope_of_entries([self.REF, ["project:-a"]], HARNESS)
        assert scope.reference and scope.keys == {"-a"}

    def test_standing_withholds_reference_only_beliefs(self) -> None:
        ref = REFERENCE_SCOPE
        assert not standing(ref, "-a")
        assert not standing(ref, None)
        both = BeliefScope(is_global=True, reference=True, keys=frozenset({"-a"}))
        assert standing(both, "-a") and not standing(both, "-b") and standing(both, None)

    def test_standing_matches_visible_without_a_reference_reading(self) -> None:
        for scope in (GLOBAL_SCOPE, keyed("-a"), UNATTRIBUTED_SCOPE, LAPSED_SCOPE):
            for observer in ("-a", "-b"):
                assert standing(scope, observer) == visible(scope, observer)
            assert standing(scope, None)

    def test_a_widening_admits_a_reference_belief(self) -> None:
        widened = BeliefScope(is_global=True, reference=True, widened=True)
        assert standing(widened, "-a") and standing(widened, None)

    def test_meet_over_global_and_reference_premises_is_reference(self) -> None:
        derived = meet([GLOBAL_SCOPE, REFERENCE_SCOPE])
        assert derived.reference and derived.is_global
        assert meet([GLOBAL_SCOPE, GLOBAL_SCOPE]) == GLOBAL_SCOPE

    def test_meet_with_a_keyed_premise_treats_reference_as_global(self) -> None:
        assert meet([REFERENCE_SCOPE, keyed("-a")]) == keyed("-a")

    def test_join_with_a_global_member_is_global_and_otherwise_pools_keys(self) -> None:
        assert join([REFERENCE_SCOPE, GLOBAL_SCOPE]) == GLOBAL_SCOPE
        pooled = join([REFERENCE_SCOPE, keyed("-a")])
        assert pooled.reference and pooled.keys == {"-a"}

    def test_same_scope_tells_reference_from_global(self) -> None:
        assert not same_scope(REFERENCE_SCOPE, GLOBAL_SCOPE)
        assert same_scope(REFERENCE_SCOPE, REFERENCE_SCOPE)

    def test_the_description_says_reference(self) -> None:
        assert describe_scope(REFERENCE_SCOPE).startswith("reference")


class TestWithReferenceMarker:
    """The document routes' marking rule."""

    def test_a_document_naming_no_project_is_marked(self) -> None:
        assert with_reference_marker(["notes"], HARNESS, standing=False) == [
            "notes",
            "scope:reference",
        ]

    def test_standing_keeps_it_global_and_drops_a_passed_marker(self) -> None:
        assert with_reference_marker(["scope:reference"], HARNESS, standing=True) == []

    def test_keyed_harness_and_unattributed_deposits_are_left_alone(self) -> None:
        for tags in (["project:-a"], ["claude-code"], ["scope:unattributed"]):
            assert with_reference_marker(tags, HARNESS, standing=False) == tags

    def test_marking_twice_is_a_no_op(self) -> None:
        once = with_reference_marker([], HARNESS, standing=False)
        assert with_reference_marker(once, HARNESS, standing=False) == once


_ref_scopes = st.builds(
    BeliefScope, is_global=st.booleans(), keys=_keys, reference=st.booleans()
).map(lambda s: s if s.is_global or not s.reference else BeliefScope(keys=s.keys))


@given(_ref_scopes, st.sampled_from(["-a", "-b", "-c", None]))
def test_standing_never_admits_what_visible_does_not(
    scope: BeliefScope, observer: str | None
) -> None:
    """The digest predicate only narrows the question surfaces' view."""
    if standing(scope, observer):
        assert visible(scope, observer)


@given(st.lists(_ref_scopes, min_size=1, max_size=4), st.sampled_from(["-a", "-b", "-c"]))
def test_a_derived_belief_is_standing_only_where_some_premise_is(
    premises: list[BeliefScope], observer: str
) -> None:
    """Every premise is in view, and one admits it as standing memory.

    A premise keyed to the observer admits a belief derived with a reference
    premise: an inconsistency between a project's claim and an article stays
    in that project's digest, because the project's own claim is contested.
    """
    if standing(meet(premises), observer):
        assert all(visible(premise, observer) for premise in premises)
        assert any(standing(premise, observer) for premise in premises)


def test_a_reference_premise_alone_never_admits_a_derived_belief() -> None:
    assert not standing(meet([REFERENCE_SCOPE, GLOBAL_SCOPE]), "-a")
    assert standing(meet([REFERENCE_SCOPE, keyed("-a")]), "-a")
