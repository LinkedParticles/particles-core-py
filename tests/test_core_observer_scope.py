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
    UNATTRIBUTED_SCOPE,
    BeliefScope,
    EntryScope,
    PairPrecondition,
    classify_entry,
    meet,
    pair_precondition,
    project_keys,
    project_tag,
    scope_of_entries,
    visible,
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
