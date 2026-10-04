# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The vocabulary document: the pure model, its revision rules, the
profile book it composes into, the proposal planner, and the JSON-LD codec.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pytest
from pydantic import ValidationError

from particles.core.predicate_profile import (
    MULTI_VALUE,
    SINGLE_VALUE,
    PredicateRole,
    SlotKind,
)
from particles.core.vocabulary import (
    Alignment,
    CensusRow,
    MatchStrength,
    Ruling,
    VocabularyDocument,
    WikidataConstraint,
    add_alias,
    add_alignment,
    effective_kind,
    find_term,
    form_stats,
    new_document,
    plan_proposals,
    proposal_key,
    set_profile,
)
from particles.core.vocabulary_jsonld import (
    DOCUMENT_TYPE,
    dumps,
    is_vocabulary_document,
    loads,
    to_jsonld,
)

T0 = datetime(2026, 10, 2, tzinfo=UTC)
RULING = Ruling(confirmed_by="reviewer", confirmed_at=T0, evidence={"basis": "test"})
FILE = "artifact:file"
RECORD = "artifact:record"


def _doc(name: str = "acme") -> VocabularyDocument:
    return new_document(
        name=name,
        prefix="acme",
        namespace="https://vocab.example.org/acme/",
        publisher="Acme",
        issued=T0,
    )


class TestDocumentModel:
    def test_new_document_is_version_one_with_no_terms(self) -> None:
        doc = _doc()
        assert doc.version == 1
        assert doc.terms == []
        assert doc.kind == "VocabularyDocument"

    @pytest.mark.parametrize("name", ["Acme", "acme name", "", "-acme"])
    def test_name_must_be_a_slug(self, name: str) -> None:
        with pytest.raises(ValidationError):
            new_document(name=name, prefix="acme", namespace="https://x.example/")

    @pytest.mark.parametrize("ns", ["https://x.example/v", "ftp://x.example/", "x.example/"])
    def test_namespace_must_be_an_iri_base(self, ns: str) -> None:
        with pytest.raises(ValidationError):
            new_document(name="acme", prefix="acme", namespace=ns)

    def test_prefix_must_be_a_curie_prefix(self) -> None:
        with pytest.raises(ValidationError):
            new_document(name="acme", prefix="1bad:", namespace="https://x.example/")

    def test_a_form_belongs_to_one_term_per_class(self) -> None:
        doc = add_alias(
            _doc(), form="mov to", subject_class=FILE, aliases=["relocat to"], ruling=RULING
        )
        with pytest.raises(ValueError, match="already"):
            add_alias(
                doc, form="transfer to", subject_class=FILE, aliases=["mov to"], ruling=RULING
            )

    def test_the_same_form_on_two_classes_is_two_terms(self) -> None:
        doc = set_profile(
            _doc(), form="mov to", subject_class=FILE, kind=SlotKind.ONE_AT_A_TIME, ruling=RULING
        )
        doc = set_profile(
            doc, form="mov to", subject_class=RECORD, kind=SlotKind.TIMELESS_SINGLE, ruling=RULING
        )
        assert len(doc.terms) == 2
        assert {t.local_name for t in doc.terms} == {
            "artifact-file/mov-to",
            "artifact-record/mov-to",
        }

    def test_equivalence_is_never_an_estimate(self) -> None:
        with pytest.raises(ValidationError, match="asserted"):
            Alignment(
                target="wdt:P551", match=MatchStrength.EQUIVALENT, confidence=0.9, ruling=RULING
            )

    def test_a_role_must_name_a_member_form(self) -> None:
        with pytest.raises(ValueError, match="does not cover"):
            set_profile(
                _doc(),
                form="mov to",
                subject_class=FILE,
                kind=SlotKind.ONE_AT_A_TIME,
                roles={"mov from": PredicateRole.PAST},
                ruling=RULING,
            )


class TestRevision:
    def test_every_revision_is_the_next_version(self) -> None:
        v1 = _doc()
        v2 = add_alias(v1, form="contain", subject_class=FILE, aliases=["includ"], ruling=RULING)
        v3 = set_profile(
            v2, form="contain", subject_class=FILE, kind=SlotKind.MANY_AT_ONCE, ruling=RULING
        )
        assert (v1.version, v2.version, v3.version) == (1, 2, 3)
        assert v1.terms == []  # the earlier version is untouched
        term = find_term(v3, "includ", FILE)
        assert term is not None and term.form == "contain"
        assert term.profile is not None and term.profile.kind is SlotKind.MANY_AT_ONCE

    def test_alignment_maps_outward_and_never_renames(self) -> None:
        doc = set_profile(
            _doc(),
            form="liv in",
            subject_class="person",
            kind=SlotKind.ONE_AT_A_TIME,
            ruling=RULING,
        )
        local = doc.terms[0].local_name
        aligned = add_alignment(
            doc,
            term_ref=f"acme:{local}",
            alignment=Alignment(
                target="wdt:P551", match=MatchStrength.EXACT, confidence=0.9, ruling=RULING
            ),
        )
        assert aligned.terms[0].local_name == local
        assert aligned.terms[0].form == "liv in"
        assert aligned.terms[0].alignments[0].target == "wdt:P551"

    def test_a_second_alignment_to_one_target_replaces_the_first(self) -> None:
        doc = set_profile(
            _doc(),
            form="liv in",
            subject_class="person",
            kind=SlotKind.ONE_AT_A_TIME,
            ruling=RULING,
        )
        ref = doc.terms[0].local_name
        for match in (MatchStrength.CLOSE, MatchStrength.EXACT):
            doc = add_alignment(
                doc,
                term_ref=ref,
                alignment=Alignment(target="wdt:P551", match=match, confidence=0.8, ruling=RULING),
            )
        assert [a.match for a in doc.terms[0].alignments] == [MatchStrength.EXACT]

    def test_aligning_an_unknown_term_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no term"):
            add_alignment(
                _doc(),
                term_ref="nope",
                alignment=Alignment(
                    target="wdt:P551", match=MatchStrength.EXACT, confidence=0.9, ruling=RULING
                ),
            )

    def test_an_existing_alias_is_not_repeated(self) -> None:
        doc = add_alias(
            _doc(), form="contain", subject_class=FILE, aliases=["includ"], ruling=RULING
        )
        again = add_alias(
            doc, form="contain", subject_class=FILE, aliases=["includ", "hold"], ruling=RULING
        )
        assert [a.form for a in again.terms[0].aliases] == ["includ", "hold"]


class TestEffectiveKind:
    def _term_doc(self, **alignment: object) -> VocabularyDocument:
        doc = set_profile(
            _doc(),
            form="hav author",
            subject_class="work",
            kind=SlotKind.ONE_AT_A_TIME,
            ruling=RULING,
        )
        return add_alignment(
            doc,
            term_ref=doc.terms[0].local_name,
            alignment=Alignment(target="wdt:P50", ruling=RULING, **alignment),
        )  # type: ignore[arg-type]

    def test_the_profile_supplies_the_kind_when_the_source_states_none(self) -> None:
        doc = self._term_doc(match=MatchStrength.EXACT, confidence=0.9)
        assert effective_kind(doc.terms[0]) is SlotKind.ONE_AT_A_TIME

    def test_the_source_wins_where_it_states_a_kind(self) -> None:
        doc = self._term_doc(
            match=MatchStrength.EQUIVALENT,
            wikidata_constraints=[WikidataConstraint(item=SINGLE_VALUE)],
        )
        assert effective_kind(doc.terms[0]) is SlotKind.TIMELESS_SINGLE

    def test_a_close_match_is_never_read_for_a_kind(self) -> None:
        doc = self._term_doc(
            match=MatchStrength.CLOSE,
            confidence=0.6,
            wikidata_constraints=[WikidataConstraint(item=MULTI_VALUE)],
        )
        assert effective_kind(doc.terms[0]) is SlotKind.ONE_AT_A_TIME

    def test_a_term_with_no_profile_and_no_source_kind_has_none(self) -> None:
        doc = add_alias(
            _doc(), form="contain", subject_class=FILE, aliases=["includ"], ruling=RULING
        )
        assert effective_kind(doc.terms[0]) is None


def _stats(*rows: tuple[str, str, int]) -> list:  # type: ignore[type-arg]
    return form_stats(CensusRow(subject_class=c, predicate=p, claims=n) for c, p, n in rows)


def _unit(*xs: float) -> np.ndarray:  # type: ignore[type-arg]
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


class TestPlanProposals:
    def test_form_stats_groups_surfaces_by_normalised_form_and_class(self) -> None:
        stats = _stats((FILE, "contains", 5), (FILE, "contained", 2), (RECORD, "contains", 1))
        by_key = {(s.form, s.subject_class): s for s in stats}
        assert by_key[("contain", FILE)].claims == 7
        assert by_key[("contain", FILE)].display == "contains"
        assert by_key[("contain", RECORD)].claims == 1

    def test_close_forms_on_one_class_become_an_alias_proposal(self) -> None:
        stats = _stats((FILE, "contains", 9), (FILE, "includes", 3), (FILE, "deletes", 4))
        vectors = {
            ("contain", FILE): _unit(1, 0.05, 0),
            ("includ", FILE): _unit(1, 0.1, 0),
            ("delet", FILE): _unit(0, 0, 1),
        }
        plan = plan_proposals(stats, vectors, threshold=0.85)
        assert len(plan.aliases) == 1
        alias = plan.aliases[0]
        assert (alias.canonical, alias.aliases, alias.claims) == ("contain", ("includ",), 12)
        assert alias.min_similarity > 0.85
        assert alias.key == proposal_key("alias", FILE, ["contain", "includ"])

    def test_opposite_direction_or_polarity_is_never_one_alias(self) -> None:
        stats = _stats(
            (RECORD, "renamed to", 5),
            (RECORD, "renamed from", 4),
            (RECORD, "includes", 3),
            (RECORD, "does not include", 2),
        )
        rename, include = _unit(1, 0, 0), _unit(0, 1, 0)
        vectors = {
            ("renam to", RECORD): rename,
            ("renam from", RECORD): rename,
            ("includ", RECORD): include,
            ("not include", RECORD): include,
        }
        plan = plan_proposals(stats, vectors, threshold=0.85)
        # Two clusters by embedding, each split in two by direction or
        # polarity, so no part has two forms.
        assert plan.aliases == []

    def test_classes_never_mix(self) -> None:
        stats = _stats((FILE, "contains", 5), (RECORD, "includes", 5))
        same = _unit(1, 0, 0)
        plan = plan_proposals(
            stats, {(s.form, s.subject_class): same for s in stats}, threshold=0.85
        )
        assert plan.aliases == []

    def test_profiles_are_ranked_by_claims_and_name_the_past_partner(self) -> None:
        stats = _stats((RECORD, "moved to", 11), (RECORD, "moved from", 6), (RECORD, "has", 2))
        plan = plan_proposals(stats, {}, threshold=0.85)
        assert [p.form for p in plan.profiles] == ["mov to", "mov from", "hav"]
        assert plan.profiles[0].past_partners == ("mov from",)
        assert plan.aliases == []  # no vectors, no alias candidates

    def test_what_a_document_already_holds_is_not_proposed_again(self) -> None:
        stats = _stats((FILE, "contains", 9), (FILE, "includes", 3), (FILE, "holds", 2))
        vectors = {
            ("contain", FILE): _unit(1, 0.05, 0),
            ("includ", FILE): _unit(1, 0.1, 0),
            ("hold", FILE): _unit(1, 0.08, 0),
        }
        doc = add_alias(
            _doc(), form="contain", subject_class=FILE, aliases=["includ"], ruling=RULING
        )
        doc = set_profile(
            doc, form="contain", subject_class=FILE, kind=SlotKind.MANY_AT_ONCE, ruling=RULING
        )
        plan = plan_proposals(stats, vectors, threshold=0.85, documents=[doc])
        assert [(a.canonical, a.aliases) for a in plan.aliases] == [("contain", ("hold",))]
        # contain is profiled and includ is its alias: only hold is offered.
        assert [p.form for p in plan.profiles] == ["hold"]

    def test_excluded_keys_are_skipped(self) -> None:
        stats = _stats((RECORD, "moved to", 11))
        key = proposal_key("profile", RECORD, ["mov to"])
        assert plan_proposals(stats, {}, threshold=0.85, excluded=frozenset({key})).profiles == []

    def test_two_minted_terms_are_never_joined(self) -> None:
        stats = _stats((FILE, "contains", 9), (FILE, "includes", 3))
        same = _unit(1, 0, 0)
        doc = set_profile(
            _doc(), form="contain", subject_class=FILE, kind=SlotKind.MANY_AT_ONCE, ruling=RULING
        )
        doc = set_profile(
            doc, form="includ", subject_class=FILE, kind=SlotKind.MANY_AT_ONCE, ruling=RULING
        )
        plan = plan_proposals(
            stats, {(s.form, s.subject_class): same for s in stats}, threshold=0.85, documents=[doc]
        )
        assert plan.aliases == []

    def test_coverage_counts_surface_predicates(self) -> None:
        stats = _stats((FILE, "contains", 9), (FILE, "contained", 1), (FILE, "includes", 3))
        vectors = {("contain", FILE): _unit(1, 0, 0), ("includ", FILE): _unit(1, 0.1, 0)}
        plan = plan_proposals(stats, vectors, threshold=0.85)
        assert plan.predicates_in_alias_candidates() == 3
        assert (plan.classed_predicates, plan.classed_forms, plan.classes) == (3, 2, 1)


class TestJsonLd:
    def _rich(self) -> VocabularyDocument:
        doc = add_alias(
            _doc(),
            form="mov to",
            subject_class=RECORD,
            aliases=["relocat to", "mov from"],
            ruling=RULING,
            label="moved to",
        )
        doc = set_profile(
            doc,
            form="mov to",
            subject_class=RECORD,
            kind=SlotKind.ONE_AT_A_TIME,
            roles={"mov from": PredicateRole.PAST},
            ruling=RULING,
        )
        doc = set_profile(
            doc,
            form="hav birth dat",
            subject_class="person",
            kind=SlotKind.TIMELESS_SINGLE,
            ruling=RULING,
        )
        return add_alignment(
            doc,
            term_ref=doc.terms[1].local_name,
            alignment=Alignment(
                target="wdt:P569",
                match=MatchStrength.EQUIVALENT,
                ruling=RULING,
                wikidata_constraints=[
                    WikidataConstraint(item="Q52060874", separators=["P1480", "P3831"])
                ],
            ),
        )

    def test_round_trip_is_exact(self) -> None:
        doc = self._rich()
        assert loads(dumps(doc)) == doc
        assert dumps(loads(dumps(doc))) == dumps(doc)

    def test_serialisation_is_canonical(self) -> None:
        text = dumps(self._rich())
        assert text.endswith("}\n")
        assert (
            text
            == json.dumps(json.loads(text), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )

    def test_it_speaks_the_w3c_vocabularies(self) -> None:
        obj = to_jsonld(self._rich())
        assert obj["@context"][0] == "https://linkedparticles.org/schemas/context.jsonld"
        assert obj["@context"][1]["acme"] == "https://vocab.example.org/acme/"
        assert DOCUMENT_TYPE in obj["@type"] and "skos:ConceptScheme" in obj["@type"]
        assert obj["owl:versionInfo"] == 5
        moved, birth = obj["ppx:terms"]
        assert moved["@id"] == "acme:artifact-record/mov-to"
        assert moved["skos:prefLabel"] == "mov to"
        assert moved["skos:altLabel"] == ["relocat to", "mov from"]
        assert moved["ppx:profile"]["@type"] == "sh:PropertyShape"
        assert moved["ppx:profile"]["ppx:oneAtATime"] is True
        assert "sh:maxCount" not in moved["ppx:profile"]
        assert moved["ppx:profile"]["sh:targetClass"] == RECORD
        assert birth["ppx:profile"]["sh:maxCount"] == 1
        assert birth["owl:equivalentProperty"] == ["wdt:P569"]
        ruling = birth["ppx:ruling"]
        assert ruling["prov:wasAttributedTo"] == "reviewer"
        assert ruling["prov:generatedAtTime"] == T0.isoformat()
        assert ruling["ppx:evidence"] == {"basis": "test"}

    def test_estimated_alignments_export_by_band(self) -> None:
        doc = set_profile(
            _doc(),
            form="liv in",
            subject_class="person",
            kind=SlotKind.ONE_AT_A_TIME,
            ruling=RULING,
        )
        for target, match in (
            ("wdt:P551", MatchStrength.EXACT),
            ("schema:homeLocation", MatchStrength.CLOSE),
        ):
            doc = add_alignment(
                doc,
                term_ref=doc.terms[0].local_name,
                alignment=Alignment(target=target, match=match, confidence=0.7, ruling=RULING),
            )
        term = to_jsonld(doc)["ppx:terms"][0]
        assert term["skos:exactMatch"] == ["wdt:P551"]
        assert term["skos:closeMatch"] == ["schema:homeLocation"]
        assert "owl:equivalentProperty" not in term

    def test_the_corpus_entry_id_never_serialises(self) -> None:
        doc = _doc().model_copy(update={"corpus_entry_id": "store-local"})
        assert "store-local" not in dumps(doc)

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda o: o.update({"@type": ["skos:ConceptScheme"]}),
            lambda o: o.pop("ppx:name"),
            lambda o: o["ppx:terms"][0]["ppx:aliasRulings"].clear(),
            lambda o: o["ppx:terms"][0].update({"@id": "other:artifact-record/mov-to"}),
        ],
    )
    def test_malformed_documents_are_refused(self, mutate) -> None:  # type: ignore[no-untyped-def]
        obj = to_jsonld(self._rich())
        mutate(obj)
        with pytest.raises(ValueError):
            loads(json.dumps(obj))

    def test_not_json_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not JSON"):
            loads(b"\xff\xfe")

    def test_detection_reads_the_type(self) -> None:
        assert is_vocabulary_document(to_jsonld(_doc()))
        assert not is_vocabulary_document({"@type": "skos:ConceptScheme"})
        assert not is_vocabulary_document([1, 2])
