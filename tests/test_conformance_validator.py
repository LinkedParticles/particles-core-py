"""Tests for the extractor conformance validator.

Coverage:
  * end-to-end run of the Numista coin extractor against a real fixture
  * ExtractorNotFoundError on an unknown extractor_id
  * empty corpus produces a report with zero particles and REQUIRED failures
  * the uncertainty_nature DIVERSITY rule correctly flags an
    always-EPISTEMIC extractor and passes a mixed extractor — as an
    ADVISORY, so it reports without adjudicating, while a
    FAIL-severity rule still flips the verdict
  * RECOMMENDED fields produce warnings below threshold
  * the validator's ``passed`` property and failure/warning subsets behave
    as documented

The Numista coin extractor is used as the realistic case because it is
fully structured (no LLM, no httpx, no subject-resolver dependency) so its
``extract()`` is deterministic against a JSON blob.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from particles.conformance.contract import CONTRACT, DIVERSITY
from particles.conformance.fixtures import (
    Fixture,
    compute_corpus_hash,
    iter_fixtures,
)
from particles.conformance.types import (
    ConformanceReport,
    DiversityRule,
    DiversitySeverity,
    FieldStat,
    FieldTier,
)
from particles.conformance.validator import (
    ExtractorNotFoundError,
    _bare_properties_keys,
    _compute_field_stat,
    select_fixtures,
    validate_extractor,
)
from particles.core.schema import (
    ExtractionStatus,
    Snapshot,
    UncertaintyNature,
    WarcRecordType,
)
from particles.extraction.general import CandidateParticle, ExtractionResult
from particles.extraction.registry import get_extractors, select_extractor

FIXTURE_DIR = Path(__file__).parent / "conformance" / "fixtures"


# ---------------------------------------------------------------------------
# Stub extractors — injected via monkeypatch to exercise diversity / tier rules
# without coupling tests to real extractor behaviour.
# ---------------------------------------------------------------------------


class _UniformEpistemicStub:
    EXTRACTOR_ID = "test-uniform-epistemic"
    EXTRACTOR_VERSION = "0.1.0"

    def accepts(self, source_type: str) -> bool:
        return source_type == "NUMISTA_API_COIN"

    async def extract(
        self, snapshot: Snapshot, content: bytes, **kwargs: object
    ) -> ExtractionResult:
        return ExtractionResult(
            candidates=[
                CandidateParticle(
                    content="Coin A is round.",
                    confidence_value=0.9,
                    uncertainty_nature=UncertaintyNature.EPISTEMIC,
                    subjects=["Coin A"],
                ),
                CandidateParticle(
                    content="Coin B is round.",
                    confidence_value=0.9,
                    uncertainty_nature=UncertaintyNature.EPISTEMIC,
                    subjects=["Coin B"],
                ),
            ]
        )


class _MixedNatureStub:
    """Emits one ALEATORY and one EPISTEMIC — passes the diversity rule."""

    EXTRACTOR_ID = "test-mixed-nature"
    EXTRACTOR_VERSION = "0.1.0"

    def accepts(self, source_type: str) -> bool:
        return source_type == "NUMISTA_API_COIN"

    async def extract(
        self, snapshot: Snapshot, content: bytes, **kwargs: object
    ) -> ExtractionResult:
        return ExtractionResult(
            candidates=[
                CandidateParticle(
                    content="Stochastic measurement.",
                    confidence_value=0.8,
                    uncertainty_nature=UncertaintyNature.ALEATORY,
                    subjects=["Sample"],
                ),
                CandidateParticle(
                    content="Model belief.",
                    confidence_value=0.8,
                    uncertainty_nature=UncertaintyNature.EPISTEMIC,
                    subjects=["Sample"],
                ),
            ]
        )


class _NoSubjectsStub:
    """Emits a candidate with no subjects — REQUIRED subject_ids fails."""

    EXTRACTOR_ID = "test-no-subjects"
    EXTRACTOR_VERSION = "0.1.0"

    def accepts(self, source_type: str) -> bool:
        return source_type == "NUMISTA_API_COIN"

    async def extract(
        self, snapshot: Snapshot, content: bytes, **kwargs: object
    ) -> ExtractionResult:
        return ExtractionResult(
            candidates=[
                CandidateParticle(
                    content="Claim about something.",
                    confidence_value=0.5,
                    uncertainty_nature=UncertaintyNature.ALEATORY,
                    subjects=[],
                ),
                CandidateParticle(
                    content="Claim about something else.",
                    confidence_value=0.5,
                    uncertainty_nature=UncertaintyNature.EPISTEMIC,
                    subjects=[],
                ),
            ]
        )


class _BarePropertiesKeyStub:
    """Emits a candidate whose properties dict has a bare (no-prefix) key.

    Every key must use a ``prefix:LocalName`` shape; the
    validator must surface bare keys as warning-level quality notes.
    """

    EXTRACTOR_ID = "test-bare-properties-key"
    EXTRACTOR_VERSION = "0.1.0"

    def accepts(self, source_type: str) -> bool:
        return source_type == "NUMISTA_API_COIN"

    async def extract(
        self, snapshot: Snapshot, content: bytes, **kwargs: object
    ) -> ExtractionResult:
        return ExtractionResult(
            candidates=[
                CandidateParticle(
                    content="Story has 200 points.",
                    confidence_value=0.9,
                    uncertainty_nature=UncertaintyNature.EPISTEMIC,
                    subjects=["thread"],
                    properties={
                        "hn:hasPoints": 200,
                        "score": 200,  # a bare key
                        "social:hasScore": 200,
                    },
                ),
                CandidateParticle(
                    content="Stochastic claim.",
                    confidence_value=0.9,
                    uncertainty_nature=UncertaintyNature.ALEATORY,
                    subjects=["thread"],
                ),
            ]
        )


class _StampedStub:
    """Emits candidates carrying a ``provider_model`` stamp.

    Stands in for any LLM-backed extractor: the real ones acquire the stamp
    inside ``_call_llm`` rather than setting it themselves, but what reaches
    ``candidate_to_particle`` is identical. Pass more than one pairing to
    simulate the mixed-run anomaly.
    """

    EXTRACTOR_ID = "test-stamped"
    EXTRACTOR_VERSION = "0.1.0"

    def __init__(self, *pairings: str) -> None:
        self._pairings = pairings

    def accepts(self, source_type: str) -> bool:
        return source_type == "NUMISTA_API_COIN"

    async def extract(
        self, snapshot: Snapshot, content: bytes, **kwargs: object
    ) -> ExtractionResult:
        natures = [UncertaintyNature.EPISTEMIC, UncertaintyNature.ALEATORY]
        return ExtractionResult(
            candidates=[
                CandidateParticle(
                    content=f"Claim {idx} from {pairing}.",
                    confidence_value=0.7,
                    uncertainty_nature=natures[idx % 2],
                    subjects=["Sample"],
                    provider_model=pairing,
                )
                for idx, pairing in enumerate(self._pairings * 2)
            ]
        )


class _AcceptsNothingStub:
    EXTRACTOR_ID = "test-accepts-nothing"
    EXTRACTOR_VERSION = "0.1.0"

    def accepts(self, source_type: str) -> bool:
        return False

    async def extract(
        self, snapshot: Snapshot, content: bytes, **kwargs: object
    ) -> ExtractionResult:  # pragma: no cover — never invoked
        return ExtractionResult()


def _inject_stub(monkeypatch: pytest.MonkeyPatch, stub: object) -> None:
    """Make ``stub`` the whole registry, for lookup *and* for routing.

    Two names must move together since: ``validator.get_extractors``
    (how ``_find_extractor`` resolves the id) and ``registry.get_extractors``
    (what ``select_extractor`` / ``selects`` walk to decide routing). Patching
    only the first leaves the stub unroutable, so every fixture would be
    filtered out and the report would cover zero particles.
    """

    def fake_get_extractors() -> list[object]:
        return [stub]

    monkeypatch.setattr("particles.conformance.validator.get_extractors", fake_get_extractors)
    monkeypatch.setattr("particles.extraction.registry.get_extractors", fake_get_extractors)


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------


class TestFixtureLoading:
    def test_iter_fixtures_loads_numista_coin(self) -> None:
        loaded = list(iter_fixtures(FIXTURE_DIR))
        ids = [f.fixture_id for f in loaded]
        assert "numista-coin-001" in ids
        f = next(x for x in loaded if x.fixture_id == "numista-coin-001")
        assert f.source_type == "NUMISTA_API_COIN"
        assert "numista-coin-extractor" in f.expected_acceptors
        assert isinstance(f.snapshot, Snapshot)
        assert f.snapshot.warc_record_type == WarcRecordType.RESPONSE
        assert f.snapshot.extraction_status == ExtractionStatus.PENDING
        assert len(f.content) > 0

    def test_iter_fixtures_loads_journal_entry(self) -> None:
        # Wiring guard for the journal-extractor fixture:
        # deterministic, no LLM — confirms the JOURNAL source_type matches the
        # extractor that should accept it. Catches a renamed source_type or a
        # deleted fixture without needing a live ANTHROPIC_API_KEY.
        from particles.extraction.journal import JournalExtractor

        loaded = {f.fixture_id: f for f in iter_fixtures(FIXTURE_DIR)}
        assert "journal-entry-001" in loaded
        f = loaded["journal-entry-001"]
        assert f.source_type == "JOURNAL"
        assert "journal-extractor" in f.expected_acceptors
        assert JournalExtractor().accepts(f.source_type) is True
        assert len(f.content) > 0

    def test_iter_fixtures_skips_corpus_manifest_and_hidden(self, tmp_path: Path) -> None:
        # Build a synthetic corpus with a real fixture, a hidden dir, a
        # plain file, and a partial dir missing manifest.yaml. Only the
        # real fixture should be yielded.
        real = tmp_path / "real"
        real.mkdir()
        (real / "manifest.yaml").write_text(
            "fixture_id: real\nsource_type: X\nexpected_acceptors: []\n"
        )
        (real / "content.bin").write_bytes(b"hello")
        (real / "snapshot.json").write_text('{"content_hash": "' + "a" * 64 + '"}')

        partial = tmp_path / "partial"
        partial.mkdir()
        (partial / "content.bin").write_bytes(b"x")

        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "manifest.yaml").write_text("fixture_id: hidden\nsource_type: X\n")

        (tmp_path / "MANIFEST.yaml").write_text("fixtures: []\n")

        ids = [f.fixture_id for f in iter_fixtures(tmp_path)]
        assert ids == ["real"]

    def test_iter_fixtures_empty_dir_returns_nothing(self, tmp_path: Path) -> None:
        assert list(iter_fixtures(tmp_path)) == []

    def test_iter_fixtures_missing_dir_returns_nothing(self, tmp_path: Path) -> None:
        assert list(iter_fixtures(tmp_path / "does-not-exist")) == []

    def test_compute_corpus_hash_is_stable(self) -> None:
        h1 = compute_corpus_hash(FIXTURE_DIR)
        h2 = compute_corpus_hash(FIXTURE_DIR)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_compute_corpus_hash_changes_on_content_change(self, tmp_path: Path) -> None:
        d = tmp_path / "fx"
        d.mkdir()
        (d / "manifest.yaml").write_text("fixture_id: a\nsource_type: X\nexpected_acceptors: []\n")
        (d / "snapshot.json").write_text('{"content_hash": "' + "a" * 64 + '"}')
        (d / "content.bin").write_bytes(b"first")
        h_first = compute_corpus_hash(tmp_path)

        (d / "content.bin").write_bytes(b"second")
        h_second = compute_corpus_hash(tmp_path)

        assert h_first != h_second


# ---------------------------------------------------------------------------
# End-to-end run against the real Numista coin extractor
# ---------------------------------------------------------------------------


class TestValidateExtractorNumista:
    @pytest.mark.asyncio
    async def test_runs_end_to_end(self) -> None:
        report = await validate_extractor("numista-coin-extractor", fixture_dir=FIXTURE_DIR)
        assert isinstance(report, ConformanceReport)
        assert report.extractor_id == "numista-coin-extractor"
        assert report.fixture_count == 1
        # Numista coin extractor emits: 1 structured + obverse + reverse +
        # edge + 1 mint + 2 catalog refs = 7 candidates for the fixture.
        assert report.particle_count == 7

    @pytest.mark.asyncio
    async def test_covers_every_contract_field(self) -> None:
        report = await validate_extractor("numista-coin-extractor", fixture_dir=FIXTURE_DIR)
        assert len(report.fields) == len(CONTRACT)
        # Every contract field appears exactly once in the report.
        seen = {s.field for s in report.fields}
        expected = {c.field for c in CONTRACT}
        assert seen == expected

    @pytest.mark.asyncio
    async def test_numista_diversity_finding_is_advisory_not_a_failure(self) -> None:
        """Numista hardcodes EPISTEMIC — flagged, but ADVISORY.

        The structured-extractor reading: a parser reports what a source
        states, so its residual uncertainty is never about sampling. The rule
        still fires and still rides the report; it just does not adjudicate,
        which is what keeps the extractor out of ``failures`` and out of the trust cap.
        """
        report = await validate_extractor("numista-coin-extractor", fixture_dir=FIXTURE_DIR)
        un_stat = next(s for s in report.fields if s.field == "uncertainty_nature")
        assert un_stat.rate == 1.0  # field is populated everywhere
        assert un_stat.distinct_values == 1  # but only one value seen
        assert un_stat.passes_threshold  # ADVISORY never flips the verdict
        assert un_stat.failure_reason is None
        assert "Diversity rule violated" in (un_stat.advisory_reason or "")
        assert un_stat in report.advisories
        assert un_stat not in report.failures
        # The whole extractor now passes: this was its only REQUIRED failure.
        assert report.passed
        assert report.failures == []


# ---------------------------------------------------------------------------
# Error paths and edge cases
# ---------------------------------------------------------------------------


class TestErrorPaths:
    @pytest.mark.asyncio
    async def test_unknown_extractor_raises(self) -> None:
        with pytest.raises(ExtractorNotFoundError) as ei:
            await validate_extractor("does-not-exist", fixture_dir=FIXTURE_DIR)
        # Helpful message includes the list of registered extractors
        assert "does-not-exist" in str(ei.value)
        assert "Registered extractors" in str(ei.value)

    @pytest.mark.asyncio
    async def test_empty_corpus_produces_no_particle_failures(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _inject_stub(monkeypatch, _UniformEpistemicStub())
        report = await validate_extractor("test-uniform-epistemic", fixture_dir=tmp_path)
        assert report.particle_count == 0
        assert report.fixture_count == 0
        # Every REQUIRED field reports "No particles produced"; OPTIONAL pass.
        required = [s for s in report.fields if s.tier == FieldTier.REQUIRED]
        assert all(not s.passes_threshold for s in required)
        assert all("No particles produced" in (s.failure_reason or "") for s in required)
        optional = [s for s in report.fields if s.tier == FieldTier.OPTIONAL]
        assert all(s.passes_threshold for s in optional)

    @pytest.mark.asyncio
    async def test_quality_note_emitted_when_no_fixture_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stub accepts nothing, but the corpus has fixtures.
        _inject_stub(monkeypatch, _AcceptsNothingStub())
        report = await validate_extractor("test-accepts-nothing", fixture_dir=FIXTURE_DIR)
        assert report.particle_count == 0
        assert report.fixture_count == 0
        assert any("No corpus fixture is scored" in n for n in report.quality_notes)


# ---------------------------------------------------------------------------
# Diversity rule + tier semantics via stubs
# ---------------------------------------------------------------------------


class TestDiversityAndTiers:
    @pytest.mark.asyncio
    async def test_uniform_epistemic_stub_raises_an_advisory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _inject_stub(monkeypatch, _UniformEpistemicStub())
        report = await validate_extractor("test-uniform-epistemic", fixture_dir=FIXTURE_DIR)
        assert report.particle_count == 2
        un = next(s for s in report.fields if s.field == "uncertainty_nature")
        assert un.distinct_values == 1
        assert un.passes_threshold  # reported, not adjudicated
        assert un.failure_reason is None
        assert "Diversity rule violated" in (un.advisory_reason or "")
        assert report.advisory_count == 1

    @pytest.mark.asyncio
    async def test_mixed_nature_stub_passes_diversity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _inject_stub(monkeypatch, _MixedNatureStub())
        report = await validate_extractor("test-mixed-nature", fixture_dir=FIXTURE_DIR)
        un = next(s for s in report.fields if s.field == "uncertainty_nature")
        assert un.distinct_values == 2
        assert un.passes_threshold
        # No REQUIRED failures => report.passed is True
        assert report.passed
        assert report.failures == []
        # Satisfying the rule means no advisory either.
        assert report.advisories == []

    @pytest.mark.asyncio
    async def test_value_counts_carries_the_diversity_margin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """the histogram behind ``distinct_values``.

        A bare ``distinct_values: 2`` cannot say whether the second value held
        by one particle or by half of them, and on a non-deterministic
        LLM-backed extractor that margin is the whole signal.
        """
        _inject_stub(monkeypatch, _MixedNatureStub())
        report = await validate_extractor("test-mixed-nature", fixture_dir=FIXTURE_DIR)
        un = next(s for s in report.fields if s.field == "uncertainty_nature")
        assert un.value_counts == {"ALEATORY": 1, "EPISTEMIC": 1}
        assert un.distinct_values == len(un.value_counts)
        assert sum(un.value_counts.values()) == report.particle_count
        # Non-enum fields carry no histogram.
        content = next(s for s in report.fields if s.field == "content")
        assert content.value_counts == {}

    @pytest.mark.asyncio
    async def test_fail_severity_still_adjudicates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The mechanism keeps its teeth for a future FAIL-severity rule.

        The ``uncertainty_nature`` rule is demoted specifically, on a
        finding about that field. It does not soften ``DIVERSITY`` itself.
        """
        monkeypatch.setattr(
            "particles.conformance.validator.DIVERSITY",
            [
                DiversityRule(
                    field="uncertainty_nature",
                    min_distinct_values=2,
                    severity=DiversitySeverity.FAIL,
                    rationale="test rule",
                )
            ],
        )
        _inject_stub(monkeypatch, _UniformEpistemicStub())
        report = await validate_extractor("test-uniform-epistemic", fixture_dir=FIXTURE_DIR)
        un = next(s for s in report.fields if s.field == "uncertainty_nature")
        assert not un.passes_threshold
        assert "Diversity rule violated" in (un.failure_reason or "")
        assert un.advisory_reason is None
        assert un in report.failures
        assert not report.passed
        assert report.advisories == []

    def test_every_shipped_diversity_rule_states_its_severity(self) -> None:
        """severity has no default; a rule must decide."""
        assert DIVERSITY, "the contract ships at least one diversity rule"
        for rule in DIVERSITY:
            assert isinstance(rule.severity, DiversitySeverity)

    @pytest.mark.asyncio
    async def test_required_field_fails_when_unpopulated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _inject_stub(monkeypatch, _NoSubjectsStub())
        report = await validate_extractor("test-no-subjects", fixture_dir=FIXTURE_DIR)
        sid = next(s for s in report.fields if s.field == "subject_ids")
        assert sid.rate == 0.0
        assert not sid.passes_threshold
        assert sid in report.failures
        assert sid.failure_reason is not None
        assert "100%" in sid.failure_reason  # expected 100% wording


# ---------------------------------------------------------------------------
# Recommended-threshold semantics
# ---------------------------------------------------------------------------


class TestRecommendedThreshold:
    @pytest.mark.asyncio
    async def test_recommended_passes_above_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Numista populates extractor_ref + provenance[].snapshot_id +
        # confidence.calibration_source on every particle: RECOMMENDED all
        # pass and warnings is empty.
        report = await validate_extractor(
            "numista-coin-extractor",
            fixture_dir=FIXTURE_DIR,
            recommended_threshold=0.8,
        )
        recommended = [s for s in report.fields if s.tier == FieldTier.RECOMMENDED]
        assert all(s.passes_threshold for s in recommended)
        assert report.warnings == []
        assert report.warning_count == 0

    def test_recommended_below_threshold_is_warning_not_failure(self) -> None:
        """_compute_field_stat: rate below recommended threshold → not-passes."""
        from particles.conformance.types import FieldContract

        entry = FieldContract(field="extractor_ref", tier=FieldTier.RECOMMENDED, rationale="x")
        # Two particles, one with extractor_ref, one without.
        from particles.core.schema import (
            Confidence,
            Particle,
            ProvenanceRef,
            ProvenanceRefType,
        )
        from particles.core.scoring.confidence import CalibrationSource

        p_with = Particle(
            content="x",
            confidence=Confidence(value=0.5, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            provenance=[
                ProvenanceRef(
                    type=ProvenanceRefType.SOURCE,
                    corpus_entry_id="e",
                    snapshot_id="s",
                )
            ],
            asserted_by="stub",
            extractor_ref={"name": "stub", "version": "1"},
            subject_ids=["s"],
        )
        p_without = Particle(
            content="y",
            confidence=Confidence(value=0.5, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            provenance=[
                ProvenanceRef(
                    type=ProvenanceRefType.SOURCE,
                    corpus_entry_id="e",
                    snapshot_id="s",
                )
            ],
            asserted_by="stub",
            extractor_ref=None,
            subject_ids=["s"],
        )

        stat = _compute_field_stat(
            entry, [p_with, p_without], diversity_rules={}, recommended_threshold=0.8
        )
        assert isinstance(stat, FieldStat)
        assert stat.rate == 0.5
        assert not stat.passes_threshold
        assert stat.tier == FieldTier.RECOMMENDED
        assert "threshold" in (stat.failure_reason or "").lower()


# ---------------------------------------------------------------------------
# Fixture dataclass smoke test
# ---------------------------------------------------------------------------


class TestPropertiesKeyShape:
    """every Particle.properties key must use prefix:LocalName."""

    def test_helper_flags_bare_key_only(self) -> None:
        cand = CandidateParticle(
            content="x",
            confidence_value=0.5,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            properties={
                "hn:hasPoints": 1,
                "score": 1,
                "social:hasScore": 1,
            },
        )
        assert _bare_properties_keys(cand) == ["score"]

    def test_helper_tolerates_none_and_empty(self) -> None:
        cand_none = CandidateParticle(
            content="x",
            confidence_value=0.5,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            properties=None,
        )
        cand_empty = CandidateParticle(
            content="x",
            confidence_value=0.5,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            properties={},
        )
        assert _bare_properties_keys(cand_none) == []
        assert _bare_properties_keys(cand_empty) == []

    @pytest.mark.asyncio
    async def test_bare_properties_key_emits_quality_note(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _inject_stub(monkeypatch, _BarePropertiesKeyStub())
        report = await validate_extractor("test-bare-properties-key", fixture_dir=FIXTURE_DIR)
        notes_text = " ".join(report.quality_notes)
        assert "prefix:LocalName" in notes_text
        assert "'score'" in notes_text
        # Prefixed keys must not produce notes.
        assert "'hn:hasPoints'" not in notes_text
        assert "'social:hasScore'" not in notes_text


class TestFixtureSelection:
    """fixture selection follows production routing precedence."""

    def test_every_fixture_routes_to_a_declared_expected_acceptor(self) -> None:
        """The drift guard.

        ``expected_acceptors`` is hand-maintained operator intent and is not a
        selection input; this is what keeps it from silently disagreeing with
        the registry that actually does the routing.
        """
        for fixture in iter_fixtures(FIXTURE_DIR):
            routed = select_extractor(fixture.source_type).EXTRACTOR_ID
            assert routed in fixture.expected_acceptors, (
                f"Fixture {fixture.fixture_id} ({fixture.source_type}) is routed to "
                f"{routed!r}, which its manifest does not list in expected_acceptors="
                f"{fixture.expected_acceptors}"
            )

    def test_each_fixture_is_selected_by_exactly_one_extractor(self) -> None:
        corpus = list(iter_fixtures(FIXTURE_DIR))
        for fixture in corpus:
            owners = [p for p in get_extractors() if select_fixtures(p, [fixture])]
            assert len(owners) == 1, (
                f"{fixture.fixture_id} selected by {[p.EXTRACTOR_ID for p in owners]}"
            )

    def test_general_extractor_selects_only_its_own_fixture(self) -> None:
        """The whole point: the fallback accepts everything, is routed one thing."""
        corpus = list(iter_fixtures(FIXTURE_DIR))
        general = next(p for p in get_extractors() if p.EXTRACTOR_ID == "general-extractor")

        assert [f.fixture_id for f in select_fixtures(general, corpus)] == [
            "web-article-001",
            "web-article-002",
            "web-essay-001",
            "web-interview-001",
        ]
        # accepts() is unconditionally True for the fallback — the pre-0231 set.
        assert all(general.accepts(f.source_type) for f in corpus)

    def test_all_accepted_restores_the_wide_set(self) -> None:
        corpus = list(iter_fixtures(FIXTURE_DIR))
        general = next(p for p in get_extractors() if p.EXTRACTOR_ID == "general-extractor")
        widened = select_fixtures(general, corpus, all_accepted=True)
        assert len(widened) == len(corpus)

    def test_domain_extractor_selection_is_unchanged_by_the_rule(self) -> None:
        """Only the fallback's set narrows; every domain extractor keeps its fixture."""
        corpus = list(iter_fixtures(FIXTURE_DIR))
        for plugin in get_extractors():
            if plugin.EXTRACTOR_ID == "general-extractor":
                continue
            routed = select_fixtures(plugin, corpus)
            accepted = select_fixtures(plugin, corpus, all_accepted=True)
            assert routed == accepted, f"{plugin.EXTRACTOR_ID} routed != accepted"

    @pytest.mark.asyncio
    async def test_report_records_which_fixture_set_was_scored(self) -> None:
        """``ConformanceReport``'s shape is unchanged, so the mode rides quality_notes."""
        routed = await validate_extractor("numista-coin-extractor", fixture_dir=FIXTURE_DIR)
        assert "routing precedence" in routed.quality_notes[0]
        assert routed.fixture_count == 1

        widened = await validate_extractor(
            "numista-coin-extractor", fixture_dir=FIXTURE_DIR, all_accepted=True
        )
        assert "--all-accepted" in widened.quality_notes[0]


class TestExtractionProviderModel:
    """the report names the pairing that produced its particles."""

    @pytest.mark.asyncio
    async def test_deterministic_extractor_reports_none(self) -> None:
        """A parser makes no completion call, so the honest value is null.

        This null is the discriminator the baseline README reads: it is what
        separates a byte-reproducible baseline from an LLM-derived one.
        """
        report = await validate_extractor("numista-coin-extractor", fixture_dir=FIXTURE_DIR)
        assert report.particle_count > 0
        assert report.extraction_provider_model is None

    @pytest.mark.asyncio
    async def test_stamped_candidates_surface_the_pairing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The value is aggregated off the particles, not read from config.

        ``_call_llm`` stamps ``candidate.provider_model`` at the completion
        seam and ``candidate_to_particle`` threads it onto the
        Particle, so a stub that stamps its candidates is exactly what an
        LLM-backed extractor looks like to the validator.
        """
        _inject_stub(monkeypatch, _StampedStub("anthropic:claude-sonnet-4-6"))
        report = await validate_extractor("test-stamped", fixture_dir=FIXTURE_DIR)
        assert report.particle_count == 2
        assert report.extraction_provider_model == "anthropic:claude-sonnet-4-6"
        # Nothing anomalous about a single pairing — no note is added for it.
        assert not any("Mixed extraction pairings" in n for n in report.quality_notes)

    @pytest.mark.asyncio
    async def test_mixed_pairings_are_recorded_and_flagged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run whose provider changed mid-pass is a capture error, said out loud.

        Sorted and joined rather than dropped — silently reporting one of two
        pairings would be a false provenance record, which is the failure this
        field exists to prevent.
        """
        _inject_stub(
            monkeypatch, _StampedStub("openai:gpt-5.6-luna", "anthropic:claude-sonnet-4-6")
        )
        report = await validate_extractor("test-stamped", fixture_dir=FIXTURE_DIR)
        assert report.extraction_provider_model == (
            "anthropic:claude-sonnet-4-6, openai:gpt-5.6-luna"
        )
        assert any("Mixed extraction pairings" in n for n in report.quality_notes)
        assert any("not a valid baseline capture" in n for n in report.quality_notes)

    @pytest.mark.asyncio
    async def test_zero_particle_run_reports_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No particles ⇒ no pairing to report; the unknown path stays unknown."""
        _inject_stub(monkeypatch, _AcceptsNothingStub())
        report = await validate_extractor("test-accepts-nothing", fixture_dir=FIXTURE_DIR)
        assert report.particle_count == 0
        assert report.extraction_provider_model is None


def test_fixture_is_frozen() -> None:
    f = list(iter_fixtures(FIXTURE_DIR))[0]
    with pytest.raises((AttributeError, TypeError)):
        # frozen dataclass — assignment must fail
        f.fixture_id = "tamper"  # type: ignore[misc]
    assert isinstance(f, Fixture)
