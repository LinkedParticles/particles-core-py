"""Event-anchored validity extraction — the deterministic parser gate.

Per tests/AGENTS.md, extractor *parsing* (the deterministic map from a fixed LLM
reply to candidates) is required-to-test; the LLM call itself is out of scope
(the integration tier drives the real model). These tests pin:

  * ``_gate_valid_until`` — the pure three-condition gate: an explicit boundary
    cue AND a confidence floor AND a future date, biased toward under-emission,
    with a named drop reason on every rejection (never silent). This is the
    over-eager-expiry guard, so the durable-mention and born-expired paths are
    the load-bearing cases.
  * ``_parse_extraction_response`` — the gate wired into the real parser: a
    future-bounded item populates ``CandidateParticle.valid_until`` + the audit
    crumbs; a durable mention / below-floor / born-expired / unparseable item
    carries no boundary and emits a quality note; the disabled config leaves
    every candidate unbounded.
  * ``candidate_to_particle`` — the boundary flows through to
    ``Particle.valid_until`` (the field the §9.3 lint + the as-of read consume).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from particles.config import ExtractionValidityConfig, ParticlesConfig
from particles.extraction import general
from particles.extraction.general import (
    VALIDITY_BASIS_KEY,
    VALIDITY_CONFIDENCE_KEY,
    _gate_valid_until,
    _parse_extraction_response,
    candidate_to_particle,
)

NOW = datetime(2026, 6, 1, tzinfo=UTC)
FUTURE = "2026-12-31"
PAST = "2020-01-01"


# ---------------------------------------------------------------------------
# The pure gate
# ---------------------------------------------------------------------------


class TestGateValidUntil:
    def test_emits_future_bounded_above_floor(self) -> None:
        vu, conf, basis, note = _gate_valid_until(
            FUTURE, 0.9, "runs through 2026", floor=0.75, now=NOW
        )
        assert vu is not None
        assert vu.date().isoformat() == FUTURE
        assert vu.tzinfo is not None  # normalized to aware UTC
        assert conf == 0.9
        assert basis == "runs through 2026"
        assert note is None

    def test_no_cue_emits_nothing_silently(self) -> None:
        # The durable-mention path: the model set valid_until null — no boundary,
        # and (correctly) no note, since nothing was dropped.
        for raw in (None, "", "   "):
            vu, conf, basis, note = _gate_valid_until(raw, None, None, floor=0.75, now=NOW)
            assert (vu, conf, basis, note) == (None, None, None, None)

    def test_below_floor_is_dropped_with_note(self) -> None:
        vu, _conf, _basis, note = _gate_valid_until(FUTURE, 0.5, "maybe", floor=0.75, now=NOW)
        assert vu is None
        assert note is not None and "VALIDITY_BELOW_FLOOR" in note

    def test_missing_confidence_is_dropped_with_note(self) -> None:
        vu, _conf, _basis, note = _gate_valid_until(FUTURE, None, "cue", floor=0.75, now=NOW)
        assert vu is None
        assert note is not None and "VALIDITY_BELOW_FLOOR" in note

    def test_born_expired_past_date_is_dropped(self) -> None:
        vu, _conf, _basis, note = _gate_valid_until(PAST, 0.99, "until 2020", floor=0.75, now=NOW)
        assert vu is None
        assert note is not None and "VALIDITY_BORN_EXPIRED" in note

    def test_equal_to_now_is_born_expired(self) -> None:
        # A boundary exactly at ``now`` provides no forward-looking value.
        vu, _conf, _basis, note = _gate_valid_until(
            NOW.date().isoformat(), 0.99, "today", floor=0.75, now=NOW
        )
        assert vu is None
        assert note is not None and "VALIDITY_BORN_EXPIRED" in note

    def test_unparseable_date_is_dropped(self) -> None:
        # An unresolved relative expression must not be guessed into a date.
        vu, _conf, _basis, note = _gate_valid_until("tomorrow", 0.99, "cue", floor=0.75, now=NOW)
        assert vu is None
        assert note is not None and "VALIDITY_UNPARSEABLE" in note

    def test_naive_iso_date_normalized_to_utc(self) -> None:
        vu, _conf, _basis, _note = _gate_valid_until(FUTURE, 0.9, None, floor=0.75, now=NOW)
        assert vu is not None and vu.tzinfo is UTC


# ---------------------------------------------------------------------------
# The parser, with the gate wired in
# ---------------------------------------------------------------------------


def _patch_config(monkeypatch: pytest.MonkeyPatch, *, enabled: bool, floor: float = 0.75) -> None:
    cfg = ParticlesConfig(
        extraction_validity=ExtractionValidityConfig(enabled=enabled, min_boundary_confidence=floor)
    )
    monkeypatch.setattr(general, "get_config", lambda: cfg)


def _reply(items: list[dict[str, object]]) -> str:
    return json.dumps(items)


class TestParserValidity:
    def test_future_boundary_populates_candidate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_config(monkeypatch, enabled=True)
        raw = _reply(
            [
                {
                    "content": "The contract runs through 2026.",
                    "subjects": [],
                    "confidence_value": 0.9,
                    "uncertainty_nature": "EPISTEMIC",
                    "valid_until": FUTURE,
                    "validity_confidence": 0.9,
                    "validity_basis": "runs through 2026",
                }
            ]
        )
        candidates, _notes = _parse_extraction_response(raw)
        assert len(candidates) == 1
        c = candidates[0]
        assert c.valid_until is not None and c.valid_until.date().isoformat() == FUTURE
        assert c.properties is not None
        assert c.properties[VALIDITY_BASIS_KEY] == "runs through 2026"
        assert c.properties[VALIDITY_CONFIDENCE_KEY] == 0.9

    def test_durable_mention_carries_no_boundary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The model correctly emitted null for a date-mentioning durable fact.
        _patch_config(monkeypatch, enabled=True)
        raw = _reply(
            [
                {
                    "content": "I met her in 2019.",
                    "subjects": [],
                    "confidence_value": 0.9,
                    "uncertainty_nature": "EPISTEMIC",
                    "valid_until": None,
                    "validity_confidence": None,
                    "validity_basis": None,
                }
            ]
        )
        candidates, _notes = _parse_extraction_response(raw)
        assert len(candidates) == 1
        assert candidates[0].valid_until is None

    def test_born_expired_dropped_with_note(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_config(monkeypatch, enabled=True)
        raw = _reply(
            [
                {
                    "content": "The order was in force until 2020.",
                    "subjects": [],
                    "confidence_value": 0.9,
                    "uncertainty_nature": "EPISTEMIC",
                    "valid_until": PAST,
                    "validity_confidence": 0.95,
                    "validity_basis": "until 2020",
                }
            ]
        )
        candidates, notes = _parse_extraction_response(raw)
        assert candidates[0].valid_until is None
        assert any("VALIDITY_BORN_EXPIRED" in n for n in notes)

    def test_below_floor_dropped_with_note(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_config(monkeypatch, enabled=True, floor=0.8)
        raw = _reply(
            [
                {
                    "content": "Maybe valid until the release.",
                    "subjects": [],
                    "confidence_value": 0.9,
                    "uncertainty_nature": "EPISTEMIC",
                    "valid_until": FUTURE,
                    "validity_confidence": 0.6,
                    "validity_basis": "until the release",
                }
            ]
        )
        candidates, notes = _parse_extraction_response(raw)
        assert candidates[0].valid_until is None
        assert any("VALIDITY_BELOW_FLOOR" in n for n in notes)

    def test_disabled_config_never_sets_boundary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_config(monkeypatch, enabled=False)
        raw = _reply(
            [
                {
                    "content": "The contract runs through 2026.",
                    "subjects": [],
                    "confidence_value": 0.9,
                    "uncertainty_nature": "EPISTEMIC",
                    "valid_until": FUTURE,
                    "validity_confidence": 0.99,
                    "validity_basis": "runs through 2026",
                }
            ]
        )
        candidates, _notes = _parse_extraction_response(raw)
        assert candidates[0].valid_until is None
        # No validity crumbs on properties when the feature is off.
        assert not (candidates[0].properties or {}).get(VALIDITY_BASIS_KEY)


# ---------------------------------------------------------------------------
# candidate_to_particle passthrough
# ---------------------------------------------------------------------------


class TestCandidateToParticle:
    def test_valid_until_flows_to_particle(self) -> None:
        from particles.core.schema import UncertaintyNature
        from particles.extraction.general import CandidateParticle

        boundary = datetime(2026, 12, 31, tzinfo=UTC)
        cand = CandidateParticle(
            content="The contract runs through 2026.",
            confidence_value=0.9,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            valid_until=boundary,
        )
        particle = candidate_to_particle(
            cand, corpus_entry_id="e1", snapshot_id="s1", subject_ids=[]
        )
        assert particle.valid_until == boundary

    def test_no_boundary_stays_none(self) -> None:
        from particles.core.schema import UncertaintyNature
        from particles.extraction.general import CandidateParticle

        cand = CandidateParticle(
            content="A durable fact.",
            confidence_value=0.9,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
        )
        particle = candidate_to_particle(
            cand, corpus_entry_id="e1", snapshot_id="s1", subject_ids=[]
        )
        assert particle.valid_until is None
