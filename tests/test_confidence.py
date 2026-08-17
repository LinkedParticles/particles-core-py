"""Tests for the two-quantity confidence math (particles/core/confidence.py).

Covers ``compute_effective_confidence`` and the read-side **uncalibrated
confidence cap**: a default-OFF, opt-in ``min`` on the
``confidence.value`` factor entering the effective-confidence formula, keyed on
``calibration_source``. The cap never mutates the stored, immutable
``confidence.value``; it only lowers the value fed into the formula.
"""

from __future__ import annotations

import pytest

from particles.config import get_config, reset_config
from particles.core.schema import Confidence
from particles.core.scoring.confidence import (
    CalibrationSource,
    compute_effective_confidence,
)


def _enable_cap(
    *,
    cap_value: float = 0.7,
    sources: list[str] | None = None,
) -> None:
    """Turn the cap on in the live config singleton.

    The autouse ``clear_subject_cache`` fixture in conftest calls
    ``reset_config()`` before every test, so these mutations never leak.
    """
    cap = get_config().confidence.uncalibrated_cap
    cap.enabled = True
    cap.cap_value = cap_value
    if sources is not None:
        cap.sources = sources


# ---------------------------------------------------------------------------
# Baseline: cap disabled by default (byte-for-byte current behaviour)
# ---------------------------------------------------------------------------


class TestCapDisabledByDefault:
    def test_default_config_cap_is_off(self) -> None:
        reset_config()
        cap = get_config().confidence.uncalibrated_cap
        assert cap.enabled is False
        assert cap.cap_value == pytest.approx(0.7)
        assert cap.sources == ["EXTRACTOR_DIRECT"]

    def test_default_no_calibration_source_unchanged(self) -> None:
        # No calibration_source passed → identical to the historical formula.
        assert compute_effective_confidence(0.95) == pytest.approx(0.95)
        assert compute_effective_confidence(
            0.95,
            extractor_trust_weight=0.5,
            source_trust_rank=0.8,
            recency_factor=0.5,
        ) == pytest.approx(0.95 * 0.5 * 0.8 * 0.5)

    def test_disabled_config_does_not_cap_even_with_source(self) -> None:
        # Cap off (default) → a high EXTRACTOR_DIRECT value is untouched.
        assert compute_effective_confidence(
            0.95, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
        ) == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Enabled: cap bites the configured sources only
# ---------------------------------------------------------------------------


class TestCapEnabled:
    def test_extractor_direct_value_capped(self) -> None:
        _enable_cap(cap_value=0.7)
        # value 0.95 → capped to 0.7 as the value factor; trust factors are 1.0.
        assert compute_effective_confidence(
            0.95, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
        ) == pytest.approx(0.7)

    def test_value_below_cap_is_not_raised(self) -> None:
        _enable_cap(cap_value=0.7)
        # min(0.5, 0.7) == 0.5 — the cap never *raises* a value.
        assert compute_effective_confidence(
            0.5, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
        ) == pytest.approx(0.5)

    def test_value_equal_to_cap_unchanged(self) -> None:
        _enable_cap(cap_value=0.7)
        assert compute_effective_confidence(
            0.7, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
        ) == pytest.approx(0.7)

    def test_calibrated_benchmark_never_capped(self) -> None:
        _enable_cap(cap_value=0.7)
        # Not in default sources → calibrated values keep their full value.
        assert compute_effective_confidence(
            0.95, calibration_source=CalibrationSource.CALIBRATED_BENCHMARK
        ) == pytest.approx(0.95)

    def test_human_review_never_capped(self) -> None:
        _enable_cap(cap_value=0.7)
        assert compute_effective_confidence(
            0.95, calibration_source=CalibrationSource.HUMAN_REVIEW
        ) == pytest.approx(0.95)

    def test_agent_asserted_not_capped_by_default(self) -> None:
        # AGENT_ASSERTED is absent from the default sources list.
        _enable_cap(cap_value=0.7)
        assert compute_effective_confidence(
            0.95, calibration_source=CalibrationSource.AGENT_ASSERTED
        ) == pytest.approx(0.95)

    def test_agent_asserted_capped_when_added_to_sources(self) -> None:
        _enable_cap(cap_value=0.7, sources=["EXTRACTOR_DIRECT", "AGENT_ASSERTED"])
        assert compute_effective_confidence(
            0.95, calibration_source=CalibrationSource.AGENT_ASSERTED
        ) == pytest.approx(0.7)
        # EXTRACTOR_DIRECT still capped too.
        assert compute_effective_confidence(
            0.95, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
        ) == pytest.approx(0.7)

    def test_custom_cap_value(self) -> None:
        _enable_cap(cap_value=0.4)
        assert compute_effective_confidence(
            0.95, calibration_source=CalibrationSource.EXTRACTOR_DIRECT
        ) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Read-side only: the stored Confidence.value is never mutated
# ---------------------------------------------------------------------------


class TestReadSideOnly:
    def test_stored_confidence_value_unchanged(self) -> None:
        _enable_cap(cap_value=0.7)
        conf = Confidence(value=0.95, calibration_source=CalibrationSource.EXTRACTOR_DIRECT)
        eff = compute_effective_confidence(conf.value, calibration_source=conf.calibration_source)
        assert eff == pytest.approx(0.7)
        # The cap is a read-side min on the value fed into the formula; the
        # stored, immutable confidence.value is untouched.
        assert conf.value == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Composition: cap the value, THEN multiply by the trust / recency factors
# (a particle may be subject to BOTH the trust-weight cap and this
# value cap)
# ---------------------------------------------------------------------------


class TestComposition:
    def test_cap_then_multiply(self) -> None:
        _enable_cap(cap_value=0.7)
        # min(0.95, 0.7) = 0.7, then × 0.5 (extractor trust) × 0.8 (source) ×
        # 0.5 (recency).
        eff = compute_effective_confidence(
            0.95,
            extractor_trust_weight=0.5,
            source_trust_rank=0.8,
            recency_factor=0.5,
            calibration_source=CalibrationSource.EXTRACTOR_DIRECT,
        )
        assert eff == pytest.approx(0.7 * 0.5 * 0.8 * 0.5)

    def test_value_cap_composes_with_trust_weight_cap(self) -> None:
        # The trust-weight cap arrives pre-clamped via the
        # extractor_trust_weight argument (the store's get_trust_weight_map does
        # that min); the value cap is applied here on the value. Both
        # bite the same particle: value 0.9 → 0.7, weight already clamped to 0.5.
        _enable_cap(cap_value=0.7)
        eff = compute_effective_confidence(
            0.9,
            extractor_trust_weight=0.5,  # already clamped upstream
            calibration_source=CalibrationSource.EXTRACTOR_DIRECT,
        )
        assert eff == pytest.approx(0.7 * 0.5)

    def test_result_still_clamped_to_unit_interval(self) -> None:
        # An over-unit product (e.g. a >1 weight) is still clamped to 1.0 after
        # the value cap, preserving the existing [0, 1] contract.
        _enable_cap(cap_value=0.7)
        eff = compute_effective_confidence(
            0.95,
            extractor_trust_weight=10.0,
            calibration_source=CalibrationSource.EXTRACTOR_DIRECT,
        )
        assert eff == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Config validation: sources must name real CalibrationSource members
# ---------------------------------------------------------------------------


class TestSourcesValidation:
    def test_unknown_source_rejected(self) -> None:
        from particles.config import UncalibratedCapConfig

        with pytest.raises(ValueError):
            UncalibratedCapConfig(sources=["NOT_A_SOURCE"])

    def test_all_known_sources_accepted(self) -> None:
        from particles.config import UncalibratedCapConfig

        cfg = UncalibratedCapConfig(
            sources=[
                "EXTRACTOR_DIRECT",
                "AGENT_ASSERTED",
                "CALIBRATED_BENCHMARK",
                "HUMAN_REVIEW",
            ]
        )
        assert len(cfg.sources) == 4


# ---------------------------------------------------------------------------
# DERIVED calibration source + min-of-premises derivation
# ---------------------------------------------------------------------------


class TestDerivedAbstractionConfidence:
    """stored value for a derived particle is min over premises."""

    def test_min_of_premises(self) -> None:
        from particles.core.scoring.confidence import derive_abstraction_confidence

        assert derive_abstraction_confidence([0.9, 0.6, 0.8]) == 0.6

    def test_single_premise_passthrough(self) -> None:
        from particles.core.scoring.confidence import derive_abstraction_confidence

        assert derive_abstraction_confidence([0.42]) == 0.42

    def test_clamped_to_unit_interval(self) -> None:
        from particles.core.scoring.confidence import derive_abstraction_confidence

        assert derive_abstraction_confidence([1.5, 2.0]) == 1.0
        assert derive_abstraction_confidence([-0.5, 0.9]) == 0.0

    def test_empty_premises_raises(self) -> None:
        from particles.core.scoring.confidence import derive_abstraction_confidence

        with pytest.raises(ValueError, match="at least one premise"):
            derive_abstraction_confidence([])

    def test_derived_enum_value_exists(self) -> None:
        assert CalibrationSource.DERIVED.value == "DERIVED"

    def test_derived_in_json_schema_artifact(self) -> None:
        """The normative artifact enum stays in sync (spec_impact)."""
        import json
        from pathlib import Path

        schema_file = Path(__file__).parents[1] / "artifacts" / "schemas" / "particle.schema.json"
        schema = json.loads(schema_file.read_text())
        artifact_enum = set(
            schema["$defs"]["Confidence"]["properties"]["calibration_source"]["enum"]
        )
        assert artifact_enum == {m.value for m in CalibrationSource}

    def test_derived_confidence_model_accepts_source(self) -> None:
        c = Confidence(value=0.5, calibration_source=CalibrationSource.DERIVED)
        assert c.calibration_source is CalibrationSource.DERIVED
