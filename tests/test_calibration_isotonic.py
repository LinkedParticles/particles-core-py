# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The isotonic calibration method.

Pinned: the fit is monotone and respects its minimum block, the map moves
values a temperature cannot (a flat 0.9 stated at a 0.6 hit rate), a stored
record round-trips through ``scaler_for_record`` and ``candidate_to_particle``,
and an isotonic record without breakpoints is refused like any unappliable one.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from particles.core.schema import ExtractorCalibration, UncertaintyNature
from particles.core.scoring.confidence import CalibrationSource
from particles.extraction.calibration import (
    TRANSFORM_ISOTONIC,
    IsotonicScaler,
    TemperatureScaler,
    describe_calibration,
    heldout_check,
    scaler_for_record,
)
from particles.extraction.general import CandidateParticle, candidate_to_particle


def _pairs() -> tuple[list[float], list[bool]]:
    raws = [0.9] * 40 + [0.6] * 40
    correct = [True] * 24 + [False] * 16 + [True] * 12 + [False] * 28
    return raws, correct


def test_fit_is_monotone_and_maps_to_observed_rates() -> None:
    raws, correct = _pairs()
    scaler = IsotonicScaler().fit(raws, correct)
    assert [(pytest.approx(x), pytest.approx(y)) for x, y in scaler.breakpoints] == [
        (0.6, 0.3),
        (0.9, 0.6),
    ]
    assert scaler.calibrate(0.9) == pytest.approx(0.6)
    assert scaler.calibrate(0.75) == pytest.approx(0.45)
    assert scaler.calibrate(0.2) == pytest.approx(0.3)
    assert scaler.method == "isotonic_regression"


def test_violators_are_pooled_and_small_blocks_merged() -> None:
    raws = [0.1 * i for i in range(1, 10) for _ in range(5)]
    correct = [(i // 5) % 2 == 0 for i in range(len(raws))]
    scaler = IsotonicScaler(min_block=10).fit(raws, correct)
    ys = [y for _, y in scaler.breakpoints]
    assert ys == sorted(ys)
    assert scaler.diagnostics is not None
    assert scaler.diagnostics.n == len(raws)


def test_isotonic_corrects_what_a_temperature_cannot() -> None:
    raws = [0.9] * 50
    correct = [True] * 30 + [False] * 20
    iso = IsotonicScaler().fit(raws, correct)
    check = heldout_check(iso, raws, correct)
    assert check.ece_after < 0.01 < check.ece_before
    # A temperature leaves a single stated value's rank and can only move it
    # along one curve; the diagnostics refuse a one-level fit outright.
    temp = TemperatureScaler().fit(raws, correct)
    assert temp.diagnostics is not None and temp.diagnostics.predictor_degenerate


def _record(**kw: object) -> ExtractorCalibration:
    base: dict[str, object] = {
        "temperature": 1.0,
        "transform": TRANSFORM_ISOTONIC,
        "fitted_at": datetime(2026, 10, 9, tzinfo=UTC),
        "benchmark_suite_id": "prose-calibration-001",
        "sample_size": 80,
        "calibration_error_before": 0.2,
        "calibration_error_after": 0.05,
        "extractor_version": "0.1.0",
        "breakpoints": [(0.6, 0.3), (0.9, 0.6)],
    }
    base.update(kw)
    return ExtractorCalibration(**base)  # type: ignore[arg-type]


def test_record_round_trips_and_stamps_the_method() -> None:
    record = _record()
    scaler = scaler_for_record(record)
    assert isinstance(scaler, IsotonicScaler)
    assert describe_calibration(record) == "isotonic, 2 breakpoint(s)"
    particle = candidate_to_particle(
        CandidateParticle(
            content="The release ships on Friday.",
            confidence_value=0.9,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
        ),
        "entry",
        "snapshot",
        "general-extractor",
        calibration=record,
    )
    assert particle.confidence.value == pytest.approx(0.6)
    assert particle.confidence.calibration_source is CalibrationSource.CALIBRATED_BENCHMARK
    assert particle.confidence.calibration_method == "isotonic_regression"


def test_an_isotonic_record_without_breakpoints_is_refused() -> None:
    assert scaler_for_record(_record(breakpoints=None)) is None


def test_a_temperature_record_still_describes_its_temperature() -> None:
    assert describe_calibration(_record(transform="logit", temperature=2.5, breakpoints=None)) == (
        "T=2.5000"
    )
