# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Temperature scaling calibration utility (§6.3, §C.4).

Temperature scaling maps extractor raw confidence values to empirical probabilities.
It is the recommended calibration method for cross-extractor confidence interoperability.

Usage:
    calibrator = TemperatureScaler()
    calibrator.fit(raw_values, labels)  # fit on validation set
    calibrated_value = calibrator.calibrate(raw_value)

The transform is **logit-space** — ``sigmoid(logit(p) / T)``, Guo et al. (2017)
as written — since. See :class:`TemperatureScaler` for why the
scalar-domain ``clamp(p / T, 0, 1)`` approximation it replaces was retired, and
:data:`TRANSFORM_LOGIT` for how a stored record declares which form fitted it.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]
from scipy.special import expit  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from particles.core.schema import ExtractorCalibration

log = logging.getLogger(__name__)

# Separator `_extractor_calibrate` joins contributing suite ids with when it
# writes ``ExtractorCalibration.benchmark_suite_id``.
SUITE_ID_SEPARATOR = "+"

#: Optimizer bounds for the NLL fit. A fitted T that lands *on* either bound is
#: not a temperature — it is the optimizer running out of room, which is what a
#: degenerate label set produces (see :class:`FitDiagnostics`). Widening them
#: does not fix such a fit; it only moves where it stops.
T_MIN = 0.01
T_MAX = 10.0

#: How close to a bound counts as "on" it. ``minimize_scalar(method="bounded")``
#: converges to within its xatol of the bound rather than landing exactly on it.
_BOUND_TOL = 1e-3

#: Value of ``ExtractorCalibration.transform`` for a fit produced by the
#: logit-space form this module implements. Records written from
#: v1.115.0 onward carry it.
TRANSFORM_LOGIT = "logit"

#: The retired pre-ADR-0238 form, ``clamp(raw / T, 0, 1)``. Never *written* —
#: named only so the refusal to apply an unlabelled record has a word for what
#: such a record is. A stored ``transform`` of ``None`` means this.
TRANSFORM_LINEAR = "linear"

# Numerical guard for log(0) / logit(0) in the fit.
_EPS = 1e-7


def is_saturated(raw_value: float) -> bool:
    """True when ``raw_value`` is an exact fixed point of :meth:`TemperatureScaler.calibrate`.

    ``0.0`` and ``1.0`` map to themselves under ``sigmoid(logit(p) / T)`` for
    every ``T > 0``, so no temperature can move them. The
    tolerance is :data:`_EPS`, the same guard the fit clips with — which is
    precisely where the two halves used to disagree.
    """
    return raw_value <= _EPS or raw_value >= 1.0 - _EPS


@dataclass(frozen=True)
class FitDiagnostics:
    """Whether a fitted temperature can be believed.

    A temperature is a one-parameter regression of stated confidence onto
    observed correctness. It is identified only when **both** variables vary,
    and the result is worth storing only if it actually improves anything.
    Four conditions each return a number while measuring nothing:

    * **Degenerate labels** — no outcome contrast; every label the
      same. NLL is monotone in T, so the optimizer walks to whichever bound
      sharpens hardest (all-False → :data:`T_MAX`, all-True → :data:`T_MIN`).
    * **Bound-hit** — T lands on :data:`T_MIN` or :data:`T_MAX`. The
      data wants a temperature outside the admissible range, so what got fitted
      is the range's edge. Degenerate labels always produce this; ordinary data
      can too (a single stated confidence whose empirical accuracy sits on the
      far side of 0.5 is unreachable by any positive T).
    * **Predictor degeneracy** — no *confidence* contrast among the
      movable pairs. The mirror of degenerate labels in the other variable, and
      the one a benchmark suite cannot fix: a gold set creates label contrast,
      but spread is a property of the extractor's output.
    * **Non-improving** — ``ece_after >= ece_before``. A calibration
      that does not reduce calibration error is not a calibration. This is the
      only condition about the fit's *quality* rather than its input *shape*,
      and it needs the caller to supply both figures via :meth:`with_ece`.

    Reported separately because the operator's next move differs for each: a
    degenerate label set means the suite's gold coverage is total (author a
    partial one), predictor degeneracy means the *extractor* states too few
    distinct confidences (no suite will help), a bound-hit means stated
    confidence cannot be reconciled with accuracy by temperature at all, and a
    non-improving fit means the NLL optimum simply is not a calibration.
    """

    n: int
    """Pairs supplied to the fit, before the saturation exclusion."""
    n_fitted: int
    """Pairs the fit actually consumed — ``n`` minus ``n_saturated``."""
    n_saturated: int
    """Pairs dropped as exact fixed points of the apply."""
    distinct_raw: int
    """Distinct raw confidence values among the fitted pairs."""
    positive_rate: float
    """Fraction of the *fitted* pairs labelled correct."""
    temperature: float
    ece_before: float | None = None
    ece_after: float | None = None

    def with_ece(self, before: float, after: float) -> FitDiagnostics:
        """Return a copy carrying the ECE pair the non-improvement check needs.

        Separate from :meth:`TemperatureScaler.fit` because the figures are
        computed over the *full* emitted population (saturated pairs included —
        the calibration is applied to them too, unchanged), which the fit
        deliberately does not see.
        """
        return replace(self, ece_before=before, ece_after=after)

    @property
    def degenerate_labels(self) -> bool:
        """True when every fitted label is False or every fitted label is True."""
        return self.n_fitted > 0 and self.positive_rate in (0.0, 1.0)

    @property
    def hit_bound(self) -> bool:
        """True when the fitted T converged onto an optimizer bound."""
        return self.temperature <= T_MIN + _BOUND_TOL or self.temperature >= T_MAX - _BOUND_TOL

    @property
    def predictor_degenerate(self) -> bool:
        """True when the movable pairs carry fewer than two distinct confidences."""
        return self.distinct_raw < 2

    @property
    def non_improving(self) -> bool:
        """True when the ECE pair was supplied and calibration did not improve it."""
        if self.ece_before is None or self.ece_after is None:
            return False
        return self.ece_after >= self.ece_before

    @property
    def is_trustworthy(self) -> bool:
        """True when nothing disqualifies the fit."""
        return not (
            self.degenerate_labels
            or self.hit_bound
            or self.predictor_degenerate
            or self.non_improving
        )

    def reasons(self) -> list[str]:
        """Operator-facing sentences for each disqualifying condition (empty if fine)."""
        out: list[str] = []
        if self.degenerate_labels:
            which = "matched" if self.positive_rate == 1.0 else "failed to match"
            out.append(
                f"degenerate labels: all {self.n_fitted} fittable particle(s) {which} a "
                "gold particle, so there is no correct-vs-incorrect contrast to fit a "
                "temperature against. Author a gold set that deliberately covers only "
                "part of what the extractor emits."
            )
        if self.predictor_degenerate:
            if self.n_fitted == 0:
                out.append(
                    f"no fittable pairs: all {self.n} particle(s) stated a saturated "
                    "confidence (0.0 or 1.0), which temperature scaling cannot move. "
                    "This is a property of the extractor's output, not of the suite."
                )
            else:
                out.append(
                    f"predictor degeneracy: the {self.n_fitted} fittable particle(s) carry "
                    f"only {self.distinct_raw} distinct confidence value(s). A temperature "
                    "is regressed from confidence spread; one level identifies nothing. No "
                    "gold set can fix this — spread is a property of the extractor."
                )
        if self.hit_bound:
            bound = T_MIN if self.temperature <= T_MIN + _BOUND_TOL else T_MAX
            out.append(
                f"fit landed on the optimizer bound (T={self.temperature:.4f}, "
                f"bound={bound}). Widening the bound does not fix this — it only moves "
                "where the fit stops."
            )
        if self.non_improving:
            assert self.ece_before is not None and self.ece_after is not None
            out.append(
                f"non-improving fit: calibration error would go from {self.ece_before:.4f} "
                f"to {self.ece_after:.4f}. A calibration that does not reduce calibration "
                "error is not a calibration. This figure is in-sample, so failing it here "
                "means it would fail a held-out estimate a fortiori."
            )
        return out


@dataclass
class TemperatureScaler:
    """Platt/Temperature Scaling calibrator (Guo et al. 2017).

    Fits a single temperature T minimising NLL on labelled validation data and
    applies it in **logit space**::

        calibrated = sigmoid(logit(raw_value) / T)

    T = 1 is the identity; T > 1 pulls every value toward 0.5 (an overconfident
    extractor); T < 1 pushes toward the ends (an underconfident one). 0.0 and
    1.0 are exact fixed points for every T > 0.

    **Why logit space**. Through v1.114.x this class advertised the
    formula above in its docstring but computed ``clamp(raw / T, 0, 1)`` — a
    scalar-domain approximation adopted on the reasoning that extractors expose
    a confidence, not logits. The approximation has two defects the logit form
    does not:

    * It is **not order-preserving at the top**: for T < 1 every value above T
      clamps to exactly 1.0, so 0.90 and 0.99 become indistinguishable.
    * It **degrades without bound**: at T = 10 every confidence is divided by
      ten. That is what shipped when the labelling bug drove the optimizer to
      :data:`T_MAX` — the calibration wrote 0.09 for a raw 0.9, immutably, at
      particle creation. The logit form's worst case at T = 10 is a
      squash toward 0.5, which is wrong but recoverable-looking, and the
      literature-standard behaviour.

    A confidence *is* a probability, and ``logit`` of a probability is defined;
    the premise that a scalar domain forced the approximation was the error.
    """

    temperature: float = 1.0
    diagnostics: FitDiagnostics | None = field(default=None, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(self, raw_values: list[float], correct: list[bool]) -> TemperatureScaler:
        """Fit temperature parameter on a labelled validation set.

        **Saturated pairs are excluded**: a raw confidence at
        0.0 or 1.0 is an exact fixed point of :meth:`calibrate`, so including
        one lets evidence the transform can never act on set the temperature.
        That asymmetry — the fit clipping a stated ``1.0`` to ``1 - 1e-7`` and
        weighting its logit of ≈16.1, while the apply returns it untouched —
        is what produced a measured ``T=6.29`` regressed almost entirely from
        immovable points and applied to the handful it knew least about.

        Sets :attr:`diagnostics`; callers that persist the result **must**
        consult it, and should first attach the ECE pair via
        :meth:`FitDiagnostics.with_ece` so the non-improvement check can run.

        Args:
            raw_values: extractor confidence values [0, 1]
            correct: ground-truth correctness labels
        """
        if len(raw_values) != len(correct):
            raise ValueError("raw_values and correct must have the same length")

        n = len(raw_values)
        movable = [(r, c) for r, c in zip(raw_values, correct, strict=True) if not is_saturated(r)]
        n_fitted = len(movable)

        # An empty movable set has no NLL to minimise; leave T at the identity
        # and let `predictor_degenerate` carry the refusal. Optimising over an
        # empty array would return nan and look like a fit.
        if n_fitted > 0:
            probs = np.clip(np.array([r for r, _ in movable], dtype=np.float64), _EPS, 1 - _EPS)
            labels = np.array([c for _, c in movable], dtype=np.float64)
            logits = np.log(probs / (1 - probs))

            def nll(T: float) -> float:
                scaled = np.clip(expit(logits / T), _EPS, 1 - _EPS)
                return float(-np.mean(labels * np.log(scaled) + (1 - labels) * np.log(1 - scaled)))

            result = minimize_scalar(nll, bounds=(T_MIN, T_MAX), method="bounded")
            self.temperature = float(result.x)

        self._fitted = True
        self.diagnostics = FitDiagnostics(
            n=n,
            n_fitted=n_fitted,
            n_saturated=n - n_fitted,
            distinct_raw=len({r for r, _ in movable}),
            positive_rate=(sum(1 for _, c in movable if c) / n_fitted) if n_fitted else 0.0,
            temperature=self.temperature,
        )
        log.info(
            "TemperatureScaler fitted: T=%.4f (N=%d of %d fittable, %d saturated, "
            "%d distinct confidence(s), positive rate %.3f)",
            self.temperature,
            n_fitted,
            n,
            self.diagnostics.n_saturated,
            self.diagnostics.distinct_raw,
            self.diagnostics.positive_rate,
        )
        return self

    def calibrate(self, raw_value: float) -> float:
        """Apply temperature scaling to a single confidence value."""
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        # 0 and 1 are fixed points of the logit transform for every T > 0
        # (logit(0) = -inf, logit(1) = +inf), so short-circuiting them is exact
        # rather than a clamp — and it keeps a raw 0.0 from coming back as 1e-7.
        if raw_value <= 0.0:
            return 0.0
        if raw_value >= 1.0:
            return 1.0
        logit = math.log(raw_value / (1.0 - raw_value))
        return _sigmoid(logit / self.temperature)

    def calibrate_batch(self, raw_values: list[float]) -> list[float]:
        return [self.calibrate(v) for v in raw_values]


def _sigmoid(x: float) -> float:
    """Overflow-free logistic. ``exp`` is taken of a negative number only."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def scaler_for_record(calibration: ExtractorCalibration) -> TemperatureScaler | None:
    """Scaler for a stored calibration, or ``None`` if it must not be applied.

    ``None`` is returned for any record whose ``transform`` is not
    :data:`TRANSFORM_LOGIT` — which today means every record fitted before. Such a record must not be applied for two independent reasons,
    either of which alone would be sufficient:

    1. Its T was fitted against **all-False labels** (the bug: the
       calibrate verb labelled a second extraction run's particles against the
       first run's id set, so no particle could ever be labelled correct). Every
       such fit ran to :data:`T_MAX`. There is no record of the era that escaped
       this — the bug was unconditional.
    2. Its T means ``clamp(raw / T, 0, 1)``, not ``sigmoid(logit(raw) / T)``.
       Reading one as the other silently changes what every future particle's
       stored confidence means.

    Callers fall back to ``calibration_source=EXTRACTOR_DIRECT`` — the raw
    stated value, honestly labelled — which is strictly better than either
    misreading. Re-fit with ``particles extractor calibrate <id> --regenerate``.
    """
    if calibration.transform != TRANSFORM_LOGIT:
        return None
    return TemperatureScaler(temperature=calibration.temperature)


def fitted_suite_ids(benchmark_suite_id: str) -> set[str]:
    """Split a stored ``benchmark_suite_id`` into its contributing suite ids.

    ``ExtractorCalibration.benchmark_suite_id`` is one suite id, or a
    ``+``-joined concatenation when several benchmark suites contributed
    ``(raw_confidence, correct)`` pairs to the fit. Empty segments are dropped,
    so an empty or malformed value yields the empty set rather than a set
    containing ``""``.
    """
    return {s for s in (part.strip() for part in benchmark_suite_id.split(SUITE_ID_SEPARATOR)) if s}


def is_suite_stale(benchmark_suite_id: str, selected_suite_ids: Iterable[str]) -> bool:
    """Return True if a fit's suite set differs from the extractor's current one.

    ``selected_suite_ids`` is what the auto-filter selects for the
    extractor *now* — the suites a re-fit would run. A stale record was fitted
    over a different set, so its temperature answers a question the extractor
    is no longer asked; a change stranded every general-extractor record this
    way when it narrowed the filter to routing precedence.

    Deliberately set inequality rather than "was this fitted before release X":
    it stays correct across the *next* suite-set change (a suite added for an
    extractor that had none, a ``source_type`` rerouted, a suite retired)
    instead of encoding one release date, and it does not misreport a record
    whose suite set never moved. An extractor that auto-matches nothing today
    (11 of 17 do) makes any non-empty fit stale, which is the
    honest reading: nothing in tree now measures what that fit measured.

    A record produced by an explicit ``calibrate --suite X`` is reported stale
    whenever the extractor's set is wider, which is accurate — the fit does
    cover less than the extractor's suites. Callers report; nothing blocks.
    """
    return fitted_suite_ids(benchmark_suite_id) != set(selected_suite_ids)


def expected_calibration_error(
    confidences: list[float],
    correctness: list[bool],
    *,
    bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) over equal-width confidence bins.

    The single ECE implementation shared by the extractor
    calibration tooling (:func:`calibration_error`) and the benchmark harness
    (:func:`particles.benchmark.metrics.compute_calibration_error`), which were
    two slightly-divergent copies before they were reconciled here.

    For each of ``bins`` equal-width bins over ``[0, 1]``:

      * ``conf_b`` = mean confidence of the items whose confidence falls in the bin
      * ``acc_b``  = fraction of those items that are correct
      * the bin contributes ``(|b| / N) × |conf_b - acc_b|`` to the ECE

    The **last bin is closed on the right** (``[lo, 1.0]``) so a confidence of
    exactly ``1.0`` lands in it rather than being silently dropped — the bug
    the pre-reconciliation ``calibration_error`` carried (its right-open final
    bin excluded ``1.0`` from every bin yet still divided by the full count,
    biasing ECE low). ``bins=10`` is the Guo et al. 2017 convention. Returns
    ``0.0`` for an empty input or ``bins <= 0`` — no claims is vacuously
    perfect calibration.
    """
    if len(confidences) != len(correctness):
        raise ValueError("Length mismatch")
    n = len(confidences)
    if n == 0 or bins <= 0:
        return 0.0

    width = 1.0 / bins
    ece = 0.0
    for i in range(bins):
        lo = i * width
        hi = lo + width
        last = i == bins - 1
        members = [j for j, c in enumerate(confidences) if (lo <= c < hi) or (last and c == 1.0)]
        if not members:
            continue
        conf_b = sum(confidences[j] for j in members) / len(members)
        acc_b = sum(1 for j in members if correctness[j]) / len(members)
        ece += (len(members) / n) * abs(conf_b - acc_b)
    return ece


def calibration_error(
    calibrated_values: list[float],
    correct: list[bool],
    n_bins: int = 10,
) -> float:
    """Mean absolute calibration error (ECE) across confidence bins.

    Thin adapter over :func:`expected_calibration_error` (the canonical ECE
    reconciled); kept for the ``extractor calibrate`` CLI and
    existing call sites that pass ``(values, labels, n_bins=…)``.
    """
    return expected_calibration_error(calibrated_values, correct, bins=n_bins)
