# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The dollar-budget decision: run or skip a unit of LLM work.

A run that carries a spend budget (``consolidation.budget_usd``) asks this
question before each LLM-priced pass and before each Message Batches chunk:
given what the run has spent so far, the budget, and what the next unit is
estimated to cost, does it run? The answer is a pure function over plain
values, so the orchestrator gathers the inputs, this decides, and the shell
applies the verdict (gather / decide / apply).

Spend is list price over the token counts the provider reported, never a
billing API; an estimate is list price over estimated tokens. A unit whose
cost cannot be estimated (an unpriced model) is never skipped on its estimate,
only once the recorded spend has already reached the budget.

:func:`admit_round` is the second decision here: which snapshots one
round of pass 1 reads under ``consolidation.extract_budget_usd``, at each
snapshot's worst case until tonight's rounds have measured its class, then at
its estimate scaled by that class's measured factor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

BudgetVerdict = Literal["run", "skip"]


def decide_spend(
    *,
    spent_usd: float,
    budget_usd: float | None,
    estimate_usd: float | None,
) -> BudgetVerdict:
    """Whether the next unit of LLM work runs under a dollar budget.

    Args:
        spent_usd: what the run has spent so far, in US$ at list price.
        budget_usd: the run's budget in US$, or ``None`` for no budget.
        estimate_usd: the next unit's estimated cost in US$, or ``None`` when
            it cannot be priced.

    Returns:
        ``"skip"`` when the budget is already spent, or when the estimate
        would carry the run past it; ``"run"`` otherwise, and always when there
        is no budget.
    """
    if budget_usd is None:
        return "run"
    if spent_usd >= budget_usd:
        return "skip"
    if estimate_usd is not None and spent_usd + estimate_usd > budget_usd:
        return "skip"
    return "run"


# ---------------------------------------------------------------------------
# Pass 1's nightly extraction budget, filled in rounds
# ---------------------------------------------------------------------------

#: Priced snapshots of a class tonight's completed rounds must hold before its
#: factor is defined; below it, the class is admitted at its worst case.
MIN_CALIBRATION_SNAPSHOTS = 5
#: The clamp on a class's factor (measured spend over estimate).
FACTOR_MIN = 0.1
FACTOR_MAX = 10.0
#: A calibrated round's summed worst case may be at most this multiple of what is
#: left, so a factor fitted on cheap snapshots cannot admit expensive ones without
#: bound: a night then spends at most twice its limit through planned calls.
CALIBRATED_WORST_CASE_MULTIPLE = 2.0
#: Rounding slack on dollar comparisons, so twenty US$0.20 charges fit US$4.00.
_EPSILON = 1e-9


@dataclass(frozen=True)
class RoundCandidate:
    """One snapshot a round may admit: its entry's oldest pending snapshot."""

    snapshot_id: str
    #: The class a factor is kept for: the priority class, else a source type.
    klass: str
    #: The read plan's high-end estimate in US$ (batched); ``None`` unpriced.
    estimate_usd: float | None
    #: Every planned call at its full output budget, input projected; ``None`` unpriced.
    worst_case_usd: float | None


@dataclass(frozen=True)
class ClassRate:
    """What tonight's completed rounds measured for one class."""

    snapshots: int = 0
    spent_usd: float = 0.0
    estimated_usd: float = 0.0

    def plus(self, *, snapshots: int, spent_usd: float, estimated_usd: float) -> ClassRate:
        """This rate with one more completed round of the class folded in."""
        return ClassRate(
            snapshots=self.snapshots + snapshots,
            spent_usd=self.spent_usd + spent_usd,
            estimated_usd=self.estimated_usd + estimated_usd,
        )

    def factor(self) -> float | None:
        """Measured spend over estimate, clamped; ``None`` until enough is measured."""
        if self.snapshots < MIN_CALIBRATION_SNAPSHOTS or self.estimated_usd <= 0:
            return None
        return min(FACTOR_MAX, max(FACTOR_MIN, self.spent_usd / self.estimated_usd))


@dataclass(frozen=True)
class RoundAdmission:
    """One round's admission: which snapshots, of which class, at what charge."""

    admitted: tuple[str, ...] = ()
    klass: str | None = None
    #: True when the round is charged worst cases (the class has no factor yet).
    bounded: bool = True
    factor: float | None = None
    #: The admitted snapshots' summed charge (worst cases, or factor × estimates).
    charged_usd: float = 0.0
    #: Their summed unscaled estimates, the denominator of the next factor.
    estimated_usd: float = 0.0
    #: Their summed worst cases, the round's bound on planned output.
    worst_case_usd: float = 0.0
    #: Candidates whose worst case exceeds the whole limit: never admitted.
    too_large: tuple[str, ...] = ()
    #: Candidates of the round's class that did not fit this round.
    passed_over: tuple[str, ...] = ()


def admit_round(
    candidates: Sequence[RoundCandidate],
    *,
    remaining_usd: float,
    limit_usd: float,
    rates: Mapping[str, ClassRate],
    slots: int,
) -> RoundAdmission:
    """The snapshots one round reads under pass 1's nightly budget.

    Pure: the shell gathers the candidates (in queue order, one per entry),
    the remaining budget and tonight's measured rates, and applies the answer.

    A round reads one class, the class of the first candidate that fits. A
    class with no factor yet is **bounded**: each snapshot is charged its worst
    case and the round may take all of ``remaining_usd``. A class with a
    factor is **calibrated**: each snapshot is charged ``factor × estimate``
    and the round may take half of ``remaining_usd``, except that its first
    snapshot may take up to all of it (that snapshot then forms a round of its
    own). A calibrated round's summed worst case is also held to
    ``CALIBRATED_WORST_CASE_MULTIPLE`` times ``remaining_usd``. A candidate
    that does not fit is passed over and stays a candidate.
    A candidate whose worst case exceeds ``limit_usd`` is never admitted.

    Args:
        candidates: in queue order, each its entry's oldest pending snapshot.
        remaining_usd: the budget left tonight.
        limit_usd: tonight's whole limit.
        rates: tonight's measurements per class.
        slots: snapshots the run's ceiling still allows.

    Returns:
        The admission; empty when nothing fits or no slot is left.
    """
    too_large = tuple(
        c.snapshot_id
        for c in candidates
        if c.worst_case_usd is None or c.worst_case_usd > limit_usd + _EPSILON
    )
    fits = [c for c in candidates if c.snapshot_id not in too_large]
    if slots <= 0 or remaining_usd <= 0:
        return RoundAdmission(too_large=too_large)
    for klass in dict.fromkeys(c.klass for c in fits):
        admission = _admit_class(
            [c for c in fits if c.klass == klass],
            klass=klass,
            remaining_usd=remaining_usd,
            factor=rates.get(klass, ClassRate()).factor(),
            slots=slots,
        )
        if admission.admitted:
            return RoundAdmission(
                admitted=admission.admitted,
                klass=admission.klass,
                bounded=admission.bounded,
                factor=admission.factor,
                charged_usd=admission.charged_usd,
                estimated_usd=admission.estimated_usd,
                worst_case_usd=admission.worst_case_usd,
                too_large=too_large,
                passed_over=admission.passed_over,
            )
    return RoundAdmission(too_large=too_large)


def _admit_class(
    candidates: Sequence[RoundCandidate],
    *,
    klass: str,
    remaining_usd: float,
    factor: float | None,
    slots: int,
) -> RoundAdmission:
    bounded = factor is None
    share = remaining_usd if bounded else remaining_usd / 2
    worst_cap = remaining_usd * CALIBRATED_WORST_CASE_MULTIPLE + _EPSILON
    admitted: list[str] = []
    passed: list[str] = []
    charged = estimated = worst = 0.0
    for c in candidates:
        if len(admitted) >= slots:
            break
        worst_case = c.worst_case_usd or 0.0
        estimate = c.estimate_usd if c.estimate_usd is not None else worst_case
        charge = worst_case if factor is None else factor * estimate
        if not bounded and worst + worst_case > worst_cap:
            # The calibrated round's hard ceiling on planned output.
            passed.append(c.snapshot_id)
            continue
        if not admitted and charge <= remaining_usd + _EPSILON and charge > share + _EPSILON:
            # A lone first snapshot between the share and all of what is left.
            return RoundAdmission(
                admitted=(c.snapshot_id,),
                klass=klass,
                bounded=bounded,
                factor=factor,
                charged_usd=charge,
                estimated_usd=estimate,
                worst_case_usd=worst_case,
                passed_over=tuple(passed),
            )
        if charged + charge <= share + _EPSILON:
            admitted.append(c.snapshot_id)
            charged += charge
            estimated += estimate
            worst += worst_case
        else:
            passed.append(c.snapshot_id)
    return RoundAdmission(
        admitted=tuple(admitted),
        klass=klass,
        bounded=bounded,
        factor=factor,
        charged_usd=charged,
        estimated_usd=estimated,
        worst_case_usd=worst,
        passed_over=tuple(passed) if admitted else (),
    )
