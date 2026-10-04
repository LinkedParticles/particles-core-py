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
"""

from __future__ import annotations

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
