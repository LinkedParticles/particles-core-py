# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Pure cadence decisions of the consolidation cycle.

The cycle reads its own run records, decides whether a run or a pass is due,
and applies the decision. The reading and the applying are I/O in
``particles.operations.consolidation``; the deciding is here, over plain
values, so each rule is testable without a store (D2):

* :func:`is_due` is the ``--if-due`` guard: the whole cycle runs when the last
  successful run completed at least ``consolidation.min_interval_hours`` ago
  (lifted from the orchestrator).
* :func:`decide_census` is the census cadence: pass 3 runs when the last
  census started at least ``consolidation.census.interval_hours`` ago, and
  otherwise the run discloses when it last ran and when it is next due.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

#: How early a run may start and still find the census due. A scheduler that
#: fires a few seconds or minutes earlier than it did a week ago would
#: otherwise slip the census to the following night, every week.
CENSUS_DRIFT_TOLERANCE = timedelta(hours=1)


def is_due(last_completed: datetime | None, now: datetime, min_interval_hours: float) -> bool:
    """Whether the cycle is due under ``--if-due``.

    A store with no successful run on record is always due.
    """
    if last_completed is None:
        return True
    return now - last_completed >= timedelta(hours=min_interval_hours)


@dataclass(frozen=True)
class CensusDecision:
    """Whether pass 3 runs this cycle, and what the report says if it does not.

    ``last_ran`` and ``next_due`` are disclosed either way. ``reason`` is
    ``None`` when the census runs and the disclosure text when it is skipped.
    """

    run: bool
    reason: str | None = None
    last_ran: datetime | None = None
    next_due: datetime | None = None


def decide_census(
    *,
    enabled: bool,
    interval_hours: int,
    last_ran: datetime | None,
    now: datetime,
    store_wide: bool = False,
) -> CensusDecision:
    """Decide whether the census runs on this cycle.

    Args:
        enabled: ``consolidation.census.enabled``. Off skips the census on
            every run, a ``store_wide`` request included: the switch is the
            operator's standing decision, the flag a single run's.
        interval_hours: ``consolidation.census.interval_hours``; ``0`` runs the
            census on every cycle.
        last_ran: when the last census on this store started, from its run
            record, or ``None`` when none is on record. A store with no
            census on record always runs one.
        now: the decision instant.
        store_wide: the run asked for ``--scope store``, the deliberate
            whole-store census. It runs regardless of the
            interval.
    """
    if not enabled:
        return CensusDecision(
            run=False,
            reason="consolidation.census.enabled is false",
            last_ran=last_ran,
        )
    if last_ran is None:
        return CensusDecision(run=True)
    next_due = last_ran + timedelta(hours=interval_hours)
    if store_wide or now + CENSUS_DRIFT_TOLERANCE >= next_due:
        return CensusDecision(run=True, last_ran=last_ran, next_due=next_due)
    return CensusDecision(
        run=False,
        reason=(
            f"census skipped: last ran {last_ran:%Y-%m-%d %H:%M} UTC, "
            f"next due {next_due:%Y-%m-%d %H:%M} UTC "
            f"(consolidation.census.interval_hours = {interval_hours})"
        ),
        last_ran=last_ran,
        next_due=next_due,
    )
