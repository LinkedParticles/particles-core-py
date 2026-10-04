# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The closure measure: two numbers a consolidation run records about itself.

A store that is never curated does not collapse. Ingest still runs the trust
ladder, query returns both sides under the contested badge, and the
curation queue stays bounded. Contradictions are disclosed, though,
and only closed by an explicit act or an automated pass, so the share of the
store under a badge can drift upward. These two numbers say whether it does,
and who did the closing when it did not:

- the **contested fraction**: ACTIVE beliefs carrying the composed contested
  badge, over all ACTIVE beliefs, read store-wide each run; and
- the **autonomous share**: of the lifecycle transitions in the run's window,
  the share no explicit write verb caused.

A transition is one of three kinds:

- ``retirement``: a belief left ACTIVE (the write-once ``retired_at`` stamp,
  so each belief retires at most once). It is a *gesture* when an explicit
  write event (supersede, retract, source retract, review) names the belief,
  and *autonomous* otherwise: the trust ladder, update supersession,
  duplicate merge, re-anchor, staleness, abstraction revalidation.
- ``contradiction_closed``: an INCONSISTENCY record closed, by a review
  (gesture) or by the nightly disclosure pass (autonomous).
- ``promotion``: an abstraction asserted, by the accept gesture or
  by the pass itself in ``auto`` mode.

A gesture is an explicit write verb, whether an operator or an agent ran it;
``gesture_by_actor`` keeps the two apart for a reader who needs them apart.

Pure: the Engine gathers plain values and this module decides the tally.
Nothing here is stored on a particle; the result rides the
``CONSOLIDATION_RUN`` event.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

#: Event types whose PARTICLE refs name the beliefs an explicit write retired.
GESTURE_RETIREMENT_EVENTS: frozenset[str] = frozenset(
    {"PARTICLE_SUPERSEDED", "PARTICLE_RETRACTED", "SOURCE_RETRACTED", "REVIEW_RESOLVED"}
)
#: The event type a review writes once per INCONSISTENCY record it resolves.
REVIEW_EVENT = "REVIEW_RESOLVED"

#: The three transition kinds, in report order.
TRANSITION_KINDS: tuple[str, ...] = ("retirement", "contradiction_closed", "promotion")

#: The reason key for a retired row that carries no ``status_reason``.
UNSPECIFIED_REASON = "UNSPECIFIED"


@dataclass(frozen=True)
class Retirement:
    """One belief that left ACTIVE inside the window."""

    particle_id: str
    reason: str | None


@dataclass(frozen=True)
class GestureEvent:
    """One explicit write event inside the window, reduced to what the tally reads."""

    event_type: str
    actor: str
    particle_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TransitionTally:
    """Lifecycle transitions in one window, split autonomous versus gesture."""

    #: kind → {"autonomous": n, "gesture": n}, one entry per :data:`TRANSITION_KINDS`.
    by_kind: dict[str, dict[str, int]] = field(default_factory=dict)
    #: Autonomous retirements by ``status_reason``: which pass did the closing.
    autonomous_by_reason: dict[str, int] = field(default_factory=dict)
    #: Gesture transitions by the event's actor (the verb, route, or agent).
    gesture_by_actor: dict[str, int] = field(default_factory=dict)

    @property
    def autonomous(self) -> int:
        return sum(k.get("autonomous", 0) for k in self.by_kind.values())

    @property
    def gesture(self) -> int:
        return sum(k.get("gesture", 0) for k in self.by_kind.values())

    @property
    def total(self) -> int:
        return self.autonomous + self.gesture

    @property
    def autonomous_share(self) -> float | None:
        """Autonomous over all transitions; ``None`` when the window held none."""
        return share(self.autonomous, self.total)


def share(part: int, whole: int) -> float | None:
    """``part / whole``, or ``None`` when ``whole`` is zero (no denominator, no claim)."""
    return part / whole if whole > 0 else None


def contested_fraction(contested: int, active: int) -> float | None:
    """ACTIVE beliefs under the badge over ACTIVE beliefs; ``None`` on an empty store."""
    return share(contested, active)


def tally_transitions(
    *,
    retirements: Sequence[Retirement],
    gestures: Sequence[GestureEvent],
    autonomous_closures: int,
    promoted_ids: Sequence[str],
    accepted_promotions: Sequence[GestureEvent],
) -> TransitionTally:
    """Split one window's transitions into autonomous and gesture.

    ``retirements`` are the beliefs whose ``retired_at`` falls in the window.
    ``gestures`` are the explicit write events in the window; a retirement any
    of them names in :data:`GESTURE_RETIREMENT_EVENTS` is a gesture, and every
    :data:`REVIEW_EVENT` among them also closed one INCONSISTENCY record.
    ``autonomous_closures`` counts the records the disclosure pass closed.
    ``promoted_ids`` are the abstractions asserted in the window, and
    ``accepted_promotions`` the accept events among them: an asserted
    abstraction an accept event names is a gesture, and the rest are the
    pass's own.
    """
    by_actor: Counter[str] = Counter()
    retired_by: dict[str, str] = {}
    for event in gestures:
        if event.event_type in GESTURE_RETIREMENT_EVENTS:
            for pid in event.particle_ids:
                retired_by.setdefault(pid, event.actor)

    reasons: Counter[str] = Counter()
    retired_gesture = 0
    for retirement in retirements:
        actor = retired_by.get(retirement.particle_id)
        if actor is None:
            reasons[retirement.reason or UNSPECIFIED_REASON] += 1
        else:
            retired_gesture += 1
            by_actor[actor] += 1

    reviews = [e for e in gestures if e.event_type == REVIEW_EVENT]
    by_actor.update(e.actor for e in reviews)

    accepted_ids = {pid for e in accepted_promotions for pid in e.particle_ids}
    by_actor.update(e.actor for e in accepted_promotions)
    autonomous_promotions = sum(1 for pid in set(promoted_ids) if pid not in accepted_ids)

    return TransitionTally(
        by_kind={
            "retirement": {
                "autonomous": sum(reasons.values()),
                "gesture": retired_gesture,
            },
            "contradiction_closed": {
                "autonomous": autonomous_closures,
                "gesture": len(reviews),
            },
            "promotion": {
                "autonomous": autonomous_promotions,
                "gesture": len(accepted_promotions),
            },
        },
        autonomous_by_reason=dict(sorted(reasons.items())),
        gesture_by_actor=dict(sorted(by_actor.items())),
    )
