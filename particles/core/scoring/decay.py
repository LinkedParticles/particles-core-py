# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Content age decay for effective confidence (§6.3).

The multiplier is the ``recency_factor`` of the §6.3 effective-confidence
formula — the technical specification states this exact expression under
"Recency factor (content age decay)", and the constants are per-``source_type``
operator configuration rather than part of the standard.

recency_factor() returns a multiplier in [floor, 1.0] based on how many days
have elapsed since a snapshot's content_published_at timestamp.  Sources with
no entry in config.content_age_decay.sources return 1.0 (no decay).

Formula: max(floor, 0.5 ** (age_days / half_life_days))

This module is pure (no I/O beyond the cached config singleton). The
``(half_life_days, floor)`` parameters are resolved by the caller — from the
process-global ``content_age_decay`` config (``recency_factor`` below) or from
the per-observer, lens-composed ``DecayPolicy``
(``particles.operations.query.decay_policy``), which calls
``recency_factor_from_params`` with the resolved parameters. The exponential
math lives here, in exactly one place.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime


def recency_factor_from_params(
    content_published_at: datetime | None,
    half_life_days: float | None,
    floor: float,
    now: datetime | None = None,
) -> float:
    """Pure exponential decay: ``max(floor, 0.5 ** (age_days / half_life_days))``.

    Returns 1.0 (no decay) when ``content_published_at`` is None (publication
    date unknown) or ``half_life_days`` is None (no decay configured for this
    scope). No I/O — the caller resolves the parameters (the global config or a
    lens-composed :class:`DecayPolicy`).

    Args:
        content_published_at: When the source content was originally published.
        half_life_days: Decay half-life in days, or None for "no decay".
        floor: Minimum multiplier; very old content never decays below this.
        now: Override for current time (used in tests).
    """
    if content_published_at is None or half_life_days is None:
        return 1.0

    reference = now if now is not None else datetime.now(UTC)
    if content_published_at.tzinfo is None:
        content_published_at = content_published_at.replace(tzinfo=UTC)

    age_days = (reference - content_published_at).total_seconds() / 86400.0
    if age_days <= 0:
        return 1.0

    factor = math.pow(0.5, age_days / half_life_days)
    return max(floor, factor)


def recency_factor(
    content_published_at: datetime | None,
    source_type: str,
    now: datetime | None = None,
) -> float:
    """Recency multiplier from the **process-global** ``content_age_decay`` config.

    The process-global path, unchanged in behaviour: returns 1.0 when
    ``content_published_at`` is None or ``source_type`` has no entry in
    ``config.content_age_decay.sources``. Per-observer (lens-composed) decay
    goes through :class:`particles.operations.query.decay_policy.DecayPolicy`
    instead; both delegate to :func:`recency_factor_from_params`.

    Args:
        content_published_at: When the source content was originally published.
        source_type: Corpus source type string (e.g. "REDDIT_POST").
        now: Override for current time (used in tests).
    """
    if content_published_at is None:
        return 1.0

    from particles.config import get_config

    decay_cfg = get_config().content_age_decay.sources.get(source_type)
    if decay_cfg is None:
        return 1.0
    return recency_factor_from_params(
        content_published_at, decay_cfg.half_life_days, decay_cfg.floor, now
    )
