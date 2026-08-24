# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Normative read-time scoring-formula surface.

The pure, no-I/O modules that decide **read-time** projection ranking:
:mod:`~particles.core.scoring.confidence` (two-quantity confidence + noisy-OR
merge, §6.3 / §6.9), :mod:`~particles.core.scoring.decay` (content-age recency
decay, §6.3), :mod:`~particles.core.scoring.utility` (usefulness
reinforcement + additive rank-lift, §6.4), and
:mod:`~particles.core.scoring.relevance` (owner-aboutness rank-lift; the one
axis the specification does not define). The last three are the
three read-time axes — **truth**, **use**, **aboutness** — composed as
separate addends on one ordering score:

    rank_score = effective_confidence + λ·ln(1 + R) + ω·A

Only the first term is a confidence; the sum is an ordering key and may exceed
``1.0``. The sections cited above are in the technical specification (see
``particles.core``); the quantities several of these functions must reproduce
bit-for-bit are pinned as test vectors in the Conformance Profile §4. These are
Client-layer math: the Engine composition that reads the store and
calls them lives in ``operations/query/effective_confidence.py``.

Symbols are re-exported here so ``from particles.core.scoring import <symbol>``
works alongside the per-module path.
"""

from __future__ import annotations

from particles.core.scoring.confidence import (
    CalibrationSource,
    compute_effective_confidence,
    derive_abstraction_confidence,
    merge_co_evidential_confidence,
)
from particles.core.scoring.decay import recency_factor, recency_factor_from_params
from particles.core.scoring.relevance import owner_rank_bonus
from particles.core.scoring.utility import reinforcement_score, utility_rank_bonus

__all__ = [
    "CalibrationSource",
    "compute_effective_confidence",
    "derive_abstraction_confidence",
    "merge_co_evidential_confidence",
    "owner_rank_bonus",
    "recency_factor",
    "recency_factor_from_params",
    "reinforcement_score",
    "utility_rank_bonus",
]
