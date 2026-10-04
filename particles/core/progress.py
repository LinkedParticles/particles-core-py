# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The shared per-operation progress event.

A long-running operation that knows its denominator reports ``done`` of
``total`` through a callback its caller passes in; the Surface renders it (the
CLI heartbeat's status line, a stderr line). The operation never prints, and
never imports the Surface: the callback is the whole seam, so the Engine and
Client layers stay below the CLI.

The liveness heartbeat is the floor every verb gets without asking; this is the
opt-in ceiling for an operation that can say how far along it is. An operation
that needs more than the four fields subclasses the event (the memory audit's
per-unit extraction outcome, the consolidation cycle's pass-end tally), so one
callback type serves every operation instead of one bespoke type each.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressEvent:
    """``done`` of ``total`` units of ``phase``, with a short ``label``.

    ``phase`` names the stage of the operation that emitted the event; the
    operation documents its own phase names. ``total`` may be 0 when a stage has
    nothing to do.
    """

    phase: str
    done: int
    total: int
    label: str


ProgressCallback = Callable[[ProgressEvent], None]
