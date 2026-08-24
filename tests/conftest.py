# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for the store-free Client distribution.

Every fixture lives in ``tests/_client_fixtures.py``, which the development
upstream and this repo share verbatim — one body, so the two suites cannot
drift. This file only re-exports it; add Client-layer fixtures there, not here.
"""

from __future__ import annotations

# pytest picks the `pytest_configure` hook and the autouse fixtures up off this
# module's namespace, so the import *is* the wiring. Keep the list exhaustive —
# a name dropped here silently disables that fixture for the whole suite.
from tests._client_fixtures import (  # noqa: F401
    no_embedding_model,
    no_env_leak,
    pytest_configure,
    reset_client_state,
    restore_logger_levels,
)
