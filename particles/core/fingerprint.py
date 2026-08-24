# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The §16.1 context-fingerprint digest — pure, store-free.

Spec §16.1 states the procedure in three steps:

  1. identify all ``ACTIVE`` particles in the asserting agent's store,
  2. sort their UUIDs lexicographically,
  3. SHA-256 the concatenated sorted UUIDs (no delimiter).

Step 1 is a store query; steps 2–3 are the deterministic digest this module
owns. Splitting them here is what lets both the store
(:func:`particles.store.particle_store.compute_context_fingerprint`, which
supplies step 1 in SQL) and the Client-layer Conformance Profile runner
(the L2 conformance runner, which supplies it from a test vector)
run the *same* code —
the procedure "MUST be followed exactly to ensure cross-agent fingerprint
compatibility", so it must exist once.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


def context_fingerprint(active_particle_ids: Iterable[str]) -> str:
    """Return the §16.1 Merkle-root digest over an ACTIVE-particle baseline.

    Args:
        active_particle_ids: The UUIDs of every ACTIVE particle in the
            baseline, in any order — this function performs the normative
            lexicographic sort itself, so the digest is order-independent.

    Returns:
        A 64-character lowercase hex SHA-256 digest. An empty baseline
        returns the SHA-256 of the empty string, the canonical value for a
        fresh store.
    """
    return hashlib.sha256("".join(sorted(active_particle_ids)).encode()).hexdigest()
