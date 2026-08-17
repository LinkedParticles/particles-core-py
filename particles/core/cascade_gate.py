"""Pure §15.1 cascade-gating decisions (Extension B).

The trust cascade auto-resolves open INCONSISTENCY particles when a new
:class:`~particles.core.schema.SourceTrustStatement` lands. Two deterministic
gates bound it, and the Conformance Profile names both as
L2-normative:

  - the **policy gate** — ``OPERATOR_DIRECT`` / ``REGISTRY_ENDORSED`` always
    cascade; ``REVIEWER_DERIVED`` only once N distinct reviewer confirmations
    exist for the same ``source_ref`` in the domain;
  - the **per-run cap** — at most ``cascade_max_per_run`` INCONSISTENCY
    particles are resolved in one cascade, the rest left for manual review.

Both are pure functions of already-resolved inputs, so they live here rather
than in :mod:`particles.operations.cascade`, which owns the I/O half (the
confirmation count query, the status writes). Same split as
:mod:`particles.core.conflict_resolution`: decision here, effect there. The runner is Client-layer and recomputes its published vectors through
these functions.
"""

from __future__ import annotations

from particles.core.schema import PolicyProvenance


def cascade_gate_passes(
    policy_provenance: PolicyProvenance,
    reviewer_confirmations: int = 0,
    min_reviewer_confirmations: int = 3,
) -> bool:
    """Whether a statement's provenance permits an automatic cascade (§15.1).

    Args:
        policy_provenance: How the SourceTrustStatement came to exist.
        reviewer_confirmations: Count of distinct reviewer-derived statements
            for the same ``(domain, source_ref)``. Only consulted on the
            ``REVIEWER_DERIVED`` branch.
        min_reviewer_confirmations: The N threshold that branch must reach.

    Returns:
        True when the cascade may run unattended.
    """
    if policy_provenance in (PolicyProvenance.OPERATOR_DIRECT, PolicyProvenance.REGISTRY_ENDORSED):
        return True
    if policy_provenance == PolicyProvenance.REVIEWER_DERIVED:
        return reviewer_confirmations >= min_reviewer_confirmations
    return False


def apply_cascade_cap(candidate_count: int, max_per_run: int) -> tuple[int, bool]:
    """Apply the §15.1 per-run cap to a batch of cascade candidates.

    Args:
        candidate_count: How many open INCONSISTENCY particles matched.
        max_per_run: The ``trust.cascade_max_per_run`` safety cap.

    Returns:
        ``(processed, capped)`` — how many the run may resolve, and whether
        the cap truncated the batch (the remainder is left for manual review).
    """
    processed = min(candidate_count, max_per_run)
    return processed, candidate_count > max_per_run
