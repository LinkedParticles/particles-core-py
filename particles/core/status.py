"""Status and StatusReason enums plus the §6.6 normative transition validator."""

from __future__ import annotations

from enum import StrEnum


class Status(StrEnum):
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    RETRACTED = "RETRACTED"
    PROVENANCE_STALE = "PROVENANCE_STALE"
    INCONSISTENCY = "INCONSISTENCY"


class StatusReason(StrEnum):
    RETRACTED_DEPENDENCY = "RETRACTED_DEPENDENCY"
    CORPUS_ENTRY_MISSING = "CORPUS_ENTRY_MISSING"
    TRUST_DEMOTED = "TRUST_DEMOTED"
    LOWER_TRUST_SOURCE = "LOWER_TRUST_SOURCE"  # auto-resolved by trust score
    SUPERSEDED_BY_REINDEX = "SUPERSEDED_BY_REINDEX"
    VALIDITY_EXPIRED = "VALIDITY_EXPIRED"
    CONFLICT_RESOLVED = "CONFLICT_RESOLVED"
    CONFLICT_PENDING = "CONFLICT_PENDING"  # quarantined conflict loser awaiting review
    EXPLICIT_RETRACTION = "EXPLICIT_RETRACTION"
    SOURCE_RETRACTED = "SOURCE_RETRACTED"  # bulk corpus retract
    EXPLICIT_SUPERSESSION = (
        "EXPLICIT_SUPERSESSION"  # deliberate revision by the asserting principal
    )
    DOCUMENT_SUPERSEDED = "DOCUMENT_SUPERSEDED"  # cap. 2, §6.6 rung 1.5 — the
    # conflicting claim's provenance document is (transitively) superseded by the winner's
    DUPLICATE_MERGED = "DUPLICATE_MERGED"  # redundant byte-identical copy folded
    # into its group's survivor by exact-duplicate auto-merge. Deliberately distinct from
    # EXPLICIT_SUPERSESSION so a revert can select precisely auto-merge's own writes.


# §6.6 normative transition table.
# None as from_status represents initial particle creation.
# PROVENANCE_STALE → ACTIVE is NOT here: Reindex creates a NEW ACTIVE particle;
# the stale one stays stale.
_ALLOWED: frozenset[tuple[Status | None, Status]] = frozenset(
    {
        (None, Status.ACTIVE),  # Extract: new particle
        (None, Status.INCONSISTENCY),  # Extract/Lint: new INCONSISTENCY particle
        # the losing candidate of an INCONSISTENT verdict is persisted
        # quarantined. Permitted ONLY with status_reason = CONFLICT_PENDING —
        # enforced at the persistence seam (store.insert_particle), since the
        # transition table is keyed on status alone.
        (None, Status.PROVENANCE_STALE),
        (Status.ACTIVE, Status.SUPERSEDED),
        (Status.ACTIVE, Status.RETRACTED),
        (Status.ACTIVE, Status.PROVENANCE_STALE),
        (Status.INCONSISTENCY, Status.PROVENANCE_STALE),  # PREFER A/B: loser demoted
        (Status.INCONSISTENCY, Status.RETRACTED),  # BOTH VALID: INCONSISTENCY particle retracted
        (Status.INCONSISTENCY, Status.INCONSISTENCY),  # DEFER: re-set same status
        (Status.PROVENANCE_STALE, Status.SUPERSEDED),  # Reindex supersession of stale particle
        (Status.PROVENANCE_STALE, Status.RETRACTED),  # Explicit operator cleanup
        # the only exit from a terminal state, and the only reversible
        # transition in the table. Permitted ONLY when the row's *current*
        # status_reason is DUPLICATE_MERGED — enforced at the persistence seam
        # (store.update_particle_status), since this table is keyed on status
        # alone, exactly as CONFLICT_PENDING birth gate is. The
        # governing test: a status transition is reversible only if it encoded
        # no judgment. An auto-merge is a hash predicate over
        # byte-identical content — no LLM, no principal's opinion — so it
        # qualifies; every other supersession reason is a judgment and stays
        # terminal, as does RETRACTED.
        (Status.SUPERSEDED, Status.ACTIVE),
    }
)

# the reason a SUPERSEDED row must currently carry for the
# un-supersede edge above to be legal. Kept beside the table so the gate and
# the edge are read together.
REVERSIBLE_SUPERSESSION_REASON: StatusReason = StatusReason.DUPLICATE_MERGED


def validate_transition(from_status: Status | None, to_status: Status) -> None:
    """Raise ValueError for any transition not in the §6.6 normative table."""
    if (from_status, to_status) not in _ALLOWED:
        raise ValueError(
            f"Invalid status transition: {from_status!r} → {to_status!r}. See §6.6 normative table."
        )
