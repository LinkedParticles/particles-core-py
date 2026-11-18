# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Status and StatusReason enums plus the §6.6 normative transition validator.

``Status`` and ``StatusReason`` are Core fields (§6.2); the transitions
permitted between them are the normative table of §6.6, reproduced below as
``_ALLOWED``. Both sections are in the technical specification (see this
package's ``__init__``).
"""

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
    # Reserved, and deliberately unwritten. §6.6 assigns this reason to a *Lint*
    # trust pass — a post-extraction re-scan that demotes an ACTIVE particle when
    # the trust of its source changes — and no such pass exists: nothing in
    # ``operations/lint/`` is trust-driven. Every trust demotion this SDK performs
    # happens on the extract-time §6.4 ladder and writes ``LOWER_TRUST_SOURCE``
    # instead (``ingest/pipeline.py``). Kept rather than retired, because the
    # value is published in the normative artifacts and the pass it names is an
    # open pending decision rather than a rejected one.
    # Audited 2026-09-19; if you are re-deriving this, stop and read those rows.
    TRUST_DEMOTED = "TRUST_DEMOTED"
    # Every member below is a normative §6.2 ``status_reason`` value; the
    # transition each one accompanies is a row of the §6.6 table.
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
    DOCUMENT_SUPERSEDED = "DOCUMENT_SUPERSEDED"  # §6.4 rung 1.5 (cap. 2):
    # the conflicting claim's provenance document is (transitively) superseded by
    # the winner's
    SUPERSEDED_BY_UPDATE = "SUPERSEDED_BY_UPDATE"  # §6.4 rung 2.5: an older
    # claim retired by a strictly newer, contradicting claim from the same source
    # lineage — a record that the world changed, not a verdict on the value, so it
    # is deliberately outside the judgment set (a later revert re-mints)
    SUPERSEDED_BY_REANCHOR = "SUPERSEDED_BY_REANCHOR"  #  a claim whose
    # meaning relied on a state an update retired, replaced by a dated restatement
    # anchored to that state. Records that its referent moved, not that it was
    # wrong, so it is outside the judgment set as well
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
        # A candidate the §6.6 ladder decided loses before anything is
        # written is persisted already demoted (§6.6, the born-PROVENANCE_STALE
        # row): the INCONSISTENT quarantine and the two supersession
        # mirrors. Permitted ONLY with a reason in
        # BORN_DEMOTED_REASONS, checked by validate_born_demoted() at the
        # persistence seam (store.insert_particle), since the transition table
        # is keyed on status alone.
        (None, Status.PROVENANCE_STALE),
        (Status.ACTIVE, Status.SUPERSEDED),
        (Status.ACTIVE, Status.RETRACTED),
        (Status.ACTIVE, Status.PROVENANCE_STALE),
        # Every non-DEFER resolution retracts the record (§6.6 wrapper-terminal
        # rule): review, the trust cascade, the held-claim release and the
        # disclosure sweep. There is no INCONSISTENCY → PROVENANCE_STALE edge;
        # the cascade wrote one until 1.192.15 (migration 053).
        (Status.INCONSISTENCY, Status.RETRACTED),
        (Status.INCONSISTENCY, Status.INCONSISTENCY),  # DEFER: re-set same status
        (Status.PROVENANCE_STALE, Status.SUPERSEDED),  # Reindex supersession of stale particle
        (Status.PROVENANCE_STALE, Status.RETRACTED),  # Explicit operator cleanup
        # Unmerge — the only exit from a terminal state, and the only
        # reversible transition in the table (§6.6, the SUPERSEDED → ACTIVE
        # row). Permitted ONLY when the row's *current*
        # status_reason is DUPLICATE_MERGED — enforced at the persistence seam
        # (store.update_particle_status), since this table is keyed on status
        # alone, exactly as the CONFLICT_PENDING birth gate above is. The
        # governing test: a status transition is reversible only if it encoded
        # no judgment. An auto-merge is a hash predicate over identical
        # content — the §6.10 normalized key, creation path 4 —
        # no LLM, no principal's opinion, so it qualifies; every other
        # supersession reason is a judgment and stays terminal, as does
        # RETRACTED.
        (Status.SUPERSEDED, Status.ACTIVE),
    }
)

# The reason a SUPERSEDED row must currently carry for the un-supersede edge
# above to be legal (§6.6). Kept beside the table so the gate and
# the edge are read together.
REVERSIBLE_SUPERSESSION_REASON: StatusReason = StatusReason.DUPLICATE_MERGED

#: The reasons a particle may be born ``PROVENANCE_STALE`` with (§6.6, the
#: birth row). Each names a loser a §6.4 rung decided, from the
#: pair alone and before the write, against a claim the store holds ACTIVE:
#: the INCONSISTENT quarantine, rung 1.5's mirror and
#: rung 2.5's mirror. A reason joins only by a later record that
#: amends this rule.
BORN_DEMOTED_REASONS: frozenset[StatusReason] = frozenset(
    {
        StatusReason.CONFLICT_PENDING,
        StatusReason.DOCUMENT_SUPERSEDED,
        StatusReason.SUPERSEDED_BY_UPDATE,
    }
)


def validate_born_demoted(
    reason: StatusReason | None, *, mirror_reason: StatusReason | None = None
) -> None:
    """Raise ValueError unless a ``PROVENANCE_STALE`` birth with ``reason`` is permitted.

    ``CONFLICT_PENDING`` is the quarantine birth, open to every caller that
    writes one. The two supersession reasons are admitted only when
    ``mirror_reason`` is the same reason: the one the mirror verdict that
    decided the birth writes. Any other caller offering either,
    a restore of a bundle's retired unit among them, is refused, because the
    reason there describes a belief once held.
    """
    if reason is StatusReason.CONFLICT_PENDING:
        return
    if reason in BORN_DEMOTED_REASONS and reason is mirror_reason:
        return
    raise ValueError(
        "A particle may be born PROVENANCE_STALE only as a quarantined conflict"
        " loser (status_reason = CONFLICT_PENDING) or as the loser of a"
        " supersession mirror verdict, with that verdict's reason"
        f" (§6.6); got status_reason = {reason!r}"
    )


def validate_transition(from_status: Status | None, to_status: Status) -> None:
    """Raise ValueError for any transition not in the §6.6 normative table."""
    if (from_status, to_status) not in _ALLOWED:
        raise ValueError(
            f"Invalid status transition: {from_status!r} → {to_status!r}. See §6.6 normative table."
        )
