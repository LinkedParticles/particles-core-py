# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Re-anchoring a claim that relied on a superseded state: the pure decisions.

When an update retires a state claim R ("the user lives in Delhi, in Lajpat
Nagar"), a claim cut from the same passage may have relied on R being current
("Sandeep's Curry House is a ten-minute walk from the user's flat"). Its
referent moved with the update, so it reads as current and is false. The pass
in :mod:`particles.operations.reanchor` asks an LLM which such claims depended
on R, has a second reading confirm each verdict and the restatement it wrote,
and replaces the claim with a dated restatement anchored to R.

This module holds the decisions over plain values, with no I/O: the reply
parsers, the outcome of one candidate, the restatement's record, and the cursor
the nightly pass resumes from (D2).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from particles.core.schema import (
    CanonicalForm,
    ContributorRef,
    Particle,
    ProvenanceRef,
    ProvenanceRefType,
)
from particles.core.status import Status

#: The contributor identity a restatement carries, beside the source's own
#: attribution. Particle-level contributors are not part of the rung 2.5
#: lineage key (which reads the corpus entry), so the restatement stays in its
#: source's lineage.
REANCHOR_ACTOR = "consolidation:reanchor"


class DependencyVerdict(StrEnum):
    """The probe's verdict on one candidate claim."""

    HOLDS = "HOLDS"
    DEPENDS = "DEPENDS"


class Outcome(StrEnum):
    """What the pass did with one candidate the probe judged dependent."""

    #: A restatement was written ACTIVE and the original retired.
    RESTATED = "restated"
    #: An ACTIVE claim already states the restatement; the original is retired.
    MATCHED_EXISTING = "matched_existing"
    #: Nothing written; the original stays ACTIVE and is shown to the owner.
    UNRESTATED = "unrestated"
    #: The second reading found the original did not depend on the state.
    DEPENDENCY_REJECTED = "dependency_rejected"


@dataclass(frozen=True)
class ProbeVerdict:
    """One candidate's line of the probe reply."""

    candidate_id: str
    verdict: DependencyVerdict
    restatement: str | None
    reason: str


@dataclass(frozen=True)
class Reading:
    """The second reading of one restatement: two questions, both must hold."""

    depended: bool
    faithful: bool
    reason: str


@dataclass(frozen=True)
class Cursor:
    """Where the nightly pass stopped: the last trigger it examined."""

    retired_at: datetime
    particle_id: str

    def payload(self) -> dict[str, str]:
        return {"retired_at": self.retired_at.isoformat(), "particle_id": self.particle_id}

    @classmethod
    def from_payload(cls, raw: object) -> Cursor | None:
        if not isinstance(raw, dict):
            return None
        at, pid = raw.get("retired_at"), raw.get("particle_id")
        if not isinstance(at, str) or not isinstance(pid, str) or not pid:
            return None
        try:
            when = datetime.fromisoformat(at)
        except ValueError:
            return None
        return cls(retired_at=when, particle_id=pid)


def parse_json_object(reply: str | None) -> dict[str, Any] | None:
    """Tolerantly isolate and parse one JSON object from an LLM reply."""
    if not reply:
        return None
    text = reply.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text
        text = text.removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        return None
    try:
        raw: Any = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return None
    return raw if isinstance(raw, dict) else None


def parse_probe_reply(reply: str | None, labels: dict[str, str]) -> list[ProbeVerdict] | None:
    """The probe's per-claim verdicts, keyed back to particle ids.

    ``labels`` maps the short label each candidate was shown under (``c1``,
    ``c2``, …) to its particle id. A line naming an unknown label, or carrying
    an unknown verdict, is dropped: that candidate simply gets no verdict this
    run. ``None`` means the reply was unusable as a whole.
    """
    data = parse_json_object(reply)
    if data is None or not isinstance(data.get("claims"), list):
        return None
    out: list[ProbeVerdict] = []
    seen: set[str] = set()
    for item in data["claims"]:
        if not isinstance(item, dict):
            continue
        pid = labels.get(str(item.get("claim", "")).strip())
        if pid is None or pid in seen:
            continue
        try:
            verdict = DependencyVerdict(str(item.get("verdict", "")).strip().upper())
        except ValueError:
            continue
        restatement = str(item.get("restatement") or "").strip() or None
        seen.add(pid)
        out.append(
            ProbeVerdict(
                candidate_id=pid,
                verdict=verdict,
                restatement=restatement if verdict is DependencyVerdict.DEPENDS else None,
                reason=str(item.get("reason") or "").strip(),
            )
        )
    return out


def parse_reading_reply(reply: str | None) -> Reading | None:
    """The second reading's two answers, or ``None`` when the reply is unusable."""
    data = parse_json_object(reply)
    if data is None:
        return None
    depended, faithful = data.get("depended"), data.get("faithful")
    if not isinstance(depended, bool) or not isinstance(faithful, bool):
        return None
    return Reading(depended=depended, faithful=faithful, reason=str(data.get("reason") or ""))


def decide_after_reading(reading: Reading | None) -> Outcome | None:
    """The outcome the second reading settles, or ``None`` to go on to the writes.

    An unusable reading keeps the original and shows it to the owner: a check
    that could not run is not a passed check. A NO on the dependency means the
    probe was wrong and the claim holds, so nothing is written and no card is
    raised.
    """
    if reading is None:
        return Outcome.UNRESTATED
    if not reading.depended:
        return Outcome.DEPENDENCY_REJECTED
    if not reading.faithful:
        return Outcome.UNRESTATED
    return None


def decide_write(*, duplicate_of: str | None, conflict: bool) -> Outcome:
    """What a restatement that passed its reading does to the store.

    A restatement never changes another belief: any confirmed contradiction
    with a standing claim (or a contradiction check that could not run, which
    the caller reports as ``conflict``) writes nothing. The rule is stricter
    than the ladder's own: a truth-apt claim whose contradiction is confirmed
    reaches a rung that either quarantines it or retires something, so no
    confirmed contradiction can leave both standing.
    """
    if duplicate_of is not None:
        return Outcome.MATCHED_EXISTING
    if conflict:
        return Outcome.UNRESTATED
    return Outcome.RESTATED


def is_before_or_at(candidate_date: datetime | None, retired_date: datetime | None) -> bool:
    """Candidacy condition 5: the candidate was not observed after the retired claim.

    Both dates must be known. A claim a later source restated may have been
    restated after the change, so it is left alone.
    """
    if candidate_date is None or retired_date is None:
        return False
    return _aware(candidate_date) <= _aware(retired_date)


def anchor_date(value: datetime | None) -> str | None:
    """The date a restatement anchors its circumstance to, as ``YYYY-MM-DD``."""
    return _aware(value).date().isoformat() if value is not None else None


def build_restatement(
    original: Particle,
    retired: Particle,
    content: str,
    *,
    provider_model: str | None,
    now: datetime,
) -> Particle:
    """The restatement's record, not yet written.

    Its evidence is the original's source, so it keeps the original's SOURCE
    refs, asserter, extractor and lineage; it states the retired claim's
    content, so it carries the retired claim as a ``PARTICLE`` premise ref in
    the blessed convention. Its confidence is the lower of the
    two stored values with that claim's calibration fields: a conjunction is
    no more credible than its weaker part. ``calibration_source`` is never
    ``DERIVED``: the read path discounts a derived claim whose premise is not
    ACTIVE, and the retired claim never is.
    """
    weaker = retired if retired.confidence.value < original.confidence.value else original
    sources = [r for r in original.provenance if r.type is ProvenanceRefType.SOURCE]
    premise = ProvenanceRef(
        type=ProvenanceRefType.PARTICLE,
        corpus_entry_id=retired.id,
        snapshot_id=retired.id,
    )
    subjects = list(dict.fromkeys([*original.subject_ids, *retired.subject_ids]))
    contributors = [
        *(original.contributors or []),
        ContributorRef(id=REANCHOR_ACTOR, role="agent", at=now),
    ]
    return original.model_copy(
        update={
            "id": str(uuid.uuid4()),
            "content": content,
            "confidence": weaker.confidence,
            "provenance": [*sources, premise],
            "asserted_at": now,
            "status": Status.ACTIVE,
            "status_reason": None,
            "supersedes": original.id,
            "subject_ids": subjects,
            "contributors": contributors,
            "extraction_provider_model": provider_model,
            "structured_claim": None,
            "canonical_form": CanonicalForm.PROSE,
            "valid_until": None,
        }
    )


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
