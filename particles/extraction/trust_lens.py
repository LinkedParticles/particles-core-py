"""Trust-lens extractor.

Accepts ``source_type == "TRUST_LENS_DEFINITION"`` corpus entries deposited
from local JSON files. The blob is parsed as a
:class:`~particles.core.schema.TrustLensDefinition` and materialised into the
``trust_lenses`` + ``trust_lens_entries`` tables, replacing any
lower-versioned materialisation of the same lens name.

The extractor produces **zero particles** — a lens is policy, not a knowledge
claim. ``ExtractionResult.candidates`` is always empty. Mirrors the
``TaxonomyExtractor`` in every mechanism.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import ApplicabilityClause, Snapshot, TrustLensDefinition
from particles.extraction.general import ExtractionResult

log = logging.getLogger(__name__)

# Inverted lens-persistence coupling (same seam as the taxonomy
# sink). Materialising a parsed ``TrustLensDefinition`` writes the lens tables
# — a graph-aware Engine step. The Engine registers the real
# ``lens_store.materialise_lens`` at import time; when unregistered (pure
# Client, store-free) or when no session is supplied, persistence is skipped.
# The sink returns a human-readable rejection reason (e.g. stale version) or
# ``None`` on success.
LensSink = Callable[[AsyncSession, TrustLensDefinition], Awaitable[str | None]]
_lens_sink: LensSink | None = None


def register_lens_sink(sink: LensSink) -> None:
    """Register the Engine-side lens persistence sink."""
    global _lens_sink
    _lens_sink = sink


SOURCE_TYPE = "TRUST_LENS_DEFINITION"
EXTRACTOR_ID = "trust-lens-extractor"
EXTRACTOR_VERSION = "0.1.0"
# Lenses are operator-curated policy, not knowledge claims, so the trust
# weight is irrelevant — this extractor emits no particles. 1.0 keeps the
# extractor record well-defined for the Extension A registry.
DEFAULT_TRUST_WEIGHT = 1.0

APPLICABILITY = [
    ApplicabilityClause(
        keyword="MUST",
        domain_uri="https://example.org/particles/trust-lens",
        domain_label="trust lens",
        source_types=[SOURCE_TYPE],
    )
]


class TrustLensExtractor:
    EXTRACTOR_ID: str = EXTRACTOR_ID
    EXTRACTOR_VERSION: str = EXTRACTOR_VERSION
    DEFAULT_TRUST_WEIGHT: float = DEFAULT_TRUST_WEIGHT
    APPLICABILITY = APPLICABILITY

    def accepts(self, source_type: str) -> bool:
        return source_type == SOURCE_TYPE

    async def extract(
        self,
        snapshot: Snapshot,
        content: bytes,
        **kwargs: object,
    ) -> ExtractionResult:
        session: AsyncSession | None = kwargs.get("session")  # type: ignore[assignment]
        corpus_entry_id: str | None = kwargs.get("corpus_entry_id")  # type: ignore[assignment]

        try:
            lens = TrustLensDefinition.model_validate_json(content.decode("utf-8"))
        except (UnicodeDecodeError, ValidationError, ValueError) as exc:
            log.warning("TrustLensExtractor: invalid TrustLensDefinition JSON: %s", exc)
            return ExtractionResult(quality_notes=[f"Invalid TrustLensDefinition JSON: {exc}"])

        n_entries = len(lens.statements) + len(lens.url_rules) + len(lens.extractor_weights)
        if session is None or _lens_sink is None:
            # JSON parsed cleanly but we cannot persist: the conformance-
            # validator path (no DB session), or pure-Client / store-free mode
            # where the Engine has not registered a sink.
            return ExtractionResult(
                candidates=[],
                quality_notes=[
                    f"TrustLensDefinition {lens.name!r} v{lens.version} parsed "
                    f"({n_entries} entries); not persisted (no DB session)."
                ],
            )

        lens.corpus_entry_id = corpus_entry_id
        rejection = await _lens_sink(session, lens)
        if rejection is not None:
            log.warning("TrustLensExtractor: lens %r not materialised: %s", lens.name, rejection)
            return ExtractionResult(candidates=[], quality_notes=[rejection])

        log.info(
            "Materialised trust lens %s (%s v%d, %d entries) from corpus entry %s",
            lens.lens_id,
            lens.name,
            lens.version,
            n_entries,
            corpus_entry_id,
        )
        return ExtractionResult(candidates=[])
