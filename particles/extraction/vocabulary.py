# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Vocabulary-document extractor.

Accepts ``source_type == "VOCABULARY_DOCUMENT"`` corpus entries: a JSON-LD
vocabulary document (``@type`` ``ppx:VocabularyDocument``), deposited from a
file or published by ``particles vocab``. The extractor hands the bytes to the
Engine's sink once the codec has parsed and validated them.

The extractor produces **zero particles**: a vocabulary is modelling policy,
not a knowledge claim. It mirrors the trust-lens extractor in every
mechanism.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import ApplicabilityClause, Snapshot
from particles.core.vocabulary import VocabularyDocument
from particles.core.vocabulary_jsonld import loads
from particles.extraction.general import ExtractionResult

log = logging.getLogger(__name__)

# Inverted persistence coupling (the trust-lens seam). The Engine
# registers ``vocabulary_store.materialise_document`` at import time; the sink
# returns a human-readable rejection reason, or ``None`` on success.
VocabularySink = Callable[[AsyncSession, VocabularyDocument, str | None], Awaitable[str | None]]
_vocabulary_sink: VocabularySink | None = None


def register_vocabulary_sink(sink: VocabularySink) -> None:
    """Register the Engine-side vocabulary persistence sink."""
    global _vocabulary_sink
    _vocabulary_sink = sink


SOURCE_TYPE = "VOCABULARY_DOCUMENT"
EXTRACTOR_ID = "vocabulary-extractor"
EXTRACTOR_VERSION = "0.1.0"
# A vocabulary is reviewed modelling policy and emits no particles, so the
# trust weight is irrelevant; 1.0 keeps the Extension A record well-defined.
DEFAULT_TRUST_WEIGHT = 1.0

APPLICABILITY = [
    ApplicabilityClause(
        keyword="MUST",
        domain_uri="https://example.org/particles/vocabulary",
        domain_label="vocabulary",
        source_types=[SOURCE_TYPE],
    )
]


class VocabularyExtractor:
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
            doc = loads(content)
        except ValueError as exc:
            log.warning("VocabularyExtractor: invalid vocabulary document: %s", exc)
            return ExtractionResult(quality_notes=[f"Invalid vocabulary document: {exc}"])

        if session is None or _vocabulary_sink is None:
            return ExtractionResult(
                quality_notes=[
                    f"Vocabulary {doc.name!r} v{doc.version} parsed ({len(doc.terms)} terms); "
                    "not persisted (no DB session)."
                ]
            )

        rejection = await _vocabulary_sink(session, doc, corpus_entry_id)
        if rejection is not None:
            log.warning("VocabularyExtractor: %r not materialised: %s", doc.name, rejection)
            return ExtractionResult(quality_notes=[rejection])
        log.info(
            "Materialised vocabulary %s v%d (%d terms) from corpus entry %s",
            doc.name,
            doc.version,
            len(doc.terms),
            corpus_entry_id,
        )
        return ExtractionResult()
