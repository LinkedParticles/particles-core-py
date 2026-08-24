# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Taxonomy extractor (Phase A).

Accepts ``source_type == "TAXONOMY_DEFINITION"`` corpus entries deposited
from local JSON files. The blob is parsed as a :class:`TaxonomyDefinition`,
its parent-path consistency is validated by the Pydantic model, and the
rows are materialised into the ``taxonomies`` + ``tag_nodes`` tables.

The extractor produces **zero particles** — taxonomies are configuration,
not knowledge claims. ``ExtractionResult.candidates`` is always empty.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import ApplicabilityClause, Snapshot, TaxonomyDefinition
from particles.extraction.general import ExtractionResult

log = logging.getLogger(__name__)

# Inverted taxonomy-persistence coupling. Persisting a
# parsed ``TaxonomyDefinition`` is the one graph-aware step of taxonomy
# extraction — it writes the ``taxonomies`` / ``tag_nodes`` tables. To keep this
# module on the Client layer, the store write is *injected*, not imported: the
# Engine registers the real ``taxonomy_store.insert_taxonomy`` at import time via
# :func:`register_taxonomy_sink`. When unregistered (pure Client, store-free) or
# when no session is supplied, persistence is skipped — identical to the
# conformance-validator path. Mirrors ``incremental``'s carry-forward hook.
TaxonomySink = Callable[[AsyncSession, TaxonomyDefinition], Awaitable[None]]
_taxonomy_sink: TaxonomySink | None = None


def register_taxonomy_sink(sink: TaxonomySink) -> None:
    """Register the Engine-side taxonomy persistence sink."""
    global _taxonomy_sink
    _taxonomy_sink = sink


SOURCE_TYPE = "TAXONOMY_DEFINITION"
EXTRACTOR_ID = "taxonomy-extractor"
EXTRACTOR_VERSION = "0.1.0"
# Taxonomies are operator-curated config, not knowledge claims, so the
# trust weight is irrelevant — this extractor emits no particles. Setting
# 1.0 keeps the extractor record well-defined for the Extension A registry.
DEFAULT_TRUST_WEIGHT = 1.0

APPLICABILITY = [
    ApplicabilityClause(
        keyword="MUST",
        domain_uri="https://example.org/particles/taxonomy",
        domain_label="taxonomy",
        source_types=[SOURCE_TYPE],
    )
]


class TaxonomyExtractor:
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
            td = TaxonomyDefinition.model_validate_json(content.decode("utf-8"))
        except (UnicodeDecodeError, ValidationError, ValueError) as exc:
            log.warning("TaxonomyExtractor: invalid TaxonomyDefinition JSON: %s", exc)
            return ExtractionResult(quality_notes=[f"Invalid TaxonomyDefinition JSON: {exc}"])

        if session is None or _taxonomy_sink is None:
            # JSON parsed cleanly but we cannot persist. Two cases: the
            # conformance-validator path (no DB session), or pure-Client /
            # store-free mode where the Engine has not registered a sink. In
            # both, the fixture's real value is "the artefact parses", not
            # "rows were materialised".
            return ExtractionResult(
                candidates=[],
                quality_notes=[
                    f"TaxonomyDefinition {td.name!r} v{td.version} parsed "
                    f"({len(td.tags)} tags); not persisted (no DB session)."
                ],
            )

        td.corpus_entry_id = corpus_entry_id
        await _taxonomy_sink(session, td)
        log.info(
            "Materialised taxonomy %s (%s v%s, %d tags) from corpus entry %s",
            td.taxonomy_id,
            td.name,
            td.version,
            len(td.tags),
            corpus_entry_id,
        )
        return ExtractionResult(candidates=[])
