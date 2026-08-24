# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""GitHub Pages extractor.

Reads ``GITHUB_PAGES`` blobs — posts hosted at ``{username}.github.io``,
fetched as raw HTML — strips the HTML to plain text, and asks the LLM to
extract claims. Long pages are chunked.
"""

from __future__ import annotations

import logging

from particles.config import get_config
from particles.core.schema import Snapshot
from particles.extraction.general import (
    CandidateParticle,
    ExtractionResult,
    PageStat,
    _call_llm,
    _split_into_chunks,
)
from particles.extraction.github._shared import (
    APPLICABILITY_PAGES,
    DEFAULT_TRUST_WEIGHT_PAGES,
    EXTRACTOR_ID_PAGES,
    EXTRACTOR_VERSION_PAGES,
    SOURCE_TYPE_PAGES,
    _inject_subjects,
    _llm_extract_with_subjects,
)

log = logging.getLogger(__name__)


class GitHubPagesExtractor:
    """Extract claim-granularity particles from a stored GITHUB_PAGES HTML blob."""

    EXTRACTOR_ID: str = EXTRACTOR_ID_PAGES
    EXTRACTOR_VERSION: str = EXTRACTOR_VERSION_PAGES
    DEFAULT_TRUST_WEIGHT: float = DEFAULT_TRUST_WEIGHT_PAGES
    APPLICABILITY = APPLICABILITY_PAGES

    def accepts(self, source_type: str) -> bool:
        return source_type == SOURCE_TYPE_PAGES

    async def extract(
        self,
        snapshot: Snapshot,
        content: bytes,
        **kwargs: object,
    ) -> ExtractionResult:
        import html2text as _h2t

        html_text = content.decode("utf-8", errors="replace")
        h = _h2t.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        h.body_width = 0
        text = h.handle(html_text)
        if not text.strip():
            return ExtractionResult(quality_notes=["Empty content"])

        extras = [snapshot.author_id] if snapshot.author_id else []
        cfg = get_config().extraction
        if len(text) > cfg.html_chunk_size:
            return await _chunked_pages_extract(text, extras)
        return await _llm_extract_with_subjects(text, extras)


async def _chunked_pages_extract(text: str, extras: list[str]) -> ExtractionResult:
    # Deliberate carry-forward holdout: Pages
    # chunks line-by-line *with overlap*, so chunk boundaries shift on almost
    # any edit and hashing them would carry forward next to nothing. Unlike the
    # six extractors that route through ``extract_with_carry_forward`` over
    # stable (paragraph / structural) chunks, Pages keeps a plain per-chunk
    # loop. Adopting carry-forward here would first require switching to a
    # stable chunker — an extraction-output change with its own version bump,
    # out of scope for the R1.9 reuse pass.
    cfg = get_config().extraction
    chunks = _split_into_chunks(text, cfg.html_chunk_size, cfg.html_chunk_overlap_lines)
    total = len(chunks)
    all_candidates: list[CandidateParticle] = []
    all_notes: list[str] = []
    page_stats: list[PageStat] = []
    transient_errors = 0
    for i, chunk_text in enumerate(chunks, start=1):
        candidates, notes, transient = await _call_llm(chunk_text)
        count = len(candidates)
        if transient:
            transient_errors += 1
        if count == 0:
            all_notes.append(f"ZERO_PAGE_YIELD: chunk {i} produced 0 particles")
        if notes:
            all_notes.extend(f"Chunk {i}: {n}" for n in notes)
        all_candidates.extend(candidates)
        page_stats.append(PageStat(page_number=i, candidate_count=count))
        log.info("GitHub Pages chunk %d/%d: %d particles", i, total, count)
    for c in all_candidates:
        _inject_subjects(c, extras)
    return ExtractionResult(
        candidates=all_candidates,
        quality_notes=all_notes,
        page_stats=page_stats,
        transient_error_count=transient_errors,
    )
