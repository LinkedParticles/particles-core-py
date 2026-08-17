"""GitHub repo extractor.

Reads ``GITHUB_REPO`` blobs — single files fetched via the contents API and
stored verbatim — and asks the LLM to extract claims, injecting the
``github:{owner}`` author and the repo name as additional subjects.
"""

from __future__ import annotations

from typing import cast

from particles.core.schema import Snapshot
from particles.extraction.general import ExtractionResult
from particles.extraction.github._shared import (
    APPLICABILITY_REPO,
    DEFAULT_TRUST_WEIGHT_REPO,
    EXTRACTOR_ID_REPO,
    EXTRACTOR_VERSION_REPO,
    SOURCE_TYPE_REPO,
    _llm_extract_with_subjects,
    _normalize_raw_url,
    _parse_repo_url,
)


class GitHubRepoExtractor:
    """Extract claim-granularity particles from a stored GITHUB_REPO file blob."""

    EXTRACTOR_ID: str = EXTRACTOR_ID_REPO
    EXTRACTOR_VERSION: str = EXTRACTOR_VERSION_REPO
    DEFAULT_TRUST_WEIGHT: float = DEFAULT_TRUST_WEIGHT_REPO
    APPLICABILITY = APPLICABILITY_REPO

    def accepts(self, source_type: str) -> bool:
        return source_type == SOURCE_TYPE_REPO

    async def extract(
        self,
        snapshot: Snapshot,
        content: bytes,
        **kwargs: object,
    ) -> ExtractionResult:
        entry_uri_r = cast("str | None", kwargs.get("entry_uri_r"))
        owner, repo, path = _resolve_repo_url(entry_uri_r)
        text = content.decode("utf-8", errors="replace")
        if not text.strip():
            return ExtractionResult(quality_notes=["Empty content"])
        header_path = path or "README"
        repo_label = f"{owner}/{repo}" if owner and repo else "(unknown repo)"
        header = f"# GitHub: {repo_label} — {header_path}\n\n"
        extras = [repo] if repo else []
        if owner:
            extras.append(f"github:{owner}")
        return await _llm_extract_with_subjects(header + text, extras)


def _resolve_repo_url(uri_r: str | None) -> tuple[str, str, str | None]:
    """Resolve (owner, repo, path) from the corpus entry's URI-R.

    The Engine pipeline passes the entry's ``uri_r`` via the ``entry_uri_r``
    kwarg, so this Client-layer extractor parses the
    repo coordinates without reading the store — the prior implementation
    queried ``corpus.store`` for the snapshot's entry, a Client→Engine edge.
    Pure-Client callers that omit the kwarg get blanks.
    """
    if not uri_r:
        return ("", "", None)
    parsed = _parse_repo_url(_normalize_raw_url(uri_r))
    if parsed is None:
        return ("", "", None)
    owner, repo, _branch, path = parsed
    return owner, repo, path
