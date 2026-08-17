"""GitHub extractors and importer (importer naming).

This package handles three content surfaces under a shared ``github:{login}``
author identity:

- ``GITHUB_REPO``  — files in a repository, fetched via the contents API
- ``GITHUB_GIST``  — gists, fetched via the gists API
- ``GITHUB_PAGES`` — posts hosted at ``{username}.github.io``, fetched via HTTP

Each surface has its own extractor class. Cross-cutting code (auth, HTTP
retry, URL parsing) lives in ``_shared``; per-extractor code lives in
``repo`` / ``gist`` / ``pages``. The importer (``GitHubImporter`` and its
private fetch helpers) moved to the Engine layer at
``particles.ingest.importers.github``.

The constants and classes re-exported here are the stable public surface —
``SOURCE_TYPE_*`` strings in particular are stored in the database and must
not change. See the `particles.extraction.AGENTS.md` package-layout note for
why this directory looks the way it does.
"""

from __future__ import annotations

from particles.extraction.github._shared import (
    APPLICABILITY_GIST,
    APPLICABILITY_PAGES,
    APPLICABILITY_REPO,
    DEFAULT_TRUST_WEIGHT_GIST,
    DEFAULT_TRUST_WEIGHT_PAGES,
    DEFAULT_TRUST_WEIGHT_REPO,
    EXTRACTOR_ID_GIST,
    EXTRACTOR_ID_PAGES,
    EXTRACTOR_ID_REPO,
    EXTRACTOR_VERSION_GIST,
    EXTRACTOR_VERSION_PAGES,
    EXTRACTOR_VERSION_REPO,
    SOURCE_TYPE_GIST,
    SOURCE_TYPE_PAGES,
    SOURCE_TYPE_REPO,
    _api_headers,
    _date_from_path,
    _normalize_raw_url,
    _parse_gist_url,
    _parse_iso_utc,
    _parse_pages_url,
    _parse_repo_url,
)
from particles.extraction.github.gist import GitHubGistExtractor
from particles.extraction.github.pages import GitHubPagesExtractor
from particles.extraction.github.repo import GitHubRepoExtractor

__all__ = [
    "APPLICABILITY_GIST",
    "APPLICABILITY_PAGES",
    "APPLICABILITY_REPO",
    "DEFAULT_TRUST_WEIGHT_GIST",
    "DEFAULT_TRUST_WEIGHT_PAGES",
    "DEFAULT_TRUST_WEIGHT_REPO",
    "EXTRACTOR_ID_GIST",
    "EXTRACTOR_ID_PAGES",
    "EXTRACTOR_ID_REPO",
    "EXTRACTOR_VERSION_GIST",
    "EXTRACTOR_VERSION_PAGES",
    "EXTRACTOR_VERSION_REPO",
    "GitHubGistExtractor",
    "GitHubPagesExtractor",
    "GitHubRepoExtractor",
    "SOURCE_TYPE_GIST",
    "SOURCE_TYPE_PAGES",
    "SOURCE_TYPE_REPO",
    # Test-only re-exports — small URL/auth helpers asserted on directly
    # by tests/test_github_extractor.py.
    "_api_headers",
    "_date_from_path",
    "_normalize_raw_url",
    "_parse_gist_url",
    "_parse_iso_utc",
    "_parse_pages_url",
    "_parse_repo_url",
]
