"""Shared helpers for the GitHub extractors and importer.

Cross-cutting code lives here:

- Identity constants for all three source types (``GITHUB_REPO`` / ``_GIST`` /
  ``_PAGES``) — strings stored in the DB; **never** rename.
- Per-source applicability clauses (Extension A).
- URL regex patterns + small parser helpers.
- HTTP auth / retry helpers around ``particles.http.get_with_retry`` and the
  ``GITHUB_API_KEY`` secret read.
- ``_inject_subjects`` / ``_llm_extract_with_subjects`` — small post-LLM
  helpers reused by the repo and pages extractors.

Per-extractor code (repo, gist, pages) and the importer live in sibling
modules.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from particles.core.schema import ApplicabilityClause
from particles.extraction.general import CandidateParticle, ExtractionResult, _call_llm
from particles.http import TransientHttpError, get_with_retry
from particles.secrets import get_github_api_key_optional

# ---------------------------------------------------------------------------
# Identity constants — one set per source type (multi-extractor module)
# ---------------------------------------------------------------------------

SOURCE_TYPE_REPO = "GITHUB_REPO"
SOURCE_TYPE_GIST = "GITHUB_GIST"
SOURCE_TYPE_PAGES = "GITHUB_PAGES"

EXTRACTOR_ID_REPO = "github-repo-extractor"
EXTRACTOR_ID_GIST = "github-gist-extractor"
EXTRACTOR_ID_PAGES = "github-pages-extractor"
EXTRACTOR_VERSION_REPO = "0.1.0"
EXTRACTOR_VERSION_GIST = "0.6.1"
# 0.2.0: include gist comments + commenter subjects (gh/ prefix, LLM-emitted)
# 0.3.0: deterministic auxiliary attribution by content-token overlap
# 0.4.0: synthesize fallback particle for substantive commenters not yet covered
# 0.5.0: adaptive chunked LLM extraction for large comment threads
# 0.5.1: shrink chunk size 20K→10K to avoid the 8K output-token cap
# 0.5.2: normalize gh/{login} → @{login} in particle content for readability
# 0.5.3: synthesis fallback off by default; chunked extraction suffices
# 0.6.0: chunk-hash carry-forward via extract_with_carry_forward;
#        Link-header pagination of comments; GIST_COMMENT_LIMIT_HIT note
# 0.6.1: _rewrite_gh_subjects now also rewrites bare logins (LLM occasionally
#        strips the `gh/` prefix it was shown) → `github:{login}` so the
#        Obsidian exporter routes them under `github.com/{login}.md`.
EXTRACTOR_VERSION_PAGES = "0.1.0"

DEFAULT_TRUST_WEIGHT_REPO = 0.75
DEFAULT_TRUST_WEIGHT_GIST = 0.65
DEFAULT_TRUST_WEIGHT_PAGES = 0.70

_GH_DOMAIN_URI = "http://schema.org/SoftwareSourceCode"
_GH_DOMAIN_LABEL = "software"

GITHUB_API_BASE = "https://api.github.com"


# ---------------------------------------------------------------------------
# URL patterns
# ---------------------------------------------------------------------------

_REPO_BLOB_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)"
    r"/blob/(?P<branch>[^/]+)/(?P<path>.+?)(?:[?#].*)?$"
)
_REPO_ROOT_RE = re.compile(r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/?(?:[?#].*)?$")
_RAW_RE = re.compile(
    r"^https?://raw\.githubusercontent\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)"
    r"/(?P<branch>[^/]+)/(?P<path>.+?)(?:[?#].*)?$"
)
_GIST_RE = re.compile(
    r"^https?://gist\.github\.com/(?P<user>[^/]+)/(?P<gist_id>[a-f0-9]+)/?(?:[?#].*)?$"
)
# username.github.io subdomain; exclude empty username (just "github.io" is GitHub itself)
_PAGES_RE = re.compile(
    r"^https?://(?P<username>[A-Za-z0-9][A-Za-z0-9-]*)\.github\.io(?P<path>/.*)?$"
)
_DATE_PATH_RE = re.compile(r"/(?P<y>\d{4})/(?P<m>\d{2})/(?P<d>\d{2})(?:/|$)")


# ---------------------------------------------------------------------------
# Applicability — shared "software" domain across all three extractors
# ---------------------------------------------------------------------------


def _applicability(source_type: str) -> list[ApplicabilityClause]:
    return [
        ApplicabilityClause(
            keyword="MUST",
            domain_uri=_GH_DOMAIN_URI,
            domain_label=_GH_DOMAIN_LABEL,
            source_types=[source_type],
        )
    ]


APPLICABILITY_REPO = _applicability(SOURCE_TYPE_REPO)
APPLICABILITY_GIST = _applicability(SOURCE_TYPE_GIST)
APPLICABILITY_PAGES = _applicability(SOURCE_TYPE_PAGES)


# ---------------------------------------------------------------------------
# URL parsing helpers
# ---------------------------------------------------------------------------


def _normalize_raw_url(url: str) -> str:
    """raw.githubusercontent.com/... → github.com/.../blob/... (else passthrough)."""
    m = _RAW_RE.match(url)
    if not m:
        return url
    return f"https://github.com/{m['owner']}/{m['repo']}/blob/{m['branch']}/{m['path']}"


def _parse_repo_url(url: str) -> tuple[str, str, str | None, str | None] | None:
    """Return (owner, repo, branch_or_None, path_or_None). None for non-repo URLs."""
    m = _REPO_BLOB_RE.match(url)
    if m:
        return m["owner"], m["repo"], m["branch"], m["path"]
    m = _REPO_ROOT_RE.match(url)
    if m:
        return m["owner"], m["repo"], None, None
    return None


def _parse_gist_url(url: str) -> tuple[str, str] | None:
    m = _GIST_RE.match(url)
    return (m["user"], m["gist_id"]) if m else None


def _parse_pages_url(url: str) -> tuple[str, str] | None:
    """Return (username, path) where path includes leading slash or is empty."""
    m = _PAGES_RE.match(url)
    return (m["username"], m["path"] or "") if m else None


def _date_from_path(path: str) -> datetime | None:
    """Extract a UTC date from a /YYYY/MM/DD/ segment in the URL path."""
    m = _DATE_PATH_RE.search(path)
    if not m:
        return None
    try:
        return datetime(int(m["y"]), int(m["m"]), int(m["d"]), tzinfo=UTC)
    except ValueError:
        return None


def _parse_iso_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


# ---------------------------------------------------------------------------
# HTTP — auth headers, retry wrapper, error mapping
# ---------------------------------------------------------------------------


def _api_headers() -> dict[str, str]:
    """Headers for api.github.com requests. Bearer auth added when key is set."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    api_key = get_github_api_key_optional()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def _github_get(client: Any, url: str, **kwargs: Any) -> Any:
    """GET via the shared retry helper; convert exhaustion to ValueError.

    Delegates retry to ``particles.http.get_with_retry`` and re-raises
    ``TransientHttpError`` as a CLI-friendly ``ValueError``. Other status
    codes (incl. 404, 403, 500) are returned to the caller for handling.
    """
    try:
        return await get_with_retry(client, url, label="GitHub API", **kwargs)
    except TransientHttpError as exc:
        raise ValueError(str(exc)) from exc


def _raise_for_github_error(resp: Any, url: str) -> None:
    """Convert remaining 4xx/5xx into a user-friendly ValueError.

    Called after ``_github_get`` (which has already retried transient 5xx).
    """
    sc = resp.status_code
    if 200 <= sc < 300:
        return
    if sc == 403:
        raise ValueError(
            f"GitHub API access forbidden (403) for {url}. "
            "Check GITHUB_API_KEY scopes or the hourly rate limit "
            "(60 req/hr without a key, 5000 req/hr with one)."
        )
    if sc == 404:
        raise ValueError(f"GitHub resource not found (404): {url}")
    if 500 <= sc < 600:
        raise ValueError(f"GitHub server error {sc} for {url}")
    raise ValueError(f"GitHub request failed (HTTP {sc}) for {url}")


# ---------------------------------------------------------------------------
# Small post-LLM helpers shared by repo + pages
# ---------------------------------------------------------------------------


def _inject_subjects(candidate: CandidateParticle, extras: list[str]) -> None:
    for s in extras:
        if s and s not in candidate.subjects:
            candidate.subjects.append(s)


async def _llm_extract_with_subjects(text: str, extras: list[str]) -> ExtractionResult:
    candidates, notes, transient = await _call_llm(text)
    for c in candidates:
        _inject_subjects(c, extras)
    return ExtractionResult(
        candidates=candidates,
        quality_notes=notes,
        transient_error_count=1 if transient else 0,
    )
