# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Reddit post extractor and its Client-safe fetch helpers.

The importer lives in ``particles.ingest.importers.reddit`` (Engine). It fetches
a thread from the Arctic Shift archive by default, or from Reddit's own ``.json``
endpoint when ``reddit.fetch_backend`` is ``reddit``, and stores a blob in
Reddit's two-listing shape either way. This module keeps what both halves share:

* the share-link resolver and the direct ``.json`` curl fetch;
* the pure helpers that build and read the stored blob, including the archive
  provenance record the importer appends as a third element;
* ``RedditExtractor``, which parses the blob, assembles an LLM-ready text from
  the post and its top comments, and calls ``_call_llm()`` for claim-granularity
  extraction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Iterator
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import ApplicabilityClause, Snapshot
from particles.extraction.general import (  # noqa: F401
    ExtractionResult,
    NormalizedDocument,
    _call_llm,
)
from particles.extraction.incremental import ChunkUnit, extract_with_carry_forward
from particles.http import SourceFetchError
from particles.url_safety import (
    Resolver,
    UnsafeUrlError,
    format_connect_pin,
    resolve_and_pin,
)

log = logging.getLogger(__name__)

SOURCE_TYPE = "REDDIT_POST"
EXTRACTOR_ID = "reddit-extractor"
EXTRACTOR_VERSION = "0.3.1"
# 0.2.0: chunked LLM extraction via extract_with_carry_forward;
#        raised top_comment_count default 30→200 and comment_body_limit
#        500→1000; emits REDDIT_COMMENT_LIMIT_HIT quality note.
# 0.3.0: walk the comment tree to arbitrary depth (BFS), not just
#        top-level + first-replies. Replies-to-replies are now captured.
# 0.3.1: canonicalise bare reddit user / subreddit subjects emitted by
#        the LLM (`ExtremeAddict` → `u/ExtremeAddict`,
#        `scifi` → `r/scifi`) so the Obsidian exporter's path-nesting
#        rule routes them under `reddit.com/u/…` / `reddit.com/r/…`.
DEFAULT_TRUST_WEIGHT = 0.40

APPLICABILITY = [
    ApplicabilityClause(
        keyword="MUST",
        domain_uri="http://www.wikidata.org/entity/Q202833",
        domain_label="social media",
        source_types=[SOURCE_TYPE],
    )
]

_REDDIT_URL_RE = re.compile(
    r"https?://(?:www\.|old\.)?reddit\.com"
    r"/r/(?P<sub>[^/]+)/comments/(?P<id>[^/]+)(?:/[^/]*)?"
    r"/?$"
)

# Reddit share links (`/r/<sub>/s/<shareid>`) — the form the iOS/Android share
# sheet produces. They are opaque redirects to the canonical `/comments/`
# permalink and carry no post id, so the importer must resolve the redirect
# before it can build the `.json` API URL (see ``_resolve_reddit_redirect``).
# Kept SEPARATE from ``_REDDIT_URL_RE`` on purpose: that regex stays the
# canonical-permalink validator (the open-redirect oracle) and must continue to
# reject `/s/` forms so the post-resolution guard actually proves something.
_REDDIT_SHARE_URL_RE = re.compile(
    r"https?://(?:www\.|old\.)?reddit\.com/r/(?P<sub>[^/]+)/s/(?P<shareid>[^/?#]+)/?(?:[?#].*)?$"
)

# Matches the leading post-permalink segments of a resolved Reddit path, so a
# *comment* permalink (`/r/x/comments/<postid>/<slug>/<commentid>/`) can be
# reduced to the post permalink (fetching the whole thread, and passing
# ``_REDDIT_URL_RE``). Anchored at the path start; trailing segments are dropped.
_REDDIT_PERMALINK_PATH_RE = re.compile(
    r"^/r/(?P<sub>[^/]+)/comments/(?P<postid>[^/]+)(?:/(?P<slug>[^/]+))?"
)

# Stop-words that look like tickers but aren't
_TICKER_STOP = frozenset(
    [
        "THE",
        "AND",
        "FOR",
        "ARE",
        "BUT",
        "NOT",
        "YOU",
        "ALL",
        "CAN",
        "HER",
        "WAS",
        "ONE",
        "OUR",
        "OUT",
        "DAY",
        "GET",
        "HAS",
        "HIM",
        "HIS",
        "HOW",
        "ITS",
        "LET",
        "MAY",
        "NEW",
        "NOW",
        "OLD",
        "OWN",
        "SAY",
        "SHE",
        "TOO",
        "USE",
        "WAY",
        "WHO",
        "WHY",
        "DID",
        "CEO",
        "CFO",
        "SEC",
        "IPO",
        "EPS",
        "ETF",
        "GDP",
        "IMO",
        "AMA",
        "TIL",
        "ELI",
        "PSA",
        "WSB",
        "IIRC",
        "AFAIK",
        "TLDR",
        "NYSE",
        "NASDAQ",
    ]
)

# ---------------------------------------------------------------------------
# HTTP fetch (curl subprocess — bypasses Cloudflare TLS fingerprint blocking)
# ---------------------------------------------------------------------------


def _curl_resolve_args(url: str, resolve: Resolver | None) -> list[str]:
    """Build the ``--resolve`` flag pair pinning ``url``'s host to vetted addresses.

    A curl subprocess never enters ``httpx``, so the ``ValidatingTransport``
    connect-time gate cannot see it; this is how the same guarantee is
    reached by a second mechanism. The addresses are resolved and
    vetted **in this process**, and curl is then pinned to them — so the validated
    address is the connected address, closing DNS rebinding / TOCTOU against the
    pinned host. curl keeps SNI and certificate verification bound to the
    original hostname, so the pin adds no downgrade.

    Raises:
        UnsafeUrlError: fail-closed, if the URL has no hostname, the host does
            not resolve, or any resolved address is private/reserved. The caller
            must let this propagate *before* spawning curl.
    """
    parts = urlsplit(url)
    host = parts.hostname
    if not host:
        raise UnsafeUrlError(f"URL {url!r} has no hostname; refusing to fetch (fail-closed).")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    addrs = resolve_and_pin(host, resolve=resolve)
    return ["--resolve", format_connect_pin(host, port, addrs)]


async def _fetch_with_curl(url: str, *, resolve: Resolver | None = None) -> bytes:
    """Fetch a URL using the system curl binary.

    Reddit's CDN (Cloudflare) blocks Python's TLS fingerprint but allows curl.
    Raises :class:`~particles.http.SourceFetchError` on a non-zero curl exit,
    carrying the parsed upstream HTTP status when the origin returned an error
    (e.g. 403) — an expected external failure the API maps to 502, not a bug.

    Hardened (security): ``--max-filesize`` bounds the downloaded body so a
    hostile or runaway response can't exhaust memory (curl does not send
    ``Accept-Encoding`` here, so the on-the-wire bytes equal the cap), and
    ``--max-time`` bounds the wall-clock so a slow-loris stall can't hang the
    extractor. Both read from config at call time. ``--resolve`` pins the
    connection to addresses this process vetted; ``resolve`` is
    injectable so unit tests stub DNS and touch no network.
    """
    cfg = get_config().http
    user_agent = cfg.user_agent
    pin = _curl_resolve_args(url, resolve)

    proc = await asyncio.create_subprocess_exec(
        "curl",
        "-sS",
        "--fail",
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
        "--max-redirs",
        "0",
        *pin,
        "--max-filesize",
        str(cfg.max_bytes),
        "--max-time",
        str(cfg.timeout_seconds),
        "-H",
        f"User-Agent: {user_agent}",
        "-H",
        "Accept: application/json",
        url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        stderr_text = stderr.decode(errors="replace")
        status = _parse_curl_http_status(stderr_text)
        if status == 403:
            # Reddit bot-walls unauthenticated access to thread content (the
            # 2025–2026 lockdown), and no longer grants API credentials to
            # scripts, so the message names the archive backend rather than an
            # OAuth route that is closed. An expected external
            # condition, not a bug: a SourceFetchError the API maps to 502.
            detail = (
                f"Reddit returned HTTP 403 for {url}: it refuses unauthenticated "
                "programmatic access to thread content. Set reddit.fetch_backend "
                "to arctic_shift (the default) to fetch the thread from the "
                "Arctic Shift archive instead."
            )
        else:
            detail = (
                f"curl failed (exit {proc.returncode}"
                f"{f', HTTP {status}' if status else ''}) fetching {url}: "
                f"{stderr_text[:200]}"
            )
        raise SourceFetchError(detail, url=url, status_code=status)
    return stdout


def _parse_curl_http_status(stderr_text: str) -> int | None:
    """Extract the HTTP status from curl's ``--fail`` stderr, if present.

    ``--fail`` reports a rejected response as ``… returned error: NNN``. Returns
    the integer status (e.g. ``403``) or ``None`` for a non-HTTP failure
    (timeout, DNS, connection reset).
    """
    m = re.search(r"returned error:\s*(\d{3})", stderr_text)
    return int(m.group(1)) if m else None


# Case-insensitive because HTTP/2 lowercases header names (``location:``) while
# HTTP/1.1 title-cases them (``Location:``).
_LOCATION_HEADER_RE = re.compile(r"^location:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def _canonicalize_reddit_permalink(url: str) -> str:
    """Reduce a Reddit permalink to its canonical post form.

    Strips the query + fragment, normalizes the host to ``www.reddit.com``, and
    truncates the path to the post permalink ``/r/<sub>/comments/<postid>/<slug>/``
    — dropping any trailing ``<commentid>`` so a *comment* share still fetches the
    whole thread. A host that is not ``(www.|old.)?reddit.com`` is left intact so
    the caller's ``_REDDIT_URL_RE`` guard rejects it (open-redirect defense).
    """
    parts = urlsplit(url)
    host = re.sub(r"^(?:www\.|old\.)?reddit\.com$", "www.reddit.com", parts.hostname or "")
    m = _REDDIT_PERMALINK_PATH_RE.match(parts.path)
    if m is None:
        # Not a comments permalink — return a host-normalized, query-stripped
        # form. It won't match _REDDIT_URL_RE, so the guard refuses to fetch it.
        return urlunsplit(("https", host, parts.path, "", ""))
    sub, postid, slug = m.group("sub"), m.group("postid"), m.group("slug")
    path = f"/r/{sub}/comments/{postid}/{slug}/" if slug else f"/r/{sub}/comments/{postid}/"
    return urlunsplit(("https", host, path, "", ""))


async def _resolve_reddit_redirect(url: str, *, resolve: Resolver | None = None) -> str:
    """Resolve a Reddit ``/s/`` share link to its canonical ``/comments/`` permalink.

    Reddit share links 3xx-redirect to the post's permalink. We read only the
    first-hop ``Location`` header via ``curl --max-redirs 0`` — no following —
    then **re-validate** the canonicalized target against ``_REDDIT_URL_RE``.
    That host-allowlist check prevents a hostile share link from steering the
    subsequent ``.json`` fetch at an arbitrary host.

    This is the one multi-hop flow in the system, and it reaches the per-hop
    guarantee by composition rather than by a transport:
    each hop is an independent fetch, so each hop pins its own connection —
    this ``--resolve``, then a second one inside ``_fetch_with_curl``. The host
    allowlist is retained; the address pin sits beneath it, not in place of it.

    Raises:
        SourceFetchError: if curl fails (e.g. the share link 404s or Reddit
            refuses the request), the response carries no ``Location``, or the
            resolved target is not a ``(www.|old.)?reddit.com`` ``/comments/``
            permalink. It subclasses ``RuntimeError``, which this raised before.
        UnsafeUrlError: if the share link's host does not resolve, or resolves
            to a private/reserved address (fail-closed, before curl is spawned).
    """
    cfg = get_config().http
    pin = _curl_resolve_args(url, resolve)
    proc = await asyncio.create_subprocess_exec(
        "curl",
        "-sS",
        "--fail",  # 4xx/5xx → non-zero exit; a 3xx redirect is NOT a failure
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
        "--max-redirs",
        "0",  # never follow — read the first-hop Location and validate it ourselves
        *pin,
        "--max-time",
        str(cfg.timeout_seconds),
        "-o",
        "/dev/null",  # discard any body
        "-D",
        "-",  # dump response headers to stdout
        "-H",
        f"User-Agent: {cfg.user_agent}",
        url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        stderr_text = stderr.decode(errors="replace")
        status = _parse_curl_http_status(stderr_text)
        if status == 403:
            detail = (
                f"Reddit returned HTTP 403 resolving the share link {url}: it refused "
                "the redirect lookup. Deposit the thread's /comments/ permalink instead."
            )
        else:
            detail = (
                f"curl failed (exit {proc.returncode}"
                f"{f', HTTP {status}' if status else ''}) resolving Reddit share link "
                f"{url}: {stderr_text[:200]}"
            )
        raise SourceFetchError(detail, url=url, status_code=status)
    match = _LOCATION_HEADER_RE.search(stdout.decode("utf-8", "replace"))
    if match is None:
        raise SourceFetchError(f"Reddit share link did not redirect to a permalink: {url}", url=url)
    resolved = urljoin(url, match.group(1))  # tolerate a relative Location
    canonical = _canonicalize_reddit_permalink(resolved)
    if not _REDDIT_URL_RE.match(canonical):
        raise SourceFetchError(
            f"Reddit share link resolved to an unexpected target, refusing to fetch: {resolved}",
            url=url,
        )
    return canonical


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class RedditExtractor:
    """Extract claim-granularity particles from a stored REDDIT_POST JSON blob."""

    EXTRACTOR_ID = EXTRACTOR_ID
    EXTRACTOR_VERSION = EXTRACTOR_VERSION
    APPLICABILITY = APPLICABILITY
    DEFAULT_TRUST_WEIGHT = DEFAULT_TRUST_WEIGHT

    def accepts(self, source_type: str) -> bool:
        return source_type == SOURCE_TYPE

    async def extract(
        self,
        snapshot: Snapshot,
        content: bytes,
        **kwargs: object,
    ) -> ExtractionResult:
        """Two-line composition convention: _normalise then _extract_claims."""
        doc = self._normalise(content, snapshot)
        return await self._extract_claims(doc, **kwargs)

    def _normalise(self, content: bytes, snapshot: Snapshot) -> NormalizedDocument:
        """Source-format parsing only: JSON → prose chunks.

        Surfaces the post's subreddit and inferred ticker as ``injected_subjects``
        so ``_extract_claims`` can stamp them on every candidate without
        re-parsing the source.
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            return NormalizedDocument(quality_notes=[f"JSON parse error: {exc}"])

        post = _get_post(data)
        if post is None:
            return NormalizedDocument(
                quality_notes=["Could not find post data in Reddit JSON blob"]
            )

        cfg = get_config()
        all_qualifying = _get_all_qualifying_comments(data)
        shown_comments = all_qualifying[: cfg.reddit.top_comment_count]

        chunks = _build_reddit_chunks(
            post=post,
            comments=shown_comments,
            body_limit=cfg.reddit.comment_body_limit,
            single_call_threshold=cfg.extraction.single_call_threshold_chars,
            chunk_chars=cfg.extraction.comment_chunk_chars,
        )

        quality_notes: list[str] = []
        unsettled_dropped, comment_total = _count_unsettled_below_floor(data)
        if unsettled_dropped:
            quality_notes.append(
                f"REDDIT_SCORES_UNSETTLED: {unsettled_dropped} of {comment_total} comments "
                f"came from the Arctic Shift archive before its second pass, so their "
                f"score still reads 0 or 1, and they fell below reddit.min_comment_score "
                f"and were left out. About 36 hours after the comments were made, run "
                f"`particles corpus refresh` on this entry to extract them with "
                f"settled scores."
            )
        if len(all_qualifying) > cfg.reddit.top_comment_count:
            quality_notes.append(
                f"REDDIT_COMMENT_LIMIT_HIT: {len(all_qualifying)} qualifying comments "
                f"exist; only the top {cfg.reddit.top_comment_count} (by score) "
                f"were sent to the LLM. Raise reddit.top_comment_count to capture more."
            )

        subreddit = post.get("subreddit", "")
        ticker = _infer_ticker(post.get("title", ""))
        injected: list[str] = []
        if subreddit:
            injected.append(f"r/{subreddit}")
        if ticker:
            injected.append(ticker)

        return NormalizedDocument(
            chunks=chunks,
            author_id=snapshot.author_id,
            content_published_at=snapshot.content_published_at,
            quality_notes=quality_notes,
            injected_subjects=injected,
        )

    async def _extract_claims(self, doc: NormalizedDocument, **kwargs: object) -> ExtractionResult:
        """LLM claim extraction over a NormalizedDocument (convention)."""

        # No chunks means _normalise rejected the input — short-circuit.
        if not doc.chunks:
            return ExtractionResult(quality_notes=doc.quality_notes)

        cfg = get_config()
        session = kwargs.get("session")
        corpus_entry_id = kwargs.get("corpus_entry_id")
        # Reindex threads its supersede set so carry-forward treats the
        # marked particles as absent (see extract_with_carry_forward).
        sup_obj = kwargs.get("supersede_ids")
        result = await extract_with_carry_forward(
            session=session if isinstance(session, AsyncSession) else None,
            chunks=doc.chunks,
            corpus_entry_id=corpus_entry_id if isinstance(corpus_entry_id, str) else None,
            extractor_id=EXTRACTOR_ID,
            extractor_version=EXTRACTOR_VERSION,
            max_llm_calls=cfg.extraction.max_llm_calls_per_source,
            supersede_ids=sup_obj if isinstance(sup_obj, frozenset) else frozenset(),
        )

        # Stamp domain-injected subjects (subreddit, ticker) on every candidate.
        for c in result.candidates:
            for subj in doc.injected_subjects:
                if subj not in c.subjects:
                    c.subjects.append(subj)

        # Canonicalise bare reddit users / subreddits back to their
        # prefixed form. The LLM sees `u/{author}: <body>` in comments
        # and usually emits `u/{author}` as a subject token — but
        # occasionally strips the prefix ("ExtremeAddict commented…"),
        # which breaks the Obsidian exporter's path-nesting rule
        # (`u/X` → `reddit.com/u/X.md`; bare `X` ends up at vault root).
        _rewrite_reddit_subjects(result.candidates, doc.chunks)

        # Propagate _normalise()'s quality notes alongside any from extraction.
        # (REDDIT_COMMENT_LIMIT_HIT is emitted by _normalise into doc.quality_notes
        # when applicable.)
        result.quality_notes = [*doc.quality_notes, *result.quality_notes]

        return result


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _get_post(data: Any) -> dict[str, Any] | None:
    """Extract the post dict from the Reddit JSON API response."""
    try:
        result = data[0]["data"]["children"][0]["data"]
        return result if isinstance(result, dict) else None
    except (IndexError, KeyError, TypeError):
        return None


def _get_all_qualifying_comments(data: Any) -> list[dict[str, Any]]:
    """Return every qualifying comment in the thread, at any depth.

    Walks the Reddit comment tree breadth-first via the nested ``replies``
    structure and keeps comments whose ``score`` meets
    ``reddit.min_comment_score``. Sorted by score descending. The caller
    decides how many to keep — today that's ``reddit.top_comment_count``.
    This function intentionally does NOT apply the cap so callers can
    measure overflow and emit a ``REDDIT_COMMENT_LIMIT_HIT`` quality note
    when the cap binds.

    Bumped from depth-1 (top-level + first-replies only) to arbitrary
    depth in 0.3.0 — on technical subreddits substantive discussion
    frequently lives deeper in reply chains and was previously dropped.
    """
    out: list[dict[str, Any]] = []
    min_score = get_config().reddit.min_comment_score
    for c in _iter_comments(data):
        body = c.get("body", "")
        if body and body not in ("[deleted]", "[removed]"):
            score = c.get("score", 0)
            if score >= min_score:
                out.append({"author": c.get("author", ""), "score": score, "body": body})

    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def _iter_comments(data: Any) -> Iterator[dict[str, Any]]:
    """Yield every comment record in the thread, breadth-first, at any depth.

    Replies are visited whether or not their parent is later kept: a
    low-scoring parent can still have high-scoring replies worth surfacing.
    """
    try:
        roots = data[1]["data"]["children"]
    except (IndexError, KeyError, TypeError):
        return
    if not isinstance(roots, list):
        return

    queue: list[Any] = list(roots)
    while queue:
        node = queue.pop(0)
        if not isinstance(node, dict):
            continue
        # Reddit uses kind "t1" for comments and "more" for "load more"
        # continuation pointers — we ignore the latter.
        if node.get("kind") != "t1":
            continue
        c = node.get("data", {})
        if not isinstance(c, dict):
            continue
        yield c
        replies = c.get("replies")
        if isinstance(replies, dict):
            reply_children = replies.get("data", {}).get("children", [])
            if isinstance(reply_children, list):
                queue.extend(reply_children)


# ---------------------------------------------------------------------------
# The stored blob: assembly from the archive, and its provenance record
# ---------------------------------------------------------------------------

#: ``reddit.fetch_backend`` value that reads the Arctic Shift archive.
ARCTIC_SHIFT_BACKEND = "arctic_shift"

#: ``kind`` of the third top-level element the importer appends to a blob it
#: assembled from the archive. Reddit's own response has two elements, and the
#: parsers here read only ``data[0]`` and ``data[1]``, so the record rides the
#: snapshot's own bytes without changing how the blob is read. The snapshot
#: model has no metadata column, and adding one would need a store migration.
FETCH_PROVENANCE_KIND = "particles_fetch_provenance"

# A record the archive first captured this long after it was created carries a
# meaningful score even without the archive's second pass. Arctic Shift makes
# that pass about 36 hours after posting; records it imported historically were
# captured long after the fact and may never get one.
_SCORE_SETTLE_SECONDS = 36 * 3600


def _assemble_reddit_listing(
    post: dict[str, Any], comment_nodes: list[Any], provenance: dict[str, Any]
) -> list[Any]:
    """Build the stored blob from an archive post and its comment tree.

    The first two elements are Reddit's ``{permalink}.json`` response shape (a
    post Listing, then a comment Listing of ``t1`` nodes), so ``_get_post``,
    ``_get_all_qualifying_comments`` and the importer's ``primary_url`` read it
    unchanged. The third is the provenance record (``FETCH_PROVENANCE_KIND``).
    """
    return [
        {"kind": "Listing", "data": {"children": [{"kind": "t3", "data": post}]}},
        {"kind": "Listing", "data": {"children": list(comment_nodes)}},
        {"kind": FETCH_PROVENANCE_KIND, "data": provenance},
    ]


def _get_fetch_provenance(data: Any) -> dict[str, Any] | None:
    """The blob's provenance record, or None for bytes Reddit itself served."""
    try:
        element = data[2]
    except (IndexError, KeyError, TypeError):
        return None
    if not isinstance(element, dict) or element.get("kind") != FETCH_PROVENANCE_KIND:
        return None
    record = element.get("data")
    return record if isinstance(record, dict) else None


def _is_score_settled(record: dict[str, Any]) -> bool:
    """Whether an archived post or comment carries a score worth filtering on.

    True once the archive's second pass has run (``_meta.retrieved_2nd_on``), or
    when the first capture was already long after the record was created.
    """
    meta = record.get("_meta")
    if isinstance(meta, dict) and meta.get("retrieved_2nd_on") is not None:
        return True
    try:
        age = float(record["retrieved_on"]) - float(record["created_utc"])
    except (KeyError, TypeError, ValueError):
        return False
    return age >= _SCORE_SETTLE_SECONDS


def _count_unsettled_below_floor(data: Any) -> tuple[int, int]:
    """Count archive comments whose unsettled score dropped them, and all comments.

    Returns ``(dropped, total)`` over live comments (not deleted or removed).
    ``dropped`` counts those below ``reddit.min_comment_score`` whose score the
    archive has not settled yet. A blob Reddit served directly returns
    ``(0, 0)``: its scores are live, and the question does not arise.
    """
    provenance = _get_fetch_provenance(data)
    if provenance is None or provenance.get("backend") != ARCTIC_SHIFT_BACKEND:
        return 0, 0
    min_score = get_config().reddit.min_comment_score
    dropped = total = 0
    for c in _iter_comments(data):
        body = c.get("body", "")
        if not body or body in ("[deleted]", "[removed]"):
            continue
        total += 1
        if c.get("score", 0) < min_score and not _is_score_settled(c):
            dropped += 1
    return dropped, total


def _render_reddit_body(post: dict[str, Any]) -> str:
    """The body chunk's text — post title and selftext only, no comments.

    Stable across re-deposits when the post body is unedited, so the body
    chunk is the prime candidate for carry-forward when only comments
    have changed.
    """
    subreddit = post.get("subreddit", "unknown")
    title = post.get("title", "")
    author = post.get("author", "")
    score = post.get("score", 0)
    body = post.get("selftext", "").strip()

    lines = [
        f"REDDIT THREAD: r/{subreddit} — {title}",
        f"Posted by u/{author} [score: {score}]",
    ]
    if body:
        lines.append("")
        lines.append(body)
    return "\n".join(lines)


def _render_reddit_comment_lines(comments: list[dict[str, Any]], body_limit: int) -> list[str]:
    return [f"[score:{c['score']}] u/{c['author']}: {c['body'][:body_limit]}" for c in comments]


def _chunk_reddit_comments_by_size(
    comments: list[dict[str, Any]], chunk_chars: int, body_limit: int
) -> list[list[dict[str, Any]]]:
    """Group reddit comments into chunks whose rendered text fits the budget."""
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_size = 0
    for c in comments:
        body = c.get("body") or ""
        # Overhead per line: "[score:N] u/{author}: " ≈ 30 chars
        rendered_size = min(len(body), body_limit) + 30
        if current and current_size + rendered_size > chunk_chars:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(c)
        current_size += rendered_size
    if current:
        chunks.append(current)
    return chunks


def _build_reddit_chunks(
    post: dict[str, Any],
    comments: list[dict[str, Any]],
    body_limit: int,
    single_call_threshold: int,
    chunk_chars: int,
) -> list[ChunkUnit]:
    """Produce ChunkUnits for a reddit thread (body + 0-or-more comment chunks).

    The body chunk is always its own unit; comment chunks are sized
    adaptively, mirroring the gist extractor's strategy. A context header
    on each comment chunk names the subreddit and post title so
    paraphrased referents stay resolvable when chunks are extracted in
    isolation.
    """
    chunks: list[ChunkUnit] = []
    body_text = _render_reddit_body(post)
    if body_text.strip():
        chunks.append(ChunkUnit(chunk_id="body", chunk_text=body_text))

    if not comments:
        return chunks

    subreddit = post.get("subreddit", "unknown")
    title = post.get("title", "")
    header = f"# Context: comments on r/{subreddit} thread — {title[:120]}"

    rendered_size = sum(min(len(c.get("body") or ""), body_limit) + 30 for c in comments)

    if rendered_size <= single_call_threshold:
        parts = [
            header,
            "TOP COMMENTS:",
            *_render_reddit_comment_lines(comments, body_limit),
        ]
        chunks.append(ChunkUnit(chunk_id="comments", chunk_text="\n".join(parts)))
        return chunks

    sub_chunks = _chunk_reddit_comments_by_size(comments, chunk_chars, body_limit)
    total = len(sub_chunks)
    for i, sub_chunk in enumerate(sub_chunks, start=1):
        text_parts = [
            header,
            f"TOP COMMENTS (chunk {i}/{total}):",
            *_render_reddit_comment_lines(sub_chunk, body_limit),
        ]
        chunks.append(
            ChunkUnit(
                chunk_id=f"comments_{i}_of_{total}",
                chunk_text="\n".join(text_parts),
            )
        )
    return chunks


def _infer_ticker(title: str) -> str | None:
    """Heuristic: first uppercase 2–5 letter token in the title not in the stop-word list."""
    tokens: list[str] = re.findall(r"\b[A-Z]{2,5}\b", title)
    for token in tokens:
        if token not in _TICKER_STOP:
            return token
    return None


# ---------------------------------------------------------------------------
# Subject canonicalisation (post-extraction step)
# ---------------------------------------------------------------------------

# Reddit username rules (per Reddit docs): 3–20 chars, letters/digits/`_`/`-`.
# Subreddit name rules: 1–21 chars, letters/digits/`_`.
_REDDIT_USER_TOKEN_RE = re.compile(r"\bu/([A-Za-z0-9_-]{3,20})\b")
_REDDIT_SUBREDDIT_TOKEN_RE = re.compile(r"\br/([A-Za-z0-9_]{1,21})\b")


def _rewrite_reddit_subjects(candidates: list[Any], chunks: list[ChunkUnit]) -> None:
    """Canonicalise bare reddit user / subreddit subjects in candidates.

    The reddit extractor renders comments as ``u/{author}: <body>`` so
    the LLM usually emits ``u/{author}`` (and ``r/{subreddit}``) as
    subject tokens — but occasionally strips the prefix ("ExtremeAddict
    commented…"), which breaks the Obsidian exporter's path-nesting
    rule (:func:`particles.exporters.markdown.subject_slug` routes
    ``u/X`` → ``reddit.com/u/X.md``; a bare ``X`` ends up at the vault
    root, indistinguishable from a real-world subject by that name).

    Scan ``chunks`` for ``u/{name}`` / ``r/{name}`` tokens to learn
    which usernames + subreddits actually exist in the source. Then
    walk each candidate's ``subjects`` list and rewrite bare matches
    back to the canonical prefixed form. Already-prefixed subjects
    pass through unchanged.

    Mirrors :func:`particles.extraction.github._rewrite_gh_subjects`'s
    role for ``gh/`` → ``github:`` canonicalisation.
    """
    known_users: set[str] = set()
    known_subs: set[str] = set()
    for chunk in chunks:
        text = chunk.chunk_text
        known_users.update(m.group(1) for m in _REDDIT_USER_TOKEN_RE.finditer(text))
        known_subs.update(m.group(1) for m in _REDDIT_SUBREDDIT_TOKEN_RE.finditer(text))

    for c in candidates:
        seen: set[str] = set()
        out: list[str] = []
        for s in c.subjects:
            if isinstance(s, str) and not s.startswith(("u/", "r/")):
                if s in known_users:
                    s = f"u/{s}"
                elif s in known_subs:
                    s = f"r/{s}"
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        c.subjects = out
