"""Hacker News thread extractor and importer (importer naming).

HackerNewsImporter — accepts ``news.ycombinator.com/item?id=N`` and the raw
                     Firebase API URL ``hacker-news.firebaseio.com/v0/item/N.json``.
                     Walks the comment tree via the public Firebase API and
                     stores the assembled JSON blob (story + comments) as a
                     single corpus entry of source type ``HACKERNEWS_THREAD``.
HackerNewsExtractor — parses the blob, renders title + story body + comment
                      tree as indented prose, and calls the chunked LLM
                      extraction helper. Additionally synthesises
                      a single *story-meta* CandidateParticle whose
                      ``properties`` dict follows URI-prefix
                      convention with the dual-emission rule (both
                      ``hn:hasPoints`` and ``social:hasScore``, etc.).

This is the schema stress test for the cross-source ``properties``
convention. The HN structured metadata (score, author, item id, comment
count, external URL) lands as a single particle with the dual-emission
``properties`` dict; the free-text claim particles produced by the LLM
follow the existing pattern and carry no ``properties``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import ApplicabilityClause, Snapshot, UncertaintyNature
from particles.extraction.general import (
    CandidateParticle,
    ExtractionResult,
    NormalizedDocument,
)
from particles.extraction.incremental import ChunkUnit, extract_with_carry_forward
from particles.http import get_capped

log = logging.getLogger(__name__)

SOURCE_TYPE = "HACKERNEWS_THREAD"
EXTRACTOR_ID = "hackernews-extractor"
EXTRACTOR_VERSION = "0.1.0"
DEFAULT_TRUST_WEIGHT = 0.50  # UGC starting weight for HN

HACKERNEWS_API_BASE = "https://hacker-news.firebaseio.com/v0"

APPLICABILITY = [
    ApplicabilityClause(
        keyword="MUST",
        domain_uri="https://news.ycombinator.com",
        domain_label="Hacker News",
        source_types=[SOURCE_TYPE],
    )
]

# Item-URL form copied by an operator from their browser.
# Examples that match:
#   https://news.ycombinator.com/item?id=12345
#   http://news.ycombinator.com/item?id=12345
#   https://news.ycombinator.com/item?id=12345&p=2  (extra params tolerated)
_HN_ITEM_URL_RE = re.compile(
    r"^https?://news\.ycombinator\.com/item\?(?:[^#]*&)?id=(?P<id>\d+)(?:&[^#]*)?$"
)

# Raw Firebase API URL form. Operators rarely paste this but the importer
# accepts it so the deposit flow is symmetric with what other tooling
# (rss-to-particles agents, scripts) might emit.
_HN_API_URL_RE = re.compile(
    r"^https?://hacker-news\.firebaseio\.com/v0/item/(?P<id>\d+)\.json(?:\?.*)?$"
)


def _parse_item_id(url: str) -> int | None:
    """Return the numeric HN item id parsed from ``url`` or None if not an HN URL."""
    m = _HN_ITEM_URL_RE.match(url) or _HN_API_URL_RE.match(url)
    return int(m.group("id")) if m else None


# ---------------------------------------------------------------------------
# Firebase API fetch helpers
# ---------------------------------------------------------------------------


def _api_url(item_id: int) -> str:
    """Return the Firebase API URL for a single HN item."""
    return f"{HACKERNEWS_API_BASE}/item/{item_id}.json"


async def _fetch_item(client: Any, item_id: int) -> dict[str, Any] | None:
    """Fetch one HN item via the Firebase API.

    Returns the decoded JSON object on success. Returns ``None`` when the
    item id is unknown (Firebase returns the literal token ``null``) or the
    item is structurally invalid — both are treated as "skip this comment
    branch" by the caller rather than raising, mirroring how Reddit's
    extractor tolerates ``[deleted]`` / ``[removed]`` bodies.
    """
    resp = await get_capped(client, _api_url(item_id))
    resp.raise_for_status()
    try:
        data = resp.json()
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


async def _fetch_thread(client: Any, root_id: int, max_comments: int) -> dict[str, Any]:
    """Fetch the story item and walk its ``kids`` tree to gather comments.

    The on-disk format is a single JSON object:

    ```json
    {
        "story": {"id": ..., "title": ..., "score": ..., "by": ..., "kids": [...]},
        "comments": {
            "12345": {"id": 12345, "by": "...", "text": "...", "kids": [...], "parent": 1},
            ...
        }
    }
    ```

    Walks depth-first (so a reply sequence stays contiguous in iteration
    order) up to ``max_comments`` total comments. Deeply-nested but
    low-engagement threads will be truncated at the cap; the extractor
    surfaces a ``HN_COMMENT_LIMIT_HIT`` quality note in that case.
    """
    story = await _fetch_item(client, root_id)
    if story is None:
        raise ValueError(f"Hacker News item {root_id} not found or returned non-dict JSON")

    comments: dict[str, dict[str, Any]] = {}

    async def _walk(kids: list[int]) -> None:
        for child_id in kids:
            if len(comments) >= max_comments:
                return
            child = await _fetch_item(client, child_id)
            if child is None:
                continue
            comments[str(child_id)] = child
            sub_kids = child.get("kids")
            if isinstance(sub_kids, list) and sub_kids:
                await _walk(sub_kids)

    root_kids = story.get("kids")
    if isinstance(root_kids, list):
        await _walk(root_kids)

    return {"story": story, "comments": comments}


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class HackerNewsExtractor:
    """Extract claim-granularity particles from a stored HACKERNEWS_THREAD JSON blob."""

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
        """Three-step composition convention.

        Departs from Reddit's strict two-line composition because HN
        emits one additional particle that is *not* LLM-derived: the
        story-meta particle carrying the dual-emission ``properties``
        dict. Building that particle requires the structured
        story dict, so ``extract()`` parses once, hands the parsed story
        to both the LLM-prose path and the meta-particle synthesiser,
        then merges the results.
        """
        doc, story = self._normalise(content, snapshot)
        result = await self._extract_claims(doc, **kwargs)
        story_meta = _build_story_meta_candidate(story, doc.injected_subjects)
        if story_meta is not None:
            result.candidates.insert(0, story_meta)
        return result

    def _normalise(
        self, content: bytes, snapshot: Snapshot
    ) -> tuple[NormalizedDocument, dict[str, Any]]:
        """Source-format parsing only: JSON → prose chunks + story dict.

        Returns a tuple ``(NormalizedDocument, story_dict)`` so the caller
        can hand the structured story to the meta-particle synthesiser
        without re-parsing the raw bytes. The story dict is empty when
        the source is malformed.
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            return (NormalizedDocument(quality_notes=[f"JSON parse error: {exc}"]), {})

        story = _get_story(data)
        if story is None:
            return (
                NormalizedDocument(
                    quality_notes=["Could not find story data in Hacker News JSON blob"]
                ),
                {},
            )

        cfg = get_config()
        comments_map = _get_comments_map(data)

        chunks = _build_hn_chunks(
            story=story,
            comments_map=comments_map,
            indent_per_level=cfg.hackernews.comment_indent,
            min_score=cfg.hackernews.min_comment_score,
            single_call_threshold=cfg.extraction.single_call_threshold_chars,
            chunk_chars=cfg.extraction.comment_chunk_chars,
        )

        quality_notes: list[str] = []
        if len(comments_map) >= cfg.hackernews.max_comments:
            quality_notes.append(
                f"HN_COMMENT_LIMIT_HIT: importer captured {len(comments_map)} comments "
                f"(at the hackernews.max_comments={cfg.hackernews.max_comments} cap). "
                "Raise the cap to capture the full thread."
            )

        author = story.get("by") or ""
        injected: list[str] = []
        if author:
            injected.append(f"hn/{author}")

        return (
            NormalizedDocument(
                chunks=chunks,
                author_id=snapshot.author_id,
                content_published_at=snapshot.content_published_at,
                quality_notes=quality_notes,
                injected_subjects=injected,
            ),
            story,
        )

    async def _extract_claims(self, doc: NormalizedDocument, **kwargs: object) -> ExtractionResult:
        """LLM claim extraction over the NormalizedDocument (convention)."""
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

        # Stamp domain-injected subjects (the story author) on every LLM-derived candidate.
        for c in result.candidates:
            for subj in doc.injected_subjects:
                if subj not in c.subjects:
                    c.subjects.append(subj)

        # Canonicalise bare HN handles back to the ``hn/<author>`` form so
        # the Obsidian exporter routes them under
        # ``news.ycombinator.com/hn/<author>.md`` (mirroring how
        # ``_rewrite_reddit_subjects`` recovers ``u/<author>`` from bare
        # names).
        _rewrite_hn_subjects(result.candidates, doc.chunks)

        result.quality_notes = [*doc.quality_notes, *result.quality_notes]
        return result


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _get_story(data: Any) -> dict[str, Any] | None:
    """Extract the story dict from the importer-shaped JSON blob."""
    if not isinstance(data, dict):
        return None
    story = data.get("story")
    return story if isinstance(story, dict) else None


def _get_comments_map(data: Any) -> dict[str, dict[str, Any]]:
    """Extract the {id_str: comment_dict} map from the importer-shaped JSON blob."""
    if not isinstance(data, dict):
        return {}
    comments = data.get("comments", {})
    if not isinstance(comments, dict):
        return {}
    return {str(k): v for k, v in comments.items() if isinstance(v, dict)}


def _render_hn_story_body(story: dict[str, Any]) -> str:
    """The story chunk's text — title + optional self-post body. Excludes comments.

    Stable across re-deposits when the story body is unedited, so this is
    the prime carry-forward candidate when only comments change (mirrors
    the Reddit body-chunk strategy).
    """
    title = story.get("title", "") or ""
    author = story.get("by", "") or ""
    score = story.get("score", 0) or 0
    url = story.get("url") or ""
    body = (story.get("text") or "").strip()

    lines = [
        f"HACKER NEWS THREAD: {title}",
        f"Posted by hn/{author} [score: {score}]",
    ]
    if url:
        lines.append(f"External URL: {url}")
    if body:
        lines.append("")
        # HN's text fields contain HTML entities and limited tags; the LLM
        # tolerates raw HTML, so we leave it unprocessed to avoid losing
        # quoted blocks rendered with ``<p>`` / ``<i>``.
        lines.append(body)
    return "\n".join(lines)


def _walk_comments_dfs(
    story: dict[str, Any],
    comments_map: dict[str, dict[str, Any]],
    min_score: int,
) -> list[tuple[int, dict[str, Any]]]:
    """Depth-first comment traversal returning ``(depth, comment)`` tuples.

    Mirrors HN's UI ordering (parents above replies, replies indented). A
    comment is dropped if its body is missing or it is flagged as deleted /
    dead — but its replies are still traversed, since a flagged parent can
    have substantive child replies (parallel to Reddit's BFS treatment).

    Per-item ``score`` is not exposed by Firebase for comments (only for
    stories), so ``min_score`` only thresholds when an item happens to
    carry a score (rare in practice — defensive handling).
    """
    out: list[tuple[int, dict[str, Any]]] = []

    def _visit(node: dict[str, Any], depth: int) -> None:
        text = (node.get("text") or "").strip()
        dead = bool(node.get("dead"))
        deleted = bool(node.get("deleted"))
        score = node.get("score")
        score_ok = not isinstance(score, int) or score >= min_score
        if text and not dead and not deleted and score_ok:
            out.append((depth, node))
        kids = node.get("kids")
        if isinstance(kids, list):
            for kid_id in kids:
                child = comments_map.get(str(kid_id))
                if child is not None:
                    _visit(child, depth + 1)

    root_kids = story.get("kids")
    if isinstance(root_kids, list):
        for kid_id in root_kids:
            child = comments_map.get(str(kid_id))
            if child is not None:
                _visit(child, 0)

    return out


def _render_hn_comment_lines(
    comments: list[tuple[int, dict[str, Any]]],
    indent_per_level: int,
) -> list[str]:
    """Render an ordered ``(depth, comment)`` list as indented prose lines."""
    lines: list[str] = []
    for depth, c in comments:
        author = c.get("by") or "anon"
        text = (c.get("text") or "").replace("\n", " ").strip()
        indent = " " * (depth * indent_per_level)
        lines.append(f"{indent}hn/{author}: {text}")
    return lines


def _chunk_hn_comments_by_size(
    comments: list[tuple[int, dict[str, Any]]],
    indent_per_level: int,
    chunk_chars: int,
) -> list[list[tuple[int, dict[str, Any]]]]:
    """Group HN comments into chunks whose rendered text fits the budget."""
    chunks: list[list[tuple[int, dict[str, Any]]]] = []
    current: list[tuple[int, dict[str, Any]]] = []
    current_size = 0
    for depth, c in comments:
        text = (c.get("text") or "").strip()
        # Overhead per line: indent + "hn/{author}: " ≈ 20 chars + indent
        rendered_size = len(text) + (depth * indent_per_level) + 20
        if current and current_size + rendered_size > chunk_chars:
            chunks.append(current)
            current = []
            current_size = 0
        current.append((depth, c))
        current_size += rendered_size
    if current:
        chunks.append(current)
    return chunks


def _build_hn_chunks(
    story: dict[str, Any],
    comments_map: dict[str, dict[str, Any]],
    indent_per_level: int,
    min_score: int,
    single_call_threshold: int,
    chunk_chars: int,
) -> list[ChunkUnit]:
    """Produce ChunkUnits for an HN thread (body + 0-or-more comment chunks).

    The body chunk is always its own unit. Comment chunks are sized
    adaptively, mirroring the Reddit / gist strategy so the chunk-hash
    carry-forward can reuse particles when only one chunk's
    text changes.
    """
    chunks: list[ChunkUnit] = []
    body_text = _render_hn_story_body(story)
    if body_text.strip():
        chunks.append(ChunkUnit(chunk_id="story", chunk_text=body_text))

    flat_comments = _walk_comments_dfs(story, comments_map, min_score)
    if not flat_comments:
        return chunks

    title = (story.get("title") or "")[:120]
    header = f"# Context: comments on Hacker News thread — {title}"

    rendered_size = sum(
        len((c.get("text") or "").strip()) + (d * indent_per_level) + 20 for d, c in flat_comments
    )

    if rendered_size <= single_call_threshold:
        parts = [
            header,
            "COMMENTS:",
            *_render_hn_comment_lines(flat_comments, indent_per_level),
        ]
        chunks.append(ChunkUnit(chunk_id="comments", chunk_text="\n".join(parts)))
        return chunks

    sub_chunks = _chunk_hn_comments_by_size(flat_comments, indent_per_level, chunk_chars)
    total = len(sub_chunks)
    for i, sub_chunk in enumerate(sub_chunks, start=1):
        text_parts = [
            header,
            f"COMMENTS (chunk {i}/{total}):",
            *_render_hn_comment_lines(sub_chunk, indent_per_level),
        ]
        chunks.append(
            ChunkUnit(
                chunk_id=f"comments_{i}_of_{total}",
                chunk_text="\n".join(text_parts),
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Story-meta particle (dual-emission)
# ---------------------------------------------------------------------------


def _build_story_meta_candidate(
    story: dict[str, Any], injected_subjects: list[str]
) -> CandidateParticle | None:
    """Synthesise the single particle carrying the dual-emission ``properties`` dict.

    The story-meta particle's ``content`` is a one-line summary of the
    thread (title + score + comment count) so cosine-similarity ranking
    can still surface it for queries like "what HN threads about LLMs got
    the most engagement". The structured ``properties`` carry the
    machine-readable values that an exporter or structured query reads
    directly (dual-emission: ``hn:hasPoints`` +
    ``social:hasScore``, etc.).

    Returns ``None`` when ``story`` is empty or its required fields
    (``id``, ``by``) are missing — in that case the source blob was
    malformed and we'd rather let the LLM-derived claims stand alone
    than fabricate a meta record from defaults.
    """
    if not story:
        return None
    item_id_raw = story.get("id")
    author = story.get("by") or ""
    if not isinstance(item_id_raw, int) or not author:
        return None

    item_id = item_id_raw
    title = story.get("title") or ""
    score_raw = story.get("score")
    score: int = score_raw if isinstance(score_raw, int) else 0
    # HN's ``descendants`` field is exactly the count of non-root items
    # in the thread tree — the importer's blob preserves it verbatim, so
    # we read it directly rather than re-counting prose lines (the prose
    # rendering drops text-less / dead comments, biasing the count low).
    desc_raw = story.get("descendants")
    comment_count: int = desc_raw if isinstance(desc_raw, int) else 0
    external_url_raw = story.get("url")
    external_url = external_url_raw if isinstance(external_url_raw, str) else None

    properties: dict[str, object] = {
        # Hacker News platform-specific keys (platform prefix).
        "hn:hasPoints": score,
        "hn:hasAuthor": author,
        "hn:hasItemId": item_id,
        "hn:hasCommentCount": comment_count,
        # Cross-platform UGC engagement keys (dual-emission).
        # Exporters / structured queries that target the social: namespace
        # see the same values without needing per-platform knowledge.
        "social:hasScore": score,
        "social:hasReplyCount": comment_count,
        "social:hasAuthorHandle": author,
        # Generic content + thread structure keys.
        "content:hasUrl": external_url,
        "thread:hasRootId": item_id,
    }

    summary = (
        f"Hacker News thread '{title or 'untitled'}' by hn/{author} "
        f"received {score} points and {comment_count} comments."
    )

    return CandidateParticle(
        content=summary,
        confidence_value=0.95,
        # Engagement metrics are observed, not inferred — EPISTEMIC.
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        subjects=list(injected_subjects),
        properties=properties,
    )


# ---------------------------------------------------------------------------
# Subject canonicalisation (post-extraction step)
# ---------------------------------------------------------------------------

# Hacker News username rules: alphanumeric + underscore + hyphen, 2–15 chars
# (Firebase allows longer; 15 is the soft cap surfaced in the UI).
_HN_USER_TOKEN_RE = re.compile(r"\bhn/([A-Za-z0-9_-]{2,32})\b")


def _rewrite_hn_subjects(candidates: list[CandidateParticle], chunks: list[ChunkUnit]) -> None:
    """Canonicalise bare HN handles in candidates back to the ``hn/<author>`` form.

    Mirrors :func:`particles.extraction.reddit._rewrite_reddit_subjects`.
    The HN extractor renders comment lines as ``hn/{author}: <body>`` so
    the LLM usually emits ``hn/{author}`` as a subject token — but
    occasionally strips the prefix, which would land the subject at the
    Obsidian vault root instead of under ``news.ycombinator.com/hn/<name>``.
    """
    known: set[str] = set()
    for chunk in chunks:
        known.update(m.group(1) for m in _HN_USER_TOKEN_RE.finditer(chunk.chunk_text))

    for c in candidates:
        seen: set[str] = set()
        out: list[str] = []
        for s in c.subjects:
            if isinstance(s, str) and not s.startswith("hn/") and s in known:
                s = f"hn/{s}"
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        c.subjects = out
