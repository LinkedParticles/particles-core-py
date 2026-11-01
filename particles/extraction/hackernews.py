# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

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
    paragraph_spans,
)
from particles.extraction.incremental import ChunkUnit, extract_with_carry_forward
from particles.http import get_capped

log = logging.getLogger(__name__)

SOURCE_TYPE = "HACKERNEWS_THREAD"
EXTRACTOR_ID = "hackernews-extractor"
EXTRACTOR_VERSION = "0.1.1"
# 0.1.1: a story text past single_call_threshold_chars is split into parts of
#        comment_chunk_chars, and every comment line is clipped to that
#        budget, so no model call is larger than the configured budget plus a
#        header (security review 2026-10-30, AB18).
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


def _is_item_id(value: object) -> bool:
    """Whether ``value`` is an HN item id: a non-negative ``int``, never a ``bool``.

    A ``kids`` entry comes from the API or the stored blob. One that is not an
    integer went into the item URL as text, so a path-shaped one reached
    another path on the API host (security review 2026-10-30, AB63).
    """
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _api_url(item_id: int) -> str:
    """Return the Firebase API URL for a single HN item.

    Raises:
        ValueError: ``item_id`` is not an HN item id (:func:`_is_item_id`).
    """
    if not _is_item_id(item_id):
        raise ValueError(f"not a Hacker News item id: {item_id!r}")
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
    except (ValueError, json.JSONDecodeError, RecursionError):
        # A body nested past the decoder's depth reads as no item
        # (security review 2026-10-28-2, AA18).
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
    order) up to ``max_comments`` comment fetches. Deeply-nested but
    low-engagement threads will be truncated at the cap; the extractor
    surfaces a ``HN_COMMENT_LIMIT_HIT`` quality note in that case. The walk
    keeps a stack rather than recursing, fetches each id once, and counts
    every fetch against the cap, so ``kids`` that revisit an item cannot
    refetch it without end (security review 2026-10-28-2, AA57).
    """
    story = await _fetch_item(client, root_id)
    if story is None:
        raise ValueError(f"Hacker News item {root_id} not found or returned non-dict JSON")

    comments: dict[str, dict[str, Any]] = {}
    seen = {str(root_id)}
    fetches = 0

    root_kids = story.get("kids")
    stack: list[int] = (
        [kid for kid in reversed(root_kids) if _is_item_id(kid)]
        if isinstance(root_kids, list)
        else []
    )
    while stack and fetches < max_comments:
        child_id = stack.pop()
        if str(child_id) in seen:
            continue
        seen.add(str(child_id))
        fetches += 1
        child = await _fetch_item(client, child_id)
        if child is None:
            continue
        comments[str(child_id)] = child
        sub_kids = child.get("kids")
        if isinstance(sub_kids, list):
            stack.extend(kid for kid in reversed(sub_kids) if _is_item_id(kid))

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
        except (ValueError, RecursionError) as exc:
            # RecursionError: a blob nested past the decoder's depth
            # (security review 2026-10-30, AB17).
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


# Characters a header line of the story chunk may hold: past HN's own
# 80-character title limit, and the URL line past any URL a browser keeps.
_HEADER_LINE_CHARS = 400
_URL_LINE_CHARS = 2100


def _render_hn_story_head(story: dict[str, Any]) -> list[str]:
    """The story's title, byline and URL lines, each clipped to its bound."""
    title = story.get("title", "") or ""
    author = story.get("by", "") or ""
    score = story.get("score", 0) or 0
    url = story.get("url") or ""
    lines = [
        _clip(f"HACKER NEWS THREAD: {title}", _HEADER_LINE_CHARS),
        _clip(f"Posted by hn/{author} [score: {score}]", _HEADER_LINE_CHARS),
    ]
    if url:
        lines.append(_clip(f"External URL: {url}", _URL_LINE_CHARS))
    return lines


def _clip(text: str, limit: int) -> str:
    """``text`` cut to ``limit`` characters, marked when it was cut."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _render_hn_story_body(story: dict[str, Any]) -> str:
    """The story chunk's text — title + optional self-post body. Excludes comments.

    Stable across re-deposits when the story body is unedited, so this is
    the prime carry-forward candidate when only comments change (mirrors
    the Reddit body-chunk strategy).
    """
    body = (story.get("text") or "").strip()
    lines = _render_hn_story_head(story)
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

    The map is the stored blob's bytes, so the walk keeps a stack rather than
    recursing and visits each id once: ``kids`` that revisit a comment cannot
    loop (security review 2026-10-28-2, AA57).
    """
    out: list[tuple[int, dict[str, Any]]] = []
    seen: set[str] = set()

    def _children(node: dict[str, Any], depth: int) -> list[tuple[str, int]]:
        kids = node.get("kids")
        if not isinstance(kids, list):
            return []
        return [(str(kid_id), depth) for kid_id in reversed(kids) if _is_item_id(kid_id)]

    stack = _children(story, 0)
    while stack:
        key, depth = stack.pop()
        node = comments_map.get(key)
        if node is None or key in seen:
            continue
        seen.add(key)
        text = (node.get("text") or "").strip()
        dead = bool(node.get("dead"))
        deleted = bool(node.get("deleted"))
        score = node.get("score")
        score_ok = not isinstance(score, int) or score >= min_score
        if text and not dead and not deleted and score_ok:
            out.append((depth, node))
        stack.extend(_children(node, depth + 1))

    return out


def _render_hn_comment_lines(
    comments: list[tuple[int, dict[str, Any]]],
    indent_per_level: int,
    line_limit: int | None = None,
) -> list[str]:
    """Render an ordered ``(depth, comment)`` list as indented prose lines.

    Each line is clipped to ``line_limit`` when one is given: a single
    comment longer than the chunk budget was one model call of its own
    (security review 2026-10-30, AB18).
    """
    lines: list[str] = []
    for depth, c in comments:
        author = c.get("by") or "anon"
        text = (c.get("text") or "").replace("\n", " ").strip()
        indent = " " * (depth * indent_per_level)
        line = f"{indent}hn/{author}: {text}"
        lines.append(line if line_limit is None else _clip(line, line_limit))
    return lines


def _chunk_lines_by_size(lines: list[str], chunk_chars: int) -> list[list[str]]:
    """Greedy line grouping such that each group fits within ``chunk_chars``."""
    chunks: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for line in lines:
        size = len(line) + 1  # +1 for the newline separator
        if current and current_size + size > chunk_chars:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(line)
        current_size += size
    if current:
        chunks.append(current)
    return chunks


def _build_hn_story_chunks(
    story: dict[str, Any], single_call_threshold: int, chunk_chars: int
) -> list[ChunkUnit]:
    """The story chunk, or the story text split into parts when it passes the threshold.

    A story text longer than ``single_call_threshold`` went to the model in
    one call (security review 2026-10-30, AB18). It is split on paragraph
    breaks into parts of at most ``chunk_chars`` (and never more than the
    threshold), each under the story's title, byline and URL.
    """
    body_text = _render_hn_story_body(story)
    if not body_text.strip():
        return []
    if len(body_text) <= single_call_threshold:
        return [ChunkUnit(chunk_id="story", chunk_text=body_text)]
    body = (story.get("text") or "").strip()
    head = "\n".join(_render_hn_story_head(story))
    spans = paragraph_spans(body, max(1, min(chunk_chars, single_call_threshold)))
    total = len(spans)
    return [
        ChunkUnit(
            chunk_id=f"story_{i}_of_{total}",
            chunk_text=f"{head}\nSTORY TEXT (part {i}/{total}):\n\n{body[start:end]}",
        )
        for i, (start, end) in enumerate(spans, start=1)
    ]


def _build_hn_chunks(
    story: dict[str, Any],
    comments_map: dict[str, dict[str, Any]],
    indent_per_level: int,
    min_score: int,
    single_call_threshold: int,
    chunk_chars: int,
) -> list[ChunkUnit]:
    """Produce ChunkUnits for an HN thread (body + 0-or-more comment chunks).

    The story is its own unit, split into parts past the threshold. Comment
    chunks are sized adaptively, mirroring the Reddit / gist strategy so the
    chunk-hash carry-forward can reuse particles when only one
    chunk's text changes. Every comment line is clipped to the chunk budget,
    so no chunk is larger than the budget plus its header (security review
    2026-10-30, AB18).
    """
    chunks = _build_hn_story_chunks(story, single_call_threshold, chunk_chars)

    flat_comments = _walk_comments_dfs(story, comments_map, min_score)
    if not flat_comments:
        return chunks

    title = (story.get("title") or "")[:120]
    header = f"# Context: comments on Hacker News thread — {title}"
    lines = _render_hn_comment_lines(
        flat_comments, indent_per_level, max(1, min(chunk_chars, single_call_threshold))
    )

    rendered_size = sum(len(line) + 1 for line in lines)

    if rendered_size <= single_call_threshold:
        parts = [header, "COMMENTS:", *lines]
        chunks.append(ChunkUnit(chunk_id="comments", chunk_text="\n".join(parts)))
        return chunks

    sub_chunks = _chunk_lines_by_size(lines, chunk_chars)
    total = len(sub_chunks)
    for i, sub_chunk in enumerate(sub_chunks, start=1):
        text_parts = [header, f"COMMENTS (chunk {i}/{total}):", *sub_chunk]
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
