# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""GitHub gist extractor.

Reads ``GITHUB_GIST`` blobs — the GitHub gists-API response stored as JSON
plus an embedded ``comments`` array — into LLM-prose chunks (body chunk +
adaptive comment chunks). Substantive commenters are attributed
via content-token overlap and ``gh/{login}`` → ``github:{login}`` subject
rewriting so the Obsidian exporter routes each commenter to its own vault
page.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, cast

from particles.config import get_config
from particles.core.schema import Snapshot, UncertaintyNature
from particles.extraction.general import CandidateParticle, ExtractionResult
from particles.extraction.github._shared import (
    APPLICABILITY_GIST,
    DEFAULT_TRUST_WEIGHT_GIST,
    EXTRACTOR_ID_GIST,
    EXTRACTOR_VERSION_GIST,
    SOURCE_TYPE_GIST,
    _inject_subjects,
)
from particles.extraction.incremental import ChunkUnit, extract_with_carry_forward

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class GitHubGistExtractor:
    """Extract claim-granularity particles from a stored GITHUB_GIST JSON blob."""

    EXTRACTOR_ID: str = EXTRACTOR_ID_GIST
    EXTRACTOR_VERSION: str = EXTRACTOR_VERSION_GIST
    DEFAULT_TRUST_WEIGHT: float = DEFAULT_TRUST_WEIGHT_GIST
    APPLICABILITY = APPLICABILITY_GIST

    def accepts(self, source_type: str) -> bool:
        return source_type == SOURCE_TYPE_GIST

    async def extract(
        self,
        snapshot: Snapshot,
        content: bytes,
        **kwargs: object,
    ) -> ExtractionResult:
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            return ExtractionResult(quality_notes=[f"JSON parse error: {exc}"])
        if not isinstance(data, dict):
            return ExtractionResult(quality_notes=["Gist blob is not a JSON object"])

        files = data.get("files") or {}
        if not isinstance(files, dict):
            files = {}
        description = (
            (data.get("description") or "").strip()
            if isinstance(data.get("description"), str)
            else ""
        )
        owner_block = data.get("owner") if isinstance(data.get("owner"), dict) else {}
        owner_login = (owner_block or {}).get("login", "")
        if not isinstance(owner_login, str):
            owner_login = ""

        comments_raw = data.get("comments") or []
        comments: list[dict[str, Any]] = (
            [c for c in comments_raw if isinstance(c, dict)]
            if isinstance(comments_raw, list)
            else []
        )

        body_text = _build_gist_body_text(description, owner_login, files)
        if not body_text and not comments:
            return ExtractionResult(quality_notes=["Empty gist content"])

        extras: list[str] = []
        if owner_login:
            extras.append(f"github:{owner_login}")
        if description and len(description) <= 120:
            extras.append(description)

        gh_cfg = get_config().github
        ext_cfg = get_config().extraction

        chunks = _build_gist_chunks(
            body_text=body_text,
            description=description,
            owner_login=owner_login,
            comments=comments,
            top_comment_count=gh_cfg.gist_top_comment_count,
            body_limit=gh_cfg.gist_comment_body_limit,
            single_call_threshold=ext_cfg.single_call_threshold_chars,
            chunk_chars=ext_cfg.comment_chunk_chars,
        )

        session = cast("AsyncSession | None", kwargs.get("session"))
        corpus_entry_id = cast("str | None", kwargs.get("corpus_entry_id"))
        # Reindex threads its supersede set so carry-forward treats the
        # marked particles as absent (see extract_with_carry_forward).
        sup_obj = kwargs.get("supersede_ids")
        result = await extract_with_carry_forward(
            session=session,
            chunks=chunks,
            corpus_entry_id=corpus_entry_id,
            extractor_id=EXTRACTOR_ID_GIST,
            extractor_version=EXTRACTOR_VERSION_GIST,
            max_llm_calls=ext_cfg.max_llm_calls_per_source,
            supersede_ids=sup_obj if isinstance(sup_obj, frozenset) else frozenset(),
        )

        for c in result.candidates:
            _inject_subjects(c, extras)

        _rewrite_gh_subjects(result.candidates, [c.chunk_text for c in chunks])
        _attribute_by_overlap(result.candidates, comments)
        if gh_cfg.gist_synthesize_commenter_particles:
            synthesized = _synthesize_missing_commenter_particles(
                result.candidates,
                comments,
                gh_cfg.gist_substantive_min_tokens,
            )
            result.candidates.extend(synthesized)
        _normalize_gh_in_content(result.candidates)

        if len(comments) > gh_cfg.gist_top_comment_count:
            result.quality_notes.append(
                f"GIST_COMMENT_LIMIT_HIT: {len(comments)} comments in envelope, "
                f"capped at {gh_cfg.gist_top_comment_count} for LLM extraction"
            )

        return result


# ---------------------------------------------------------------------------
# Gist body + comment chunk assembly
# ---------------------------------------------------------------------------


def _build_gist_body_text(description: str, owner_login: str, files: dict[str, Any]) -> str:
    parts: list[str] = []
    if description:
        parts.append(f"Gist description: {description}")
    if owner_login:
        parts.append(f"Gist by gh/{owner_login}")
    for filename, info in files.items():
        if not isinstance(filename, str):
            continue
        body = (info or {}).get("content", "") if isinstance(info, dict) else ""
        if not isinstance(body, str):
            body = ""
        parts.append(f"## {filename}\n\n{body}")
    return "\n\n".join(parts).strip()


def _render_comment_lines(comments: list[dict[str, Any]], body_limit: int) -> list[str]:
    lines: list[str] = []
    for c in comments:
        user = c.get("user") if isinstance(c.get("user"), dict) else {}
        login = (user or {}).get("login", "")
        body = c.get("body", "")
        if not isinstance(login, str) or not isinstance(body, str):
            continue
        if not login or not body.strip():
            continue
        lines.append(f"gh/{login}: {body[:body_limit]}")
    return lines


def _chunk_comments_by_size(
    comments: list[dict[str, Any]], chunk_chars: int, body_limit: int
) -> list[list[dict[str, Any]]]:
    """Group comments into chunks whose rendered text fits within ``chunk_chars``.

    A comment that exceeds the budget on its own still gets its own chunk —
    the body_limit truncation upstream means this is bounded.
    """
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_size = 0
    for c in comments:
        body = c.get("body") or ""
        rendered_size = min(len(body), body_limit) + 30  # "gh/{login}: " overhead
        if current and current_size + rendered_size > chunk_chars:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(c)
        current_size += rendered_size
    if current:
        chunks.append(current)
    return chunks


def _build_gist_chunks(
    body_text: str,
    description: str,
    owner_login: str,
    comments: list[dict[str, Any]],
    top_comment_count: int,
    body_limit: int,
    single_call_threshold: int,
    chunk_chars: int,
) -> list[ChunkUnit]:
    """Produce ChunkUnits for the gist (body + 0-or-more comment chunks).

    The body chunk is always its own unit so a re-deposit with unchanged
    body but new comments only re-runs the comment chunks. Comments are
    grouped adaptively: a single chunk if the rendered text fits under
    ``single_call_threshold`` characters, otherwise split into chunks of
    ``chunk_chars`` each.

    A context header is prepended to comment chunks so the LLM can resolve
    paraphrased referents ("this technique", "the recipe") back to the
    gist's topic.
    """
    chunks: list[ChunkUnit] = []
    if body_text:
        chunks.append(ChunkUnit(chunk_id="body", chunk_text=body_text))

    if not comments:
        return chunks

    shown = comments[:top_comment_count]
    rendered_size = sum(min(len(c.get("body") or ""), body_limit) + 30 for c in shown)

    header_parts: list[str] = []
    if owner_login:
        header_parts.append(f"Context: comments on a gist by gh/{owner_login}")
    if description:
        header_parts.append(f'titled "{description[:120]}"')
    header = "# " + ", ".join(header_parts) + "." if header_parts else "# Gist comments"

    if rendered_size <= single_call_threshold:
        parts = [header, "## Comments", *_render_comment_lines(shown, body_limit)]
        chunks.append(ChunkUnit(chunk_id="comments", chunk_text="\n\n".join(parts)))
        return chunks

    sub_chunks = _chunk_comments_by_size(shown, chunk_chars, body_limit)
    total = len(sub_chunks)
    for i, sub_chunk in enumerate(sub_chunks, start=1):
        text_parts = [
            header,
            f"## Comments (chunk {i}/{total})",
            *_render_comment_lines(sub_chunk, body_limit),
        ]
        chunks.append(
            ChunkUnit(
                chunk_id=f"comments_{i}_of_{total}",
                chunk_text="\n\n".join(text_parts),
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Post-extraction subject + content rewriting
# ---------------------------------------------------------------------------


_GH_LOGIN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,38}$")
# Unanchored variant for in-content substitution. GitHub login pattern only —
# the trailing pattern won't match arbitrary paths like "gh/api/repos" because
# `/` and `.` aren't in the login charset.
_GH_INLINE_LOGIN_RE = re.compile(r"\bgh/([A-Za-z0-9][A-Za-z0-9-]{0,38})\b")


def _normalize_gh_in_content(candidates: list[CandidateParticle]) -> None:
    """Rewrite ``gh/{login}`` → ``@{login}`` in candidate *content* text.

    The LLM prompt renders comments as ``gh/{login}: <body>`` so the model
    emits the login as a subject token; the same token also leaks into the
    extracted claim's content ("gh/7TIN's project is available at…"), which
    reads awkwardly. This pass rewrites in-content occurrences to GitHub's
    native ``@`` mention syntax. Subjects are rewritten separately by
    :func:`_rewrite_gh_subjects` (``gh/`` → ``github:`` for routing).
    """
    for c in candidates:
        c.content = _GH_INLINE_LOGIN_RE.sub(r"@\1", c.content)


def _rewrite_gh_subjects(
    candidates: list[CandidateParticle], chunks: list[str] | None = None
) -> None:
    """Rewrite ``gh/{login}`` subjects emitted by the LLM into ``github:{login}``.

    The gist extractor renders commenters as ``gh/{login}: <body>`` (mirroring
    reddit's ``u/{author}:`` pattern) so the LLM usually emits ``gh/{login}``
    as a subject token. We canonicalise to ``github:{login}`` here so the
    Obsidian exporter routes the subject to ``github.com/{login}.md`` via the
    existing rule.

    The LLM occasionally strips the ``gh/`` prefix and emits a bare login
    instead. When ``chunks`` is provided, scan it for ``gh/{login}``
    tokens to learn which logins actually exist in the source, then
    rewrite bare matches in candidates back to ``github:{login}``. This
    keeps a commenter from landing at ``<vault>/7TIN.md`` when they
    should be at ``<vault>/github.com/7TIN.md``.
    """
    known_logins: set[str] = set()
    if chunks:
        for chunk in chunks:
            for m in _GH_INLINE_LOGIN_RE.finditer(chunk):
                login = m.group(1)
                if _GH_LOGIN_RE.match(login):
                    known_logins.add(login)

    for c in candidates:
        seen: set[str] = set()
        out: list[str] = []
        for s in c.subjects:
            if isinstance(s, str):
                if s.startswith("gh/"):
                    login = s[len("gh/") :]
                    if _GH_LOGIN_RE.match(login):
                        s = f"github:{login}"
                elif s in known_logins:
                    s = f"github:{s}"
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        c.subjects = out


# Common English words that carry no claim-content signal. Kept short on
# purpose: longer stopword lists hurt recall here more than they help.
_ATTRIBUTION_STOPWORDS = frozenset(
    [
        "about",
        "after",
        "again",
        "also",
        "another",
        "because",
        "been",
        "before",
        "being",
        "both",
        "each",
        "from",
        "have",
        "having",
        "here",
        "into",
        "just",
        "like",
        "more",
        "most",
        "much",
        "only",
        "other",
        "over",
        "same",
        "should",
        "some",
        "such",
        "than",
        "that",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "thus",
        "very",
        "want",
        "well",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "with",
        "would",
        "your",
        "yours",
        "thanks",
        "thank",
        "andrej",
        "great",
        "really",
        "many",
        "make",
        "made",
        "much",
    ]
)


def _content_tokens(text: str) -> set[str]:
    """Lowercase content-bearing tokens of length ≥ 4, stopwords removed.

    Used purely for source-attribution overlap scoring — not for any
    semantic claim about the text. The minimum-length filter drops most
    function words and tiny numbers; the stopword filter drops the
    high-frequency English residue that survives.
    """
    tokens = re.findall(r"[a-z][a-z0-9_]{3,}", text.lower())
    return {t for t in tokens if t not in _ATTRIBUTION_STOPWORDS}


def _attribute_by_overlap(candidates: list[CandidateParticle], comments: list[Any]) -> None:
    """Append ``github:{login}`` as an auxiliary subject when a candidate's
    content overlaps a comment's body.

    The LLM's prompt asks for subjects the claim is *about*; for general
    technical claims the commenter is not the subject. This pass adds an
    orthogonal *source* attribution so each substantive commenter ends up
    with a vault page (per "Gist comments are rendered inline"
    note).

    Heuristic: tokenise candidate and each comment to content-bearing words
    (length ≥ 4, stopwords removed). A comment claims the candidate when
    they share ≥ 2 content tokens *and* the shared set covers ≥ 20 % of the
    candidate's tokens. Thresholds are deliberately loose — over-attribution
    only adds an extra subject; it does not corrupt the topical subjects.

    Empty / pleasantry comments ("thanks!") naturally drop out because the
    stopword filter strips them to empty token sets.
    """
    if not comments:
        return
    prepared: list[tuple[str, set[str]]] = []
    for c in comments:
        if not isinstance(c, dict):
            continue
        user = c.get("user") if isinstance(c.get("user"), dict) else {}
        login = (user or {}).get("login")
        body = c.get("body")
        if not isinstance(login, str) or not isinstance(body, str) or not login or not body:
            continue
        tokens = _content_tokens(body)
        if len(tokens) < 2:
            continue
        prepared.append((login, tokens))
    if not prepared:
        return

    for candidate in candidates:
        cand_tokens = _content_tokens(candidate.content)
        if len(cand_tokens) < 2:
            continue
        best_login: str | None = None
        best_score = 0.0
        for login, ct in prepared:
            shared = cand_tokens & ct
            if len(shared) < 2:
                continue
            coverage = len(shared) / len(cand_tokens)
            if coverage >= 0.20 and coverage > best_score:
                best_score = coverage
                best_login = login
        if best_login:
            subj = f"github:{best_login}"
            if subj not in candidate.subjects:
                candidate.subjects.append(subj)


def _synthesize_missing_commenter_particles(
    candidates: list[CandidateParticle],
    comments: list[Any],
    min_tokens: int,
) -> list[CandidateParticle]:
    """Fallback: ensure every substantive commenter ends up as a subject.

    The LLM extracts substantive technical claims and the overlap pass
    attributes them to commenters when possible; this fallback fills the
    remaining gap. For each commenter whose comment clears the substantive
    threshold (≥ ``min_tokens`` content tokens) but who has *no* candidate
    tagging them yet, we synthesize one particle quoting their comment so
    the Obsidian export renders ``github.com/{login}.md``.

    Pleasantry comments ("thank you Andrej!") fall below the token floor
    and produce no subject — consistent with the user-stated principle that
    such commenters do not need vault pages.
    """
    represented: set[str] = set()
    for c in candidates:
        for s in c.subjects:
            if isinstance(s, str) and s.startswith("github:"):
                represented.add(s)

    synthesized: list[CandidateParticle] = []
    seen_logins: set[str] = set()
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        user = comment.get("user") if isinstance(comment.get("user"), dict) else {}
        login = (user or {}).get("login")
        body = comment.get("body")
        if not isinstance(login, str) or not isinstance(body, str):
            continue
        if not login or login in seen_logins:
            continue
        body_stripped = body.strip()
        if not body_stripped:
            continue
        if len(_content_tokens(body_stripped)) < min_tokens:
            continue
        subj = f"github:{login}"
        if subj in represented:
            seen_logins.add(login)
            continue
        snippet = body_stripped if len(body_stripped) <= 300 else body_stripped[:297] + "..."
        synthesized.append(
            CandidateParticle(
                content=f'@{login} commented on this gist: "{snippet}"',
                confidence_value=0.85,
                uncertainty_nature=UncertaintyNature.EPISTEMIC,
                subjects=[subj],
            )
        )
        represented.add(subj)
        seen_logins.add(login)
    return synthesized
