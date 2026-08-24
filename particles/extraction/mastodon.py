# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Mastodon thread extractor and importer (importer naming).

MastodonImporter — accepts three URL shapes that operators copy from their
                   browser or the public API:

                   - ``https://<instance>/@<user>/<status_id>`` — viewing a
                     status local to ``<instance>``.
                   - ``https://<instance>/@<user>@<remote>/<status_id>`` — the
                     viewing instance is showing a federated status whose home
                     is ``<remote>``. **v1 routes the fetch to <remote>**, not
                     the viewing instance, so the corpus stays addressed by
                     the canonical bytes the home instance serves. The known
                     limitation: if the same status is later viewed via a
                     different mirror, the bytes — and thus the
                     ``content_hash`` — may differ, and the corpus
                     will hold two entries pointing at semantically the same
                     post. This is intentional for v1; a federation-aware
                     dedup pass would have to live above the corpus layer.
                   - ``https://<instance>/api/v1/statuses/<status_id>`` — the
                     raw API URL (rare from a human; emitted by automation).

                   The importer fetches the status + its context tree
                   (``ancestors`` + ``descendants``) via two unauthenticated
                   public API calls and stores the assembled JSON blob as a
                   single corpus entry of source type ``MASTODON_THREAD``.

MastodonExtractor — parses the blob, renders the root status's spoiler_text
                    + content as the headline, the ancestor chain above it,
                    and the descendant reply tree below it as indented prose
                    handed to the chunked LLM extraction helper.
                    Additionally synthesises a single *story-meta*
                    CandidateParticle whose ``properties`` dict follows URI-prefix convention with the dual-emission rule
                    (``mastodon:hasFavouritesCount`` + ``social:hasScore``,
                    etc.).

A second schema stress test for the cross-source
``properties`` convention. The Mastodon structured metadata is richer than
HN's: in addition to the score / author / id / replies trio that maps
cleanly to ``social:*``, Mastodon carries a content-warning field
(``spoiler_text``) and a boost / reblog indicator that HN has no
counterpart for. Both are stamped on the story-meta particle as
platform-specific keys (``mastodon:hasSpoilerText``, ``mastodon:isReblog``)
with no current cross-platform equivalent.
"""

from __future__ import annotations

import html
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

SOURCE_TYPE = "MASTODON_THREAD"
EXTRACTOR_ID = "mastodon-extractor"
EXTRACTOR_VERSION = "0.2.0"
# 0.2.0: capture ``mastodon:reblogOfStatusId``,
# ``mastodon:reblogOfAccountAcct``, ``mastodon:reblogOfStatusUri``
# when ``status.reblog`` is non-null so a future ``BOOSTS``-relation
# activation can backfill the edge from properties. Existing snapshots
# extracted under 0.1.0 are re-extracted via ADR-0057 carry-forward
# (most chunk hashes hold; only the meta-particle row changes).
DEFAULT_TRUST_WEIGHT = 0.50  # UGC starting weight for Mastodon

APPLICABILITY = [
    ApplicabilityClause(
        keyword="MUST",
        # Mastodon is a protocol with many instances rather than a single
        # site; ``joinmastodon.org`` is the canonical project home and the
        # closest stable identifier for the platform. Per-instance URIs
        # would force a clause-per-instance, which doesn't compose with
        # the Applicability table.
        domain_uri="https://joinmastodon.org",
        domain_label="Mastodon",
        source_types=[SOURCE_TYPE],
    )
]


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------

# Examples that match:
#   https://mastodon.social/@user/123456789012345678
#   https://fosstodon.org/@user@mastodon.social/123456789012345678
# The status id segment is constrained to digits; Mastodon snowflake-style
# IDs are decimal strings (often 18+ digits). Octothorpes / queries / etc.
# tolerated as a trailing fragment.
_MASTODON_STATUS_URL_RE = re.compile(
    r"^https?://(?P<viewing>[^/]+)/@(?P<user>[A-Za-z0-9_]+)"
    r"(?:@(?P<remote>[A-Za-z0-9_.\-]+))?"
    r"/(?P<id>\d+)(?:[/?#].*)?$"
)

# Raw API form: https://<instance>/api/v1/statuses/<id>
_MASTODON_API_URL_RE = re.compile(
    r"^https?://(?P<host>[^/]+)/api/v1/statuses/(?P<id>\d+)(?:\?.*)?$"
)


def _parse_status_route(url: str) -> tuple[str, str, str | None] | None:
    """Return ``(home_instance, status_id, account_acct)`` parsed from ``url``.

    The first element is the host the importer SHOULD fetch from — for the
    ``@user@remote`` form that is ``remote``, not the viewing instance.
    The third element is the operator-visible account handle (``user`` for
    local; ``user@remote`` for cross-instance). Returns ``None`` when the
    URL is not a recognised Mastodon shape.
    """
    m = _MASTODON_STATUS_URL_RE.match(url)
    if m:
        viewing = m.group("viewing")
        user = m.group("user")
        remote = m.group("remote")
        status_id = m.group("id")
        if remote:
            # Cross-instance view — fetch from the user's home instance so
            # the corpus is addressed by the canonical bytes.
            return (remote, status_id, f"{user}@{remote}")
        return (viewing, status_id, user)
    m = _MASTODON_API_URL_RE.match(url)
    if m:
        host = m.group("host")
        status_id = m.group("id")
        # No account handle without an extra fetch — defer to the API
        # response's ``account.acct`` field at deposit time.
        return (host, status_id, None)
    return None


# ---------------------------------------------------------------------------
# API fetch helpers
# ---------------------------------------------------------------------------


def _status_api_url(instance: str, status_id: str) -> str:
    """Return the public-status API URL for ``status_id`` on ``instance``."""
    return f"https://{instance}/api/v1/statuses/{status_id}"


def _context_api_url(instance: str, status_id: str) -> str:
    """Return the status-context API URL (``ancestors`` + ``descendants``)."""
    return f"https://{instance}/api/v1/statuses/{status_id}/context"


async def _fetch_json(client: Any, url: str) -> dict[str, Any] | None:
    """GET ``url`` and return the decoded JSON dict (or ``None`` on a non-dict response)."""
    resp = await get_capped(client, url)
    resp.raise_for_status()
    try:
        data = resp.json()
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


async def _fetch_thread(
    client: Any, instance: str, status_id: str, max_replies: int
) -> dict[str, Any]:
    """Fetch the status + its context tree as a single JSON blob.

    The on-disk format is:

    ```json
    {
        "status": {<full Status entity, possibly with reblog>},
        "context": {
            "ancestors": [<Status entity>, ...],
            "descendants": [<Status entity>, ...]
        },
        "instance": "<home instance hostname>"
    }
    ```

    ``ancestors`` + ``descendants`` are capped at ``max_replies`` total
    (ancestors are walked entirely first because the reply chain UP is
    typically short; remaining budget goes to descendants). When the cap
    fires, the extractor surfaces a ``MASTODON_REPLY_LIMIT_HIT`` quality
    note so operators know the captured slice is truncated.
    """
    status = await _fetch_json(client, _status_api_url(instance, status_id))
    if status is None:
        raise ValueError(f"Mastodon status {status_id}@{instance} not found or non-dict JSON")

    context_raw = await _fetch_json(client, _context_api_url(instance, status_id))
    ancestors_raw: list[Any] = []
    descendants_raw: list[Any] = []
    if context_raw is not None:
        anc = context_raw.get("ancestors")
        if isinstance(anc, list):
            ancestors_raw = anc
        desc = context_raw.get("descendants")
        if isinstance(desc, list):
            descendants_raw = desc

    # Keep all ancestors (reply chains UP are short) and trim descendants
    # to fit the remaining budget.
    ancestors = [a for a in ancestors_raw if isinstance(a, dict)]
    descendants_all = [d for d in descendants_raw if isinstance(d, dict)]
    remaining = max(0, max_replies - len(ancestors))
    descendants = descendants_all[:remaining]

    return {
        "status": status,
        "context": {"ancestors": ancestors, "descendants": descendants},
        "instance": instance,
    }


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class MastodonExtractor:
    """Extract claim-granularity particles from a stored MASTODON_THREAD JSON blob."""

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
        """Three-step composition mirrored from HN.

        Like the HN extractor, this one parses once and feeds the parsed
        status to both the LLM-prose path and the meta-particle
        synthesiser, so the dual-emission ``properties`` on
        the synthesised meta particle stay aligned with what the LLM saw.
        """
        doc, status, instance, root_id = self._normalise(content, snapshot)
        result = await self._extract_claims(doc, **kwargs)
        story_meta = _build_status_meta_candidate(status, instance, root_id, doc.injected_subjects)
        if story_meta is not None:
            result.candidates.insert(0, story_meta)
        return result

    def _normalise(
        self, content: bytes, snapshot: Snapshot
    ) -> tuple[NormalizedDocument, dict[str, Any], str, str]:
        """Source-format parsing only: JSON → prose chunks + status dict.

        Returns ``(NormalizedDocument, status_dict, home_instance, root_id)``.
        The instance string is the importer-stamped ``"instance"`` field on
        the blob (e.g. ``"mastodon.social"``); the meta-particle
        synthesiser needs it for ``mastodon:hasInstance``. The root_id
        is computed by walking the ``ancestors`` list to its head — the
        synthesiser cannot do it alone because it only sees ``status``.
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            return (NormalizedDocument(quality_notes=[f"JSON parse error: {exc}"]), {}, "", "")

        if not isinstance(data, dict):
            return (
                NormalizedDocument(quality_notes=["Top-level JSON value is not an object"]),
                {},
                "",
                "",
            )

        status = data.get("status")
        if not isinstance(status, dict):
            return (
                NormalizedDocument(
                    quality_notes=["Could not find status data in Mastodon JSON blob"]
                ),
                {},
                "",
                "",
            )

        instance_raw = data.get("instance")
        instance: str = instance_raw if isinstance(instance_raw, str) else ""

        context_obj = data.get("context")
        context: dict[str, Any] = context_obj if isinstance(context_obj, dict) else {}
        ancestors_raw = context.get("ancestors")
        descendants_raw = context.get("descendants")
        ancestors: list[dict[str, Any]] = (
            [a for a in ancestors_raw if isinstance(a, dict)]
            if isinstance(ancestors_raw, list)
            else []
        )
        descendants: list[dict[str, Any]] = (
            [d for d in descendants_raw if isinstance(d, dict)]
            if isinstance(descendants_raw, list)
            else []
        )

        cfg = get_config()
        chunks = _build_mastodon_chunks(
            status=status,
            ancestors=ancestors,
            descendants=descendants,
            instance=instance,
            indent_per_level=cfg.mastodon.reply_indent,
            min_favourites=cfg.mastodon.min_reply_favourites,
            single_call_threshold=cfg.extraction.single_call_threshold_chars,
            chunk_chars=cfg.extraction.comment_chunk_chars,
        )

        quality_notes: list[str] = []
        captured = len(ancestors) + len(descendants)
        if captured >= cfg.mastodon.max_replies:
            quality_notes.append(
                f"MASTODON_REPLY_LIMIT_HIT: importer captured {captured} context items "
                f"(at the mastodon.max_replies={cfg.mastodon.max_replies} cap). "
                "Raise the cap to capture the full thread."
            )

        injected: list[str] = []
        acct = _account_acct(status)
        if acct:
            injected.append(f"mastodon/{acct}")

        root_id = _root_id_from_ancestors(status, ancestors)

        return (
            NormalizedDocument(
                chunks=chunks,
                author_id=snapshot.author_id,
                content_published_at=snapshot.content_published_at,
                quality_notes=quality_notes,
                injected_subjects=injected,
            ),
            status,
            instance,
            root_id,
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

        # Stamp domain-injected subjects (the status author) on every LLM
        # candidate, mirroring HN.
        for c in result.candidates:
            for subj in doc.injected_subjects:
                if subj not in c.subjects:
                    c.subjects.append(subj)

        # Canonicalise bare Mastodon handles back to ``mastodon/<acct>`` so
        # the Obsidian exporter routes them under the expected vault path.
        _rewrite_mastodon_subjects(result.candidates, doc.chunks)

        result.quality_notes = [*doc.quality_notes, *result.quality_notes]
        return result


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

# Mastodon status content arrives as HTML (``<p>`` paragraphs, inline ``<a>``
# for mentions / hashtags / external links, ``<br>`` for line breaks within
# a paragraph). For the LLM prompt we want plain text with paragraph
# structure preserved. ``html2text`` is overkill (it's in the dependency
# set, but introduces Markdown markers we'd then have to undo), so a
# minimal in-module stripper is the cheaper option.

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_PARAGRAPH_END_RE = re.compile(r"</p\s*>", re.IGNORECASE)
_HTML_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


def _html_to_text(s: str) -> str:
    """Strip HTML tags while preserving paragraph + line breaks.

    Mastodon's status HTML is well-behaved: each paragraph is a ``<p>...</p>``
    and intra-paragraph line breaks are explicit ``<br>``. We collapse
    ``</p>`` to a double newline, ``<br>`` to a single newline, drop the
    remaining tags, then unescape HTML entities (``&amp;`` → ``&``).
    """
    s = _HTML_BR_RE.sub("\n", s)
    s = _HTML_PARAGRAPH_END_RE.sub("\n\n", s)
    s = _HTML_TAG_RE.sub("", s)
    s = html.unescape(s)
    # Collapse runs of >2 blank lines but otherwise preserve paragraph breaks.
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _account_acct(status: dict[str, Any]) -> str:
    """Return the ``account.acct`` for ``status`` or the empty string."""
    account = status.get("account")
    if isinstance(account, dict):
        acct = account.get("acct")
        if isinstance(acct, str):
            return acct
    return ""


def _effective_status(status: dict[str, Any]) -> dict[str, Any]:
    """If ``status`` is a boost (``reblog`` non-null), return the boosted status.

    Used by the prose renderer so the LLM sees the actual content of the
    boosted post rather than an empty ``content`` field. The meta-particle
    synthesiser does NOT call this — it operates on the wrapper status
    because ``isReblog`` / ``hasInstance`` describe the wrapper, not the
    original.
    """
    reblog = status.get("reblog")
    if isinstance(reblog, dict):
        return reblog
    return status


def _is_reblog(status: dict[str, Any]) -> bool:
    """Return True if ``status`` is a boost (the ``reblog`` field is non-null)."""
    return isinstance(status.get("reblog"), dict)


def _status_content_text(status: dict[str, Any]) -> str:
    """Return the HTML-stripped ``content`` of ``status`` (boost-aware)."""
    effective = _effective_status(status)
    raw = effective.get("content")
    if not isinstance(raw, str):
        return ""
    return _html_to_text(raw)


def _render_status_headline(status: dict[str, Any], instance: str) -> str:
    """Render the root status as the prose headline.

    Includes the spoiler text (CW) if present — even when the LLM is going
    to see the body anyway, the CW phrase often carries the author's own
    framing of what they wrote (``"long covid rant"``, ``"unpopular take
    on tabs vs spaces"``) which is a useful claim signal.
    """
    acct = _account_acct(status)
    favs = _safe_int(status.get("favourites_count"))
    reblogs = _safe_int(status.get("reblogs_count"))
    replies = _safe_int(status.get("replies_count"))
    spoiler_raw = status.get("spoiler_text")
    spoiler = spoiler_raw.strip() if isinstance(spoiler_raw, str) else ""
    boost_marker = " [BOOST]" if _is_reblog(status) else ""

    instance_label = instance or "mastodon"
    lines = [
        f"MASTODON THREAD on {instance_label}{boost_marker}",
        f"Posted by mastodon/{acct} [favourites: {favs}, boosts: {reblogs}, replies: {replies}]",
    ]
    if spoiler:
        lines.append(f"CW: {spoiler}")
    body = _status_content_text(status)
    if body:
        lines.append("")
        lines.append(body)
    return "\n".join(lines)


def _render_reply_line(status: dict[str, Any], depth: int, indent_per_level: int) -> str:
    """Render one reply as a single indented prose line."""
    indent = " " * (depth * indent_per_level)
    acct = _account_acct(status) or "anon"
    # Replies often span multiple paragraphs; for prose chunking we
    # flatten the whole body onto one line so depth indentation stays
    # readable. The LLM tolerates the loss of paragraph structure for
    # short comments better than it tolerates losing indentation cues.
    body = _status_content_text(status).replace("\n", " ").strip()
    spoiler_raw = status.get("spoiler_text")
    spoiler = spoiler_raw.strip() if isinstance(spoiler_raw, str) else ""
    if spoiler:
        body = f"[CW: {spoiler}] {body}"
    return f"{indent}mastodon/{acct}: {body}"


def _safe_int(v: object) -> int:
    """Coerce ``v`` to int or 0. Mastodon usually emits ints but defaults defensively."""
    return v if isinstance(v, int) and not isinstance(v, bool) else 0


def _filter_replies_by_favs(
    replies: list[dict[str, Any]], min_favourites: int
) -> list[dict[str, Any]]:
    """Drop replies whose ``favourites_count`` is below the threshold (0 = keep all)."""
    if min_favourites <= 0:
        return list(replies)
    return [r for r in replies if _safe_int(r.get("favourites_count")) >= min_favourites]


def _build_ancestor_lines(ancestors: list[dict[str, Any]], indent_per_level: int) -> list[str]:
    """Render the ancestor chain (oldest first) as flat depth-0 lines.

    Mastodon's API returns ``ancestors`` already in oldest-first order
    (root of the reply chain at index 0). We render each as depth 0 since
    they are all linear ancestors above the focal status — the indented
    tree starts with the focal status's descendants.
    """
    lines: list[str] = []
    for a in ancestors:
        lines.append(_render_reply_line(a, depth=0, indent_per_level=indent_per_level))
    return lines


def _build_descendant_lines(
    status: dict[str, Any],
    descendants: list[dict[str, Any]],
    indent_per_level: int,
    min_favourites: int,
) -> list[str]:
    """Render the descendant reply tree, depth-first, with indentation by depth.

    The Mastodon context API returns descendants as a flat list with
    ``in_reply_to_id`` pointing into the same list or to the focal
    status. We rebuild the tree by parent id, then walk it depth-first
    from the focal status's direct replies down.
    """
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for d in descendants:
        parent_id = d.get("in_reply_to_id")
        if isinstance(parent_id, str):
            by_parent.setdefault(parent_id, []).append(d)

    status_id_raw = status.get("id")
    if not isinstance(status_id_raw, str):
        return []

    lines: list[str] = []

    def _walk(parent_id: str, depth: int) -> None:
        children = _filter_replies_by_favs(by_parent.get(parent_id, []), min_favourites)
        for c in children:
            lines.append(_render_reply_line(c, depth, indent_per_level))
            cid = c.get("id")
            if isinstance(cid, str):
                _walk(cid, depth + 1)

    _walk(status_id_raw, depth=0)
    return lines


def _build_mastodon_chunks(
    status: dict[str, Any],
    ancestors: list[dict[str, Any]],
    descendants: list[dict[str, Any]],
    instance: str,
    indent_per_level: int,
    min_favourites: int,
    single_call_threshold: int,
    chunk_chars: int,
) -> list[ChunkUnit]:
    """Produce ChunkUnits for a Mastodon thread (status + 0-or-more reply chunks).

    The status chunk includes the ancestor chain so the LLM sees the
    conversational context the author was replying into. Reply chunks
    cover the descendant tree, sized adaptively to fit the per-chunk
    budget (parallel to the HN / gist / reddit strategy so the
    chunk-hash carry-forward reuses particles when only one
    chunk's text changes).
    """
    chunks: list[ChunkUnit] = []

    headline = _render_status_headline(status, instance)
    ancestor_lines = _build_ancestor_lines(ancestors, indent_per_level)

    status_chunk_parts = []
    if ancestor_lines:
        status_chunk_parts.append("CONTEXT (ancestors, oldest first):")
        status_chunk_parts.extend(ancestor_lines)
        status_chunk_parts.append("")
    status_chunk_parts.append(headline)
    status_chunk_text = "\n".join(status_chunk_parts).strip()
    if status_chunk_text:
        chunks.append(ChunkUnit(chunk_id="status", chunk_text=status_chunk_text))

    descendant_lines = _build_descendant_lines(
        status, descendants, indent_per_level, min_favourites
    )
    if not descendant_lines:
        return chunks

    spoiler_raw = status.get("spoiler_text")
    spoiler = spoiler_raw.strip() if isinstance(spoiler_raw, str) else ""
    header_subject = spoiler or _status_content_text(status)[:80] or "(no subject)"
    header = f"# Context: replies on Mastodon thread — {header_subject[:120]}"

    rendered_size = sum(len(line) for line in descendant_lines)

    if rendered_size <= single_call_threshold:
        parts = [header, "REPLIES:", *descendant_lines]
        chunks.append(ChunkUnit(chunk_id="replies", chunk_text="\n".join(parts)))
        return chunks

    sub_chunks = _chunk_lines_by_size(descendant_lines, chunk_chars)
    total = len(sub_chunks)
    for i, sub in enumerate(sub_chunks, start=1):
        parts = [header, f"REPLIES (chunk {i}/{total}):", *sub]
        chunks.append(
            ChunkUnit(
                chunk_id=f"replies_{i}_of_{total}",
                chunk_text="\n".join(parts),
            )
        )
    return chunks


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


# ---------------------------------------------------------------------------
# Status-meta particle (dual-emission)
# ---------------------------------------------------------------------------


def _build_status_meta_candidate(
    status: dict[str, Any],
    instance: str,
    root_id: str,
    injected_subjects: list[str],
) -> CandidateParticle | None:
    """Synthesise the single particle carrying the dual-emission ``properties`` dict.

    The meta-particle's ``content`` is a one-line summary of the status
    (instance + author + engagement counts) so cosine-similarity ranking
    can still surface it. The structured ``properties`` carry the
    machine-readable values that an exporter or structured query reads
    directly.

    ``root_id`` is passed in by the caller because computing it requires
    walking the importer's ``ancestors`` list, which lives outside the
    ``status`` dict. Empty-string ``root_id`` triggers the
    self/parent fallback for callers (e.g. unit tests) that don't have
    ancestor data on hand.

    Returns ``None`` when ``status`` is empty or its required fields
    (``id``, ``account.acct``) are missing — in that case the source
    blob was malformed and we'd rather let the LLM-derived claims stand
    alone than fabricate a meta record from defaults.
    """
    if not status:
        return None
    status_id_raw = status.get("id")
    acct = _account_acct(status)
    if not isinstance(status_id_raw, str) or not acct:
        return None
    status_id = status_id_raw

    favs = _safe_int(status.get("favourites_count"))
    reblogs = _safe_int(status.get("reblogs_count"))
    replies = _safe_int(status.get("replies_count"))
    spoiler_raw = status.get("spoiler_text")
    spoiler = spoiler_raw.strip() if isinstance(spoiler_raw, str) and spoiler_raw.strip() else None
    is_reblog = _is_reblog(status)
    # when this status is a boost, capture the boosted
    # status's identity so a future ``BOOSTS``-relation activation can
    # backfill the edge. Without this, the only signal that the boost
    # happened is the ``isReblog`` boolean — the IDENTITY of what was
    # boosted is lost.
    reblog_payload = status.get("reblog") if is_reblog else None
    reblog_of_status_id: str | None = None
    reblog_of_account_acct: str | None = None
    reblog_of_status_uri: str | None = None
    if isinstance(reblog_payload, dict):
        raw_id = reblog_payload.get("id")
        if isinstance(raw_id, str) and raw_id:
            reblog_of_status_id = raw_id
        raw_acct = _account_acct(reblog_payload) or None
        if raw_acct:
            reblog_of_account_acct = raw_acct
        raw_uri = reblog_payload.get("uri")
        if isinstance(raw_uri, str) and raw_uri:
            reblog_of_status_uri = raw_uri
    language_raw = status.get("language")
    language = language_raw if isinstance(language_raw, str) and language_raw else None

    # External URL: Mastodon represents external link cards as the ``card``
    # field — we read ``card.url`` if present, otherwise omit the key. We
    # intentionally do NOT fall back to the status's own ``url`` (which is
    # the canonical permalink of the post itself; that's identity, not an
    # external resource the post references).
    card = status.get("card")
    external_url: str | None = None
    if isinstance(card, dict):
        card_url = card.get("url")
        if isinstance(card_url, str) and card_url:
            external_url = card_url

    in_reply_to_raw = status.get("in_reply_to_id")
    parent_id = in_reply_to_raw if isinstance(in_reply_to_raw, str) else None

    # Caller supplies the actual root id (walking ancestors lives outside
    # this function). Fall back to a parent/self heuristic when the
    # caller passes the empty string — useful for unit tests that
    # construct a meta particle without ancestor context.
    if not root_id:
        root_id = status_id if parent_id is None else parent_id

    properties: dict[str, object] = {
        # Mastodon platform-specific keys (platform prefix).
        "mastodon:hasFavouritesCount": favs,
        "mastodon:hasReblogsCount": reblogs,
        "mastodon:hasRepliesCount": replies,
        "mastodon:hasInstance": instance,
        "mastodon:hasAccountHandle": acct,
        "mastodon:hasStatusId": status_id,
        "mastodon:isReblog": is_reblog,
        # Cross-platform UGC engagement keys (dual-emission).
        # social:hasReactionCount is dual of reblogs — Mastodon's
        # "amplification" reaction. Favourites are the score / like
        # analog (matches HN's hn:hasPoints → social:hasScore dual).
        "social:hasScore": favs,
        "social:hasReplyCount": replies,
        "social:hasReactionCount": reblogs,
        "social:hasAuthorHandle": acct,
        # Generic content + thread structure keys. Omit absent values
        # entirely rather than stamping None so consumers can use plain
        # ``"key" in props`` checks.
        "thread:hasRootId": root_id,
    }
    if spoiler is not None:
        properties["mastodon:hasSpoilerText"] = spoiler
    if language is not None:
        properties["content:hasLanguage"] = language
    if external_url is not None:
        properties["content:hasUrl"] = external_url
    if parent_id is not None:
        properties["thread:hasParentId"] = parent_id
    # reblog-target identity, captured for future ``BOOSTS``
    # relation backfill. Only present when this status IS a boost.
    if reblog_of_status_id is not None:
        properties["mastodon:reblogOfStatusId"] = reblog_of_status_id
    if reblog_of_account_acct is not None:
        properties["mastodon:reblogOfAccountAcct"] = reblog_of_account_acct
    if reblog_of_status_uri is not None:
        properties["mastodon:reblogOfStatusUri"] = reblog_of_status_uri

    cw_note = f" [CW: {spoiler}]" if spoiler else ""
    boost_note = " (boost)" if is_reblog else ""
    summary = (
        f"Mastodon status by mastodon/{acct} on {instance or 'unknown'}{boost_note}{cw_note} "
        f"received {favs} favourites, {reblogs} boosts, {replies} replies."
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

# Mastodon handles: alphanumeric + underscore for the local part; the
# optional remote suffix is host-shaped (alphanumeric + dot + hyphen).
_MASTODON_USER_TOKEN_RE = re.compile(r"\bmastodon/([A-Za-z0-9_]+(?:@[A-Za-z0-9_.\-]+)?)\b")


def _rewrite_mastodon_subjects(
    candidates: list[CandidateParticle], chunks: list[ChunkUnit]
) -> None:
    """Canonicalise bare Mastodon handles in candidates to ``mastodon/<acct>``.

    Mirrors :func:`particles.extraction.hackernews._rewrite_hn_subjects`.
    Reply lines are rendered as ``mastodon/{acct}: <body>`` so the LLM
    usually emits ``mastodon/{acct}`` as a subject token, but
    occasionally strips the prefix — which would land the subject at
    the Obsidian vault root instead of under the per-account path.
    """
    known: set[str] = set()
    for chunk in chunks:
        known.update(m.group(1) for m in _MASTODON_USER_TOKEN_RE.finditer(chunk.chunk_text))

    for c in candidates:
        seen: set[str] = set()
        out: list[str] = []
        for s in c.subjects:
            if isinstance(s, str) and not s.startswith("mastodon/") and s in known:
                s = f"mastodon/{s}"
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        c.subjects = out


# ---------------------------------------------------------------------------
# Module-private helper for ancestors-aware root id stamping
# ---------------------------------------------------------------------------


def _root_id_from_ancestors(status: dict[str, Any], ancestors: list[dict[str, Any]]) -> str:
    """Return the root status id of the reply chain.

    Mastodon's context API returns ``ancestors`` in oldest-first order,
    so ``ancestors[0].id`` is the root. When ``ancestors`` is empty the
    focal ``status`` IS the root.
    """
    if ancestors:
        first = ancestors[0]
        first_id = first.get("id")
        if isinstance(first_id, str):
            return first_id
    status_id = status.get("id")
    return status_id if isinstance(status_id, str) else ""


__all__ = [
    "APPLICABILITY",
    "DEFAULT_TRUST_WEIGHT",
    "EXTRACTOR_ID",
    "EXTRACTOR_VERSION",
    "MastodonExtractor",
    "SOURCE_TYPE",
    "_build_mastodon_chunks",
    "_build_status_meta_candidate",
    "_html_to_text",
    "_parse_status_route",
    "_render_status_headline",
    "_rewrite_mastodon_subjects",
    "_root_id_from_ancestors",
]
