# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The append-only delta read: its text arithmetic, store-free.

An ``APPEND_ONLY`` entry's new snapshot is its predecessor plus new text, so the
general extractor reads only the new text, with the end of the earlier text
shown as context it extracts nothing from. The Engine pipeline chooses the base
snapshot and the offset it was read to; everything here is a pure function of
bytes and text, so the Client layer can compute it and the pipeline can call
the same functions to map a stop point back to raw bytes.

Offsets come in two units, and the difference matters:

* **raw bytes** of a snapshot's content. ``snapshots.extracted_through`` is in
  these, because raw bytes do not change and the append-only promise is a
  byte-prefix property.
* **characters of the extraction text**, the content decoded as the general
  extractor reads it (:func:`~particles.extraction.general.extraction_text`).
  The decoder can change between extractor versions, so a character offset
  recorded under one decoder would cut the text early or late under another.

A raw prefix is therefore always decoded again with the current decoder and
checked against the decoded snapshot; a decoder change that is not
prefix-stable shows up as a failed check, and the extraction falls back.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Collection
from dataclasses import dataclass

from particles.config import get_config
from particles.extraction.general import (
    _normalise_for_hashing,
    extraction_text,
    paragraph_spans,
)

#: Why a handed-over prefix was not used, as the quality note names it (§5).
FALLBACK_DECODED_PREFIX = "the decoded prefix does not prefix the decoded snapshot"
FALLBACK_PREFIX_UNDECODABLE = "the prefix could not be decoded"

_PARAGRAPH_BREAK = re.compile(rb"\n\n")


@dataclass(frozen=True)
class DeltaChunk:
    """One chunk of a delta read: its span in the extraction text, and its context."""

    start: int
    end: int
    text: str
    context: str | None


def context_before(text: str, end: int, chars: int) -> str | None:
    """The last ``chars`` characters of ``text[:end]``, cut back to a paragraph boundary.

    The window's first paragraph is dropped when the window starts inside it,
    so the context opens on a whole paragraph; failing a paragraph break, on a
    whole line; failing both, the window is kept as it is. ``None`` when
    ``chars`` is 0 or nothing but whitespace is left.
    """
    if chars <= 0 or end <= 0:
        return None
    start = max(0, end - chars)
    window = text[start:end]
    if start > 0:
        for separator in ("\n\n", "\n"):
            cut = window.find(separator)
            if cut >= 0 and window[cut + len(separator) :].strip():
                window = window[cut + len(separator) :]
                break
    window = window.strip()
    return window or None


def plan_delta_chunks(
    text: str, prefix_len: int, *, chunk_chars: int, context_chars: int
) -> list[DeltaChunk]:
    """Split ``text[prefix_len:]`` into delta chunks, each with its context (§1).

    The delta is cut on paragraph boundaries into chunks of at most
    ``chunk_chars``. The first chunk's context is the end of the text before
    the delta; each later chunk's is the end of the chunk before it. Spans are
    positions in ``text``, so a read that stops early can say where.
    """
    delta = text[prefix_len:]
    chunks: list[DeltaChunk] = []
    for start, end in paragraph_spans(delta, chunk_chars):
        if chunks:
            prev = chunks[-1].text
            context = context_before(prev, len(prev), context_chars)
        else:
            context = context_before(text, prefix_len, context_chars)
        chunks.append(
            DeltaChunk(
                start=prefix_len + start,
                end=prefix_len + end,
                text=delta[start:end],
                context=context,
            )
        )
    return chunks


def decoded_prefix_length(
    prefix: bytes, text: str, *, is_markdown: bool, mark_tools: bool
) -> tuple[int | None, str | None]:
    """How much of ``text`` the raw ``prefix`` accounts for, or why it cannot say.

    Decodes ``prefix`` with the current decoder and checks that the result
    prefixes ``text``, the snapshot decoded the same way. Returns
    ``(length, None)`` when it does and ``(None, reason)`` when it does not.
    """
    try:
        decoded, _ = extraction_text(prefix, is_markdown=is_markdown, mark_tools=mark_tools)
    except Exception:  # noqa: BLE001 — any decode failure is a fallback, never an error
        return None, FALLBACK_PREFIX_UNDECODABLE
    if not text.startswith(decoded):
        return None, FALLBACK_DECODED_PREFIX
    return len(decoded), None


def raw_offset_for(
    content: bytes, decoded_through: int, *, is_markdown: bool, mark_tools: bool
) -> int:
    """The raw byte offset of a stop point given in extraction-text characters (§3).

    The largest raw paragraph boundary (just past a ``\\n\\n``, or the start)
    whose decoded prefix is no longer than ``decoded_through`` and prefixes
    the decoded content: text up to there was read in full. The content's
    length when ``decoded_through`` reaches the end of the decoded text.
    """
    text, _ = extraction_text(content, is_markdown=is_markdown, mark_tools=mark_tools)
    if decoded_through >= len(text):
        return len(content)

    def decoded(boundary: int) -> str | None:
        try:
            out, _ = extraction_text(
                content[:boundary], is_markdown=is_markdown, mark_tools=mark_tools
            )
        except Exception:  # noqa: BLE001 — an undecodable prefix is simply not a candidate
            return None
        return out

    boundaries = [0] + [m.end() for m in _PARAGRAPH_BREAK.finditer(content)]
    # The decoded length grows with the boundary, so a binary search finds the
    # last boundary short enough; the prefix check then walks back past any
    # boundary whose decoding does not prefix the text.
    lo, hi = 0, len(boundaries) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        out = decoded(boundaries[mid])
        if out is not None and len(out) <= decoded_through:
            lo = mid
        else:
            hi = mid - 1
    for index in range(lo, -1, -1):
        out = decoded(boundaries[index])
        if out is not None and len(out) <= decoded_through and text.startswith(out):
            return boundaries[index]
    return 0


def derive_base_offset(
    content: bytes, carried_hashes: Collection[str], *, is_markdown: bool, mark_tools: bool
) -> int | None:
    """The raw offset a base extracted before was read to, where hashes show it (§4).

    ``carried_hashes`` are the chunk hashes the entry's claims carry. When
    every one is a chunk of ``content`` under the current chunker, the entry
    was extracted only by the current chunking, and the base counts as read
    through the end of the last chunk some claim carries. Chunks are read in
    order, so a claim on a chunk shows the read reached it; a chunk before it
    that carries no claim was read and yielded nothing.
    Returns ``None``, meaning *fully extracted*, otherwise: when a hash is not a
    current chunk (an earlier chunking read the text, and how far is not
    recorded), when no claim carries a hash, when the last chunk is carried,
    and when the base was read in one call (at or below
    ``extraction.html_chunk_size``, which is never cut).
    """
    size = get_config().extraction.html_chunk_size
    text, _ = extraction_text(content, is_markdown=is_markdown, mark_tools=mark_tools)
    if len(text) <= size:
        return None
    carried = set(carried_hashes)
    if not carried:
        return None
    normalised = _normalise_for_hashing(text)
    spans = paragraph_spans(normalised, size)
    hashes = [
        hashlib.sha256(normalised[start:end].encode("utf-8")).hexdigest() for start, end in spans
    ]
    if not carried <= set(hashes):
        return None
    last = max(i for i, h in enumerate(hashes) if h in carried)
    if last == len(hashes) - 1:
        return None
    # The chunker cuts the normalised text, which only deletes characters from
    # the decoded text, so this offset is a lower bound on the same point there.
    return raw_offset_for(
        content, spans[last + 1][0], is_markdown=is_markdown, mark_tools=mark_tools
    )
