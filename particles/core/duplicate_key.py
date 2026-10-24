# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Exact-duplicate identity key — the shared predicate.

Both rungs are the standard's: §9.2 step 6a suppresses an exact duplicate at
extract time against the §6.10 re-observation rule, and §6.10 creation
path 4 folds one that was already written. Both are defined over the same
§6.10 *normalized key*, which is what this module computes — the single
"same claim, twice" test they share.

One notion of "the same claim, twice" for the whole SDK:

* :func:`normalize_content` — the conservative content key. Whitespace runs
  collapse, sentence-final punctuation is trimmed, and typographic variants
  fold (quotation marks dropped, curly apostrophes, dashes and the ellipsis
  spelled one way); **case and wording are preserved**. This is
  *exact*-content identity, not near-duplicate merging.
* :func:`content_hash` — SHA-256 over the normalized key, stored on
  ``particles.content_norm_hash`` so the extract-time suppression lookup
  is one indexed probe instead of a scan.
* :func:`duplicate_key` — the full comparison tuple: normalized content,
  subject-id set, and ``stance:holder``.

**Why normalized rather than raw bytes.** The extract path already normalizes
for its intra-pass fold (:mod:`particles.ingest.candidate_dedup`), so
keying the cross-pass rung on raw bytes would mean two strings that dedupe
*within* one pass fail to dedupe *across* two. The measured cost is small: the
auto-merge activation-day census found 328 normalized groups against 322 raw
(+1.9 %).

An asymmetry was recorded here: the suppression rung normalized
while the Tier-A merge still keyed on raw ``content``, which left *prevention
strictly wider than cleanup* — a trailing-period twin could never be minted
twice, yet an already-minted one was permanently unreachable by
``links dedup``. That asymmetry was closed: the mop
(:func:`particles.operations.links_suggest._content_hash`) now keys on this
function, so both rungs reach exactly the same pairs and the mop's "0 groups"
means what an operator reads it to mean.

**Why this lives in Core.** Both the Engine-side intra-pass fold
(``ingest/``) and the ORM row (``store/``) need the same function, and
``store`` may not import ``ingest`` — that edge would add a subpackage cycle
the ``acyclic_siblings`` import contract rejects. Core is the shared
substrate
both already depend on. Pure — no I/O, no logging, per ``core/AGENTS.md``.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable

# Trailing punctuation trimmed during normalization. Conservative: sentence-final
# marks only, so "worse." and "worse" collapse but distinct claims never do.
_TRAILING_PUNCT = ".,;:!?\"'"

# Double quotation marks, in every script's spelling, are dropped outright: a
# restatement that quotes a title or a literal and one that does not are the
# same sentence (measured a vault note's claim retired and re-minted
# for exactly that difference). Single quotes are *folded*, never dropped,
# because the same mark is an apostrophe and "the dog's bowl" is not "the
# dogs' bowl"; a claim that quotes with single marks therefore still differs
# from its unquoted twin. The prime is not folded: as ``'`` it would be trimmed
# as a trailing mark, and ``f′`` would key equal to ``f``.
_DROPPED_QUOTES = '"\u201c\u201d\u201e\u201f\u00ab\u00bb\u2039\u203a\u300c\u300d\u300e\u300f\uff02'
_SINGLE_QUOTES = "\u2018\u2019\u201a\u201b"
# Hyphen, non-breaking hyphen, figure dash, en and em dash, horizontal bar, and
# the minus sign all read as one dash: "options 1–3" is "options 1-3".
_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
# Invisible characters that ``str.split`` does not treat as whitespace: zero-width
# space, word joiner, byte-order mark, soft hyphen.
_INVISIBLE = "\u200b\u2060\ufeff\u00ad"

_TYPOGRAPHIC_FOLD = str.maketrans(
    {
        **dict.fromkeys(_DROPPED_QUOTES + _INVISIBLE),
        **dict.fromkeys(_SINGLE_QUOTES, "'"),
        **dict.fromkeys(_DASHES, "-"),
        "\u2026": "...",
    }
)


def normalize_content(content: str) -> str:
    """Conservative exact-content key: fold typography, collapse whitespace, trim.

    Case and word order are intentionally preserved — this is *exact*-content
    identity, not near-duplicate merging: cosine was measured not to order
    duplicate-likelihood below identity, so there is no safe
    similarity tier below this one. Two strings collapse only when they differ
    by nothing more than:

    * Unicode composition (NFC: a precomposed and a decomposed ``é``);
    * typography — double quotation marks present or absent, a curly or a
      straight apostrophe, a dash spelled as hyphen, en dash, em dash or minus,
      an ellipsis as one character or three, an invisible zero-width mark;
    * whitespace runs; or
    * a sentence-final punctuation mark.

    A trailing ellipsis is trimmed like any trailing run of periods, which the
    key has always done for three dots. Compatibility folding (NFKC) is
    deliberately not applied: it rewrites
    ``m²`` as ``m2`` and ``½`` as ``1/2``, which change what a claim says.
    The fold is versioned with the stored ``content_norm_hash`` it keys
    (migration 055 re-keyed the rows the typographic fold changed).
    """
    folded = unicodedata.normalize("NFC", content).translate(_TYPOGRAPHIC_FOLD)
    collapsed = " ".join(folded.split())
    return collapsed.rstrip(_TRAILING_PUNCT).rstrip()


def content_hash(content: str) -> str:
    """SHA-256 hex digest of :func:`normalize_content` — the stored index key.

    Stable across processes and platforms (UTF-8, no salt), because it is
    persisted on the particle row and compared against values written by other
    runs.
    """
    return hashlib.sha256(normalize_content(content).encode("utf-8")).hexdigest()


def duplicate_key(
    content: str,
    subject_ids: Iterable[str],
    stance_holder: str | None,
) -> tuple[str, frozenset[str], str | None]:
    """The full identity tuple two claims must share to be one claim (§6.10).

    Three components, each carrying its own reason:

    * **normalized content** — the claim itself.
    * **subject-id set** — Tier A's "same Subject", generalized to the
      whole set. Keying on resolved *ids* (not names) means the empty set
      matches the empty set, so subject-less duplicates are covered too — the
      blind spot recorded for the per-Subject finder.
    * **``stance:holder``**. Identical text held by different
      principals is not one claim; merging would collapse the per-holder
      distribution.

    Truth-aptness, asserted-ness, and ACTIVE status are
    *caller* gates rather than key components: they decide whether a particle is
    eligible to participate at all, not which claim it is.
    """
    return (normalize_content(content), frozenset(subject_ids), stance_holder)
