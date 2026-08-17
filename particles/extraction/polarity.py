"""Claim-polarity classification contract (capability 1).

The general extractor classifies each candidate's *polarity* — how the source
document **itself presents** a proposition it states:

* ``ASSERTED`` (the default, and the meaning of the key's absence) — the
  document puts the proposition forward as a decision it makes or a claim it
  holds.
* ``DECLINED`` — the document presents it as rejected, superseded, deferred, or
  out-of-scope (a *Rejected Alternatives* entry, a *Why superseded* note, a
  *Deferred* item, "X is out of scope; do Y for now").
* ``HYPOTHETICAL`` — the document presents it as a counterfactual, conditional,
  motivational supposition, future projection, or worked example ("without a
  single source of truth, audit trails *will be* unreliable").

This module holds the small, dependency-free contract shared by the producer
(``particles.extraction.general``) and the consumers that must keep
**non-asserted** (``DECLINED`` / ``HYPOTHETICAL``) particles off the default
factual surface — mirroring the ``scope`` contract:

* ``particles.operations.query.main`` — the default result set excludes them
  (overridable via ``QueryRequest.include_non_asserted``).
* ``particles.ingest.pipeline`` — §6.6 conflict resolution skips them, so a
  rejected alternative never manufactures an ``INCONSISTENCY`` against the
  chosen decision.
* ``particles.operations.lint.contradictions`` (L-SEM-01),
  ``particles.operations.lint.coverage`` / ``contestedness``, and
  ``particles.operations.links_suggest`` (L-IDX-01) — all skip them.
* the one-way projection exporters (Obsidian / Logseq / wiki / Anki / JSONL) —
  exclude them from the rendered surface by default (the round-trippable
  interchange export keeps everything, since polarity rides on ``properties``).

The signal lives on a particle's Extension-side ``properties`` dict,
so Core modules never branch on it; only the operation / extraction layers do.
``confidence`` is **never** the polarity lever — a ``DECLINED`` claim may be
perfectly true (the rejection *did* happen), so demoting ``confidence.value``
would lie about truth-likelihood. Polarity governs visibility and engine
participation, never the scalar — the binding invariant.
"""

from __future__ import annotations

# Key on the ``properties`` dict carrying the polarity classification. The
# ``extraction:`` prefix is the requirement, taken up —
# the same code path already writes ``extraction:validity_basis`` /
# ``extraction:source_modality``, and ``core.stance`` writes ``stance:holder``.
# Particles minted before 1.111.0 carry the bare ``polarity`` spelling; Alembic
# 035 rewrites them in place and the interchange codec normalises the legacy key
# on import, so nothing downstream needs to know both spellings.
POLARITY_KEY = "extraction:polarity"
# Default value (also the meaning of the key's absence): the document asserts it.
POLARITY_ASSERTED = "ASSERTED"
# The two non-asserted values that trigger exclusion from the default surface.
POLARITY_DECLINED = "DECLINED"
POLARITY_HYPOTHETICAL = "HYPOTHETICAL"

# The polarity values that mark a particle non-asserted (off the default
# factual surface). A particle with no ``extraction:polarity`` key is
# ``ASSERTED``.
NON_ASSERTED_POLARITIES = frozenset({POLARITY_DECLINED, POLARITY_HYPOTHETICAL})


def is_non_asserted(properties: dict[str, object] | None) -> bool:
    """Return True if a particle is non-asserted (``DECLINED`` or ``HYPOTHETICAL``).

    True only when the particle carries ``extraction:polarity`` ∈ {``DECLINED``,
    ``HYPOTHETICAL``} (set by the extraction-time classifier). Absence of the
    key ⇒ ``ASSERTED`` ⇒ False, so every particle minted before this axis
    existed behaves exactly as before. This is the single predicate the query,
    pipeline, lint, link-suggestion, and exporter consumers share — keeping the
    key/value strings in one place. Mirrors ``is_excluded_document_meta``
    (scope).
    """
    if not properties:
        return False
    return properties.get(POLARITY_KEY) in NON_ASSERTED_POLARITIES
