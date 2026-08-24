# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Extractor plugin registry (Client layer).

To add a new domain-specific extractor:
  1. Create particles/extraction/<name>.py implementing ExtractorPlugin.
  2. Add an entry to _make_extractors() below.
     That is the only file outside the new module that needs editing.

Ordering matters: the first extractor whose accepts() returns True wins.
GeneralExtractor must be last — it accepts every source type.

The complementary *importer* registry — ``ImporterPlugin``, ``get_importers``,
``ensure_extractor_records`` — lives in the Engine layer at
``particles.ingest.importers.registry``: importers fetch a URL
and write the resulting blob into the corpus, so they reach
``particles.corpus`` / ``particles.store`` and cannot live in the Client
layer. The generic HTTP fetch in deposit.py is the implicit importer
fallback and does not appear in the importer list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from particles.config import get_config

if TYPE_CHECKING:
    from particles.core.schema import Snapshot
    from particles.extraction.general import ExtractionResult


#: Extractor ids whose particles are structured code-symbol readings — one
#: FALSIFIABLE claim per documented symbol. Documentation projection
#: demotes these in conceptual sections via ``code_symbol_rank_weight``;
#: ``query``, the drift lint, and exporters see them at full strength.
#: Keyed on a *capability* so a future doc-comment extractor joins by adding its
#: id here. Kept a literal (not an import of ``docstrings.EXTRACTOR_ID``) to
#: preserve this module's deferred-plugin-import design (see ``_make_extractors``),
#: so it must stay in sync with that constant.
CODE_SYMBOL_EXTRACTOR_IDS: frozenset[str] = frozenset({"docstring-extractor"})


@runtime_checkable
class ExtractorPlugin(Protocol):
    """Protocol every domain-specific extractor must satisfy."""

    EXTRACTOR_ID: str
    EXTRACTOR_VERSION: str

    def accepts(self, source_type: str) -> bool:
        """Return True if this extractor handles the given source_type string."""
        ...

    async def extract(
        self,
        snapshot: Snapshot,
        content: bytes,
        **kwargs: object,
    ) -> ExtractionResult:
        """Extract candidate particles from the stored blob."""
        ...


# ---------------------------------------------------------------------------
# Lazy singleton — built on first call to avoid circular imports
# ---------------------------------------------------------------------------

_extractors: list[ExtractorPlugin] | None = None


def get_extractors() -> list[ExtractorPlugin]:
    """Return the ordered extractor list (cached after first call)."""
    global _extractors
    if _extractors is None:
        _extractors = _make_extractors()
    return _extractors


def _make_extractors() -> list[ExtractorPlugin]:
    # defer: lazy-init — plugin classes are imported inside the factory so
    # adding or removing a plugin file (or one of its own top-level imports)
    # does not cascade-fail registry import-time and break every consumer of
    # ``get_extractors()``. Pay the plugin-import cost on first registry use,
    # not on every ``import particles.extraction.registry``. See root
    # AGENTS.md § Code conventions → Deferred imports (case 2: lazy-init).
    from particles.extraction.docstrings import DocstringExtractor
    from particles.extraction.general import GeneralExtractor
    from particles.extraction.github import (
        GitHubGistExtractor,
        GitHubPagesExtractor,
        GitHubRepoExtractor,
    )
    from particles.extraction.hackernews import HackerNewsExtractor
    from particles.extraction.journal import JournalExtractor
    from particles.extraction.mastodon import MastodonExtractor
    from particles.extraction.mcp_memory import McpMemoryExtractor
    from particles.extraction.nomisma import NomismaExtractor
    from particles.extraction.numista import (
        NumistaCoinExtractor,
        NumistaIssuerExtractor,
        NumistaListingExtractor,
    )
    from particles.extraction.rdf import RdfExtractor
    from particles.extraction.reddit import RedditExtractor
    from particles.extraction.taxonomy import TaxonomyExtractor
    from particles.extraction.trust_lens import TrustLensExtractor
    from particles.extraction.wikidata import WikidataExtractor

    return [
        TaxonomyExtractor(),
        TrustLensExtractor(),
        NumistaCoinExtractor(),
        NumistaIssuerExtractor(),
        NumistaListingExtractor(),
        WikidataExtractor(),
        NomismaExtractor(),
        RdfExtractor(),  # RDF_GRAPH structured extractor
        McpMemoryExtractor(),  # MCP_MEMORY_EXPORT migration extractor
        RedditExtractor(),
        HackerNewsExtractor(),
        MastodonExtractor(),
        GitHubRepoExtractor(),
        GitHubGistExtractor(),
        GitHubPagesExtractor(),
        DocstringExtractor(),  # PYTHON_SOURCE structured extractor; before the fallback
        JournalExtractor(),  # JOURNAL genre; before the fallback
        GeneralExtractor(),  # fallback — must be last
    ]


# ---------------------------------------------------------------------------
# Extension A: applicability enforcement and trust weight cache
# ---------------------------------------------------------------------------


def infer_domain(source_type: str) -> str | None:
    """Return the domain_label for a source_type from the first MUST applicability clause.

    Reads from the in-process extractor list — no DB round-trip (Extension B).
    Falls back to the configured ``trust.source_type_domains`` map for source
    types no extractor MUST-claims — this is what makes the
    AUTHOR-scoped trust tier reachable for directly-asserted (CONVERSATION)
    content so the agent_trust_rank seed binds. Returns None when neither the
    extractors nor the map cover it (e.g. WEB_PAGE, PDF), exactly as before.
    """
    from particles.core.schema import ApplicabilityClause

    for plugin in get_extractors():
        clauses: list[ApplicabilityClause] = getattr(plugin, "APPLICABILITY", [])
        for c in clauses:
            if c.keyword == "MUST" and source_type in c.source_types:
                return c.domain_label
    return get_config().trust.source_type_domains.get(source_type)


def is_must_not(plugin: ExtractorPlugin, source_type: str) -> bool:
    """Return True if the plugin has a MUST_NOT clause covering source_type."""
    from particles.core.schema import ApplicabilityClause

    clauses: list[ApplicabilityClause] = getattr(plugin, "APPLICABILITY", [])
    return any(c.keyword == "MUST_NOT" and source_type in c.source_types for c in clauses)


def select_extractor(source_type: str) -> ExtractorPlugin:
    """Return the extractor the pipeline would route ``source_type`` to.

    The one definition of "which extractor handles this source type": the
    first registered plugin, in registry order, that has no MUST_NOT clause
    for the type (Extension A) and whose ``accepts()`` returns True.
    ``GeneralExtractor`` is last and accepts everything, so this always
    resolves unless an extractor's MUST_NOT covers the fallback itself.

    Read it — do not re-derive it. The extract pipeline, the memory
    benchmark's per-session routing, and the benchmark suite
    auto-filter all call this, so a benchmark cannot silently disagree with
    the pipeline about whose contract a suite measures.

    Raises:
        LookupError: if no registered extractor accepts ``source_type``.
    """
    for plugin in get_extractors():
        if is_must_not(plugin, source_type):
            continue
        if plugin.accepts(source_type):
            return plugin
    raise LookupError(f"No extractor accepted source_type {source_type!r}")


def selects(plugin: ExtractorPlugin, source_types: list[str]) -> bool:
    """Return True if ``plugin`` is the routing choice for any of ``source_types``.

    The benchmark suite auto-filter: a suite auto-matches an
    extractor iff the production registry would select that extractor for at
    least one of the suite's declared source types. This is deliberately
    *not* ``any(plugin.accepts(st) ...)`` — ``GeneralExtractor.accepts()`` is
    unconditionally True (it is the fallback), so the accepts()
    predicate hands the general extractor every domain suite in the project
    and reports the resulting artifact as its score.

    An unroutable source type (no extractor at all) matches nothing rather
    than raising — a malformed suite should narrow a run, never abort it.
    """
    for st in source_types:
        try:
            if select_extractor(st).EXTRACTOR_ID == plugin.EXTRACTOR_ID:
                return True
        except LookupError:
            continue
    return False
