# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Symbol-aware docstring extractor.

A deterministic, **LLM-free** Client-layer extractor that turns a
``PYTHON_SOURCE`` corpus entry (deposited by ``import project``)
into one structural particle per **documented** code symbol, attached to a
**bare-local code-symbol Subject**.

Like the other structured extractors (Wikidata, Numista, Nomisma), it overrides
:meth:`extract` directly — no ``_normalise`` / ``_extract_claims`` prose stage,
no LLM call — so it is exempt from the two-/three-step shapes.
A Python docstring is rigidly structured: ``ast`` gives module / class /
function nodes and their docstrings deterministically, and Google-style sections
(summary, ``Args:``, ``Returns:``, ``Raises:``) parse with a small section
grammar. The output is calibration-free: a deterministic extractor has no
probabilistic output to temperature-scale (calibration is identity),
so it carries no benchmark / ECE gate.

**Granularity: one particle per documented symbol.** Content is the
docstring's *summary* (first paragraph); the structured sections fold into the ``properties`` dict under ``docstring:`` keys. Each particle is a
``FALSIFIABLE`` ``CLAIM`` — load-bearing, because truth-gating
**excludes** non-``FALSIFIABLE`` particles from contradiction-checking, and the
whole code/design-drift payoff (surfaces a docstring claiming *X*
against an ADR deciding *not-X*) depends on the docstring claim *being* compared.

The code-symbol Subject is the symbol's fully-qualified dotted path
(``particles.core.scoring.confidence.effective_confidence``), derived from the file's
location via the ``entry_uri_r`` pipeline kwarg — Client-self-contained
(path string + ``__init__.py`` walk, no store access). The Engine resolver finds
no external authority for a dotted Python path and creates a bare-local Subject,
which is exactly right: code symbols are first-party entities with no Wikidata
identity. (non-entity gate **exempts** ``PYTHON_SOURCE`` candidates so
it does not strip these subjects — see ``particles/ingest/pipeline.py``.)
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from particles.core.schema import (
    ApplicabilityClause,
    AssertionModality,
    ParticleType,
    Snapshot,
    UncertaintyNature,
)
from particles.extraction.general import CandidateParticle, ExtractionResult

log = logging.getLogger(__name__)

SOURCE_TYPE = "PYTHON_SOURCE"
EXTRACTOR_ID = "docstring-extractor"
EXTRACTOR_VERSION = "0.1.0"
# First-party authoritative project statements rank high; tunable.
DEFAULT_TRUST_WEIGHT = 0.80
APPLICABILITY = [
    ApplicabilityClause(
        keyword="MUST",
        domain_uri="http://www.wikidata.org/entity/Q28865",  # Python (programming language)
        domain_label="Python source code",
        source_types=[SOURCE_TYPE],
    )
]

# Docstrings are read directly from the authoritative source; the value is
# deterministic and calibration-free.
_SUMMARY_CONFIDENCE = 0.95


# ---------------------------------------------------------------------------
# Module-path resolution (Client-self-contained)
# ---------------------------------------------------------------------------


def _module_path_from_uri(uri_r: str | None) -> str:
    """Resolve a file's dotted module path from the corpus entry's ``uri_r``.

    Parses ``file:///…/particles/core/scoring/confidence.py`` →
    ``particles.core.scoring.confidence`` by walking up while an ``__init__.py``
    exists in each parent directory (the
    package root), then joining the run of package directory names with the file
    stem. A standalone ``.py`` (no ``__init__.py`` sibling) yields just its stem.
    Derived from the path string + filesystem ``__init__.py`` checks alone — no
    store access. Returns ``""`` when ``uri_r`` is absent or not a
    ``file://`` URI (e.g. a ``.py`` deposited as raw text), in which case symbols
    fall back to their bare qualified names.
    """
    if not uri_r:
        return ""
    parsed = urlparse(uri_r)
    if parsed.scheme != "file":
        return ""
    file_path = Path(unquote(parsed.path))
    parts: list[str] = []
    parent = file_path.parent
    while (parent / "__init__.py").exists():
        parts.append(parent.name)
        parent = parent.parent
    parts.reverse()
    parts.append(file_path.stem)
    return ".".join(p for p in parts if p)


def _symbol_subject(module_path: str, qualname: str) -> str:
    """Join a module path and a symbol's AST-nested qualified name with a dot."""
    if module_path and qualname:
        return f"{module_path}.{qualname}"
    return module_path or qualname


# ---------------------------------------------------------------------------
# AST walk — module / class / function docstrings in document order
# ---------------------------------------------------------------------------


def _walk_documented_symbols(tree: ast.Module, module_path: str) -> list[tuple[str, str, str]]:
    """Return ``(subject, kind, docstring)`` for each *documented* symbol.

    Recurses ``Module`` → ``ClassDef`` / ``FunctionDef`` / ``AsyncFunctionDef``,
    calling :func:`ast.get_docstring`. A node with no docstring emits nothing —
    the analog of mkdocstrings' ``show_if_no_docstring: false``. Results are in
    document order (module docstring first, then body order, depth-first), so
    snapshot tests and summary-reading consumers see a stable sequence.
    """
    results: list[tuple[str, str, str]] = []

    mod_doc = ast.get_docstring(tree, clean=True)
    if mod_doc and mod_doc.strip():
        results.append((_symbol_subject(module_path, ""), "module", mod_doc))

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                qual = f"{prefix}{child.name}"
                doc = ast.get_docstring(child, clean=True)
                if doc and doc.strip():
                    results.append((_symbol_subject(module_path, qual), "class", doc))
                visit(child, f"{qual}.")
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                qual = f"{prefix}{child.name}"
                doc = ast.get_docstring(child, clean=True)
                if doc and doc.strip():
                    results.append((_symbol_subject(module_path, qual), "function", doc))
                visit(child, f"{qual}.")

    visit(tree, "")
    return results


# ---------------------------------------------------------------------------
# Google-style section grammar (no LLM)
# ---------------------------------------------------------------------------

# Canonical section name keyed by lowercased header word(s).
_SECTION_ALIASES = {
    "args": "args",
    "arguments": "args",
    "parameters": "args",
    "returns": "returns",
    "return": "returns",
    "raises": "raises",
    "exceptions": "raises",
    "yields": "yields",
    "yield": "yields",
    "attributes": "attributes",
}
# A bare ``Word:`` / ``Word Word:`` line with nothing after the colon — a header.
_HEADER_RE = re.compile(r"^(?P<name>[A-Za-z][A-Za-z ]*?):\s*$")
# A Google ``name: desc`` / ``name (type): desc`` mapping entry (key may be
# ``*args`` / ``**kwargs`` / a dotted attribute).
_ENTRY_RE = re.compile(r"^(?P<key>\*{0,2}[A-Za-z_][\w.]*)\s*(?:\([^)]*\))?\s*:\s*(?P<desc>.*)$")


def _section_header_name(stripped_line: str) -> str | None:
    m = _HEADER_RE.match(stripped_line)
    if m is None:
        return None
    return _SECTION_ALIASES.get(m.group("name").strip().lower())


def _first_paragraph(lines: list[str]) -> str:
    """The docstring summary — leading lines up to the first blank / section header."""
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or _section_header_name(stripped) is not None:
            break
        out.append(stripped)
    return " ".join(out).strip()


def _split_sections(lines: list[str]) -> dict[str, list[str]]:
    """Partition docstring lines into ``{section_name: body_lines}``.

    Everything before the first recognised header is ignored (it is the summary
    + extended description). A header opens a new section; subsequent lines fold
    into it until the next header.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    buf: list[str] = []
    for line in lines:
        header = _section_header_name(line.strip())
        if header is not None:
            if current is not None:
                sections[current] = buf
            current = header
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = buf
    return sections


def _parse_mapping(body: list[str]) -> dict[str, str]:
    """Parse a ``name: description`` section (Args / Raises / Attributes).

    Entry detection is indentation-based: a line at the section's base indent
    matching ``key: desc`` opens a new entry; deeper lines continue the previous
    entry's description. Returns ``{name: description}`` in source order.
    """
    entries: dict[str, str] = {}
    base: int | None = None
    cur_key: str | None = None
    parts: list[str] = []
    for line in body:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if base is None:
            base = indent
        m = _ENTRY_RE.match(line.strip())
        if indent <= base and m is not None:
            if cur_key is not None:
                entries[cur_key] = " ".join(parts).strip()
            cur_key = m.group("key")
            desc = m.group("desc").strip()
            parts = [desc] if desc else []
        elif cur_key is not None:
            parts.append(line.strip())
    if cur_key is not None:
        entries[cur_key] = " ".join(parts).strip()
    return entries


def _join_freeform(body: list[str]) -> str:
    """Collapse a free-form section (Returns / Yields) into one description string."""
    return " ".join(line.strip() for line in body if line.strip()).strip()


def _parse_google_docstring(docstring: str) -> tuple[str, dict[str, object]]:
    """Return ``(summary, sections)`` from a cleaned Google-style docstring.

    ``summary`` is the first paragraph; ``sections`` maps the recognised section
    names (``args`` / ``raises`` / ``attributes`` → ``{name: desc}`` dicts;
    ``returns`` / ``yields`` → a single description string), omitting empties.
    """
    lines = docstring.strip("\n").split("\n")
    summary = _first_paragraph(lines)
    raw_sections = _split_sections(lines)
    out: dict[str, object] = {}
    for name, section_lines in raw_sections.items():
        if name in ("args", "raises", "attributes"):
            mapping = _parse_mapping(section_lines)
            if mapping:
                out[name] = mapping
        else:  # returns, yields
            text = _join_freeform(section_lines)
            if text:
                out[name] = text
    return summary, out


def _candidate_for_symbol(subject: str, kind: str, docstring: str) -> CandidateParticle:
    """Build the one structural particle for a documented symbol (4)."""
    summary, sections = _parse_google_docstring(docstring)
    properties: dict[str, object] = {"docstring:kind": kind}
    if "args" in sections:
        properties["docstring:args"] = sections["args"]
    if "returns" in sections:
        properties["docstring:returns"] = sections["returns"]
    if "raises" in sections:
        properties["docstring:raises"] = sections["raises"]
    # Content is the summary; fall back to the collapsed docstring when a symbol
    # opens directly with a section (no first paragraph) so content is never empty.
    content = summary or _join_freeform(docstring.split("\n"))
    return CandidateParticle(
        content=content,
        confidence_value=_SUMMARY_CONFIDENCE,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        subjects=[subject] if subject else [],
        # FALSIFIABLE (the default) is load-bearing — see module docstring.
        assertion_modality=AssertionModality.FALSIFIABLE,
        particle_type=ParticleType.CLAIM,
        properties=properties,
    )


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class DocstringExtractor:
    """Deterministic, LLM-free extractor over ``PYTHON_SOURCE`` blobs."""

    EXTRACTOR_ID: str = EXTRACTOR_ID
    EXTRACTOR_VERSION: str = EXTRACTOR_VERSION
    DEFAULT_TRUST_WEIGHT: float = DEFAULT_TRUST_WEIGHT
    APPLICABILITY = APPLICABILITY

    def accepts(self, source_type: str) -> bool:
        return source_type == SOURCE_TYPE

    async def extract(
        self,
        snapshot: Snapshot,
        content: bytes,
        **kwargs: object,
    ) -> ExtractionResult:
        """Parse the source blob's AST → one particle per documented symbol.

        A file that fails to parse (syntax error, or Python 3.x-only syntax under
        an older host) yields zero candidates plus a ``quality_note``, never an
        exception that would fail the snapshot.
        """
        entry_uri_r = kwargs.get("entry_uri_r")
        module_path = _module_path_from_uri(entry_uri_r if isinstance(entry_uri_r, str) else None)
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return ExtractionResult(
                candidates=[], quality_notes=["Content is not valid UTF-8; no particles emitted"]
            )
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError) as exc:
            return ExtractionResult(
                candidates=[],
                quality_notes=[f"Python parse error ({exc}); no docstring particles emitted"],
            )

        documented = _walk_documented_symbols(tree, module_path)
        candidates = [_candidate_for_symbol(subj, kind, doc) for subj, kind, doc in documented]
        notes = [] if candidates else ["No documented symbols found"]
        log.info(
            "Docstring extractor: %d candidate(s) from %s",
            len(candidates),
            module_path or "(unknown module)",
        )
        return ExtractionResult(candidates=candidates, quality_notes=notes)
