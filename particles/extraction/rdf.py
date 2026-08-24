# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""RDF parsing extractor — structure-canonical particles.

Turns a deposited RDF document (Turtle / N-Triples / TriG / N-Quads / JSON-LD /
RDF-XML) into CandidateParticles whose **structured claim is the assertion** and
whose ``content`` is a derived verbalization — the inverse of every other
extractor in this package, and the first producer of
``CanonicalForm.STRUCTURED`` (minted the marker for exactly this
population).

A *structured extractor* in the sense: no prose stage, so it is exempt
from the ``_normalise`` / ``_extract_claims`` shapes and overrides ``extract()``
directly, like the Wikidata / Nomisma / Numista extractors.

Two constraints are load-bearing rather than incidental:

1. **No LLM call.** Not for extraction, not for polishing the verbalization.
   The whole value of this path is being exact and free.
2. **No network call.** The Wikidata extractor fetches labels live; this one
   must not, because a structure-canonical particle's ``content`` is *derived*,
   and a derivation depending on a remote service is not reproducible. Every
   label comes from the deposited bytes. This is also what keeps the module
   trivially Client-clean — no hook, no cache, no session.
   The rule is enforced structurally rather than by scrubbing what a document
   asks for: :func:`no_remote_retrieval` closes rdflib's retrieval seams for
   the duration of every parse, so *nothing outside the deposited bytes* is
   read — no scheme, no parser, no exceptions.

``content`` is never empty: the label ladder terminates at the full IRI, so
``Particle.content``'s ``min_length=1`` is satisfied by construction and is
never loosened (ruled that out — it would be a breaking
``SCHEMA_VERSION`` change).
"""

from __future__ import annotations

import json
import logging
import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from particles.config import get_config
from particles.core.schema import (
    ApplicabilityClause,
    CanonicalForm,
    ClaimTerm,
    ExternalRef,
    Snapshot,
    StructuredClaim,
    TermKind,
    UncertaintyNature,
)
from particles.extraction.general import CandidateParticle, ExtractionResult

if TYPE_CHECKING:  # pragma: no cover — typing only
    from collections.abc import Iterator

    from rdflib import Dataset, Graph
    from rdflib.term import Node

log = logging.getLogger(__name__)

SOURCE_TYPE = "RDF_GRAPH"
EXTRACTOR_ID = "rdf-extractor"
#: 0.2.0 closed the §3.10 retrieval seams structurally. A document naming a
#: non-inline ``@context`` under any scheme now yields only its absolute-IRI
#: triples, where 0.1.0 yielded the context-expanded ones for every scheme but
#: ``http(s)``, so prior-version particles are re-extractable by
#: ``particles reindex --extractor-version 0.1.0``.
EXTRACTOR_VERSION = "0.2.0"
#: Publishing RDF signals deliberate curation, but says nothing about *who*
#: published it — so this sits between a generic web page and Wikidata's 0.90.
#: Distinct from ``rdf.default_confidence``, which is about the reading, not the
#: source (two-quantity separation).
DEFAULT_TRUST_WEIGHT = 0.60
APPLICABILITY = [
    ApplicabilityClause(
        keyword="MUST",
        domain_uri="http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        domain_label="linked data",
        source_types=[SOURCE_TYPE],
    )
]

# --------------------------------------------------------------------------
# Syntax detection
# --------------------------------------------------------------------------

#: Suffix → rdflib parser format. ``.json`` is deliberately absent: that slot is
#: already contended (taxonomy / trust-lens definitions) and this SDK's own
#: interchange bundles carry an ``@context``, so an ``@context`` shape
#: sniff would route a store export into the RDF path.
_FORMAT_BY_SUFFIX: dict[str, str] = {
    ".ttl": "turtle",
    ".turtle": "turtle",
    ".nt": "nt",
    ".trig": "trig",
    ".nq": "nquads",
    ".jsonld": "json-ld",
    ".rdf": "xml",
    ".owl": "xml",
}

_FORMAT_BY_CONTENT_TYPE: dict[str, str] = {
    "text/turtle": "turtle",
    "application/n-triples": "nt",
    "application/trig": "trig",
    "application/n-quads": "nquads",
    "application/ld+json": "json-ld",
    "application/rdf+xml": "xml",
}

#: Consumed by ``corpus/deposit.py::_detect_source_type`` (Engine → Client is
#: allowed; the constants stay here so the syntax table has one home).
RDF_SUFFIXES: frozenset[str] = frozenset(_FORMAT_BY_SUFFIX)
RDF_CONTENT_TYPES: frozenset[str] = frozenset(_FORMAT_BY_CONTENT_TYPE)

#: Fallback parse order when the hint is absent or wrong. TriG is a superset of
#: Turtle, which is a superset of N-Triples, so one parser covers the whole text
#: family; N-Quads, JSON-LD and RDF/XML each need their own. Fixed order keeps
#: the resolution deterministic.
_FALLBACK_FORMATS: tuple[str, ...] = ("trig", "nquads", "json-ld", "xml")

_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_RDF_TYPE = f"{_RDF}type"
_RDF_STATEMENT = f"{_RDF}Statement"
_RDF_SUBJECT = f"{_RDF}subject"
_RDF_PREDICATE = f"{_RDF}predicate"
_RDF_OBJECT = f"{_RDF}object"
_OWL_ONTOLOGY = "http://www.w3.org/2002/07/owl#Ontology"

#: Label predicates the verbalization ladder reads, in preference order. Kept
#: separate from the configurable ``skip_predicates`` (which decides what is
#: *emitted*) because reading and skipping are different questions: an operator
#: may want ``rdfs:label`` emitted as a claim and still want it used as a label.
_LABEL_PREDICATES: tuple[str, ...] = (
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2004/02/skos/core#prefLabel",
    "http://purl.org/dc/terms/title",
    "http://purl.org/dc/elements/1.1/title",
    "http://xmlns.com/foaf/0.1/name",
)

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _format_from_uri(uri: str | None) -> str | None:
    """Map a source URI or file path onto an rdflib parser format."""
    if not uri:
        return None
    path = urlparse(uri).path if "://" in uri else uri
    lowered = path.lower()
    for suffix, fmt in _FORMAT_BY_SUFFIX.items():
        if lowered.endswith(suffix):
            return fmt
    return None


def _format_candidates(uri: str | None, content: bytes) -> list[str]:
    """Ordered parser formats to try, most-likely first.

    The URI suffix is the hint; a cheap content sniff promotes the two syntaxes
    the Turtle family cannot possibly parse; the fixed fallback chain covers the
    rest. Deterministic by construction — no input produces a different order on
    a second run.
    """
    ordered: list[str] = []
    hinted = _format_from_uri(uri)
    if hinted:
        ordered.append(hinted)

    head = content[:512].lstrip()
    if head[:1] in (b"{", b"["):
        ordered.append("json-ld")
    if head.startswith(b"<?xml") or b"<rdf:RDF" in content[:2048]:
        ordered.append("xml")

    ordered.extend(_FALLBACK_FORMATS)
    return list(dict.fromkeys(ordered))


# --------------------------------------------------------------------------
# Hostile input
# --------------------------------------------------------------------------


class RemoteRetrievalRefused(RuntimeError):
    """A parse tried to fetch something outside the deposited bytes.

    Raised by the :func:`no_remote_retrieval` guard. The extractor converts it
    into a quality note, so it never escapes ``extract()`` as an exception
    .
    """


def _forbid_external_entities(parser: Any, refused: list[str]) -> None:
    """Turn off external entity and DTD resolution on one SAX parser.

    An RDF/XML document is the one syntax whose retrieval happens in a second
    parser that the JSON scrub cannot see: ``<!ENTITY x SYSTEM "file:///…">``
    is a local file read at parse time. CPython's expat reader already defaults
    ``feature_external_ges`` off and refuses external parameter entities
    outright, so on a stock interpreter the entity expands to nothing — but
    that is an implicit default of whichever reader ``xml.sax.make_parser``
    happened to pick, and ``xml.sax.default_parser_list`` is writable by any
    installed package. Setting the feature explicitly makes the guarantee ours;
    the refusing resolver covers a reader that does not recognise the feature.
    """
    from xml.sax import SAXNotRecognizedException, SAXNotSupportedException
    from xml.sax.handler import feature_external_ges, feature_external_pes

    for feature in (feature_external_ges, feature_external_pes):
        try:
            parser.setFeature(feature, False)
        except (SAXNotRecognizedException, SAXNotSupportedException):  # pragma: no cover
            # A reader that cannot express the feature still gets the resolver.
            log.debug("SAX reader does not support %s; relying on the entity resolver", feature)

    class _RefusingEntityResolver:
        def resolveEntity(self, public_id: str | None, system_id: str) -> str:
            refused.append(system_id)
            raise RemoteRetrievalRefused(
                f"refused to resolve external XML entity {system_id!r} during an RDF parse"
            )

    parser.setEntityResolver(_RefusingEntityResolver())


@contextmanager
def no_remote_retrieval() -> Iterator[list[str]]:
    """Refuse every retrieval rdflib would perform during a parse, any scheme.

    The §3.2 no-network rule is what a structure-canonical ``content``
    derivation's reproducibility rests on, and rdflib will otherwise reach the
    network — or the local filesystem — from inside ``parse()``. Scrubbing the
    document first (:func:`strip_remote_contexts`) is not sufficient on its
    own: it can only see the shapes it knows about, in a payload it can parse
    as JSON. This guard closes the seams themselves, so a retrieval the scrub
    missed is refused rather than performed.

    Two seams, one per parser:

    * **JSON-LD** — ``rdflib.plugins.shared.jsonld.context.source_to_json`` is
      called from exactly one place, ``Context._fetch_context``, and only to
      dereference a string ``@context`` or an ``@import``. Patching it there
      leaves the JSON-LD parser's own call (which turns the deposited bytes
      into JSON) untouched, so a legitimate parse is unaffected. rdflib 7.6
      exposes no injectable document loader, which is why this is a patch.
    * **RDF/XML** — ``rdflib.plugins.parsers.rdfxml.create_parser`` builds the
      SAX reader; the wrapper hardens each one as it is created.

    Both are module attributes restored on exit, so nothing leaks past the
    ``with`` block; nesting restores correctly because each level saves the
    value it actually replaced. The patch is process-wide for its duration —
    a parse running concurrently in another thread is guarded too, which errs
    in the safe direction.

    Yields:
        The list of refused sources, appended to as refusals happen. Non-empty
        means something in the document tried to reach outside itself.
    """
    from rdflib.plugins.parsers import rdfxml
    from rdflib.plugins.shared.jsonld import context as jsonld_context

    refused: list[str] = []

    def _refuse_context_fetch(source: Any, *args: Any, **kwargs: Any) -> Any:
        target = source if isinstance(source, str) else repr(source)
        refused.append(target)
        raise RemoteRetrievalRefused(
            f"refused to retrieve JSON-LD context {target!r} during an RDF parse"
        )

    # ``context.py`` re-binds the name with ``from .util import source_to_json``,
    # which mypy does not count as an explicit re-export — but that binding is
    # precisely the one that must be replaced (patching ``util`` would not reach
    # the already-resolved reference).
    original_source_to_json = jsonld_context.source_to_json  # type: ignore[attr-defined]
    original_create_parser = rdfxml.create_parser

    def _hardened_create_parser(target: Any, store: Any) -> Any:
        parser = original_create_parser(target, store)
        _forbid_external_entities(parser, refused)
        return parser

    jsonld_context.source_to_json = _refuse_context_fetch  # type: ignore[attr-defined]
    rdfxml.create_parser = _hardened_create_parser
    try:
        yield refused
    finally:
        jsonld_context.source_to_json = original_source_to_json  # type: ignore[attr-defined]
        rdfxml.create_parser = original_create_parser


def strip_remote_contexts(content: bytes) -> tuple[bytes, list[str]]:
    """Drop every non-inline ``@context`` / ``@import`` from a JSON-LD document.

    Defense in depth in front of :func:`no_remote_retrieval`: the guard makes a
    retrieval impossible, and this keeps the common case from *attempting* one,
    so a document that merely names a well-known context still parses into its
    absolute-IRI triples instead of failing the whole JSON-LD candidate.

    Two rules, both deliberately scheme-blind:

    * A **string-valued** ``@context`` is dropped whatever it looks like.
      ``http(s)://`` is the obvious case, but ``file:///etc/hostname`` is a
      local file read, ``ftp://`` is still a fetch, and a *relative* reference
      is resolved against the document base into either — so the only safe
      reading of "the context is not inline" is the type of the value, not its
      prefix. Inline (object) contexts are kept, and the term definitions
      inside them — including relative IRIs — are left exactly as written.
    * An ``@import`` inside a context object is dropped. It is a JSON-LD 1.1
      retrieval key that lives *inside* the context rather than beside it, so
      a pass that only inspects ``@context`` values never sees it.

    Both are found at any depth: a context can hang off any node object, and
    JSON-LD 1.1 scoped contexts put one inside a term definition.

    Returns:
        ``(document_bytes, dropped_refs)``. Non-JSON input is returned
        unchanged with no drops, so this is safe to call on any payload.
    """
    try:
        doc = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return content, []

    dropped: list[str] = []

    def _clean_context(ctx: Any) -> Any:
        if isinstance(ctx, str):
            dropped.append(ctx)
            return None
        if isinstance(ctx, list):
            kept = [c for c in (_clean_context(item) for item in ctx) if c is not None]
            return kept or None
        if isinstance(ctx, dict):
            return _clean_context_object(ctx)
        # null (a context reset) and anything malformed go through untouched;
        # deciding what they mean is rdflib's job, not the scrubber's.
        return ctx

    def _clean_context_object(node: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "@import":
                if isinstance(value, str):
                    dropped.append(value)
                continue
            if key == "@context":
                cleaned = _clean_context(value)
                if cleaned is not None:
                    out[key] = cleaned
                continue
            out[key] = _clean_context_object(value) if isinstance(value, dict) else value
        return out

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            out: dict[str, Any] = {}
            for key, value in node.items():
                if key == "@context":
                    cleaned = _clean_context(value)
                    if cleaned is not None:
                        out[key] = cleaned
                else:
                    out[key] = _walk(value)
            return out
        if isinstance(node, list):
            return [_walk(item) for item in node]
        return node

    cleaned_doc = _walk(doc)
    if not dropped:
        # Avoid a re-serialization round-trip when there is nothing to strip.
        return content, []
    return json.dumps(cleaned_doc).encode("utf-8"), dropped


# --------------------------------------------------------------------------
# Verbalization
# --------------------------------------------------------------------------


def prettify_local_name(local: str) -> str:
    """Render an IRI local name as readable words.

    ``wasMintedAt`` → ``was minted at``; ``BerlinMint`` → ``Berlin mint``;
    ``has_weight`` → ``has weight``; ``P571`` → ``P571``.

    camelCase is split, separators become spaces, and every word *after the
    first* is lower-cased while the first keeps its original case. That single
    rule reads correctly for both conventions at once: predicates are
    lower-camel by convention and come out as a verb phrase, classes and
    entities are upper-camel and keep their capital.
    """
    spaced = _CAMEL_BOUNDARY.sub(" ", local).replace("_", " ").replace("-", " ")
    words = spaced.split()
    if not words:
        return local
    return " ".join([words[0]] + [w.lower() for w in words[1:]])


def _local_name(iri: str) -> str:
    for sep in ("#", "/"):
        if sep in iri:
            tail = iri.rsplit(sep, 1)[1]
            if tail:
                return tail
    return ""


class _Verbalizer:
    """Renders terms as prose using only what the deposited document supplies."""

    def __init__(self, labels: dict[str, str], graph: Graph) -> None:
        self._labels = labels
        self._graph = graph

    def label(self, term: Node) -> str:
        """Label a term via the four-rung ladder; never returns empty for a URI."""
        from rdflib.term import BNode, Literal, URIRef

        if isinstance(term, Literal):
            text = str(term)
            lang = term.language
            # The Wikidata extractor's monolingualtext convention: flag a
            # non-English literal rather than silently presenting it as English.
            return f"{text} ({lang})" if lang and lang.split("-")[0] != "en" else text

        if isinstance(term, BNode):
            # A blank node has no IRI to fall back on, so an unlabelled one is
            # genuinely unrenderable — the caller drops the triple.
            return self._labels.get(str(term), "")

        if isinstance(term, URIRef):
            iri = str(term)
            # Rung 1: a label carried by this very document.
            labelled = self._labels.get(iri)
            if labelled:
                return labelled
            # Rung 2: the IRI's local name, made readable.
            local = _local_name(iri)
            if local:
                return prettify_local_name(local)
            # Rung 3: the CURIE, when the document bound a prefix for it.
            curie = self._curie(iri)
            if curie:
                return curie
            # Rung 4: the IRI itself. Always non-empty — this is what makes
            # ``content``'s min_length=1 hold by construction.
            return iri

        return str(term)

    def _curie(self, iri: str) -> str | None:
        try:
            prefix, _, name = self._graph.namespace_manager.compute_qname(iri, generate=False)
        except (KeyError, ValueError):
            return None
        return f"{prefix}:{name}" if prefix else None


def _build_label_index(dataset: Dataset) -> dict[str, str]:
    """Index every label the document carries, English preferred.

    Preference: an ``@en`` literal, then an untagged one, then the first seen —
    and, at equal language rank, the earlier predicate in ``_LABEL_PREDICATES``.
    Deterministic given the same document.
    """
    from rdflib.term import Literal

    best: dict[str, tuple[int, int]] = {}
    labels: dict[str, str] = {}
    for rank, predicate in enumerate(_LABEL_PREDICATES):
        pred_ref = _uri(predicate)
        for graph in dataset.graphs():
            for subject, _p, obj in graph.triples((None, pred_ref, None)):
                if not isinstance(obj, Literal):
                    continue
                text = str(obj).strip()
                if not text:
                    continue
                lang = obj.language
                lang_rank = 0 if (lang and lang.split("-")[0] == "en") else (1 if not lang else 2)
                key = str(subject)
                score = (lang_rank, rank)
                if key not in best or score < best[key]:
                    best[key] = score
                    labels[key] = text
    return labels


def _uri(iri: str) -> Any:
    from rdflib.term import URIRef

    return URIRef(iri)


# --------------------------------------------------------------------------
# Subject binding
# --------------------------------------------------------------------------


def external_ref_for(iri: str, namespaces: dict[str, str]) -> ExternalRef | None:
    """Map an absolute IRI onto a Subject Authority external ref, or ``None``.

    A Client-side table rather than a call into the registry, which is
    Engine-side and unreachable from here; deriving it would need a
    ``recognize_uri`` Protocol addition plus a registration hook.
    The longest matching prefix wins, so a more specific namespace is never
    shadowed by a shorter one.
    """
    match: tuple[str, str] | None = None
    for prefix, namespace in namespaces.items():
        if not iri.startswith(prefix) or len(iri) <= len(prefix):
            continue
        if match is None or len(prefix) > len(match[0]):
            match = (prefix, namespace)
    if match is None:
        return None
    prefix, namespace = match
    return ExternalRef(namespace=namespace, id=iri[len(prefix) :], uri=iri)


def _claim_term(term: Node, *, position: str) -> ClaimTerm | None:
    """Render one RDF term as a :class:`ClaimTerm`."""
    from rdflib.term import BNode, Literal, URIRef

    if isinstance(term, URIRef):
        return ClaimTerm(kind=TermKind.URI, value=str(term))
    if isinstance(term, Literal):
        if position != "object":
            return None
        datatype = str(term.datatype) if term.datatype else None
        return ClaimTerm(
            kind=TermKind.LITERAL,
            value=str(term),
            datatype=datatype,
            language=term.language if not datatype else None,
        )
    if isinstance(term, BNode):
        # No stable identity outside this parse, but honest as a lexical token.
        return ClaimTerm(kind=TermKind.TOKEN, value=f"_:{term}")
    return None


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


class _Annotation:
    """Confidence carried by a reification bundle or a named graph."""

    __slots__ = ("confidence",)

    def __init__(self, confidence: float | None = None) -> None:
        self.confidence = confidence


def _read_confidence(value: Node, predicates: set[str]) -> float | None:
    """Read a confidence literal, tolerantly. Out-of-range or non-numeric → None."""
    from rdflib.term import Literal

    if not isinstance(value, Literal):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if 0.0 <= number <= 1.0 else None


class RdfExtractor:
    """Parses a deposited RDF document into structure-canonical candidates."""

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
        cfg = get_config().rdf
        notes: list[str] = []

        if len(content) > cfg.max_bytes:
            # Fail before the parser rather than inside it: a
            # compressed-expansion document should cost a length check.
            return ExtractionResult(
                candidates=[],
                quality_notes=[
                    f"RDF document is {len(content)} bytes, over the "
                    f"rdf.max_bytes limit of {cfg.max_bytes}; not parsed"
                ],
            )

        entry_uri = kwargs.get("entry_uri_r")
        payload, dropped = strip_remote_contexts(content)
        if dropped:
            listed = ", ".join(sorted(set(dropped)))
            notes.append(
                "Dropped non-inline JSON-LD @context / @import reference(s) before "
                f"parsing (no network at extraction time): {listed}"
            )

        hint = entry_uri if isinstance(entry_uri, str) else None
        # Every parse the extractor performs runs inside the guard, so a
        # retrieval the scrub above did not recognise is refused rather than
        # performed.
        with no_remote_retrieval() as refused:
            dataset, fmt, parse_note = _parse(payload, hint)
        if refused:
            listed = ", ".join(sorted(set(refused)))
            notes.append(
                "Refused a non-inline retrieval during the RDF parse "
                f"(no network at extraction time): {listed}"
            )
        if dataset is None:
            notes.append(parse_note or "RDF parse failed")
            return ExtractionResult(candidates=[], quality_notes=notes)

        candidates, emit_notes = _emit(dataset, cfg)
        notes.extend(emit_notes)
        log.info(
            "RDF extractor: %d candidate(s) from %s (format=%s)",
            len(candidates),
            snapshot.snapshot_id,
            fmt,
        )
        return ExtractionResult(candidates=candidates, quality_notes=notes)


def _parse(content: bytes, entry_uri: str | None) -> tuple[Dataset | None, str | None, str | None]:
    """Parse into a Dataset, trying the candidate formats in order."""
    from rdflib import Dataset

    errors: list[str] = []
    for fmt in _format_candidates(entry_uri, content):
        dataset = Dataset()
        try:
            dataset.parse(data=content, format=fmt)
        except Exception as exc:  # rdflib raises a wide variety on bad syntax
            errors.append(f"{fmt}: {exc}")
            continue
        if len(dataset) == 0 and not any(len(g) for g in dataset.graphs()):
            errors.append(f"{fmt}: parsed but yielded no triples")
            continue
        return dataset, fmt, None
    return None, None, "RDF parse failed for every candidate syntax — " + "; ".join(errors[:3])


def _emit(dataset: Dataset, cfg: Any) -> tuple[list[CandidateParticle], list[str]]:
    """Turn a parsed dataset into candidates."""
    from rdflib.term import URIRef

    labels = _build_label_index(dataset)
    skip = set(cfg.skip_predicates)
    conf_predicates = set(cfg.confidence_predicates)
    notes: list[str] = []

    # --- reification bundles: fold, never double-emit (§3.3a) ---------------
    reifiers: set[Node] = set()
    reified: dict[tuple[Node, Node, Node], _Annotation] = {}
    for graph in dataset.graphs():
        for reifier in set(graph.subjects(_uri(_RDF_TYPE), _uri(_RDF_STATEMENT))):
            subj = graph.value(reifier, _uri(_RDF_SUBJECT))
            pred = graph.value(reifier, _uri(_RDF_PREDICATE))
            obj = graph.value(reifier, _uri(_RDF_OBJECT))
            # A reifier node is *definitionally* document plumbing, so every
            # triple on it is consumed — unlike a named-graph IRI, which may
            # legitimately also be a real-world entity.
            reifiers.add(reifier)
            if subj is None or pred is None or obj is None:
                continue
            bundle = _Annotation()
            for _s, p, annotated in graph.triples((reifier, None, None)):
                if str(p) in conf_predicates:
                    bundle.confidence = _read_confidence(annotated, conf_predicates)
            reified[(subj, pred, obj)] = bundle

    # --- graph-level annotations (§3.3b) ------------------------------------
    graph_confidence: dict[str, float] = {}
    consumed_graph_annotations: set[tuple[Node, Node, Node]] = set()
    graph_names = {str(g.identifier) for g in dataset.graphs()}
    for graph in dataset.graphs():
        for subj, pred, obj in graph:
            if str(pred) in conf_predicates and str(subj) in graph_names:
                value = _read_confidence(obj, conf_predicates)
                if value is not None:
                    graph_confidence[str(subj)] = value
                # Only the annotation triple itself is consumed; other triples
                # about a graph IRI stay ordinary assertions.
                consumed_graph_annotations.add((subj, pred, obj))

    candidates: list[CandidateParticle] = []
    emitted: set[tuple[Node, Node, Node]] = set()
    truncated = False

    def _emit_one(
        subj: Node,
        pred: Node,
        obj: Node,
        *,
        graph_id: str | None,
        confidence: float | None,
    ) -> None:
        nonlocal truncated
        if len(candidates) >= cfg.max_triples:
            truncated = True
            return
        verbalizer = _Verbalizer(labels, next(iter(dataset.graphs())))
        candidate = _build_candidate(
            subj,
            pred,
            obj,
            verbalizer=verbalizer,
            graph_id=graph_id,
            confidence=confidence if confidence is not None else cfg.default_confidence,
            namespaces=cfg.uri_namespaces,
            allow_blank_subjects=cfg.include_blank_node_subjects,
        )
        if candidate is not None:
            candidates.append(candidate)
            emitted.add((subj, pred, obj))

    for graph in dataset.graphs():
        graph_id = str(graph.identifier)
        named = graph_id in graph_names and not graph_id.startswith("urn:x-rdflib:")
        for subj, pred, obj in sorted(graph, key=lambda t: (str(t[0]), str(t[1]), str(t[2]))):
            if subj in reifiers:
                continue
            if (subj, pred, obj) in consumed_graph_annotations:
                continue
            if str(pred) in skip:
                continue
            if isinstance(obj, URIRef) and str(obj) == _OWL_ONTOLOGY and str(pred) == _RDF_TYPE:
                continue
            reification = reified.get((subj, pred, obj))
            confidence = reification.confidence if reification is not None else None
            if confidence is None:
                confidence = graph_confidence.get(graph_id)
            _emit_one(
                subj,
                pred,
                obj,
                graph_id=graph_id if named else None,
                confidence=confidence,
            )

    # A triple that exists *only* inside a reification bundle is still asserted
    # by that bundle — emit it once, after the bare triples, so ordering stays
    # stable for snapshot tests.
    for (subj, pred, obj), annotation in sorted(
        reified.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), str(kv[0][2]))
    ):
        if (subj, pred, obj) in emitted:
            continue
        _emit_one(subj, pred, obj, graph_id=None, confidence=annotation.confidence)

    if truncated:
        notes.append(
            f"RDF document exceeded rdf.max_triples ({cfg.max_triples}); "
            f"kept the first {len(candidates)} candidate(s) and stopped"
        )
    if not candidates:
        notes.append("RDF document parsed but yielded no emittable triples")
    return candidates, notes


def _build_candidate(
    subj: Node,
    pred: Node,
    obj: Node,
    *,
    verbalizer: _Verbalizer,
    graph_id: str | None,
    confidence: float,
    namespaces: dict[str, str],
    allow_blank_subjects: bool,
) -> CandidateParticle | None:
    """Build one structure-canonical candidate, or ``None`` when unrenderable."""
    from rdflib.term import BNode, URIRef

    if isinstance(subj, BNode) and not allow_blank_subjects:
        # No identity outside this parse: cannot be re-identified on
        # re-extraction, cannot bind to a Subject, cannot be exported (§3.3).
        return None

    subject_label = verbalizer.label(subj)
    predicate_label = verbalizer.label(pred)
    object_label = verbalizer.label(obj)
    if not subject_label or not object_label or not predicate_label:
        return None

    claim = _structured_claim(subj, pred, obj)
    if claim is None:
        return None

    content = f"{subject_label} {predicate_label}: {object_label}"

    subjects: list[str] = [subject_label]
    external_refs: dict[str, ExternalRef] = {}
    if isinstance(subj, URIRef):
        ref = external_ref_for(str(subj), namespaces)
        if ref is not None:
            external_refs[subject_label] = ref
    # A URI object makes this a graph edge between two entities — the same shape
    # the Wikidata extractor produces with ``[entity_id] + related_qids``.
    if isinstance(obj, URIRef) and object_label != subject_label:
        subjects.append(object_label)
        ref = external_ref_for(str(obj), namespaces)
        if ref is not None:
            external_refs[object_label] = ref

    properties: dict[str, object] | None = {"rdf:graph": graph_id} if graph_id else None

    return CandidateParticle(
        content=content,
        confidence_value=confidence,
        # A parse reports what a source states; the residual uncertainty is
        # about the world, not about sampling noise.
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        subjects=subjects,
        structured_claim=claim,
        canonical_form=CanonicalForm.STRUCTURED,
        external_refs=external_refs,
        properties=properties,
    )


def _structured_claim(subj: Node, pred: Node, obj: Node) -> StructuredClaim | None:
    subject_term = _claim_term(subj, position="subject")
    predicate_term = _claim_term(pred, position="predicate")
    object_term = _claim_term(obj, position="object")
    if subject_term is None or predicate_term is None or object_term is None:
        return None
    return StructuredClaim(
        subject=subject_term,
        predicate=predicate_term,
        object=object_term,
        # For a STRUCTURED particle the stamp records what parser *read* this
        # triple, not what tooling derived it — the same three fields, read in
        # the other direction.
        structurizer_id=EXTRACTOR_ID,
        structurizer_version=EXTRACTOR_VERSION,
    )
