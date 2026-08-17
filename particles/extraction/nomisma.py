"""Nomisma Linked Open Data extractor and importer (importer naming).

Nomisma (nomisma.org) publishes dereferenceable LOD URIs for numismatic
concepts: materials, denominations, mints, object types, and issuing
authorities. Each URI resolves to JSON-LD via content negotiation.

The extractor reads stored JSON-LD and produces one structured particle per
entity, mapping SKOS/GEO predicates to properties and attaching a Nomisma
ExternalRef to the matching subject.

Actual JSON-LD format from Nomisma:
  - Top-level dict with "@context" (prefix map) and "@graph" (node list)
  - Entity @id uses the "nm:" prefix: "nm:al", "nm:berlin", etc.
  - Predicates use prefixed forms: "skos:prefLabel", "geo:lat", etc.
  - @type uses prefixed forms: "nmo:Material", "nmo:Mint", etc.
  - Geo coordinates live on a separate "#this" node linked via "geo:location"
"""

from __future__ import annotations

import json
import logging
import re

from particles.core.schema import (
    ApplicabilityClause,
    ClaimTerm,
    ExternalRef,
    Snapshot,
    StructuredClaim,
    TermKind,
    UncertaintyNature,
)
from particles.extraction.general import CandidateParticle, ExtractionResult

log = logging.getLogger(__name__)

EXTRACTOR_ID = "nomisma-extractor"
EXTRACTOR_VERSION = "0.3.1"  # bumped: numismatics domain QID Q8148→Q631286
SOURCE_TYPE = "NOMISMA_API"
DEFAULT_TRUST_WEIGHT = 0.95
APPLICABILITY = [
    ApplicabilityClause(
        keyword="MUST",
        domain_uri="http://www.wikidata.org/entity/Q631286",
        domain_label="numismatics",
        source_types=[SOURCE_TYPE],
    )
]

_NOMISMA_URL_RE = re.compile(r"https?://nomisma\.org/id/([\w\-]+)")

#: Spelled as an absolute IRI rather than the ``skos:`` CURIE the ``properties``
#: dict uses. Originally forced: only ``nmo:`` and
#: ``nuds:`` were published, and a CURIE is admitted as a ``URI`` term only when the
#: published context can expand it, so the CURIE would have degraded to
#: ``TOKEN``. Since then ``skos:`` has been published, so both spellings are now
#: legal ``URI`` terms — and this one is kept anyway. Switching would rewrite a
#: stored ``structured_claim`` predicate on every nomisma particle, which costs
#: an ``EXTRACTOR_VERSION`` bump, a reindex, and a baseline re-capture to buy
#: nothing: the absolute form is what an RDF export emits either way.
_SKOS_DEFINITION = "http://www.w3.org/2004/02/skos/core#definition"

# Nomisma @type values (prefixed) → subject_class strings
_TYPE_TO_CLASS: dict[str, str] = {
    "nmo:Material": "nmo:Material",
    "nmo:Denomination": "nmo:Denomination",
    "nmo:Mint": "nmo:Mint",
    "nmo:Authority": "nmo:Authority",
    "nmo:Issuer": "nmo:Issuer",
    "nmo:ObjectType": "nmo:ObjectType",
    "nmo:NumismaticObject": "nmo:NumismaticObject",
    "nmo:Region": "nmo:Region",
    "nmo:Collection": "nmo:Collection",
}


def _get_en(value: object) -> str | None:
    """Return the English @value from a JSON-LD value (list or single dict)."""
    if isinstance(value, dict):
        if value.get("@language") == "en":
            v = value.get("@value", "")
            return str(v) if v else None
        return None
    if isinstance(value, list):
        for item in value:
            result = _get_en(item)
            if result:
                return result
    return None


def _get_ids(value: object) -> list[str]:
    """Return @id strings from a JSON-LD reference value (list or single dict)."""
    if isinstance(value, dict) and "@id" in value:
        return [str(value["@id"])]
    if isinstance(value, list):
        return [str(v["@id"]) for v in value if isinstance(v, dict) and "@id" in v]
    return []


def _get_typed_decimal(value: object) -> float | None:
    """Return a float from a JSON-LD typed literal {"@type": "xsd:decimal", "@value": "..."}."""
    if isinstance(value, dict) and "@value" in value:
        try:
            return float(value["@value"])
        except (TypeError, ValueError):
            pass
    if isinstance(value, int | float):
        return float(value)
    return None


def _parse_graph(content: bytes) -> tuple[list[dict[str, object]], dict[str, str]]:
    """Parse Nomisma JSON-LD; return (nodes, prefix_map).

    Handles both {"@context": ..., "@graph": [...]} and bare list formats.
    """
    doc = json.loads(content)
    if isinstance(doc, dict):
        ctx = doc.get("@context", {})
        prefix_map: dict[str, str] = {k: v for k, v in ctx.items() if isinstance(v, str)}
        graph = doc.get("@graph", [])
        nodes = graph if isinstance(graph, list) else [graph]
    elif isinstance(doc, list):
        prefix_map = {}
        nodes = doc
    else:
        return [], {}
    return [n for n in nodes if isinstance(n, dict)], prefix_map


def _expand_id(node_id: str, prefix_map: dict[str, str]) -> str:
    """Expand a prefixed ID like 'nm:al' to 'http://nomisma.org/id/al'."""
    if node_id.startswith("http://") or node_id.startswith("https://"):
        return node_id
    if ":" in node_id:
        prefix, local = node_id.split(":", 1)
        base = prefix_map.get(prefix, "")
        if base:
            return base + local
    return node_id


def _as_str_list(value: object) -> list[str]:
    """Coerce a JSON-LD @type value (str or list) to a list of strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _find_main_node(
    nodes: list[dict[str, object]], prefix_map: dict[str, str]
) -> dict[str, object] | None:
    """Return the primary Nomisma concept node (nm:xxx without a fragment)."""
    nm_base = prefix_map.get("nm", "http://nomisma.org/id/")
    for node in nodes:
        node_id = str(node.get("@id", ""))
        expanded = _expand_id(node_id, prefix_map)
        if expanded.startswith(nm_base) and "#" not in expanded:
            return node
    # Fallback: first node with any nomisma.org/id/ @id
    for node in nodes:
        if _NOMISMA_URL_RE.search(str(node.get("@id", ""))):
            return node
    return None


def _find_geo_node(nodes: list[dict[str, object]], main_id: str) -> dict[str, object] | None:
    """Return the #this geo node linked from the main entity, if present."""
    geo_id = main_id + "#this" if not main_id.endswith("#this") else main_id
    for node in nodes:
        if str(node.get("@id", "")) == geo_id:
            return node
    return None


class NomismaExtractor:
    """Extracts a structured particle from a stored Nomisma JSON-LD blob."""

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
        nodes, prefix_map = _parse_graph(content)
        if not nodes:
            return ExtractionResult(quality_notes=["Empty or unparseable JSON-LD"])

        node = _find_main_node(nodes, prefix_map)
        if node is None:
            return ExtractionResult(quality_notes=["No Nomisma concept node found in JSON-LD"])

        # Derive canonical URI and concept ID from the node's @id
        raw_id = str(node.get("@id", ""))
        entity_uri = _expand_id(raw_id, prefix_map)
        concept_id = _concept_id_from_uri(entity_uri) or raw_id

        # English label from skos:prefLabel
        label_raw = node.get("skos:prefLabel")
        label = _get_en(label_raw) if label_raw is not None else None
        if not label:
            label = concept_id.replace("_", " ")

        # Nomisma class from @type (prefixed)
        nmo_class: str = "nmo:Concept"
        for t in _as_str_list(node.get("@type")):
            if t in _TYPE_TO_CLASS:
                nmo_class = _TYPE_TO_CLASS[t]
                break

        # Build properties
        properties: dict[str, object] = {}

        defn = _get_en(node.get("skos:definition"))
        if defn:
            properties["skos:definition"] = defn

        exact_ids = _get_ids(node.get("skos:exactMatch"))
        wikidata = next((u for u in exact_ids if "wikidata.org" in u), None)
        if wikidata:
            properties["skos:exactMatch"] = wikidata
        elif exact_ids:
            properties["skos:exactMatch"] = exact_ids

        close_ids = _get_ids(node.get("skos:closeMatch"))
        getty = next((u for u in close_ids if "getty.edu" in u), None)
        if getty:
            properties["skos:closeMatch"] = getty
        elif close_ids:
            properties["skos:closeMatch"] = close_ids[0] if len(close_ids) == 1 else close_ids

        # Geo coordinates from the linked #this node
        geo_node = _find_geo_node(nodes, entity_uri)
        if geo_node:
            lat = _get_typed_decimal(geo_node.get("geo:lat"))
            lon = _get_typed_decimal(geo_node.get("geo:long"))
            if lat is not None:
                properties["geo:lat"] = lat
            if lon is not None:
                properties["geo:long"] = lon

        # Human-readable content
        class_name = nmo_class.split(":")[-1]
        defn_str = str(properties.get("skos:definition", ""))
        if defn_str:
            particle_content = f"{label} (Nomisma {class_name}): {defn_str}"
        else:
            particle_content = f"{label} is a {class_name} in the Nomisma numismatic ontology."

        external_ref = ExternalRef(namespace="nomisma", id=concept_id, uri=entity_uri)

        # The subject term is the real entity IRI, not a label — Nomisma
        # publishes dereferenceable LOD URIs, so `bind_subject_id`'s URI rung
        # matches it against this candidate's external ref and
        # binds `subject_id`. That is the join back into the graph, and it is
        # why this extractor is the best-keyed member of the family
        #.
        #
        # PROSE-canonical either way: with a definition, `content` renders the
        # class *and* the definition — two facts, one triple — so it fails the
        # §2.1 test. The bare form renders exactly the type triple, but its
        # `content` is likewise the SDK's sentence rather than a derived
        # rendering of an asserted triple, and holding both branches at PROSE
        # keeps one extractor from emitting two canonical forms by accident of
        # which fields a source happened to fill in.
        subject_term = ClaimTerm(kind=TermKind.URI, value=entity_uri)
        if defn_str:
            claim = StructuredClaim(
                subject=subject_term,
                predicate=ClaimTerm(kind=TermKind.URI, value=_SKOS_DEFINITION),
                object=ClaimTerm(kind=TermKind.LITERAL, value=defn_str, language="en"),
                structurizer_id=EXTRACTOR_ID,
                structurizer_version=EXTRACTOR_VERSION,
            )
        else:
            claim = StructuredClaim(
                subject=subject_term,
                predicate=ClaimTerm(kind=TermKind.URI, value="rdf:type"),
                object=ClaimTerm(kind=TermKind.URI, value=nmo_class),
                structurizer_id=EXTRACTOR_ID,
                structurizer_version=EXTRACTOR_VERSION,
            )

        candidate = CandidateParticle(
            content=particle_content,
            confidence_value=0.99,
            uncertainty_nature=UncertaintyNature.EPISTEMIC,
            subjects=[label],
            properties=properties,
            subject_classes={label: nmo_class},
            external_refs={label: external_ref},
            structured_claim=claim,
        )

        log.info("Nomisma extractor: 1 candidate from %s (%s / %s)", concept_id, label, nmo_class)
        return ExtractionResult(candidates=[candidate])


def _concept_id_from_uri(uri: str) -> str:
    """Extract concept ID from a Nomisma URI or prefixed ID."""
    if uri.startswith("nm:"):
        return uri[3:]
    m = _NOMISMA_URL_RE.search(uri)
    return m.group(1) if m else uri
