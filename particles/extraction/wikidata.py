# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Wikidata-specific extractor.

Reads the Wikibase REST API JSON blob stored at deposit time and converts
each statement into a CandidateParticle. No LLM is involved.

**Structure-canonical**. A Wikibase statement
*is* a triple: the blob names the subject (its own Q-id), the predicate (the
P-id) and the object (a Q-id or a typed value). Every candidate therefore
carries the ``structured_claim`` built from those identifiers directly
— ``wd:`` entity IRIs and ``wdt:`` truthy property IRIs, no round-trip through
the English labels — and is marked ``CanonicalForm.STRUCTURED``: the triple is
the assertion Wikidata published, and ``content`` is this SDK's rendering of it.
Labels keep serving ``content`` exactly as before, so the prose is unchanged.

The one place this diverges from the RDF extractor's shape (``rdf.py``, the
other structure-canonical producer) is the network. ``rdf.py`` may not fetch,
because a deposited RDF document carries its own labels and a derivation that
reached a remote service would be irreproducible for no gain. A Wikibase entity
blob carries labels for *itself* and for nothing it references, so verbalizing
``P19 → Q350`` as "place of birth: Cambridge" is only possible from the API that
published the statement in the first place. That fetch is confined to
``content``: the triple — the half that is now the assertion — is built purely
from the deposited bytes and is bit-identical on every re-extraction. The
label-derived prose was already network-derived and already immutable once
minted; marking the triple canonical changes which half a reader
treats as authoritative, and nothing else.

Property and item labels are fetched from the REST API on first use and
cached in memory for the process lifetime.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from collections.abc import Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

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

log = logging.getLogger(__name__)

EXTRACTOR_ID = "wikidata-extractor"
#: 0.2.0 — structure-canonical emission. Prose ``content`` is
#: byte-identical to 0.1.0's; the particles gain the triple and the
#: ``STRUCTURED`` marker, which is exactly what
#: ``particles reindex --extractor-version 0.1.0`` exists to backfill.
EXTRACTOR_VERSION = "0.2.0"
SOURCE_TYPE = "WIKIDATA_API"
DEFAULT_TRUST_WEIGHT = 0.90
APPLICABILITY = [
    ApplicabilityClause(
        keyword="MUST",
        domain_uri="http://www.wikidata.org/entity/Q2013",
        domain_label="Wikidata",
        source_types=[SOURCE_TYPE],
    )
]

_WIKIDATA_URL_RE = re.compile(r"https://www\.wikidata\.org/(?:wiki|entity)/(Q\d+)")
_WIKIDATA_REST = "https://www.wikidata.org/w/rest.php/wikibase/v1"
_ABSOLUTE_IRI = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")

# In-process label caches (property labels and item labels rarely change)
_property_label_cache: dict[str, str] = {}
_item_label_cache: dict[str, str] = {}

# Data types we skip (no useful text rendering)
_SKIP_DATA_TYPES = {"commonsMedia", "geo-shape", "tabular-data", "globe-coordinate", "math"}

# Properties we skip regardless of value type
# P910 topic's main category, P301 category's main topic, P373 Commons category,
# P935 Commons gallery, P18 image, P948 page banner, P94 coat of arms image
_SKIP_PROPERTY_IDS = {"P910", "P301", "P373", "P935", "P18", "P948", "P94"}


# ---------------------------------------------------------------------------
# Structured claim — the triple Wikidata published
# ---------------------------------------------------------------------------

#: ``wd:`` — every Wikibase entity, item and property alike, lives here.
WD_ENTITY_PREFIX = "http://www.wikidata.org/entity/"
#: ``wdt:`` — the *truthy* predicate, i.e. the one-triple statement form.
#: Deprecated-rank statements are skipped before this is reached, matching
#: Wikidata's own rule that they get no truthy triple. A normal-rank statement
#: sharing a property with a preferred one is a smaller deviation the same rule
#: would exclude; it is emitted anyway, because dropping it would silence a
#: claim the prose half has always carried, and ``rank`` already reaches
#: confidence through ``wikidata.rank_confidence``.
WDT_PROP_PREFIX = "http://www.wikidata.org/prop/direct/"
_XSD = "http://www.w3.org/2001/XMLSchema#"

#: The namespace the Wikidata authority mints its refs under; reused
#: here so the ref this extractor attaches collides with — rather than
#: duplicates — the one subject resolution produces.
EXTERNAL_REF_NAMESPACE = "wikidata"


def entity_uri(entity_id: str) -> str:
    """Absolute IRI for a Wikibase entity id (``Q42`` → ``wd:Q42``)."""
    return f"{WD_ENTITY_PREFIX}{entity_id}"


def external_ref_for(entity_id: str) -> ExternalRef:
    """The Subject external ref for a Q-id, keyed to give ``bind_subject_id``
    its URI rung a key to match the triple's subject term on."""
    return ExternalRef(namespace=EXTERNAL_REF_NAMESPACE, id=entity_id, uri=entity_uri(entity_id))


def _time_literal(raw: str) -> str | None:
    """Normalise a Wikibase time value to an ``xsd:dateTime`` lexical form.

    ``+1949-10-07T00:00:00Z`` → ``1949-10-07T00:00:00Z``. At year or month
    precision Wikidata writes ``00`` for the unknown components; its own truthy
    serialisation normalises those to ``01`` so the literal stays a valid
    ``xsd:dateTime``, and so does this. The BCE sign is preserved.
    """
    if not raw:
        return None
    sign = "-" if raw.startswith("-") else ""
    date_part, _, time_part = raw.lstrip("+-").partition("T")
    bits = date_part.split("-")
    if len(bits) != 3 or not all(bit.isdigit() for bit in bits):
        return None
    year, month, day = bits
    month = "01" if month == "00" else month
    day = "01" if day == "00" else day
    return f"{sign}{year}-{month}-{day}T{time_part or '00:00:00Z'}"


def _object_term(data_type: str, content: object) -> ClaimTerm | None:
    """Render a statement's value as the object term of its triple.

    Follows Wikidata's own published RDF mapping for the truthy form, so the
    stored triple is the one an ``https://query.wikidata.org`` reader would see:
    items become ``wd:`` IRIs, times ``xsd:dateTime``, quantities
    ``xsd:decimal``, monolingual text a language-tagged literal, and plain /
    external-id strings untyped literals. ``None`` when the value has no honest
    term, which leaves the particle prose-canonical rather than losing it.
    """
    if data_type == "wikibase-item":
        qid = str(content)
        return ClaimTerm(kind=TermKind.URI, value=entity_uri(qid)) if qid else None

    if data_type == "time":
        if not isinstance(content, dict):
            return None
        lexical = _time_literal(str(content.get("time", "")))
        if lexical is None:
            return None
        return ClaimTerm(kind=TermKind.LITERAL, value=lexical, datatype=f"{_XSD}dateTime")

    if data_type == "quantity":
        if not isinstance(content, dict):
            return None
        amount = _render_quantity(content)
        if not amount:
            return None
        return ClaimTerm(kind=TermKind.LITERAL, value=amount, datatype=f"{_XSD}decimal")

    if data_type == "monolingualtext":
        if not isinstance(content, dict):
            return None
        text = str(content.get("text", ""))
        language = str(content.get("language", "")) or None
        if not text:
            return None
        return ClaimTerm(kind=TermKind.LITERAL, value=text, language=language)

    if data_type == "url":
        url = str(content)
        if not url:
            return None
        # Wikidata serialises the ``url`` datatype as an IRI node, not a literal.
        if _ABSOLUTE_IRI.match(url):
            return ClaimTerm(kind=TermKind.URI, value=url)
        return ClaimTerm(kind=TermKind.LITERAL, value=url, datatype=f"{_XSD}anyURI")

    if data_type in ("string", "external-id"):
        text = str(content)
        return ClaimTerm(kind=TermKind.LITERAL, value=text) if text else None

    return None


def _structured_claim(
    entity_id: str, prop_id: str, data_type: str, content: object
) -> StructuredClaim | None:
    """Build the whole triple, or ``None`` when the object has no term."""
    object_term = _object_term(data_type, content)
    if object_term is None:
        return None
    return StructuredClaim(
        subject=ClaimTerm(kind=TermKind.URI, value=entity_uri(entity_id)),
        predicate=ClaimTerm(kind=TermKind.URI, value=f"{WDT_PROP_PREFIX}{prop_id}"),
        object=object_term,
        # For a STRUCTURED particle the stamp records what *read* the triple
        # from the source rather than what derived it from prose.
        structurizer_id=EXTRACTOR_ID,
        structurizer_version=EXTRACTOR_VERSION,
    )


# ---------------------------------------------------------------------------
# Label fetching
# ---------------------------------------------------------------------------

# Inverted persistent-label-cache coupling. The L1
# in-process dict and the L3 live REST fetch are Client-safe; only the L2
# persistent DB cache (``wikidata_cache.get_label`` / ``set_label`` and its
# own ``session_scope``) reaches the Engine. So the L2 layer is *injected*: the
# Engine registers a cache that consults the store on a miss, runs the supplied
# live fetcher, and persists the result. When unregistered (pure Client,
# store-free) the lookup falls back to L1 + L3 only — labels resolve live but
# are not persisted across runs. Mirrors ``incremental``'s carry-forward hook.
#
# Signature: (entity_id, fetch_live, session) -> label. ``fetch_live`` is the
# Client-side L3 closure; ``session`` is the caller's session or None.
LabelCache = Callable[[str, Callable[[], Awaitable[str]], "AsyncSession | None"], Awaitable[str]]
_label_cache: LabelCache | None = None


def register_label_cache(cache: LabelCache) -> None:
    """Register the Engine-side persistent label cache."""
    global _label_cache
    _label_cache = cache


async def _fetch_label(
    entity_id: str, url: str, fallback: str, session: AsyncSession | None = None
) -> str:
    """Fetch a label with three-level cache: in-process → DB → API.

    L1 (in-process) and L3 (live REST) live here on the Client. The L2 DB cache
    is injected via :func:`register_label_cache`; without it, only L1 + L3 run.
    """
    # L1: in-process dict (current process lifetime)
    cache = _property_label_cache if entity_id.startswith("P") else _item_label_cache
    if entity_id in cache:
        return cache[entity_id]

    async def _fetch_live() -> str:
        # L3: Wikidata REST API
        from particles.http import get_capped, particles_client

        label = fallback
        try:
            async with particles_client(timeout=10.0) as client:
                resp = await get_capped(client, url)
                resp.raise_for_status()
                data = resp.json()
                label = str(data.get("labels", {}).get("en", fallback))
        except Exception as exc:
            log.warning("Failed to fetch label for %s: %s", entity_id, exc)
        return label

    # L2: persistent DB cache (Engine-injected); pure-Client falls back to L3.
    if _label_cache is not None:
        label = await _label_cache(entity_id, _fetch_live, session)
    else:
        label = await _fetch_live()

    cache[entity_id] = label
    return label


async def _fetch_property_label(pid: str, session: AsyncSession | None = None) -> str:
    """Return English label for a Wikidata property ID, e.g. 'P571' → 'inception'."""
    return await _fetch_label(
        pid, f"{_WIKIDATA_REST}/entities/properties/{pid}", fallback=pid, session=session
    )


async def _fetch_item_label(qid: str, session: AsyncSession | None = None) -> str:
    """Return English label for a Wikidata item QID, e.g. 'Q64' → 'Berlin'."""
    return await _fetch_label(
        qid, f"{_WIKIDATA_REST}/entities/items/{qid}", fallback=qid, session=session
    )


# ---------------------------------------------------------------------------
# Value rendering
# ---------------------------------------------------------------------------


def _render_time(content: dict[str, object]) -> str:
    """Render a Wikibase time value to a human-readable string."""
    raw = str(content.get("time", ""))
    precision_raw = content.get("precision", 11)
    precision = int(precision_raw) if isinstance(precision_raw, int | float | str) else 11
    # raw format: "+1949-10-07T00:00:00Z" or "-0044-03-15T00:00:00Z"
    sign = "-" if raw.startswith("-") else ""
    raw = raw.lstrip("+-")
    parts = raw.split("T")[0].split("-")
    year = sign + parts[0].lstrip("0") if parts else "?"
    month_names = [
        "",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    if precision >= 11 and len(parts) >= 3:
        try:
            d, m = int(parts[2]), int(parts[1])
            return f"{d} {month_names[m]} {year}"
        except (ValueError, IndexError):
            pass
    if precision >= 10 and len(parts) >= 2:
        try:
            m = int(parts[1])
            return f"{month_names[m]} {year}"
        except (ValueError, IndexError):
            pass
    return year or raw


def _render_quantity(content: dict[str, object]) -> str:
    amount = str(content.get("amount", "")).lstrip("+")
    # Strip trailing zeros after decimal point
    with contextlib.suppress(ValueError):
        amount = str(int(float(amount))) if "." not in amount else amount
    return amount


async def _render_value(
    data_type: str,
    value: dict[str, object],
    session: AsyncSession | None = None,
) -> tuple[str, list[str]]:
    """Render a statement value.

    Returns (rendered_text, related_qids).
    related_qids is non-empty only for wikibase-item values.
    """
    if value.get("type") != "value":
        return "", []

    content = value.get("content")

    if data_type == "wikibase-item":
        qid = str(content)
        label = await _fetch_item_label(qid, session=session)
        # Skip items with no English label (Wikidata meta-items, duplicates, etc.)
        if label == qid:
            return "", []
        return label, [qid]

    if data_type == "time":
        if not isinstance(content, dict):
            return "", []
        return _render_time(content), []

    if data_type in ("string", "external-id", "url"):
        return str(content), []

    if data_type == "monolingualtext":
        if isinstance(content, dict):
            text = str(content.get("text", ""))
            lang = str(content.get("language", ""))
            return f"{text} ({lang})" if lang != "en" else text, []
        return "", []

    if data_type == "quantity":
        if isinstance(content, dict):
            return _render_quantity(content), []
        return "", []

    return "", []


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------


class WikidataExtractor:
    """Extracts particles from a stored Wikibase REST API JSON blob."""

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
        session: AsyncSession | None = kwargs.get("session")  # type: ignore[assignment]
        entity = json.loads(content)
        entity_id: str = entity.get("id", "")
        label: str = entity.get("labels", {}).get("en", entity_id)
        statements: dict[str, list[dict[str, object]]] = entity.get("statements", {})

        if not label or not entity_id:
            return ExtractionResult(candidates=[], quality_notes=["Missing id or English label"])

        candidates: list[CandidateParticle] = []
        notes: list[str] = []

        for prop_id, stmt_list in statements.items():
            if prop_id in _SKIP_PROPERTY_IDS:
                continue

            prop_label = await _fetch_property_label(prop_id, session=session)

            for stmt in stmt_list:
                rank: str = str(stmt.get("rank", "normal"))
                if rank == "deprecated":
                    continue  # skip deprecated statements

                prop_info = stmt.get("property", {})
                data_type: str = (
                    str(prop_info.get("data_type", "")) if isinstance(prop_info, dict) else ""
                )

                if data_type in _SKIP_DATA_TYPES:
                    continue

                value = stmt.get("value", {})
                if not isinstance(value, dict):
                    continue

                rendered, related_qids = await _render_value(data_type, value, session=session)
                if not rendered:
                    continue

                content_str = f"{label} {prop_label}: {rendered}"
                from particles.config import get_config

                confidence = getattr(get_config().wikidata.rank_confidence, rank, 0.85)

                # Primary subject = this entity's QID; related = QIDs from item values
                subjects = [entity_id] + related_qids

                # The triple, from the deposited identifiers alone — no label
                # round-trip. A value type with no honest term (the tolerant
                # backstop) leaves the particle prose-canonical
                # rather than losing the claim.
                claim = _structured_claim(entity_id, prop_id, data_type, value.get("content"))

                candidates.append(
                    CandidateParticle(
                        content=content_str,
                        confidence_value=confidence,
                        uncertainty_nature=UncertaintyNature.EPISTEMIC,
                        subjects=subjects,
                        structured_claim=claim,
                        canonical_form=(CanonicalForm.STRUCTURED if claim else CanonicalForm.PROSE),
                        # Keyed by subject name (a Q-id here), which is how the
                        # pipeline zips refs onto resolved Subjects and how
                        # ``bind_subject_id``'s URI rung finds the triple's
                        # subject.
                        external_refs={qid: external_ref_for(qid) for qid in subjects},
                    )
                )

        log.info(
            "Wikidata extractor: %d candidates from %s (%s)",
            len(candidates),
            entity_id,
            label,
        )
        return ExtractionResult(candidates=candidates, quality_notes=notes)
