"""Shared helpers for the Numista extractors and importer.

Cross-cutting code lives here:

- Identity constants for all three source types (``NUMISTA_API_COIN`` /
  ``_ISSUER`` / ``_LISTING``) — stored in the DB; **never** rename.
- The Numista API base URL and the two URL regexes the importer dispatches on.
- The ``_ISSUER_SUFFIX`` map + ``_issuer_suffix`` / ``_coin_label`` formatters
  that build the canonical subject name for a coin.
- ``_candidate`` — the ``CandidateParticle`` constructor every extractor uses
  to keep ``uncertainty_nature`` and other defaults consistent.
- ``_claim`` / ``_predicate_term`` — the structured-claim builder.
- ``_extract_catalog_refs`` — small parser for the Numista references array.

**Which candidates are structure-canonical.** A candidate is
``STRUCTURED`` exactly when its ``content`` is a deterministic rendering of its
``structured_claim`` and of no other fact — the per-field templated particles.
The entity infobox stays ``PROSE``: its ``content`` states fifteen facts while
a triple states one, so it carries a ``rdf:type`` annotation
instead. A candidate whose ``content`` is the source's own free text (the coin
extractor's ``comments``) carries no annotation at all.
"""

from __future__ import annotations

import re

from particles.core.jsonld_context import is_published_curie
from particles.core.schema import (
    CanonicalForm,
    ClaimTerm,
    StructuredClaim,
    TermKind,
    UncertaintyNature,
)
from particles.extraction.general import CandidateParticle

# ---------------------------------------------------------------------------
# Identity constants
# ---------------------------------------------------------------------------

EXTRACTOR_ID_COIN = "numista-coin-extractor"
EXTRACTOR_ID_ISSUER = "numista-issuer-extractor"
EXTRACTOR_ID_LISTING = "numista-listing-extractor"
EXTRACTOR_VERSION_COIN = "0.3.1"  # bumped: numismatics domain QID Q8148→Q631286
EXTRACTOR_VERSION_ISSUER = "0.3.1"  # bumped: numismatics domain QID Q8148→Q631286
EXTRACTOR_VERSION_LISTING = "0.2.1"  # bumped: numismatics domain QID Q8148→Q631286
SOURCE_TYPE_COIN = "NUMISTA_API_COIN"
SOURCE_TYPE_ISSUER = "NUMISTA_API_ISSUER"
SOURCE_TYPE_LISTING = "NUMISTA_LISTING_HTML"

NUMISTA_API_BASE = "https://api.numista.com/v3"
NUMISTA_COIN_RE = re.compile(r"en\.numista\.com/(?:catalogue/pieces(\d+)\.html|(\d+)/?$)")
NUMISTA_ISSUER_RE = re.compile(r"en\.numista\.com/catalogue/(?:index\.php|[^/]+-\d+\.html)")

# Numista issuer code → short suffix used in subject names
_ISSUER_SUFFIX: dict[str, str] = {
    "ddr": "GDR",
    "de": "Germany",
    "fr": "France",
    "gb": "UK",
    "us": "USA",
}


# ---------------------------------------------------------------------------
# Subject-name + particle constructors shared by every extractor
# ---------------------------------------------------------------------------


def _coin_label(title: str, min_year: int | None, max_year: int | None, suffix: str) -> str:
    """Build the canonical subject name for a coin, e.g. '1 Pfennig (1948-1950) GDR'."""
    if min_year and max_year:
        yr = str(min_year) if min_year == max_year else f"{min_year}-{max_year}"
        return f"{title} ({yr}) {suffix}".strip()
    return f"{title} {suffix}".strip()


def _issuer_suffix(issuer_code: str, issuer_name: str) -> str:
    return _ISSUER_SUFFIX.get(issuer_code.lower(), issuer_name)


def _predicate_term(predicate: str) -> ClaimTerm:
    """Render a predicate as a URI term when published, else as a TOKEN.

    "Published" means ``artifacts/schemas/context.jsonld`` carries the prefix
     — a predicate spelled with one is a ``URI`` term in sense, "an absolute IRI, or a CURIE in a context.jsonld prefix".
    Anything else is recorded honestly as a ``TOKEN`` rather than coerced into a
    namespace we would then have to defend.

    The set is read from the artifact, not hand-listed: the hand-list
    here named ``schema:``, which the context did not publish.
    """
    kind = TermKind.URI if is_published_curie(predicate) else TermKind.TOKEN
    return ClaimTerm(kind=kind, value=predicate)


def _claim(
    subject: str,
    predicate: str,
    obj: str | float,
    *,
    extractor_version: str,
    extractor_id: str,
    object_is_entity: bool = False,
    datatype: str | None = None,
) -> StructuredClaim:
    """Build one stamped S-P-O rendering of a parsed Numista field.

    The subject is the coin's canonical label as a ``TOKEN``: Numista publishes
    catalogue *page* URLs rather than entity IRIs, so there is no honest URI to
    put here. ``bind_subject_id``'s name rung binds it, because
    these extractors inject the very subject names their triples name.

    Args:
        subject: the candidate's canonical subject name.
        predicate: a CURIE from the alignment, or a lowercase verb
            phrase where that table supplies none.
        obj: the object value.
        extractor_version: stamped as ``structurizer_version``.
        extractor_id: stamped as ``structurizer_id`` — for a STRUCTURED
            particle the stamp records what *read* the triple, not what
            derived it.
        object_is_entity: True when the object names an entity (a mint, a
            material) rather than carrying a value.
        datatype: an ``xsd:`` datatype CURIE; LITERAL objects only.

    Returns:
        The stamped annotation.
    """
    if object_is_entity:
        object_term = ClaimTerm(kind=TermKind.TOKEN, value=str(obj))
    else:
        object_term = ClaimTerm(
            kind=TermKind.LITERAL,
            value=str(obj),
            datatype=datatype or ("xsd:decimal" if isinstance(obj, float | int) else "xsd:string"),
        )
    return StructuredClaim(
        subject=ClaimTerm(kind=TermKind.TOKEN, value=subject),
        predicate=_predicate_term(predicate),
        object=object_term,
        structurizer_id=extractor_id,
        structurizer_version=extractor_version,
    )


def _candidate(
    content: str,
    subjects: list[str],
    confidence: float = 0.95,
    properties: dict[str, object] | None = None,
    subject_classes: dict[str, str] | None = None,
    structured_claim: StructuredClaim | None = None,
    canonical_form: CanonicalForm = CanonicalForm.PROSE,
) -> CandidateParticle:
    return CandidateParticle(
        content=content,
        confidence_value=confidence,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        subjects=subjects,
        properties=properties,
        subject_classes=subject_classes or {},
        structured_claim=structured_claim,
        canonical_form=canonical_form,
    )


def _extract_catalog_refs(references: list[dict[str, object]]) -> list[str]:
    """Return a list of 'CODE# NUMBER' strings from a Numista references array."""
    refs: list[str] = []
    for ref in references:
        cat = ref.get("catalogue") or {}
        code = str(cat.get("code", "")) if isinstance(cat, dict) else ""
        number = str(ref.get("number", ""))
        if code and number:
            refs.append(f"{code}# {number}")
    return refs
