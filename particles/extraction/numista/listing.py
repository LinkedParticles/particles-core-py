"""Numista listing HTML extractor.

Reads ``NUMISTA_LISTING_HTML`` blobs — public Numista catalogue listing
pages — and emits per-coin structured particles by parsing
``<div class="description_piece">`` elements. No API key, no per-coin API
calls required: composition, weight, diameter, catalog refs, and object
type all come straight out of the HTML.
"""

from __future__ import annotations

import logging
import re

from particles.core.schema import ApplicabilityClause, Snapshot
from particles.extraction.general import CandidateParticle, ExtractionResult
from particles.extraction.numista._shared import (
    EXTRACTOR_ID_LISTING,
    EXTRACTOR_VERSION_LISTING,
    SOURCE_TYPE_LISTING,
    _candidate,
    _claim,
    _coin_label,
    _issuer_suffix,
)

log = logging.getLogger(__name__)

_PHYS_RE = re.compile(
    r"^(.+?)\s*•\s*([\d.]+)\s*g\s*•\s*⌀\s*([\d.]+)\s*mm",
    re.IGNORECASE,
)
_YEARS_RE = re.compile(r"(\d{4})(?:-(\d{4}))?")


class NumistaListingExtractor:
    """Parses Numista catalogue listing HTML (NUMISTA_LISTING_HTML source type).

    Extracts per-coin structured particles from <div class="description_piece">
    elements. Parses composition, weight, diameter, catalog refs, and type
    directly from the HTML text — no API key, no per-coin API calls required.
    """

    EXTRACTOR_ID: str = EXTRACTOR_ID_LISTING
    EXTRACTOR_VERSION: str = EXTRACTOR_VERSION_LISTING
    DEFAULT_TRUST_WEIGHT: float = 0.80
    APPLICABILITY = [
        ApplicabilityClause(
            keyword="MUST",
            domain_uri="http://www.wikidata.org/entity/Q631286",
            domain_label="numismatics",
            source_types=[SOURCE_TYPE_LISTING],
        )
    ]

    def accepts(self, source_type: str) -> bool:
        return source_type == SOURCE_TYPE_LISTING

    async def extract(
        self,
        snapshot: Snapshot,
        content: bytes,
        **kwargs: object,
    ) -> ExtractionResult:
        from lxml import html as lxml_html
        from lxml.html import HtmlElement, HTMLParser

        doc: HtmlElement = lxml_html.fromstring(content, parser=HTMLParser(encoding="utf-8"))
        pieces: list[HtmlElement] = doc.xpath('//div[contains(@class,"description_piece")]')
        if not pieces:
            return ExtractionResult(quality_notes=["No description_piece divs found"])

        # Derive issuer code and name from the first flag image
        issuer_code = ""
        issuer_name = ""
        for piece in pieces:
            flag_imgs: list[HtmlElement] = piece.xpath('.//img[contains(@src,"/design/pays/")]')
            if flag_imgs:
                src = flag_imgs[0].get("src", "")
                m = re.search(r"/design/pays/([^./]+)\.", src)
                if m:
                    issuer_code = m.group(1)
                issuer_name = flag_imgs[0].get("title", "")
                break

        suffix = _issuer_suffix(issuer_code, issuer_name)

        candidates: list[CandidateParticle] = []
        notes: list[str] = []

        for piece in pieces:
            strong_links: list[HtmlElement] = piece.xpath(".//strong/a")
            if not strong_links:
                continue
            href = strong_links[0].get("href", "").strip("/")
            if not href.isdigit():
                continue
            coin_id = href

            full_text = piece.text_content()
            lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
            if len(lines) < 2:
                continue

            title = lines[0]
            years_str = lines[1]
            ym = _YEARS_RE.match(years_str)
            min_y = int(ym.group(1)) if ym else None
            max_y = int(ym.group(2)) if (ym and ym.group(2)) else min_y

            label = _coin_label(title, min_y, max_y, suffix)

            properties: dict[str, object] = {}
            subject_classes: dict[str, str] = {label: "nmo:NumismaticObject"}

            if issuer_name:
                properties["nmo:hasIssuer"] = issuer_name
                subject_classes[issuer_name] = "nmo:Issuer"

            # Type from <em> tag
            em_els: list[HtmlElement] = piece.xpath(".//em")
            if em_els:
                type_str = (
                    em_els[0]
                    .text_content()
                    .strip()
                    .replace("Coins › ", "")
                    .replace("Coins > ", "")
                    .strip()
                )
                if type_str:
                    properties["nmo:hasObjectType"] = type_str
                    subject_classes[type_str] = "nmo:ObjectType"

            # Physical and refs lines (after title, years, type)
            for line in lines[2:]:
                norm = line.replace("\xa0", " ").replace(" ", " ")
                phys = _PHYS_RE.match(norm)
                if phys:
                    material = phys.group(1).strip()
                    properties["nmo:hasMaterial"] = material
                    properties["nmo:hasWeight"] = float(phys.group(2))
                    properties["nmo:hasDiameter"] = float(phys.group(3))
                    subject_classes[material] = "nmo:Material"
                elif "#" in norm:
                    parts = [re.sub(r"\s+", " ", p).strip() for p in norm.split(",")]
                    ref_strs = [p for p in parts if "#" in p]
                    if ref_strs:
                        properties["nuds:references"] = ref_strs

            if min_y is not None:
                yr = str(min_y) if (max_y is None or max_y == min_y) else f"{min_y}-{max_y}"
                properties["nmo:hasProductionDate"] = yr

            properties["numista:id"] = coin_id
            properties["numista:url"] = f"https://en.numista.com/catalogue/pieces{coin_id}.html"

            link_subjects = [
                str(v)
                for k, v in properties.items()
                if k in ("nmo:hasMaterial", "nmo:hasObjectType") and isinstance(v, str)
            ]
            all_subjects = list(
                dict.fromkeys([label] + ([issuer_name] if issuer_name else []) + link_subjects)
            )

            summary_parts: list[str] = []
            if "nmo:hasMaterial" in properties:
                summary_parts.append(f"made of {str(properties['nmo:hasMaterial']).lower()}")
            if "nmo:hasWeight" in properties:
                summary_parts.append(f"weight {properties['nmo:hasWeight']}g")
            if "nmo:hasDiameter" in properties:
                summary_parts.append(f"diameter {properties['nmo:hasDiameter']}mm")
            summary = label + (": " + ", ".join(summary_parts) + "." if summary_parts else ".")

            # This extractor emits nothing but infobox particles, so the type
            # triple is its whole structured-claim coverage:
            # PROSE-canonical, since `content` states every parsed property.
            candidates.append(
                _candidate(
                    summary,
                    all_subjects,
                    confidence=0.92,
                    properties=properties,
                    subject_classes=subject_classes,
                    structured_claim=_claim(
                        label,
                        "rdf:type",
                        "nmo:NumismaticObject",
                        extractor_id=EXTRACTOR_ID_LISTING,
                        extractor_version=EXTRACTOR_VERSION_LISTING,
                        object_is_entity=True,
                    ),
                )
            )

        log.info(
            "Numista listing extractor: %d candidates from %d description_piece divs",
            len(candidates),
            len(pieces),
        )
        if not candidates:
            notes.append("No valid coin entries parsed from HTML listing")
        return ExtractionResult(candidates=candidates, quality_notes=notes)
