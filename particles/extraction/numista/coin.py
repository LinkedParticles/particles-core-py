# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Numista coin-level extractor.

Reads ``NUMISTA_API_COIN`` blobs — a single Numista coin-type API response
(``GET /types/{id}``) — and produces a structured Nomisma-aligned particle
for the coin itself plus descriptive particles for the obverse / reverse
text, mints, and catalog references. No network calls at extraction time.
"""

from __future__ import annotations

import json
import logging

from particles.core.schema import ApplicabilityClause, CanonicalForm, Snapshot
from particles.extraction.general import CandidateParticle, ExtractionResult
from particles.extraction.numista._shared import (
    EXTRACTOR_ID_COIN,
    EXTRACTOR_VERSION_COIN,
    SOURCE_TYPE_COIN,
    _candidate,
    _claim,
    _coin_label,
    _extract_catalog_refs,
    _issuer_suffix,
)

log = logging.getLogger(__name__)


def _structured(
    content: str,
    subjects: list[str],
    subject: str,
    predicate: str,
    obj: str | float,
    *,
    confidence: float = 0.95,
    object_is_entity: bool = False,
) -> CandidateParticle:
    """One structure-canonical candidate: ``content`` renders exactly this triple."""
    return _candidate(
        content,
        subjects,
        confidence=confidence,
        structured_claim=_claim(
            subject,
            predicate,
            obj,
            extractor_id=EXTRACTOR_ID_COIN,
            extractor_version=EXTRACTOR_VERSION_COIN,
            object_is_entity=object_is_entity,
        ),
        canonical_form=CanonicalForm.STRUCTURED,
    )


class NumistaCoinExtractor:
    """Extracts rich particles from a single Numista coin-type API response."""

    EXTRACTOR_ID: str = EXTRACTOR_ID_COIN
    EXTRACTOR_VERSION: str = EXTRACTOR_VERSION_COIN
    DEFAULT_TRUST_WEIGHT: float = 0.90
    APPLICABILITY = [
        ApplicabilityClause(
            keyword="MUST",
            domain_uri="http://www.wikidata.org/entity/Q631286",
            domain_label="numismatics",
            source_types=[SOURCE_TYPE_COIN],
        )
    ]

    def accepts(self, source_type: str) -> bool:
        return source_type == SOURCE_TYPE_COIN

    async def extract(
        self,
        snapshot: Snapshot,
        content: bytes,
        **kwargs: object,
    ) -> ExtractionResult:
        data: dict[str, object] = json.loads(content)
        candidates: list[CandidateParticle] = []
        notes: list[str] = []

        title = str(data.get("title", ""))
        min_year = data.get("min_year")
        max_year = data.get("max_year")
        min_y = int(min_year) if isinstance(min_year, int | float) else None
        max_y = int(max_year) if isinstance(max_year, int | float) else None

        issuer_obj = data.get("issuer") or {}
        if isinstance(issuer_obj, dict):
            issuer_code = str(issuer_obj.get("code", ""))
            issuer_name = str(issuer_obj.get("name", ""))
            period_obj = issuer_obj.get("period")
            period_name = (
                str(period_obj.get("name", "")).strip() if isinstance(period_obj, dict) else ""
            )
        else:
            issuer_code = issuer_name = period_name = ""

        suffix = _issuer_suffix(issuer_code, issuer_name)
        label = _coin_label(title, min_y, max_y, suffix)

        if not label.strip():
            return ExtractionResult(quality_notes=["Missing coin title"])

        # Resolve issuer subject name — prefer Wikidata QID from ruler array
        ruler_list = data.get("ruler") or []
        issuer_subjects: list[str] = []
        if isinstance(ruler_list, list) and ruler_list:
            for ruler in ruler_list:
                if isinstance(ruler, dict):
                    qid = str(ruler.get("wikidata_id", ""))
                    if qid:
                        issuer_subjects.append(qid)
                        break
        if not issuer_subjects and issuer_name:
            issuer_subjects.append(issuer_name)

        # ---------------------------------------------------------------
        # Build structured properties dict (Nomisma mapping)
        # ---------------------------------------------------------------
        properties: dict[str, object] = {}
        subject_classes: dict[str, str] = {label: "nmo:NumismaticObject"}

        if issuer_name:
            properties["nmo:hasIssuer"] = issuer_name
            subject_classes[issuer_name] = "nmo:Issuer"
        if period_name:
            properties["nmo:hasAuthority"] = period_name
            subject_classes[period_name] = "nmo:Authority"

        # nmo:hasObjectType from `type` / `object_type.name` (e.g. "Standard circulation coins").
        # `category` is the broad class ("coin") already captured by Instance of: [[Coin]].
        type_str = str(data.get("type") or "").strip()
        if not type_str:
            obj_type_obj = data.get("object_type")
            type_str = str(
                obj_type_obj.get("name", "") if isinstance(obj_type_obj, dict) else ""
            ).strip()
        if type_str:
            properties["nmo:hasObjectType"] = type_str
            subject_classes[type_str] = "nmo:ObjectType"

        value_obj = data.get("value")
        if isinstance(value_obj, dict):
            value_text = str(value_obj.get("text", "")).strip()
            currency_obj = value_obj.get("currency")
            currency_name = ""
            if isinstance(currency_obj, dict):
                currency_name = str(
                    currency_obj.get("full_name") or currency_obj.get("name") or ""
                ).strip()
            if value_text:
                properties["nmo:hasFaceValue"] = value_text
            if currency_name:
                properties["nmo:hasDenomination"] = currency_name
                subject_classes[currency_name] = "nmo:Denomination"
        elif isinstance(value_obj, str) and value_obj.strip():
            properties["nmo:hasFaceValue"] = value_obj.strip()

        composition = data.get("composition")
        comp_text = ""
        if isinstance(composition, dict):
            comp_text = str(composition.get("text", "")).strip()
        elif isinstance(composition, str):
            comp_text = composition.strip()
        if comp_text:
            properties["nmo:hasMaterial"] = comp_text
            subject_classes[comp_text] = "nmo:Material"

        weight = data.get("weight")
        if isinstance(weight, int | float):
            properties["nmo:hasWeight"] = float(weight)

        size = data.get("size")
        if isinstance(size, int | float):
            properties["nmo:hasDiameter"] = float(size)

        thickness = data.get("thickness")
        if isinstance(thickness, int | float):
            properties["nmo:hasDepth"] = float(thickness)

        shape_obj = data.get("shape")
        shape_text = (
            str(shape_obj.get("text", "")).strip()
            if isinstance(shape_obj, dict)
            else str(shape_obj or "").strip()
        )
        if shape_text:
            properties["nmo:hasShape"] = shape_text

        technique_obj = data.get("technique")
        technique_text = (
            str(technique_obj.get("text", "")).strip()
            if isinstance(technique_obj, dict)
            else str(technique_obj or "").strip()
        )
        if technique_text:
            properties["nmo:hasManufacture"] = technique_text

        orientation = str(data.get("orientation") or "").strip()
        if orientation:
            properties["nmo:hasAxis"] = orientation

        edge = data.get("edge")
        edge_desc = ""
        if isinstance(edge, dict):
            edge_desc = str(edge.get("description", "")).strip()
            if edge_desc:
                properties["nmo:hasEdge"] = edge_desc

        if min_y is not None:
            yr = str(min_y) if (max_y is None or max_y == min_y) else f"{min_y}-{max_y}"
            properties["nmo:hasProductionDate"] = yr

        demonetization_raw = data.get("demonetization")
        demonetization = ""
        if isinstance(demonetization_raw, dict):
            if demonetization_raw.get("is_demonetized"):
                demonetization = str(demonetization_raw.get("demonetization_date", "")).strip()
        elif isinstance(demonetization_raw, str):
            demonetization = demonetization_raw.strip()
        if demonetization:
            properties["nuds:demonetizationDate"] = demonetization

        references = data.get("references") or []
        ref_strs = _extract_catalog_refs(references) if isinstance(references, list) else []
        if ref_strs:
            properties["nuds:references"] = ref_strs

        numista_id = data.get("id")
        if numista_id is not None:
            properties["numista:id"] = str(numista_id)
            properties["numista:url"] = f"https://en.numista.com/catalogue/pieces{numista_id}.html"

        # Subjects for the structured particle: coin + issuer + all link targets
        link_subjects: list[str] = []
        for key in (
            "nmo:hasMaterial",
            "nmo:hasDenomination",
            "nmo:hasObjectType",
            "nmo:hasAuthority",
        ):
            val = properties.get(key)
            if isinstance(val, str) and val:
                link_subjects.append(val)

        structured_subjects = list(dict.fromkeys([label] + issuer_subjects + link_subjects))

        # Human-readable summary for the structured particle's content field
        summary_parts: list[str] = []
        if comp_text:
            summary_parts.append(f"made of {comp_text.lower()}")
        if "nmo:hasWeight" in properties:
            summary_parts.append(f"weight {properties['nmo:hasWeight']}g")
        if "nmo:hasDiameter" in properties:
            summary_parts.append(f"diameter {properties['nmo:hasDiameter']}mm")
        if demonetization:
            summary_parts.append(f"demonetized {demonetization}")
        summary = label + (": " + ", ".join(summary_parts) + "." if summary_parts else ".")

        # The infobox stays PROSE-canonical: `content` states every property in
        # `properties`, and a triple states one. Its annotation
        # is the type triple `subject_classes` already asserts — derived from
        # that metadata, not from the prose, which is exactly why the particle
        # is not STRUCTURED.
        candidates.append(
            _candidate(
                summary,
                structured_subjects,
                confidence=0.97,
                properties=properties,
                subject_classes=subject_classes,
                structured_claim=_claim(
                    label,
                    "rdf:type",
                    "nmo:NumismaticObject",
                    extractor_id=EXTRACTOR_ID_COIN,
                    extractor_version=EXTRACTOR_VERSION_COIN,
                    object_is_entity=True,
                ),
            )
        )

        # ---------------------------------------------------------------
        # Descriptive particles (free-text, no structured properties)
        # ---------------------------------------------------------------
        base_subjects = [label] + issuer_subjects

        obverse = data.get("obverse")
        if isinstance(obverse, dict):
            desc = str(obverse.get("description", "")).strip()
            if desc:
                candidates.append(
                    _structured(
                        f"The obverse of {label} depicts: {desc}",
                        base_subjects,
                        label,
                        "has obverse description",
                        desc,
                    )
                )
            lettering = str(obverse.get("lettering", "")).strip()
            if lettering:
                candidates.append(
                    _structured(
                        f"The obverse lettering of {label} reads: {lettering}",
                        base_subjects,
                        label,
                        "has obverse lettering",
                        lettering,
                    )
                )
            engravers = obverse.get("engravers") or []
            if isinstance(engravers, list):
                for e in engravers:
                    name = str(e.get("name", "")).strip() if isinstance(e, dict) else str(e).strip()
                    if name:
                        candidates.append(
                            _structured(
                                f"{label} was engraved by {name}.",
                                base_subjects + [name],
                                label,
                                "was engraved by",
                                name,
                                object_is_entity=True,
                            )
                        )

        reverse = data.get("reverse")
        if isinstance(reverse, dict):
            desc = str(reverse.get("description", "")).strip()
            if desc:
                candidates.append(
                    _structured(
                        f"The reverse of {label} depicts: {desc}",
                        base_subjects,
                        label,
                        "has reverse description",
                        desc,
                    )
                )
            lettering = str(reverse.get("lettering", "")).strip()
            if lettering:
                candidates.append(
                    _structured(
                        f"The reverse lettering of {label} reads: {lettering}",
                        base_subjects,
                        label,
                        "has reverse lettering",
                        lettering,
                    )
                )

        if edge_desc:
            candidates.append(
                _structured(
                    f"{label} has a {edge_desc.lower()} edge.",
                    base_subjects,
                    label,
                    "nmo:hasEdge",
                    edge_desc,
                )
            )

        mints = data.get("mints") or []
        if isinstance(mints, list):
            for mint in mints:
                if not isinstance(mint, dict):
                    continue
                mint_name = str(mint.get("name", "")).strip()
                if not mint_name:
                    continue
                mark = str(mint.get("mark", "") or "").strip()
                mint_subjects = base_subjects + [mint_name]
                subject_classes[mint_name] = "nmo:Mint"
                suffix_str = f" (mintmark: {mark})" if mark else ""
                mint_content = f"{label} was struck at {mint_name}{suffix_str}."
                mint_claim = _claim(
                    label,
                    "was struck at",
                    mint_name,
                    extractor_id=EXTRACTOR_ID_COIN,
                    extractor_version=EXTRACTOR_VERSION_COIN,
                    object_is_entity=True,
                )
                # With a mintmark the sentence states a second fact the triple
                # does not carry, so it fails the §2.1 test and keeps the
                # annotation at PROSE. Without one it renders the triple exactly.
                candidates.append(
                    _candidate(
                        mint_content,
                        mint_subjects,
                        structured_claim=mint_claim,
                        canonical_form=(CanonicalForm.PROSE if mark else CanonicalForm.STRUCTURED),
                    )
                )

        for ref_str in ref_strs:
            candidates.append(
                _structured(
                    f"{label} is catalogued as {ref_str}.",
                    base_subjects,
                    label,
                    "nuds:references",
                    ref_str,
                )
            )

        comments = str(data.get("comments") or "").strip()
        if comments:
            candidates.append(_candidate(comments, base_subjects, confidence=0.85))

        log.info("Numista coin extractor: %d candidates for %s", len(candidates), label)
        if not candidates:
            notes.append("No fields produced particles — check API response structure")

        return ExtractionResult(candidates=candidates, quality_notes=notes)
