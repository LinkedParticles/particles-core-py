# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Numista issuer-level extractor.

Reads ``NUMISTA_API_ISSUER`` blobs — a combined paginated
``GET /types?issuer=`` response stored by the importer — and emits one
structured particle per coin using the summary fields available from the
issuer search API. Fields only available from the individual coin endpoint
(currency, demonetization, edge, obverse / reverse text) are absent — to get
those, deposit each coin page individually.
"""

from __future__ import annotations

import json
import logging

from particles.core.schema import ApplicabilityClause, CanonicalForm, Snapshot
from particles.extraction.general import CandidateParticle, ExtractionResult
from particles.extraction.numista._shared import (
    EXTRACTOR_ID_ISSUER,
    EXTRACTOR_VERSION_ISSUER,
    SOURCE_TYPE_ISSUER,
    _candidate,
    _claim,
    _coin_label,
    _extract_catalog_refs,
    _issuer_suffix,
)

log = logging.getLogger(__name__)


class NumistaIssuerExtractor:
    """Extracts structured particles from a combined Numista issuer search response.

    Produces one structured particle per coin using the summary fields available
    from the issuer search API (composition, weight, diameter, catalog refs).
    Fields only available from the individual coin endpoint (currency, demonetization,
    edge, obverse/reverse) are absent — deposit individual coin pages for full infoboxes.
    """

    EXTRACTOR_ID: str = EXTRACTOR_ID_ISSUER
    EXTRACTOR_VERSION: str = EXTRACTOR_VERSION_ISSUER
    DEFAULT_TRUST_WEIGHT: float = 0.85
    APPLICABILITY = [
        ApplicabilityClause(
            keyword="MUST",
            domain_uri="http://www.wikidata.org/entity/Q631286",
            domain_label="numismatics",
            source_types=[SOURCE_TYPE_ISSUER],
        )
    ]

    def accepts(self, source_type: str) -> bool:
        return source_type == SOURCE_TYPE_ISSUER

    async def extract(
        self,
        snapshot: Snapshot,
        content: bytes,
        **kwargs: object,
    ) -> ExtractionResult:
        envelope: dict[str, object] = json.loads(content)
        issuer_code = str(envelope.get("issuer_code", ""))
        issuer_name = str(envelope.get("issuer_name", ""))
        suffix = _issuer_suffix(issuer_code, issuer_name)
        types = envelope.get("types") or []

        if not isinstance(types, list):
            return ExtractionResult(quality_notes=["No 'types' array in issuer blob"])

        all_candidates: list[CandidateParticle] = []
        notes: list[str] = []

        for coin in types:
            if not isinstance(coin, dict):
                continue
            title = str(coin.get("title", "")).strip()
            if not title:
                continue

            min_y_raw = coin.get("min_year")
            max_y_raw = coin.get("max_year")
            min_y = int(min_y_raw) if isinstance(min_y_raw, int | float) else None
            max_y = int(max_y_raw) if isinstance(max_y_raw, int | float) else None
            label = _coin_label(title, min_y, max_y, suffix)

            properties: dict[str, object] = {}
            subject_classes: dict[str, str] = {label: "nmo:NumismaticObject"}

            if issuer_name:
                properties["nmo:hasIssuer"] = issuer_name
                subject_classes[issuer_name] = "nmo:Issuer"

            # Type — prefer specific 'type' field over generic 'category'
            type_str = str(coin.get("type") or "").strip()
            if not type_str:
                obj_type_obj = coin.get("object_type")
                type_str = str(
                    obj_type_obj.get("name", "") if isinstance(obj_type_obj, dict) else ""
                ).strip()
            if type_str:
                properties["nmo:hasObjectType"] = type_str
                subject_classes[type_str] = "nmo:ObjectType"

            composition = coin.get("composition")
            comp_text = ""
            if isinstance(composition, dict):
                comp_text = str(composition.get("text", "")).strip()
            elif isinstance(composition, str):
                comp_text = composition.strip()
            if comp_text:
                properties["nmo:hasMaterial"] = comp_text
                subject_classes[comp_text] = "nmo:Material"

            weight = coin.get("weight")
            if isinstance(weight, int | float):
                properties["nmo:hasWeight"] = float(weight)

            size = coin.get("size")
            if isinstance(size, int | float):
                properties["nmo:hasDiameter"] = float(size)

            if min_y is not None:
                yr = str(min_y) if (max_y is None or max_y == min_y) else f"{min_y}-{max_y}"
                properties["nmo:hasProductionDate"] = yr

            references = coin.get("references") or []
            ref_strs = _extract_catalog_refs(references) if isinstance(references, list) else []
            if ref_strs:
                properties["nuds:references"] = ref_strs

            numista_id = coin.get("id")
            if numista_id is not None:
                properties["numista:id"] = str(numista_id)
                properties["numista:url"] = (
                    f"https://en.numista.com/catalogue/pieces{numista_id}.html"
                )

            link_subjects = [
                str(v)
                for k, v in properties.items()
                if k in ("nmo:hasMaterial", "nmo:hasObjectType") and isinstance(v, str)
            ]
            all_subjects = list(
                dict.fromkeys([label] + ([issuer_name] if issuer_name else []) + link_subjects)
            )

            summary_parts: list[str] = []
            if comp_text:
                summary_parts.append(f"made of {comp_text.lower()}")
            if "nmo:hasWeight" in properties:
                summary_parts.append(f"weight {properties['nmo:hasWeight']}g")
            if "nmo:hasDiameter" in properties:
                summary_parts.append(f"diameter {properties['nmo:hasDiameter']}mm")
            summary = label + (": " + ", ".join(summary_parts) + "." if summary_parts else ".")

            # PROSE-canonical: the summary states every property in
            # `properties`.
            all_candidates.append(
                _candidate(
                    summary,
                    all_subjects,
                    confidence=0.90,
                    properties=properties,
                    subject_classes=subject_classes,
                    structured_claim=_claim(
                        label,
                        "rdf:type",
                        "nmo:NumismaticObject",
                        extractor_id=EXTRACTOR_ID_ISSUER,
                        extractor_version=EXTRACTOR_VERSION_ISSUER,
                        object_is_entity=True,
                    ),
                )
            )

            for ref_str in ref_strs:
                all_candidates.append(
                    _candidate(
                        f"{label} is catalogued as {ref_str}.",
                        [label],
                        confidence=0.90,
                        structured_claim=_claim(
                            label,
                            "nuds:references",
                            ref_str,
                            extractor_id=EXTRACTOR_ID_ISSUER,
                            extractor_version=EXTRACTOR_VERSION_ISSUER,
                        ),
                        canonical_form=CanonicalForm.STRUCTURED,
                    )
                )

        log.info(
            "Numista issuer extractor: %d candidates from %d coin types (%s)",
            len(all_candidates),
            len(types) if isinstance(types, list) else 0,
            issuer_code,
        )
        return ExtractionResult(candidates=all_candidates, quality_notes=notes)
