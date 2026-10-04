# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""JSON-LD codec for the vocabulary document (conventions).

Pure: no I/O. It follows the interchange codec's conventions (the published
``@context``, W3C vocabulary terms, a canonical byte form) but lives in
``core`` rather than ``particles.interchange``: the store and the ingest
pipeline read documents, and ``interchange`` already imports both, so a codec
there would close a subpackage cycle.

The document is written as JSON-LD under the published context
(``CONTEXT_URL``, the one interchange units reference) plus a small inline
context for what that context does not carry: SHACL, the document's own
prefix, and ``ppx:``, the profile annotations needed because SHACL cannot say
"one at a time". The vocabularies are the W3C ones that already
exist for each part, so a reader who knows them can read the document without
this SDK:

* each term is an ``rdf:Property`` and a ``skos:Concept``, minted in the
  document's namespace, with its canonical form as ``skos:prefLabel`` and each
  confirmed alias as ``skos:altLabel``;
* an alignment is ``owl:equivalentProperty`` when asserted, ``skos:exactMatch``
  or ``skos:closeMatch`` when estimated, plus the source's
  cardinality as recorded;
* a profile is an ``sh:PropertyShape`` targeted at the subject class:
  ``sh:maxCount 1`` for timeless single, ``ppx:oneAtATime`` and
  ``ppx:manyAtOnce`` for the kinds SHACL has no term for;
* every ruling is PROV: ``prov:wasAttributedTo``, ``prov:generatedAtTime`` and
  the evidence it was confirmed on.

Serialisation is canonical (sorted keys, two-space indent, one trailing
newline), so one document version has one byte form and one content hash in
the corpus. :func:`loads` is the inverse and is the validation gate: whatever
it accepts is a valid :class:`~particles.core.vocabulary.VocabularyDocument`.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from particles.core.jsonld_context import CONTEXT_URL
from particles.core.predicate_profile import PredicateRole, SlotKind
from particles.core.vocabulary import (
    Alias,
    Alignment,
    MatchStrength,
    Ruling,
    TermProfile,
    VocabularyDocument,
    VocabularyTerm,
    WikidataConstraint,
)

__all__ = [
    "DOCUMENT_TYPE",
    "PPX_NAMESPACE",
    "SHACL_NAMESPACE",
    "dumps",
    "from_jsonld",
    "is_vocabulary_document",
    "loads",
    "to_jsonld",
]

#: The namespace of ``ppx:oneAtATime`` and its siblings.
#: Declared in each document's inline context; the published particle context
#: stays unchanged.
PPX_NAMESPACE = "https://linkedparticles.org/vocab/profile#"
SHACL_NAMESPACE = "http://www.w3.org/ns/shacl#"
#: The ``@type`` that marks a JSON-LD document as a vocabulary document.
DOCUMENT_TYPE = "ppx:VocabularyDocument"

#: How a match strength is written outward.
_MATCH_PROPERTY = {
    MatchStrength.EQUIVALENT: "owl:equivalentProperty",
    MatchStrength.EXACT: "skos:exactMatch",
    MatchStrength.CLOSE: "skos:closeMatch",
}
_DATETIME = "xsd:dateTime"


def _inline_context(doc: VocabularyDocument) -> dict[str, Any]:
    return {
        "sh": SHACL_NAMESPACE,
        "ppx": PPX_NAMESPACE,
        doc.prefix: doc.namespace,
        "sh:targetClass": {"@type": "@id"},
        "sh:path": {"@type": "@id"},
        "owl:equivalentProperty": {"@type": "@id"},
        "skos:exactMatch": {"@type": "@id"},
        "skos:closeMatch": {"@type": "@id"},
        "ppx:target": {"@type": "@id"},
        "ppx:evidence": {"@type": "@json"},
        "prov:generatedAtTime": {"@type": _DATETIME},
        "dc:issued": {"@type": _DATETIME},
    }


def _ruling_to_json(ruling: Ruling) -> dict[str, Any]:
    out: dict[str, Any] = {
        "@type": "prov:Activity",
        "prov:wasAttributedTo": ruling.confirmed_by,
        "prov:generatedAtTime": ruling.confirmed_at.isoformat(),
        "ppx:evidence": ruling.evidence,
    }
    if ruling.proposal_event_id is not None:
        out["ppx:proposalEvent"] = ruling.proposal_event_id
    return out


def _ruling_from_json(obj: dict[str, Any]) -> Ruling:
    return Ruling(
        confirmed_by=obj["prov:wasAttributedTo"],
        confirmed_at=datetime.fromisoformat(obj["prov:generatedAtTime"]),
        evidence=obj.get("ppx:evidence") or {},
        proposal_event_id=obj.get("ppx:proposalEvent"),
    )


def _alignment_to_json(alignment: Alignment) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ppx:target": alignment.target,
        "ppx:match": alignment.match.value,
        "ppx:matchConfidence": alignment.confidence,
        "ppx:ruling": _ruling_to_json(alignment.ruling),
    }
    if alignment.owl_functional:
        out["ppx:sourceFunctional"] = True
    if alignment.sh_max_count is not None:
        out["ppx:sourceMaxCount"] = alignment.sh_max_count
    if alignment.wikidata_constraints:
        out["ppx:wikidataConstraints"] = [
            {"ppx:item": c.item, "ppx:separators": list(c.separators)}
            for c in alignment.wikidata_constraints
        ]
    return out


def _alignment_from_json(obj: dict[str, Any]) -> Alignment:
    return Alignment(
        target=obj["ppx:target"],
        match=MatchStrength(obj["ppx:match"]),
        confidence=obj.get("ppx:matchConfidence", 1.0),
        owl_functional=bool(obj.get("ppx:sourceFunctional", False)),
        sh_max_count=obj.get("ppx:sourceMaxCount"),
        wikidata_constraints=[
            WikidataConstraint(item=c["ppx:item"], separators=c.get("ppx:separators") or [])
            for c in obj.get("ppx:wikidataConstraints") or []
        ],
        ruling=_ruling_from_json(obj["ppx:ruling"]),
    )


def _profile_to_json(doc: VocabularyDocument, term: VocabularyTerm) -> dict[str, Any]:
    profile = term.profile
    assert profile is not None
    out: dict[str, Any] = {
        "@type": "sh:PropertyShape",
        "sh:path": doc.term_iri(term),
        "sh:targetClass": term.subject_class,
        "ppx:slotKind": profile.kind.value,
        "ppx:ruling": _ruling_to_json(profile.ruling),
    }
    match profile.kind:
        case SlotKind.TIMELESS_SINGLE:
            out["sh:maxCount"] = 1
        case SlotKind.ONE_AT_A_TIME:
            out["ppx:oneAtATime"] = True
        case SlotKind.MANY_AT_ONCE:
            out["ppx:manyAtOnce"] = True
    if profile.roles:
        out["ppx:roles"] = [
            {"ppx:form": form, "ppx:role": role.value}
            for form, role in sorted(profile.roles.items())
        ]
    return out


def _profile_from_json(obj: dict[str, Any]) -> TermProfile:
    return TermProfile(
        kind=SlotKind(obj["ppx:slotKind"]),
        roles={r["ppx:form"]: PredicateRole(r["ppx:role"]) for r in obj.get("ppx:roles") or []},
        ruling=_ruling_from_json(obj["ppx:ruling"]),
    )


def _term_to_json(doc: VocabularyDocument, term: VocabularyTerm) -> dict[str, Any]:
    out: dict[str, Any] = {
        "@id": f"{doc.prefix}:{term.local_name}",
        "@type": ["rdf:Property", "skos:Concept"],
        "skos:prefLabel": term.form,
        "ppx:subjectClass": term.subject_class,
        "ppx:ruling": _ruling_to_json(term.ruling),
    }
    if term.label is not None:
        out["rdfs:label"] = term.label
    if term.aliases:
        out["skos:altLabel"] = [a.form for a in term.aliases]
        out["ppx:aliasRulings"] = [
            {"ppx:form": a.form, "ppx:ruling": _ruling_to_json(a.ruling)} for a in term.aliases
        ]
    if term.alignments:
        for strength, prop in _MATCH_PROPERTY.items():
            targets = sorted(a.target for a in term.alignments if a.match is strength)
            if targets:
                out[prop] = targets
        out["ppx:alignments"] = [_alignment_to_json(a) for a in term.alignments]
    if term.profile is not None:
        out["ppx:profile"] = _profile_to_json(doc, term)
    return out


def _local_name(doc_prefix: str, term_id: str) -> str:
    head = f"{doc_prefix}:"
    if not term_id.startswith(head):
        raise ValueError(f"term id {term_id!r} is not minted under prefix {doc_prefix!r}")
    return term_id[len(head) :]


def _term_from_json(prefix: str, obj: dict[str, Any]) -> VocabularyTerm:
    rulings = {r["ppx:form"]: r["ppx:ruling"] for r in obj.get("ppx:aliasRulings") or []}
    aliases: list[Alias] = []
    for form in obj.get("skos:altLabel") or []:
        if form not in rulings:
            raise ValueError(f"alias {form!r} carries no ruling: every alias is a reviewed ruling")
        aliases.append(Alias(form=form, ruling=_ruling_from_json(rulings[form])))
    profile = obj.get("ppx:profile")
    return VocabularyTerm(
        local_name=_local_name(prefix, obj["@id"]),
        form=obj["skos:prefLabel"],
        subject_class=obj["ppx:subjectClass"],
        label=obj.get("rdfs:label"),
        aliases=aliases,
        alignments=[_alignment_from_json(a) for a in obj.get("ppx:alignments") or []],
        profile=_profile_from_json(profile) if profile is not None else None,
        ruling=_ruling_from_json(obj["ppx:ruling"]),
    )


def to_jsonld(doc: VocabularyDocument) -> dict[str, Any]:
    """The document as a JSON-LD object under the published context."""
    out: dict[str, Any] = {
        "@context": [CONTEXT_URL, _inline_context(doc)],
        "@id": doc.namespace,
        "@type": [DOCUMENT_TYPE, "skos:ConceptScheme", "owl:Ontology"],
        "ppx:name": doc.name,
        "ppx:prefix": doc.prefix,
        "ppx:namespace": doc.namespace,
        "owl:versionInfo": doc.version,
        "dc:issued": doc.issued.isoformat(),
        "ppx:terms": [_term_to_json(doc, t) for t in doc.terms],
    }
    if doc.publisher is not None:
        out["dc:publisher"] = doc.publisher
    if doc.description is not None:
        out["dc:description"] = doc.description
    return out


def from_jsonld(obj: dict[str, Any]) -> VocabularyDocument:
    """Decode a JSON-LD vocabulary document; raises ``ValueError`` on anything malformed."""
    if not is_vocabulary_document(obj):
        raise ValueError(f"not a vocabulary document (no @type {DOCUMENT_TYPE!r})")
    try:
        prefix = obj["ppx:prefix"]
        return VocabularyDocument(
            name=obj["ppx:name"],
            prefix=prefix,
            namespace=obj["ppx:namespace"],
            version=obj["owl:versionInfo"],
            publisher=obj.get("dc:publisher"),
            description=obj.get("dc:description"),
            issued=datetime.fromisoformat(obj["dc:issued"]),
            terms=[_term_from_json(prefix, t) for t in obj.get("ppx:terms") or []],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed vocabulary document: missing or mistyped {exc}") from exc


def is_vocabulary_document(obj: object) -> bool:
    """Whether a parsed JSON value declares itself a vocabulary document."""
    if not isinstance(obj, dict):
        return False
    types = obj.get("@type")
    if isinstance(types, str):
        types = [types]
    return isinstance(types, list) and DOCUMENT_TYPE in types


def dumps(doc: VocabularyDocument) -> str:
    """Canonical JSON-LD text: sorted keys, two-space indent, one trailing newline."""
    return json.dumps(to_jsonld(doc), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def loads(data: str | bytes) -> VocabularyDocument:
    """Parse and validate JSON-LD text; raises ``ValueError`` on anything malformed."""
    try:
        obj = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"not JSON: {exc}") from exc
    return from_jsonld(obj)
