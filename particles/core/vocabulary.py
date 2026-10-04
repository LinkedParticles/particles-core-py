# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The vocabulary document: a store's local ontology, as a shareable carrier.

Pure: no I/O. Locally minted predicates are modelling decisions with
provenance, and reviewed predicate profiles need somewhere to live.
A :class:`VocabularyDocument` is that place. It has the shape of a
trust lens: a named, publisher-stamped, monotonically versioned
document, deposited into the corpus, materialised, and adopted per store. A
revision is a new version deposited beside the old one, never an edit.

What a document holds:

* **terms**: one canonical predicate each, the pair *(normalised form,
  subject class)* (§1), minted as an IRI in the document's namespace;
* **aliases**: further normalised forms a reviewer confirmed name the same
  relation on the same class, serialised as ``skos:altLabel`` (§2);
* **alignments**: outward links to an external property, ``owl:equivalentProperty``
  for an asserted equivalence and ``skos:exactMatch`` / ``skos:closeMatch`` by
  confidence band for an estimate, with the cardinality constraints that
  property publishes (§3);
* **profiles**: the reviewed kind of slot, in SHACL terms (§4).

Every one of those carries a :class:`Ruling`: who confirmed it, when, and on
what evidence.

A document is a reviewed vocabulary a store keeps and exports, never a gate
extraction must satisfy. Letting the update rung read its profiles in place
of the slot probe was proposed and declined after measurement, so
nothing on the §6.6 ladder reads a document. An adopted document's
alignments feed the vocabulary report.

The proposal step's pure half lives here too (:func:`plan_proposals`): the
census of a store's predicates by subject class, clustered by embedding
similarity and split so a proposal never joins opposite polarity or direction.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from particles.core.predicate_profile import (
    PredicateRole,
    SlotKind,
    SourceConstraints,
    kind_from_source,
    normalise_predicate,
)

__all__ = [
    "Alias",
    "Alignment",
    "AliasProposal",
    "CensusRow",
    "MatchStrength",
    "ProfileProposal",
    "ProposalPlan",
    "Ruling",
    "TermProfile",
    "VocabularyDocument",
    "VocabularyTerm",
    "WikidataConstraint",
    "add_alias",
    "add_alignment",
    "effective_kind",
    "find_term",
    "mint_local_name",
    "new_document",
    "plan_proposals",
    "proposal_key",
    "set_profile",
]

#: Names a document can be adopted by: a slug, like a lens name.
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
#: A CURIE prefix (an XML NCName, restricted to ASCII).
_PREFIX = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,31}$")
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def _utcnow() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# The document model
# ---------------------------------------------------------------------------


class Ruling(BaseModel):
    """The provenance of one modelling ruling: who confirmed it, when, on what evidence.

    ``evidence`` is free-form and travels as written: the proposal's method and
    figures (similarity, claim counts, the surface forms it saw) for a
    reviewed proposal, or the operator's stated basis for a direct ruling.
    ``proposal_event_id`` names the store-local event that proposed it, when
    there was one; it is origin metadata, like an interchange unit's source id.
    """

    model_config = {"frozen": True}

    confirmed_by: str = Field(min_length=1)
    confirmed_at: datetime = Field(default_factory=_utcnow)
    evidence: dict[str, Any] = Field(default_factory=dict)
    proposal_event_id: str | None = None


class Alias(BaseModel):
    """A further normalised form confirmed to name a term's relation."""

    model_config = {"frozen": True}

    form: str = Field(min_length=1)
    ruling: Ruling


class MatchStrength(StrEnum):
    """How strongly a local term maps to an external property."""

    EQUIVALENT = "equivalent"
    """``owl:equivalentProperty``: an asserted equivalence, never an estimate."""
    EXACT = "exact"
    """``skos:exactMatch``: an estimate in the high band."""
    CLOSE = "close"
    """``skos:closeMatch``: an estimate below it. Never read for a kind."""


class WikidataConstraint(BaseModel):
    """One Wikidata ``P2302`` statement: a constraint item and its ``P4155`` separators."""

    model_config = {"frozen": True}

    item: str = Field(pattern=r"^Q\d+$")
    separators: list[str] = Field(default_factory=list)

    @field_validator("separators")
    @classmethod
    def _check_separators(cls, value: list[str]) -> list[str]:
        for sep in value:
            if not re.fullmatch(r"P\d+", sep):
                raise ValueError(f"a separator is a Wikidata property id, not {sep!r}")
        return sorted(set(value))


class Alignment(BaseModel):
    """An outward link from a local term to an external property.

    The cardinality fields record what the *external* source publishes, so the
    kind is read from the source by :func:`~particles.core.predicate_profile.kind_from_source`
    and never re-decided here. A local term that later finds an external match
    gains an alignment; it is never renamed, so nothing already published under
    its IRI breaks.
    """

    model_config = {"frozen": True}

    target: str = Field(min_length=1)
    match: MatchStrength
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    owl_functional: bool = False
    sh_max_count: int | None = Field(default=None, ge=0)
    wikidata_constraints: list[WikidataConstraint] = Field(default_factory=list)
    ruling: Ruling

    @model_validator(mode="after")
    def _equivalence_is_not_an_estimate(self) -> Alignment:
        if self.match is MatchStrength.EQUIVALENT and self.confidence < 1.0:
            raise ValueError(
                "an owl:equivalentProperty alignment is asserted, never estimated: "
                "use match 'exact' or 'close' for a confidence below 1.0"
            )
        return self

    def constraints(self) -> SourceConstraints:
        """The source's cardinality constraints, in the form ``kind_from_source`` reads."""
        return SourceConstraints(
            owl_functional=self.owl_functional,
            sh_max_count=self.sh_max_count,
            wikidata=tuple((c.item, frozenset(c.separators)) for c in self.wikidata_constraints),
        )


class TermProfile(BaseModel):
    """The reviewed kind of slot a term fills, in SHACL terms.

    Serialised as a property shape: ``sh:maxCount 1`` for timeless single, and
    ``ppx:oneAtATime`` / ``ppx:manyAtOnce`` for the two kinds SHACL cannot say.
    ``roles`` names member forms (the canonical form or an alias) that record a
    past value of the slot; an unlisted form gives the current value.
    """

    model_config = {"frozen": True}

    kind: SlotKind
    roles: dict[str, PredicateRole] = Field(default_factory=dict)
    ruling: Ruling


class VocabularyTerm(BaseModel):
    """One canonical predicate: a normalised form on a subject class."""

    model_config = {"frozen": True}

    local_name: str = Field(min_length=1)
    form: str = Field(min_length=1)
    subject_class: str = Field(min_length=1)
    label: str | None = None
    aliases: list[Alias] = Field(default_factory=list)
    alignments: list[Alignment] = Field(default_factory=list)
    profile: TermProfile | None = None
    ruling: Ruling

    @field_validator("local_name")
    @classmethod
    def _check_local_name(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", value):
            raise ValueError(f"a term's local name must be IRI-safe, not {value!r}")
        return value

    @model_validator(mode="after")
    def _check_members(self) -> VocabularyTerm:
        forms = [self.form, *(a.form for a in self.aliases)]
        if len(forms) != len(set(forms)):
            raise ValueError(f"term {self.local_name!r} lists a form twice")
        if self.profile is not None:
            unknown = set(self.profile.roles) - set(forms)
            if unknown:
                raise ValueError(
                    f"term {self.local_name!r} gives a role to forms it does not cover: "
                    f"{sorted(unknown)}"
                )
        return self

    def forms(self) -> frozenset[str]:
        """The canonical form and every confirmed alias."""
        return frozenset({self.form, *(a.form for a in self.aliases)})


class VocabularyDocument(BaseModel):
    """An operator- or enterprise-controlled vocabulary of reviewed predicate rulings.

    ``name`` is the adoption handle and ``version`` a monotonic integer, both as
    a trust lens has them: a higher version of one name supersedes a
    lower one, and every version stays in the corpus. ``namespace`` is the
    dereferenceable IRI base terms are minted under (``prefix`` is its CURIE
    prefix), the way the standard's own schemas resolve at the apex.
    """

    model_config = {"frozen": True}

    kind: Literal["VocabularyDocument"] = "VocabularyDocument"
    name: str
    prefix: str
    namespace: str
    version: int = Field(ge=1)
    publisher: str | None = None
    description: str | None = None
    issued: datetime = Field(default_factory=_utcnow)
    terms: list[VocabularyTerm] = Field(default_factory=list)
    #: Set at materialisation; store-local, so it never serialises.
    corpus_entry_id: str | None = Field(default=None, exclude=True)

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        if not _NAME.fullmatch(value):
            raise ValueError(
                f"a vocabulary name is a lowercase slug (letters, digits, '.', '_', '-'), "
                f"not {value!r}"
            )
        return value

    @field_validator("prefix")
    @classmethod
    def _check_prefix(cls, value: str) -> str:
        if not _PREFIX.fullmatch(value):
            raise ValueError(f"a prefix is a CURIE prefix (an NCName), not {value!r}")
        return value

    @field_validator("namespace")
    @classmethod
    def _check_namespace(cls, value: str) -> str:
        if not re.fullmatch(r"https?://[^\s#?]+[/#]", value):
            raise ValueError(
                "a namespace is an http(s) IRI base ending in '/' or '#' so terms can be "
                f"minted under it, not {value!r}"
            )
        return value

    @model_validator(mode="after")
    def _check_terms(self) -> VocabularyDocument:
        names: set[str] = set()
        keys: dict[tuple[str, str], str] = {}
        for term in self.terms:
            if term.local_name in names:
                raise ValueError(f"local name {term.local_name!r} is minted twice")
            names.add(term.local_name)
            for form in term.forms():
                key = (form, term.subject_class)
                if key in keys:
                    raise ValueError(
                        f"form {form!r} on class {term.subject_class!r} belongs to both "
                        f"{keys[key]!r} and {term.local_name!r}"
                    )
                keys[key] = term.local_name
        return self

    def term_iri(self, term: VocabularyTerm) -> str:
        """The term's minted IRI: the namespace plus its local name."""
        return self.namespace + term.local_name


def new_document(
    *,
    name: str,
    prefix: str,
    namespace: str,
    publisher: str | None = None,
    description: str | None = None,
    issued: datetime | None = None,
) -> VocabularyDocument:
    """Version 1 of a document, with no terms."""
    return VocabularyDocument(
        name=name,
        prefix=prefix,
        namespace=namespace,
        version=1,
        publisher=publisher,
        description=description,
        issued=issued or _utcnow(),
    )


# ---------------------------------------------------------------------------
# Revision: each returns the next version, never an edit of this one
# ---------------------------------------------------------------------------


def _slug(text: str) -> str:
    return _SLUG_STRIP.sub("-", text.lower()).strip("-") or "term"


def mint_local_name(doc: VocabularyDocument, form: str, subject_class: str) -> str:
    """A fresh IRI-safe local name for *(form, class)*, unique in the document."""
    base = f"{_slug(subject_class)}/{_slug(form)}"
    taken = {t.local_name for t in doc.terms}
    name, n = base, 1
    while name in taken:
        n += 1
        name = f"{base}-{n}"
    return name


def find_term(doc: VocabularyDocument, form: str, subject_class: str) -> VocabularyTerm | None:
    """The term covering *(form, class)*, by canonical form or alias."""
    for term in doc.terms:
        if term.subject_class == subject_class and form in term.forms():
            return term
    return None


def _term_by_ref(doc: VocabularyDocument, ref: str) -> VocabularyTerm | None:
    """A term by local name, CURIE (``prefix:local``) or full IRI."""
    for term in doc.terms:
        if ref in (term.local_name, f"{doc.prefix}:{term.local_name}", doc.term_iri(term)):
            return term
    return None


def _revise(
    doc: VocabularyDocument, terms: list[VocabularyTerm], issued: datetime | None
) -> VocabularyDocument:
    """The next version with ``terms``, validated whole (a revision never skips the checks)."""
    data = doc.model_dump()
    data.update(
        version=doc.version + 1,
        terms=[t.model_dump() for t in terms],
        issued=issued or _utcnow(),
    )
    return VocabularyDocument.model_validate(data)


def _ensure_term(
    doc: VocabularyDocument,
    form: str,
    subject_class: str,
    ruling: Ruling,
    label: str | None,
) -> tuple[list[VocabularyTerm], VocabularyTerm]:
    """The document's terms with one covering *(form, class)*, minted if absent."""
    existing = find_term(doc, form, subject_class)
    if existing is not None:
        return list(doc.terms), existing
    term = VocabularyTerm(
        local_name=mint_local_name(doc, form, subject_class),
        form=form,
        subject_class=subject_class,
        label=label,
        ruling=ruling,
    )
    return [*doc.terms, term], term


def _replace(terms: list[VocabularyTerm], old: VocabularyTerm, new: VocabularyTerm) -> None:
    terms[terms.index(old)] = new


def add_alias(
    doc: VocabularyDocument,
    *,
    form: str,
    subject_class: str,
    aliases: Sequence[str],
    ruling: Ruling,
    label: str | None = None,
    issued: datetime | None = None,
) -> VocabularyDocument:
    """The next version, with ``aliases`` confirmed as further forms of *(form, class)*.

    The canonical term is minted if absent. An alias already covered by the
    term is skipped; one that is another term's form on the class is refused,
    since merging two minted terms would rename one, and a published IRI never
    changes (link the two with an equivalence alignment instead).
    """
    terms, term = _ensure_term(doc, form, subject_class, ruling, label)
    new_aliases = list(term.aliases)
    for alias in aliases:
        if alias in term.forms() or alias in {a.form for a in new_aliases}:
            continue
        other = find_term(doc, alias, subject_class)
        if other is not None:
            raise ValueError(
                f"{alias!r} on {subject_class!r} is already {other.local_name!r}; a minted "
                "term is never merged away. Align the two with an equivalence instead."
            )
        new_aliases.append(Alias(form=alias, ruling=ruling))
    _replace(terms, term, term.model_copy(update={"aliases": new_aliases}))
    return _revise(doc, terms, issued)


def set_profile(
    doc: VocabularyDocument,
    *,
    form: str,
    subject_class: str,
    kind: SlotKind,
    ruling: Ruling,
    roles: Mapping[str, PredicateRole] | None = None,
    label: str | None = None,
    issued: datetime | None = None,
) -> VocabularyDocument:
    """The next version, with the term for *(form, class)* profiled as ``kind``.

    The term is minted if absent. A role names a member form; a role form the
    term does not yet cover is refused rather than silently aliased, since an
    alias is its own ruling.
    """
    terms, term = _ensure_term(doc, form, subject_class, ruling, label)
    profile = TermProfile(kind=kind, roles=dict(roles or {}), ruling=ruling)
    _replace(terms, term, term.model_copy(update={"profile": profile}))
    return _revise(doc, terms, issued)


def add_alignment(
    doc: VocabularyDocument,
    *,
    term_ref: str,
    alignment: Alignment,
    issued: datetime | None = None,
) -> VocabularyDocument:
    """The next version, with an outward alignment on an existing term.

    The term keeps its local name and form: an external match is a link out,
    never a rename. A second alignment to the same target replaces
    the first, since a ruling about one target supersedes an earlier one.
    """
    term = _term_by_ref(doc, term_ref)
    if term is None:
        raise ValueError(f"no term {term_ref!r} in vocabulary {doc.name!r}")
    alignments = [a for a in term.alignments if a.target != alignment.target]
    alignments.append(alignment)
    terms = list(doc.terms)
    _replace(terms, term, term.model_copy(update={"alignments": alignments}))
    return _revise(doc, terms, issued)


# ---------------------------------------------------------------------------
# Reading the documents: the kind of each term
# ---------------------------------------------------------------------------

#: Kinds in the order that keeps more claims: a tie between documents, or
#: between sources, breaks toward keeping.
_KEEPING_ORDER = (SlotKind.TIMELESS_SINGLE, SlotKind.MANY_AT_ONCE, SlotKind.ONE_AT_A_TIME)


def _most_keeping(kinds: Iterable[SlotKind]) -> SlotKind | None:
    present = set(kinds)
    for kind in _KEEPING_ORDER:
        if kind in present:
            return kind
    return None


def effective_kind(term: VocabularyTerm) -> SlotKind | None:
    """The kind a term's slot has: read from the source where it says, else the profile.

    Only an asserted or exact alignment is read for a kind; a close match is
    too loose to carry the external property's cardinality. Where the aligned
    sources state a kind, that kind wins over a reviewed profile (
    the kind is read from the source where its constraints say it, and from a
    reviewed profile otherwise).
    """
    from_source = _most_keeping(
        kind
        for a in term.alignments
        if a.match is not MatchStrength.CLOSE
        and (kind := kind_from_source(a.constraints())) is not None
    )
    if from_source is not None:
        return from_source
    return term.profile.kind if term.profile is not None else None


# ---------------------------------------------------------------------------
# Proposals: the pure half of `vocab propose`
# ---------------------------------------------------------------------------

_NEGATIONS = frozenset({"not", "never", "no", "cannot", "can't", "won't", "nor", "without"})
_TOWARD = frozenset({"to", "into", "onto", "toward", "towards"})


def _signature(form: str) -> tuple[bool, str]:
    """A form's polarity and direction: what an alias may never mix.

    Direction is ``from`` (names the old value), ``by`` (a ``by``-passive, the
    roles reversed) or ``fwd`` (anything else, ``to`` included).
    """
    tokens = form.split()
    negated = any(t in _NEGATIONS or t.endswith("n't") for t in tokens)
    if "from" in tokens:
        direction = "from"
    elif "by" in tokens:
        direction = "by"
    else:
        direction = "fwd"
    return negated, direction


def _direction_partner(form: str) -> str | None:
    """The same form with ``to`` / ``into`` turned to ``from``: its past-value partner."""
    tokens = form.split()
    swapped = ["from" if t in _TOWARD else t for t in tokens]
    return " ".join(swapped) if swapped != tokens else None


@dataclass(frozen=True)
class CensusRow:
    """One surface predicate on one subject class, and the claims that use it."""

    subject_class: str
    predicate: str
    claims: int


@dataclass(frozen=True)
class FormStats:
    """A normalised form on a class: its claims and the surface forms that produce it."""

    form: str
    subject_class: str
    claims: int
    surfaces: tuple[tuple[str, int], ...]

    @property
    def display(self) -> str:
        """The commonest surface form: what a reviewer reads, and what is embedded."""
        return self.surfaces[0][0]


def proposal_key(kind: str, subject_class: str, forms: Iterable[str]) -> str:
    """A stable handle for a proposal: the same candidate gets the same key on every run."""
    digest = hashlib.sha256(
        "\x1f".join([kind, subject_class, *sorted(forms)]).encode("utf-8")
    ).hexdigest()
    return f"vp-{digest[:12]}"


@dataclass(frozen=True)
class AliasProposal:
    """Forms on one class that embed close together and agree in polarity and direction."""

    key: str
    subject_class: str
    canonical: str
    aliases: tuple[str, ...]
    claims: int
    min_similarity: float
    members: tuple[FormStats, ...]

    def evidence(self) -> dict[str, Any]:
        return {
            "method": "embedding",
            "min_similarity": round(self.min_similarity, 4),
            "claims": self.claims,
            "surface_forms": {m.form: [list(s) for s in m.surfaces[:8]] for m in self.members},
        }


@dataclass(frozen=True)
class ProfileProposal:
    """A canonical predicate worth a profile ruling: the reviewer chooses its kind."""

    key: str
    subject_class: str
    form: str
    claims: int
    surfaces: tuple[tuple[str, int], ...]
    past_partners: tuple[str, ...]

    def evidence(self) -> dict[str, Any]:
        return {
            "method": "census",
            "claims": self.claims,
            "surface_forms": [list(s) for s in self.surfaces[:8]],
            "past_partners": list(self.past_partners),
        }


@dataclass
class ProposalPlan:
    """Everything one proposal run found, ranked by the claims each would cover."""

    aliases: list[AliasProposal] = field(default_factory=list)
    profiles: list[ProfileProposal] = field(default_factory=list)
    classed_predicates: int = 0
    classed_forms: int = 0
    classes: int = 0

    def predicates_in_alias_candidates(self) -> int:
        """Distinct *(class, surface predicate)* pairs some alias proposal covers."""
        return sum(len(m.surfaces) for p in self.aliases for m in p.members)


def form_stats(rows: Iterable[CensusRow]) -> list[FormStats]:
    """Group a census by *(normalised form, class)*, commonest surfaces first."""
    grouped: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        grouped[(normalise_predicate(row.predicate), row.subject_class)][row.predicate] += (
            row.claims
        )
    out: list[FormStats] = []
    for (form, cls), surfaces in grouped.items():
        ranked = tuple(sorted(surfaces.items(), key=lambda s: (-s[1], s[0])))
        out.append(
            FormStats(form=form, subject_class=cls, claims=sum(surfaces.values()), surfaces=ranked)
        )
    return out


def _cluster(vectors: np.ndarray[Any, Any], threshold: float) -> list[list[int]]:
    """Average-linkage clusters whose members average at least ``threshold`` cosine apart."""
    n = len(vectors)
    if n < 2:
        return [[i] for i in range(n)]
    # Deferred: only the proposal step clusters, and scipy is a heavy import
    # (AGENTS.md § Deferred imports, case 2).
    from scipy.cluster.hierarchy import fcluster, linkage  # type: ignore[import-untyped]

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.where(norms == 0, 1.0, norms)
    tree = linkage(unit, method="average", metric="cosine")
    labels = fcluster(tree, t=1.0 - threshold, criterion="distance")
    groups: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[int(label)].append(index)
    return list(groups.values())


def _min_similarity(vectors: np.ndarray[Any, Any], members: Sequence[int]) -> float:
    sub = vectors[list(members)]
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    unit = sub / np.where(norms == 0, 1.0, norms)
    sims = unit @ unit.T
    return float(sims[np.triu_indices(len(members), k=1)].min())


def plan_proposals(
    stats: Sequence[FormStats],
    vectors: Mapping[tuple[str, str], np.ndarray[Any, Any]],
    *,
    threshold: float,
    documents: Sequence[VocabularyDocument] = (),
    excluded: frozenset[str] = frozenset(),
) -> ProposalPlan:
    """Rank alias and profile candidates from a classed census (§4).

    ``vectors`` embeds each *(form, class)*; a form with no vector takes part in
    profile proposals only. Clustering runs within one class at ``threshold``
    (average linkage on cosine), and each cluster is split by polarity and
    direction so a proposal never joins ``renamed to`` with ``renamed from``
    or ``includes`` with ``does not include``. A form a document already
    covers on that class is never re-proposed as an alias; a term a document
    already profiles is never re-proposed for a profile. ``excluded`` holds
    keys already proposed or ruled on.
    """
    covered: dict[tuple[str, str], VocabularyTerm] = {}
    for doc in documents:
        for term in doc.terms:
            for form in term.forms():
                covered[(form, term.subject_class)] = term

    plan = ProposalPlan(
        classed_predicates=sum(len(s.surfaces) for s in stats),
        classed_forms=len(stats),
        classes=len({s.subject_class for s in stats}),
    )
    by_class: dict[str, list[FormStats]] = defaultdict(list)
    for s in stats:
        by_class[s.subject_class].append(s)

    for cls, members in sorted(by_class.items()):
        embedded = [s for s in members if (s.form, cls) in vectors]
        if len(embedded) >= 2:
            matrix = np.stack([vectors[(s.form, cls)] for s in embedded])
            for cluster in _cluster(matrix, threshold):
                if len(cluster) < 2:
                    continue
                parts: dict[tuple[bool, str], list[int]] = defaultdict(list)
                for index in cluster:
                    parts[_signature(embedded[index].form)].append(index)
                for part in parts.values():
                    proposal = _alias_proposal(cls, embedded, part, matrix, covered, excluded)
                    if proposal is not None:
                        plan.aliases.append(proposal)

        present = {s.form for s in members}
        for s in members:
            covering = covered.get((s.form, cls))
            if covering is not None and (covering.profile is not None or s.form != covering.form):
                continue
            key = proposal_key("profile", cls, [s.form])
            if key in excluded:
                continue
            partner = _direction_partner(s.form)
            plan.profiles.append(
                ProfileProposal(
                    key=key,
                    subject_class=cls,
                    form=s.form,
                    claims=s.claims,
                    surfaces=s.surfaces,
                    past_partners=(partner,) if partner in present else (),
                )
            )

    plan.aliases.sort(key=lambda p: (-p.claims, p.subject_class, p.canonical))
    plan.profiles.sort(key=lambda p: (-p.claims, p.subject_class, p.form))
    return plan


def _alias_proposal(
    cls: str,
    embedded: Sequence[FormStats],
    part: Sequence[int],
    matrix: np.ndarray[Any, Any],
    covered: Mapping[tuple[str, str], VocabularyTerm],
    excluded: frozenset[str],
) -> AliasProposal | None:
    """One polarity- and direction-consistent part of a cluster, as a proposal."""
    if len(part) < 2:
        return None
    members = [embedded[i] for i in part]
    terms = {covered[(m.form, cls)].local_name for m in members if (m.form, cls) in covered}
    if len(terms) > 1:
        return None  # two minted terms: never merged by an alias (add_alias refuses it)
    anchored = [m for m in members if (m.form, cls) in covered]
    if anchored:
        canonical = covered[(anchored[0].form, cls)].form
        fresh = [m for m in members if (m.form, cls) not in covered]
        if not fresh:
            return None
        aliases = tuple(sorted(m.form for m in fresh))
    else:
        ranked = sorted(members, key=lambda m: (-m.claims, len(m.form), m.form))
        canonical = ranked[0].form
        aliases = tuple(sorted(m.form for m in ranked[1:]))
    key = proposal_key("alias", cls, [canonical, *aliases])
    if key in excluded:
        return None
    ordered = sorted(members, key=lambda m: (-m.claims, m.form))
    return AliasProposal(
        key=key,
        subject_class=cls,
        canonical=canonical,
        aliases=aliases,
        claims=sum(m.claims for m in members),
        min_similarity=_min_similarity(matrix, part),
        members=tuple(ordered),
    )
