"""Pydantic v2 models for all Core particle and corpus types.

Core fields are required for v0.2 conformance.
Extension A–E fields appear as Optional and are stored/round-tripped but
not processed by Core logic.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from particles.core.scoring.confidence import CalibrationSource
from particles.core.status import Status, StatusReason

SCHEMA_VERSION = "1.0.0"


class SchemaVersionMismatchError(Exception):
    """Raised when the SDK refuses to operate on a store carrying older particles.

    (scrap-and-re-extract upgrade policy), the query / extract /
    review / reindex operations MUST refuse to run when the store contains
    ACTIVE particles whose ``schema_version`` does not match the current SDK
    ``SCHEMA_VERSION``. The exception carries the actionable operator message
    that surfaces through the CLI, FastAPI, and MCP layers — they all let it
    bubble up; the CLI translates it to a clean exit; FastAPI/MCP emit the
    str() as the error payload.

    Lint deliberately does NOT raise this — it reports `SCHEMA_VERSION_MISMATCH`
    findings via `_report_schema_versions` so the operator can diagnose first
    (per the L-STR-07 check). The "refuse" policy applies to the operations
    that would produce wrong results on mismatched data.
    """

    def __init__(
        self,
        *,
        current_version: str,
        found_versions: dict[str, int],
    ) -> None:
        self.current_version = current_version
        self.found_versions = found_versions
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        # Sorted for deterministic output (helps tests and the operator's
        # eye). The dict is small (~1-3 entries typically); sort cost is
        # negligible.
        mismatched = sorted(
            ((v, c) for v, c in self.found_versions.items() if v != self.current_version),
            key=lambda x: x[0],
        )
        total = sum(c for _, c in mismatched)
        breakdown = ", ".join(f"{c} at v{v}" for v, c in mismatched)
        return (
            f"Particle store contains {total} ACTIVE particles whose "
            f"schema_version does not match the current SDK "
            f"(SDK is at v{self.current_version}; store has {breakdown}). "
            f" the v0.3.x → v1.0.0 upgrade path is "
            f"scrap-and-re-extract. Run:\n"
            f"  particles db init --force    # confirms before dropping\n"
            f"  particles extract --all-pending\n"
            f"The corpus is preserved; only the particle store is rebuilt."
        )


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class UncertaintyNature(StrEnum):
    ALEATORY = "ALEATORY"  # irreducible; per PSUM UncertaintyNature
    EPISTEMIC = "EPISTEMIC"  # reducible; per PSUM UncertaintyNature


class AssertionModality(StrEnum):
    """What *kind* of assertion a particle makes — its truth-aptness.

    Orthogonal to ``uncertainty_nature`` (which presupposes a fact of the
    matter) and to scope (which lives in ``properties``). The Core
    engine applies truth-semantics — §6.6 contradiction resolution, L-SEM-01,
    L-IDX-01 — only to ``FALSIFIABLE`` particles (see :func:`is_truth_apt`);
    the other modalities co-exist and are never contradiction-checked or
    trust-arbitrated. Default ``FALSIFIABLE`` keeps every existing particle
    unchanged. Closed enum: adding a modality is an additive minor change.
    """

    FALSIFIABLE = "FALSIFIABLE"  # default — observer-independent, truth-apt
    EVALUATIVE = "EVALUATIVE"  # value / preference judgement ("X is best")
    EXPERIENTIAL = "EXPERIENTIAL"  # first-person report of an inner state ("I felt happy")
    CONSTITUTIVE = "CONSTITUTIVE"  # a rule a document establishes ("a Particle MUST …")


class ParticleType(StrEnum):
    CLAIM = "CLAIM"
    REVIEW = "REVIEW"  # human review annotation records
    # Prose-level structural connective tissue: a NARRATIVE's
    # content is a one-sentence label; its prose body is derived at render
    # time by traversing PART_OF / SEQUENCE_IN edges. confidence keeps the
    # universal truth-likelihood meaning, never an
    # "accurate compression" belief.
    NARRATIVE = "NARRATIVE"
    ACTION = "ACTION"  # reserved; deferred GTD integration (techspec OQ-12 / Appendix A)
    ANNOTATION = "ANNOTATION"  # reserved


class ProvenanceRefType(StrEnum):
    SOURCE = "SOURCE"
    PARTICLE = "PARTICLE"
    AGENT = "AGENT"


class SourceType(StrEnum):
    """Core source type identifiers. Domain-specific extractors define their own strings."""

    WEB_PAGE = "WEB_PAGE"
    PDF = "PDF"
    IMAGE = "IMAGE"  # standalone image (PNG/JPEG/GIF/WebP); vision extraction
    CSV = "CSV"
    CONVERSATION = "CONVERSATION"
    DATA_EXPORT = "DATA_EXPORT"
    LOCAL_FILE = "LOCAL_FILE"
    LOCAL_MARKDOWN = "LOCAL_MARKDOWN"  # hand-written / LLM-prose Markdown vaults
    PYTHON_SOURCE = "PYTHON_SOURCE"  # Python source; symbol-aware deposit/extract
    # One type for five concrete syntaxes (Turtle / N-Triples / TriG / N-Quads /
    # JSON-LD / RDF-XML) — they serialize one data model, and a trust statement
    # should not have to be written once per serialization.
    RDF_GRAPH = "RDF_GRAPH"
    ACADEMIC_PAPER = "ACADEMIC_PAPER"
    FORUM = "FORUM"
    BLOG = "BLOG"
    TAXONOMY_DEFINITION = "TAXONOMY_DEFINITION"  # Extension C.2
    TRUST_LENS_DEFINITION = "TRUST_LENS_DEFINITION"  # Extension B


class Mutability(StrEnum):
    APPEND_ONLY = "APPEND_ONLY"
    MUTABLE = "MUTABLE"
    STABLE = "STABLE"
    EPHEMERAL = "EPHEMERAL"


class FetchPolicy(StrEnum):
    LAZY = "LAZY"
    NEVER = "NEVER"


class WarcRecordType(StrEnum):
    RESPONSE = "RESPONSE"
    REVISIT = "REVISIT"


class ExtractionStatus(StrEnum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class SourceRefType(StrEnum):
    SOURCE_TYPE = "SOURCE_TYPE"
    CORPUS_ENTRY = "CORPUS_ENTRY"
    AUTHOR = "AUTHOR"


class PolicyProvenance(StrEnum):
    OPERATOR_DIRECT = "OPERATOR_DIRECT"  # highest authority
    REVIEWER_DERIVED = "REVIEWER_DERIVED"  # from Review session
    REGISTRY_ENDORSED = "REGISTRY_ENDORSED"  # endorsed by trusted registry


class ResolutionAction(StrEnum):
    PREFER_A = "PREFER_A"
    PREFER_B = "PREFER_B"
    BOTH_VALID = "BOTH_VALID"
    DEFER = "DEFER"


class AudienceHint(StrEnum):
    EXPERT = "EXPERT"  # numeric confidence + uncertainty classification
    GENERAL = "GENERAL"  # natural language hedging
    REGULATORY = "REGULATORY"  # full provenance citations


class RelationType(StrEnum):
    """Typed edge between two particles (§6.10).

    The registry of well-known values is documented. Today
    ``CO_EVIDENTIAL`` plus the two narrative kinds (``PART_OF`` /
    ``SEQUENCE_IN``) are emitted by code paths — the other
    members are RESERVED names, pre-blessed for future extractor work
    so the kind enum can grow additively without ad-hoc string drift.
    """

    # ACTIVE — written by L-IDX-01 lint and `particles links add`.
    CO_EVIDENTIAL = "CO_EVIDENTIAL"

    # ACTIVE — narrative connective tissue. Both asymmetric:
    # PART_OF is constituent → narrative; SEQUENCE_IN is predecessor →
    # successor. Deliberately NOT in `_SYMMETRIC_KINDS` (relation_store)
    # so the endpoint order is stored verbatim and carries direction.
    PART_OF = "PART_OF"
    SEQUENCE_IN = "SEQUENCE_IN"

    # ACTIVE — endorsement stances. Both asymmetric: the edge is
    # stance → target and marks the source particle as a stance (the role
    # marker). Deliberately NOT in `_SYMMETRIC_KINDS` so the
    # direction is stored verbatim ("S endorses T" ≠ "T endorses S").
    ENDORSES = "ENDORSES"
    DISPUTES = "DISPUTES"

    # RESERVED — the name is endorsed; per-kind activation ADRs
    # ship the consumer surface (query filter, lint integration, CLI
    # parser) before any extractor starts emitting them.
    CONTRADICTS = "CONTRADICTS"
    BOOSTS = "BOOSTS"
    QUOTES = "QUOTES"
    REPLIES_TO = "REPLIES_TO"
    MENTIONS = "MENTIONS"


class RelationCreatedBy(StrEnum):
    """How a particle relation came to exist (§6.10).

    Mirrors PolicyProvenance for SourceTrustStatement: the relation
    is self-describing about its own epistemic quality.
    """

    EXTRACTOR_DIRECT = "EXTRACTOR_DIRECT"  # emitted by an aggregating extractor
    HUMAN_REVIEW = "HUMAN_REVIEW"  # operator accepted a suggested candidate
    AUTO_CLUSTER_V1 = "AUTO_CLUSTER_V1"  # future: automated linker pass
    MANUAL_CLI = "MANUAL_CLI"  # `particles links add` issued by operator
    LLM_JUDGE = "LLM_JUDGE"  # `particles links suggest --apply`
    # `particles links dedup --apply` — byte-identical content, decided by
    # hash with no verdict of any kind. Kept distinct from LLM_JUDGE (which would
    # misattribute a deterministic merge to a model) and from AUTO_CLUSTER_V1
    # (reserved for the clustering pass).
    EXACT_DUPLICATE = "EXACT_DUPLICATE"


class SuggestMode(StrEnum):
    """Operating mode for the ``links suggest`` operation.

    ``REPORT`` lists candidate pairs above the similarity threshold with no
    LLM call and no mutation. ``LLM_JUDGE`` additionally sends each Subject's
    candidate cluster to an LLM for a per-pair PARAPHRASE / DISTINCT / UNSURE
    verdict, still without mutating. ``APPLY`` implies ``LLM_JUDGE`` and
    auto-links the PARAPHRASE pairs via a CO_EVIDENTIAL relation.
    """

    REPORT = "REPORT"
    LLM_JUDGE = "LLM_JUDGE"
    APPLY = "APPLY"


class JudgeVerdictKind(StrEnum):
    """An LLM's verdict on whether a candidate pair asserts the same claim."""

    PARAPHRASE = "PARAPHRASE"  # same underlying claim — safe to link CO_EVIDENTIAL
    DISTINCT = "DISTINCT"  # genuinely different claims — never auto-linked
    UNSURE = "UNSURE"  # ambiguous — reported, never auto-linked


class TermKind(StrEnum):
    """What one term of a structured claim *is*.

    ``TOKEN`` is the honest middle: a lexical name for an entity or relation
    that no vocabulary resolved. Requiring ``URI`` in predicate position would
    collapse structured-claim coverage to the handful of ontology-aligned
    extractors, so an unresolved relation is recorded as what it is and the
    minting of a local-namespace IRI is left to the export layer.
    """

    URI = "URI"  # absolute IRI, or a CURIE in a context.jsonld prefix
    TOKEN = "TOKEN"  # an unresolved lexical name for an entity or relation
    LITERAL = "LITERAL"  # a lexical value (object position only)


class CanonicalForm(StrEnum):
    """Which of the prose/structured pair is the assertion.

    ``PROSE`` is the default and covers everything the LLM prose extractors
    produce: ``content`` is the assertion and any :class:`StructuredClaim` is
    a derived annotation. ``STRUCTURED`` is the inverse — the source asserts
    the triple and ``content`` is verbalised from it — and is emitted
    per-candidate by the structure-canonical extractors (RDF, the Numista
    family) under the rule: ``STRUCTURED`` exactly when
    ``content`` is a deterministic rendering of its ``structured_claim`` and
    of no other fact. One extractor may emit both forms.
    """

    PROSE = "PROSE"
    STRUCTURED = "STRUCTURED"


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------


class ClaimTerm(BaseModel):
    """One term (subject, predicate, or object) of a structured claim.

    Attributes:
        kind: URI, TOKEN (unresolved lexical name), or LITERAL (object only).
        value: the IRI/CURIE, the lexical name, or the literal's lexical form.
        datatype: xsd datatype IRI. LITERAL only.
        language: BCP-47 language tag. LITERAL only, and never with datatype.
    """

    model_config = {"frozen": True}

    kind: TermKind
    value: str = Field(min_length=1)
    datatype: str | None = None
    language: str | None = None

    @model_validator(mode="after")
    def _check_literal_qualifiers(self) -> ClaimTerm:
        if self.kind is not TermKind.LITERAL and (
            self.datatype is not None or self.language is not None
        ):
            raise ValueError("datatype/language are permitted only on a LITERAL term")
        if self.datatype is not None and self.language is not None:
            raise ValueError("a literal carries a datatype or a language tag, never both")
        return self


class StructuredClaim(BaseModel):
    """A derived, regenerable S-P-O rendering of a particle's ``content``.

    The core invariant generalises: the **asserted** form is immutable,
    the **derived** form is regenerable. This annotation is produced by tooling
    from ``content``, carries its own derivation stamp, and its fidelity is a
    property of the annotation — never evidence about the claim. Nothing that
    writes it may touch ``content``, ``confidence``, or provenance.

    Generated in exactly two places: the extraction pipeline
    and the ``particles structure`` backfill. Exporters report coverage and
    never generate.

    Attributes:
        subject: the subject term (never LITERAL).
        predicate: the relation term (never LITERAL).
        object: the object term.
        subject_id: the resolved Subject UUID when the subject term names one
            of the particle's subjects; None when it resolved to nothing.
        structurizer_id: what produced the triple, e.g. ``general-extractor``.
        structurizer_version: that generator's version at generation time.
        generated_at: when the triple was produced.
    """

    model_config = {"frozen": True}

    subject: ClaimTerm
    predicate: ClaimTerm
    object: ClaimTerm
    subject_id: str | None = None
    structurizer_id: str = Field(min_length=1)
    structurizer_version: str = Field(min_length=1)
    generated_at: datetime = Field(default_factory=_utcnow)

    @model_validator(mode="after")
    def _check_term_positions(self) -> StructuredClaim:
        if self.subject.kind is TermKind.LITERAL:
            raise ValueError("a literal cannot occupy the subject position")
        if self.predicate.kind is TermKind.LITERAL:
            raise ValueError("a literal cannot occupy the predicate position")
        return self


class Confidence(BaseModel):
    """The stored, immutable confidence record (§6.3).

    confidence.value is never modified after particle creation. It is the
    extractor's confidence as calibrated at creation time;
    calibration_source/calibration_method/calibration_ref record how.
    effective_confidence is computed at query time only and never stored.
    """

    model_config = {"frozen": True}

    value: float = Field(ge=0.0, le=1.0)
    variance: float | None = None
    calibration_source: CalibrationSource = CalibrationSource.EXTRACTOR_DIRECT
    calibration_method: str | None = None  # e.g. "temperature_scaling"
    calibration_ref: str | None = None  # reference to calibration_history entry


class ProvenanceRef(BaseModel):
    """Points into the corpus with snapshot-level precision (§7.3).

    Field-reuse convention (blessing the INCONSISTENCY
    precedent): when ``type`` is :data:`ProvenanceRefType.PARTICLE`, the
    referenced **particle id travels in** ``corpus_entry_id`` (and
    ``snapshot_id``, when set, carries the same particle id). Consumers:
    ``build_inconsistency_particle`` (writer), the retraction-propagation
    lint and the premise machinery (readers).
    """

    type: ProvenanceRefType
    corpus_entry_id: str
    snapshot_id: str | None = None
    location: str | None = None  # byte range, paragraph number, comment ID, …
    # SHA-256 of the LLM-prompt text that produced this particle, when an
    # extractor uses chunked extraction with carry-forward.
    # NULL for non-chunked extractors and for pre-0057 particles.
    chunk_hash: str | None = None


# Recommended (not enforced) contributor-role vocabulary. The
# set is OPEN — roles outside it are valid; conformance MAY warn, never rejects.
CONTRIBUTOR_ROLES: frozenset[str] = frozenset(
    {"author", "extractor", "curator", "reviewer", "importer", "agent"}
)
"""The canonical registry of **attributed acts on a particle / entry / subject**
(catalogued). This constant *is* the catalogue — the table is its normative description, and there is no second list.

| Role | The actor… |
|---|---|
| ``author`` | …originated the claim's content (speaker, post author, doc author). |
| ``extractor`` | …is the extractor plugin that produced the particle from a source. |
| ``importer`` | …is the importer plugin that deposited the source blob. |
| ``curator`` | …performed a curation gesture (assign-subject, supersede, retract, tag). |
| ``reviewer`` | …made a trust-bearing review judgment (§9.6 Review). |
| ``agent`` | …is an autonomous agent (e.g. an MCP client) asserting or acting. |

**The vocabulary is OPEN** (``ContributorRef.role`` is a plain ``str``): a role
outside this set validates, and conformance MAY warn but never rejects. The
registry blesses recommended values without closing the field, preserving
domain extensibility.

**Discipline (pattern).** A new contributor role is
added *here* — one place — **before** any code emits it. The cautionary tale is
concrete: an early conversation-import draft minted a fourth role vocabulary
(``speaker_role ∈ {owner, assistant, participant}``) before review noticed that
a conversation speaker is simply an ``author`` in this set. Reach for an ad-hoc
string and you have re-created that bug.

**Three axes, kept distinct** — this registry covers only (a) and
deliberately does **not** absorb its neighbours:

- **(a) Contributor role** — the *act* on a particle. ``ContributorRef.role``.
  **This registry.**
- **(b) Source-author role** — the author's relationship to the *source*
  (``author_role``, e.g. ``"maintainer"``; §6.5). Source metadata on a
  ``Snapshot``, named here only to say it is **not** a contributor role.
- **(c) Relation / stance kind** — ``ENDORSES`` / ``DISPUTES`` /
  ``CO_EVIDENTIAL`` … These are *edges* (:class:`RelationType`), owned by the
  **relation-kind registry** (for the stance kinds). A
  ``role: "author"`` is not a stance.
"""


class ContributorRef(BaseModel):
    """One contributor's attributed act on a particle / entry / subject.

    Extension D/E: stored and round-tripped, but **Core modules MUST
    NOT branch on it**. ``id`` shares the AUTHOR-scoped ``SourceRef`` namespace
    (``platform:identifier``, §6.5; e.g. ``github:torvalds``) so a per-viewer
    ``SourceTrustStatement`` lookup joins on it with no extra
    identity machinery. It is immutable attribution substrate — it carries no
    confidence / trust / endorsement (those are query-time lenses).

    Attributes:
        id: ``platform:identifier`` identity string, canonically normalized (§6.5).
        role: Open vocabulary. :data:`CONTRIBUTOR_ROLES` is the canonical
            registry of recommended values — a new role is added
            there before any code emits it.
        at: UTC timestamp of when this contributor performed the act.
    """

    model_config = {"frozen": True}

    id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    at: datetime = Field(default_factory=_utcnow)


class ExtractorRef(BaseModel):
    """Reference to the extractor that produced a particle (§6.2, §14.3).

    Core, and the join key into the extractor registry: ``name`` is what trust
    weighting and exporter attribution look up, ``version`` is what
    ``reindex --extractor-version`` and the chunk-hash carry-forward scope
    on. Absent only for particles asserted directly by an operator or an
    authorized agent (§9.1a) — ``Particle.extractor_ref`` is
    ``None`` there, never an empty ref.

    Modelled. It was an untyped ``dict[str, Any]`` through 1.109.x
    with the two key names stated only in prose, so nothing checked them at
    any of the three validation layers.

    The **runtime substrate** the extractor invoked is *not* recorded here —
    that is the sibling ``Particle.extraction_provider_model``.
    This names the code; one extractor version runs under many models.

    Attributes:
        name: Registered extractor id, e.g. ``general-extractor``.
        version: That extractor's semver version at extraction time; a
            superseded version is the re-extraction scope (§9.5, §14.3).
    """

    model_config = {"frozen": True}

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class Particle(BaseModel):
    """The minimal unit of knowledge in the Particles standard (§6.1–6.2).

    A particle is a single claim with full provenance. `content` is the
    human-readable assertion. `confidence` and `uncertainty_nature` quantify
    epistemic state. `subject_ids` links the claim to canonical entities.
    `properties` carries structured ontology-keyed data for structured
    extractors (e.g. Numista); it is never used for conflict detection.

    Attributes:
        content: The claim text (min length 1).
        confidence: Stored, immutable confidence record.
        uncertainty_nature: EPISTEMIC (reducible) or ALEATORY (irreducible).
        provenance: Corpus entry / snapshot references.
        asserted_by: Agent or extractor ID that created this particle.
        status: Lifecycle status (ACTIVE, SUPERSEDED, RETRACTED, …).
        subject_ids: UUIDs of subjects this particle is a statement about.
        properties: Nomisma ontology-keyed structured data.
    """

    # Core fields — required for v0.2 conformance
    id: str = Field(default_factory=_new_uuid)
    content: str = Field(min_length=1)
    confidence: Confidence
    uncertainty_nature: UncertaintyNature
    provenance: list[ProvenanceRef] = Field(default_factory=list)
    asserted_by: str
    asserted_at: datetime = Field(default_factory=_utcnow)
    status: Status = Status.ACTIVE
    status_reason: StatusReason | None = None
    schema_version: str = Field(default=SCHEMA_VERSION)
    particle_type: ParticleType = ParticleType.CLAIM
    # Truth-aptness axis. Additive Optional Extension field
    #: old particles deserialize to the FALSIFIABLE default, so the
    # schema freeze holds and SCHEMA_VERSION stays 1.0.0. The
    # engine applies truth-semantics only to FALSIFIABLE particles — see
    # is_truth_apt(); Core branches on exactly this one default-safe bit.
    assertion_modality: AssertionModality = AssertionModality.FALSIFIABLE

    # Extension A field — stored but not processed by Core logic
    uncertainty_kind: str | None = None
    # Core (§6.2): valid_until (lazy expiry filter, §9.3) and supersedes
    # (revision chain). Kept in declaration order for serialization
    # stability.
    valid_until: datetime | None = None
    supersedes: str | None = None
    # Extension A field — stored but not processed by Core logic
    basis: dict[str, Any] | None = None
    # Core (§6.2; owner decision 2026-06-11) — omitted only for
    # operator-asserted particles. Kept in declaration order for
    # serialization stability. Modelled as ExtractorRef; it was an
    # untyped dict through 1.109.x, so a plain {"name", "version"} mapping
    # still validates into one.
    extractor_ref: ExtractorRef | None = None
    # Core (§6.2) — the ``"<provider>:<model>"`` pairing that
    # produced this claim: the disclosure key, the calibration
    # key. A *sibling* of extractor_ref, never a key inside it:
    # extractor_ref names the code, this names the runtime substrate that
    # code invoked, and one extractor version runs under many models.
    # ``None`` means UNRECORDED, never "no model" — no completion provider
    # produced it (a deterministic extractor, or a direct assertion),
    # or it predates this field. Never backfilled, never
    # defaulted, never recomputed. Core MUST NOT branch on it.
    extraction_provider_model: str | None = None
    sequence_context: list[str] | None = None

    # Core: subjects this particle is a statement about
    subject_ids: list[str] = Field(default_factory=list)

    # Extension C fields — stored but not processed by Core logic
    tags: list[str] | None = None
    context_fingerprint: str | None = None

    # Structured properties keyed by ontology URI, e.g. {"nmo:hasWeight": 0.75}
    # Used by structured extractors (Numista); None for free-text particles.
    # Never used for conflict detection; used only by exporters.
    properties: dict[str, object] | None = None

    # Extension D/E field — who extracted / curated / asserted this claim.
    # Additive Optional; Core MUST NOT branch on it. None ≡ [].
    contributors: list[ContributorRef] | None = None

    # the derived S-P-O rendering of ``content``, and which of the
    # pair is the assertion. Additive Optional Extension fields:
    # old particles deserialize to None / PROSE, so the schema freeze
    # holds and SCHEMA_VERSION stays 1.0.0 — the same
    # posture taken before. Core MUST NOT branch on
    # ``structured_claim``; the one branch on ``canonical_form`` is the
    # validator below.
    structured_claim: StructuredClaim | None = None
    canonical_form: CanonicalForm = CanonicalForm.PROSE

    # query-time-only "contested" marker — the id of an open
    # INCONSISTENCY particle that references this one, else None. Computed by the
    # read surfaces (query / particles_list); NEVER stored on ParticleRow and
    # NEVER branched on in Core. Makes the §6/§6b contradiction ledger legible to
    # the agent at recall time, not only to the operator at Review.
    contested: str | None = None

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, v: str) -> str:
        if not v:
            raise ValueError("schema_version must not be empty")
        return v

    @model_validator(mode="after")
    def _check_canonical_form(self) -> Particle:
        # STRUCTURED means "the triple is the assertion", so there must be one.
        # The converse does not hold: a PROSE-canonical particle carrying an
        # annotation is the normal case.
        if self.canonical_form is CanonicalForm.STRUCTURED and self.structured_claim is None:
            raise ValueError("canonical_form=STRUCTURED requires a structured_claim")
        return self


def is_truth_apt(particle: Particle) -> bool:
    """Whether the engine should apply truth-semantics to this particle.

    ``True`` only for ``FALSIFIABLE`` particles — the single, default-safe bit
    the §6.6 conflict ladder, L-SEM-01 semantic-contradiction lint, and
    L-IDX-01 co-evidential suggestion gate on. Non-truth-apt particles
    (evaluative / experiential / constitutive) co-exist and are never
    contradiction-checked, trust-arbitrated, or co-evidentially clustered.
    """
    return particle.assertion_modality == AssertionModality.FALSIFIABLE


class Snapshot(BaseModel):
    """A timestamped, content-addressed capture of a corpus source (§7.3)."""

    snapshot_id: str = Field(default_factory=_new_uuid)
    captured_at: datetime = Field(default_factory=_utcnow)
    content_hash: str  # SHA-256 of raw content body
    etag: str | None = None
    last_modified: datetime | None = None
    warc_record_type: WarcRecordType = WarcRecordType.RESPONSE
    archive_path: str | None = None
    refers_to: str | None = None  # WARC-Refers-To snapshot_id for REVISIT
    extraction_status: ExtractionStatus = ExtractionStatus.PENDING
    # UGC author metadata (§6.4)
    author_id: str | None = None  # e.g. "github:torvalds"
    author_role: str | None = None  # e.g. "maintainer"
    # Original publication timestamp of the source content.
    # Set by domain importers where known; None means no decay is applied.
    content_published_at: datetime | None = None


class CorpusEntry(BaseModel):
    """Stable record of a source and its relationship to its origin (§7.3)."""

    entry_id: str = Field(default_factory=_new_uuid)
    uri_r: str | None = None  # Memento URI-R
    source_type: str  # SourceType or any extractor-defined string constant
    mutability: Mutability = Mutability.STABLE
    fetch_policy: FetchPolicy = FetchPolicy.NEVER
    created_at: datetime = Field(default_factory=_utcnow)
    deposited_by: str
    tags: list[str] = Field(default_factory=list)
    snapshots: list[Snapshot] = Field(default_factory=list)
    # Extension D/E — who deposited / imported this source. Rides
    # alongside the single-valued ``deposited_by`` audit string. None ≡ [].
    contributors: list[ContributorRef] | None = None


class SourceRef(BaseModel):
    """What a SourceTrustStatement applies to (§6.4)."""

    type: SourceRefType
    value: str  # SOURCE_TYPE name, corpus entry_id, or "platform:identifier"


class SourceTrustStatement(BaseModel):
    """Operator-defined trust policy record (§6.4).

    Stored as first-class records; consulted by conflict resolution and query.
    Demotion-only rule: may only demote confidence, never silently suppress conflicts.
    """

    statement_id: str = Field(default_factory=_new_uuid)
    domain: str
    source_ref: SourceRef
    trust_rank: float = Field(ge=0.0, le=1.0)
    policy_provenance: PolicyProvenance
    asserted_by: str
    asserted_at: datetime = Field(default_factory=_utcnow)
    basis: str | None = None
    review_id: str | None = None


class ExternalRef(BaseModel):
    """A reference to an entity in an external ontology or catalogue."""

    namespace: str  # "wikidata", "numista", "km_catalog", "mesh", "isbn", …
    id: str  # "Q16957", "N1004", "KM#8", …
    uri: str | None = None  # canonical URI if the namespace publishes one
    # Confidence that this external link is correct.
    # 1.0 = asserted by a structured extractor; < 1.0 = scored by embedding similarity.
    # Existing rows deserialise without this field and default to 1.0.
    confidence: float = 1.0


class ParticleRelation(BaseModel):
    """A typed edge between two particles (§6.10).

    The relation is symmetric: ``(a, b)`` and ``(b, a)`` denote the same
    edge. Implementations canonicalise pairs as ``(min(a, b), max(a, b))``
    at the storage layer so the unique constraint catches duplicates.
    """

    particle_a: str
    particle_b: str
    relation_type: RelationType
    created_by: RelationCreatedBy
    created_at: datetime = Field(default_factory=_utcnow)
    # How confident the *link* is — not the confidence of either particle.
    # 1.0 for HUMAN_REVIEW and EXTRACTOR_DIRECT; cosine similarity for
    # AUTO_CLUSTER_V1 candidates.
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Extension A: Extractor Registry and Trust Model
# ---------------------------------------------------------------------------


class ApplicabilityClause(BaseModel):
    """RFC 2119 domain applicability declaration for an extractor (§14.1)."""

    keyword: Literal["MUST", "SHOULD", "MAY", "MUST_NOT"]
    domain_uri: str  # Wikidata entity URI, e.g. "http://www.wikidata.org/entity/Q631286"
    domain_label: str  # human-readable, e.g. "numismatics"
    source_types: list[str]  # source_type strings this clause covers


class ExtractorCalibration(BaseModel):
    """Temperature-scaling calibration record on a registered extractor.

    A single most-recent calibration per extractor. Applied at particle-
    construction time when the extractor has one and the particle would
    otherwise carry calibration_source=EXTRACTOR_DIRECT. The math lives in
    :class:`particles.extraction.calibration.TemperatureScaler`; this model
    is just the persisted record.

    Attributes:
        temperature: The fitted T. What it *means* is declared by ``transform``.
        transform: Which functional form ``temperature`` parameterises. ``"logit"`` — ``sigmoid(logit(raw) / T)``, Guo et al. (2017),
            the only form this SDK fits or applies. ``None`` is a pre-ADR-0238
            record: its T parameterises the retired ``clamp(raw / T, 0, 1)``
            form *and* was fitted against all-False labels, so it is never
            applied — the pipeline falls back to ``EXTRACTOR_DIRECT`` and the
            operator re-fits. See
            :func:`particles.extraction.calibration.scaler_for_record`.
        fitted_at: When the fit was performed.
        benchmark_suite_id: ``suite_id`` of the benchmark suite the fit ran
            against (or a ``+``-joined concatenation when multiple suites
            contributed pairs).
        sample_size: Number of (raw, label) pairs the fit consumed.
        calibration_error_before: ECE before calibration (the value the
            benchmark runner reported).
        calibration_error_after: ECE after applying the fitted T to the
            same labelled pairs.
        provider_model: The ``"<provider>:<model>"`` pairing the benchmark ran
            under. Calibration is provider-sensitive: a temperature
            fitted against one model is not applied to another's outputs. The
            pipeline applies the record only when the current extraction
            provider/model matches; on mismatch the particle falls back to
            ``calibration_source=EXTRACTOR_DIRECT``. ``None`` is a legacy record
            (fitted before this key existed); it is treated as the historical
            default ``anthropic:claude-sonnet-4-6``.
    """

    temperature: float = Field(gt=0.0)
    transform: str | None = None
    fitted_at: datetime
    benchmark_suite_id: str
    sample_size: int = Field(ge=1)
    calibration_error_before: float = Field(ge=0.0, le=1.0)
    calibration_error_after: float = Field(ge=0.0, le=1.0)
    provider_model: str | None = None


class ExtractorRecord(BaseModel):
    """Registered extractor artifact with trust and applicability metadata (§14.3)."""

    extractor_id: str
    name: str
    version: str
    applicability: list[ApplicabilityClause] = Field(default_factory=list)
    trust_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    registered_by: str = "anthropic/particles-sdk"
    registered_at: datetime = Field(default_factory=_utcnow)
    calibration: ExtractorCalibration | None = None


class Subject(BaseModel):
    """A canonical real-world entity, identified independently of any source.

    Subjects are the nodes of the knowledge graph. Particles are statements
    (properties or edges) about subjects. Resolution is by canonical_name
    (case-insensitive) or aliases; external_ids enable cross-ontology lookup.

    Attributes:
        canonical_name: The authoritative display name (e.g. "1 Pfennig (1948-1950) GDR").
        aliases: Alternative names used during subject resolution.
        external_ids: Cross-references to Wikidata, Numista, Nomisma, etc.
        subject_class: Nomisma ontology class for exporter template selection
            (e.g. ``nmo:NumismaticObject``, ``nmo:Material``). None for generic subjects.
    """

    id: str = Field(default_factory=_new_uuid)
    canonical_name: str = Field(min_length=1)
    description: str | None = None
    aliases: list[str] = Field(default_factory=list)
    external_ids: list[ExternalRef] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    asserted_by: str
    # Nomisma ontology class name for template selection in exporters.
    # e.g. "nmo:NumismaticObject", "nmo:Material", "nmo:Denomination", "nmo:Issuer"
    subject_class: str | None = None
    # Extension D/E — who created / curated this canonical entity.
    # None ≡ []; Core MUST NOT branch on it.
    contributors: list[ContributorRef] | None = None


class ReviewParticle(BaseModel):
    """Audit record written after a Review resolution action (§9.6)."""

    review_id: str = Field(default_factory=_new_uuid)
    inconsistency_particle_id: str
    resolution: ResolutionAction
    reviewer_id: str
    reviewed_at: datetime = Field(default_factory=_utcnow)
    trust_statement_id: str | None = None
    note: str | None = None


# ---------------------------------------------------------------------------
# Extension C.2: Taxonomy and tag-aware query expansion
# ---------------------------------------------------------------------------


class TagNode(BaseModel):
    """One tag in a TaxonomyDefinition tree (spec §16.2).

    Tag paths use ``/`` as the hierarchy separator. Roots have no slash and
    ``parent`` is ``None``. For a non-root tag, ``parent`` must equal the
    tag minus its last ``/``-segment (validated by ``TaxonomyDefinition``).
    """

    tag: str = Field(min_length=1)
    parent: str | None = None
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None


class TaxonomyDefinition(BaseModel):
    """A depositable corpus artefact defining a hierarchical tag taxonomy.

    Operators publish taxonomies by depositing a JSON file; the
    ``TaxonomyExtractor`` materialises the rows into the ``taxonomies`` and
    ``tag_nodes`` query-time index. Tags carry no truth value — they are a
    curation layer over the particle store, orthogonal to the Subject
    knowledge graph.
    """

    taxonomy_id: str = Field(default_factory=_new_uuid)
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    author: str = Field(min_length=1)
    domain: str | None = None
    tags: list[TagNode]
    published_at: datetime = Field(default_factory=_utcnow)
    corpus_entry_id: str | None = None

    @model_validator(mode="after")
    def _check_parent_paths(self) -> TaxonomyDefinition:
        for node in self.tags:
            if "/" in node.tag:
                expected_parent = node.tag.rsplit("/", 1)[0]
                if node.parent != expected_parent:
                    raise ValueError(
                        f"TagNode {node.tag!r}: parent must be "
                        f"{expected_parent!r}, got {node.parent!r}"
                    )
            elif node.parent is not None:
                raise ValueError(
                    f"TagNode {node.tag!r} is a root tag (no '/') so parent "
                    f"must be None, got {node.parent!r}"
                )
        return self


# ---------------------------------------------------------------------------
# Trust lenses — Extension B
# ---------------------------------------------------------------------------


class TrustLensStatement(BaseModel):
    """One portable SOURCE_TYPE-scoped trust statement carried by a lens.

    ``CORPUS_ENTRY``-scoped statements are excluded from lenses by design —
    they key on store-local entry ids and are not portable.
    """

    domain: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    trust_rank: float = Field(ge=0.0, le=1.0)
    basis: str | None = None


class TrustLensUrlRule(BaseModel):
    """One portable URL trust rule carried by a lens."""

    scope: Literal["domain", "url_pattern"]
    pattern: str = Field(min_length=1)
    score: float | None = Field(default=None, ge=0.0, le=1.0)  # domain rows only
    modifier: float | None = None  # url_pattern rows only

    @model_validator(mode="after")
    def _check_scope_fields(self) -> TrustLensUrlRule:
        if self.scope == "domain" and (self.score is None or self.modifier is not None):
            raise ValueError("domain-scoped rule requires `score` and forbids `modifier`")
        if self.scope == "url_pattern" and (self.modifier is None or self.score is not None):
            raise ValueError("url_pattern-scoped rule requires `modifier` and forbids `score`")
        return self


class TrustLensDecayRule(BaseModel):
    """One portable content-age decay rule carried by a lens.

    Sets the recency half-life / floor for content matching ``pattern`` — an
    exact ``source_type`` string (``scope="source_type"``) or a trust-rule-style
    URL regex (``scope="url_pattern"``, e.g. a per-subreddit half-life). Unlike
    the URL *trust* rule's additive ``modifier``, a decay rule is an **absolute**
    ``(half_life_days, floor)`` pair — half-life is not meaningfully additive.
    The URL layer is more specific than the source_type layer (a per-subreddit
    rule overrides the source-type default in either direction); across adopted
    lenses the most-skeptical rule wins (shortest half-life / lowest floor).
    """

    scope: Literal["source_type", "url_pattern"]
    pattern: str = Field(min_length=1)
    half_life_days: float = Field(gt=0.0)
    floor: float = Field(ge=0.0, le=1.0)


class TrustLensUtilityRule(BaseModel):
    """One portable usefulness (outcome-learning) utility rule carried by a lens.

    Where a ``decay_rule`` is the *judgment* half of content-age
    decay, a ``utility_rule`` is the judgment half of **usefulness**: it sets
    *how far* demonstrated use may reorder the projection / digest head, and
    *how fast* unreinforced utility fades. The per-belief utility *evidence*
    (the mined reinforcement count) is store-local and lives outside the lens
     — this rule carries only the tunables.

    - ``half_life_uses_days`` — days for a single utility event's weight to halve
      (the reinforcement half-life; ``> 0``).
    - ``rank_lift`` — the ``λ`` in ``rank_score = effective_confidence
      + λ·ln(1 + R)`` (``≥ 0``; ``0`` disables the lift). **Promotion-only** by
      construction — the bonus is never negative, so lack of use withholds the
      lift rather than demoting.

    This single knob replaced the old ``weight`` / ``floor`` / ``cap`` triple
     when the bounded multiplier was superseded; a lens
    published under the old vocabulary carries no ``rank_lift`` and is treated
    as silent about utility (the store's local ``utility`` config applies).

    Across adopted lenses the most-skeptical rule wins — *less* promotion:
    shortest ``half_life_uses_days``, lowest ``rank_lift`` (symmetric with the
    decay rule). ``scope="default"`` is the
    store-wide rule; ``source_type`` / ``url_pattern`` scopes mirror the decay
    ladder for finer control.
    """

    scope: Literal["default", "source_type", "url_pattern"] = "default"
    pattern: str | None = None  # required for source_type / url_pattern scopes
    half_life_uses_days: float = Field(gt=0.0)
    rank_lift: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _check_scope_pattern(self) -> TrustLensUtilityRule:
        if self.scope == "default":
            if self.pattern is not None:
                raise ValueError("default-scoped utility rule forbids `pattern`")
        elif not self.pattern:
            raise ValueError(f"{self.scope}-scoped utility rule requires `pattern`")
        return self


class TrustLensDefinition(BaseModel):
    """A depositable corpus artefact bundling portable trust policy.

    A community publishes its view of which sources to trust by depositing a
    JSON file; the ``TrustLensExtractor`` materialises it into the
    ``trust_lenses`` / ``trust_lens_entries`` tables. Viewers adopt lenses per
    store; the query-time ``TrustPolicy`` composes the store's local
    policy over adopted lenses, most-skeptical-wins. Adopting a lens can only
    demote (ranks are in [0, 1] and silence stays neutral), never promote.

    A lens carries a fourth portable layer, ``decay_rules`` —
    per-observer content-age decay (recency half-life / floor), composed through
    the query-time ``DecayPolicy`` the same most-skeptical-wins way as the trust
    layers. A lens silent about decay leaves the store's global
    ``content_age_decay`` config untouched.

    A lens also carries ``utility_rules`` — the usefulness
    (outcome-learning) judgment, composed through the query-time ``UtilityPolicy``
    into an additive rank-lift on the projection / digest ranking path
    only. A lens silent about utility leaves the store's local ``utility`` config
    untouched; with no utility evidence the bonus is ``+0`` (cold-start).
    """

    kind: Literal["TrustLensDefinition"] = "TrustLensDefinition"
    lens_id: str = Field(default_factory=_new_uuid)
    name: str = Field(min_length=1)
    version: int = Field(ge=1)  # monotonic; a higher version supersedes
    publisher: str | None = None
    description: str | None = None
    statements: list[TrustLensStatement] = Field(default_factory=list)
    url_rules: list[TrustLensUrlRule] = Field(default_factory=list)
    extractor_weights: dict[str, float] = Field(default_factory=dict)
    decay_rules: list[TrustLensDecayRule] = Field(default_factory=list)
    utility_rules: list[TrustLensUtilityRule] = Field(default_factory=list)
    corpus_entry_id: str | None = None  # set after deposit

    @model_validator(mode="after")
    def _check_extractor_weights(self) -> TrustLensDefinition:
        for extractor_id, weight in self.extractor_weights.items():
            if not 0.0 <= weight <= 1.0:
                raise ValueError(
                    f"extractor_weights[{extractor_id!r}] must be in [0, 1], got {weight}"
                )
        return self


# ---------------------------------------------------------------------------
# Query / Lint API types
# ---------------------------------------------------------------------------


class StructuralGroupBy(StrEnum):
    """Bucket axis for the ``--group-by`` aggregate mode."""

    SUBJECT = "subject"
    PREDICATE = "predicate"
    OBJECT = "object"


class QueryRequest(BaseModel):
    # Bounded so an oversized question can't drive an arbitrarily large prompt
    # into the paid query-response completion (security review F6). 8192 chars is
    # well above any natural question; longer inputs belong in the corpus, not a
    # query. min_length=1 rejects an empty question before it reaches the LLM.
    # None selects a purely structural mode — the validator
    # below requires a question or a structural condition, never neither.
    question: str | None = Field(default=None, min_length=1, max_length=8192)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty_nature: UncertaintyNature | None = None
    # Truth-aptness filter. None ⇒ every modality (non-FALSIFIABLE
    # particles are kept *in* retrieval — only out of the engine's
    # truth-arbitration). A set value narrows: FALSIFIABLE for a pure-facts
    # view, EXPERIENTIAL for a feelings view, etc.
    assertion_modality: AssertionModality | None = None
    recency_window_days: int | None = None
    audience: AudienceHint = AudienceHint.GENERAL
    top_k: int = Field(default=40, ge=1, le=200)
    subject_id: str | None = None  # filter to particles about a specific subject
    # Tag filter — each requested tag is subtree-expanded across all active
    # taxonomies before the candidate set is filtered (Extension C.2).
    tags: list[str] = Field(default_factory=list)
    #: when True, the tag filter also walks UP each requested
    # tag's parent chain, so a query for a specific node additionally matches
    # particles tagged only with a broader ancestor term. Off by default — it
    # widens the match set and only the subtree expansion is the documented
    # default.
    include_ancestors: bool = False
    # When False (default), particles tagged DOCUMENT_META are
    # excluded from results; set True to include them in the candidate set.
    include_document_meta: bool = False
    # When False (default), non-asserted particles — polarity DECLINED /
    # HYPOTHETICAL (cap. 1): a document's rejected / superseded /
    # deferred / counterfactual prose — are excluded from the default factual
    # surface; set True to include them in the candidate set.
    include_non_asserted: bool = False
    # when True, each result carries the query-time agreement
    # distribution (holders endorsing / disputing the claim, attributed and
    # cited). Off by default — it costs per-result edge traversal on a hot path
    # (the projection at scale is deferred).
    include_agreement: bool = False
    # when True, each result carries its per-claim contestedness
    # reading — the max−min spread of effective_confidence across the viewer's
    # policy set (local policy + each adopted lens), with the per-policy
    # renderings attributed. Off by default (per-member policy evaluation cost);
    # absent entirely when the viewer has fewer than two policies (§3).
    include_contestedness: bool = False
    # the as-of reference instant — answer "what did the store
    # believe at T". Unset (None) is byte-for-byte today's read behaviour.
    # A timezone-naive value is assumed UTC (matching the store's existing
    # naive-to-UTC normalization); a bare date parses to start-of-day UTC at
    # the surface layer. A future instant is a caller bug, not a query, and
    # is rejected by the validator below (HTTP 422; CLI/MCP surface the
    # message).
    as_of: datetime | None = None
    # structural claim filters over the annotation.
    # With a question they prefilter the semantic candidate set (ranking
    # untouched, §2.4); without one they select the deterministic listing
    # mode (no embedding, no LLM call). ``predicate`` is exact-string,
    # case-insensitive — a CURIE and its expanded IRI are different strings.
    predicate: str | None = Field(default=None, min_length=1)
    object_eq: str | None = Field(default=None, min_length=1)
    object_gt: str | None = Field(default=None, min_length=1)
    object_lt: str | None = Field(default=None, min_length=1)
    object_contains: str | None = Field(default=None, min_length=1)
    # deterministic aggregate modes. ``count`` returns the
    # number of matching claims with the effective-confidence distribution;
    # ``group_by`` buckets them. Both reject a simultaneous question — an LLM
    # narrating a deterministic count adds nothing but risk (§2.1).
    count: bool = False
    group_by: StructuralGroupBy | None = None
    # the *explicit* aggregate confidence floor. There is no
    # default floor — a silently-thresholded count lies in both directions —
    # so this field is None unless the caller set it, and it applies only to
    # the aggregate modes.
    min_effective_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    # list the distinct predicate terms with kind and claim
    # count — the discovery surface for the exact-string predicate filter.
    # A standalone mode: combines with no question, filter, or aggregate.
    list_predicates: bool = False

    @property
    def claim_filter_flags(self) -> tuple[str | None, ...]:
        return (
            self.predicate,
            self.object_eq,
            self.object_gt,
            self.object_lt,
            self.object_contains,
        )

    @property
    def has_claim_filters(self) -> bool:
        """True when any structural filter is set."""
        return any(v is not None for v in self.claim_filter_flags)

    @property
    def is_aggregate(self) -> bool:
        """True in a deterministic aggregate mode."""
        return self.count or self.group_by is not None

    @property
    def is_structural_mode(self) -> bool:
        """True when the request runs a purely structural, LLM-free mode
        (deterministic listing, aggregate, or predicate-vocabulary listing —
        modes three and four)."""
        return (
            self.list_predicates
            or self.is_aggregate
            or (self.question is None and self.has_claim_filters)
        )

    @model_validator(mode="after")
    def _check_structural(self) -> QueryRequest:
        if self.question is None and not self.is_structural_mode:
            raise ValueError(
                "a query needs a question, a structural claim filter "
                "(predicate / object_*), an aggregate (count / group_by), or "
                "list_predicates"
            )
        if self.is_aggregate and self.question is not None:
            raise ValueError(
                "aggregate modes (count / group_by) are deterministic and reject "
                "a simultaneous question — an LLM narrating a deterministic "
                "count adds nothing but risk"
            )
        if self.list_predicates and (
            self.question is not None or self.has_claim_filters or self.is_aggregate
        ):
            raise ValueError(
                "list_predicates is a standalone vocabulary listing; combine it "
                "with no question, filter, or aggregate"
            )
        if self.min_effective_confidence is not None and not self.is_aggregate:
            raise ValueError(
                "min_effective_confidence applies only to the aggregate modes "
                "(count / group_by); other modes use min_confidence"
            )
        for raw_bound in (self.object_gt, self.object_lt):
            if raw_bound is None:
                continue
            # Deferred import: claims.py imports ClaimTerm from this module,
            # so a module-top import here would be a cycle (AGENTS.md
            # § Deferred imports, case 1).
            from particles.core.claims import parse_bound

            if parse_bound(raw_bound) is None:
                raise ValueError(
                    f"object comparison bound {raw_bound!r} is neither a number "
                    "nor an ISO-8601 date/datetime"
                )
        return self

    @model_validator(mode="after")
    def _check_as_of(self) -> QueryRequest:
        if self.as_of is not None:
            as_of = self.as_of
            if as_of.tzinfo is None:
                as_of = as_of.replace(tzinfo=UTC)
                self.as_of = as_of
            if as_of > datetime.now(UTC):
                raise ValueError(
                    "as_of must not be in the future: the as-of lens answers what "
                    f"the store believed at a past instant (got {as_of.isoformat()})."
                )
        return self


class CoverageGapKind(StrEnum):
    NO_SUBJECT_MATCH = "NO_SUBJECT_MATCH"  # query named a subject not in the registry
    SUBJECT_HAS_NO_PARTICLES = (
        "SUBJECT_HAS_NO_PARTICLES"  # subject exists but zero ACTIVE CLAIM particles
    )
    SUBJECT_HAS_LOW_COVERAGE = (
        "SUBJECT_HAS_LOW_COVERAGE"  # subject has particles but fewer than threshold
    )


class SubjectCoverageGap(BaseModel):
    subject_id: str | None = None  # None for NO_SUBJECT_MATCH
    subject_name: str | None = None  # canonical_name if known
    kind: CoverageGapKind
    particle_count: int = 0  # ACTIVE CLAIM particles for this subject
    detail: str


class StancePosition(BaseModel):
    """One holder's position in a claim's query-time agreement distribution.

    Computed at query time over the ``ENDORSES`` / ``DISPUTES`` edges into the
    target's CO_EVIDENTIAL group; never stored (substrate-plus-lens).
    ``effective_confidence`` is the *stance particle's own* believability
    (how sure we are the holder holds the attitude) — it is surfaced alongside,
    and never folded into, the target claim's confidence (the §4 MUST).
    """

    kind: RelationType  # ENDORSES or DISPUTES
    holder: str  # the stance:holder identifier (a platform:identifier, §3)
    stance_particle_id: str  # citation — the reified stance particle
    effective_confidence: float  # of the stance particle itself, query-time
    magnitude: float | None = None  # stance:magnitude, [0, 1]; None ⇒ unqualified


class PolicyRendering(BaseModel):
    """One nameable policy's rendering of a claim's effective confidence.

    ``policy`` is the attributable member name — ``"local"`` for the store's own
    policy or an adopted lens's name — and ``effective_confidence`` is the value
    that policy renders for the claim (the §6.9 noisy-OR merge over the claim's
    co-evidential group, evaluated under this member alone). The extremes are
    *nameable policies* by design — the range statistic exists so the max and min
    can be attributed ("local: 0.43; acme-numismatics: 0.81").
    """

    policy: str  # "local" or the adopted lens name
    effective_confidence: float  # rendered under this policy alone, [0, 1]


class ContestednessReading(BaseModel):
    """A claim's lens-divergence of effective confidence.

    ``spread`` is max − min of ``effective_confidence`` evaluated separately
    under each policy in the viewer's policy set (the local policy plus each
    adopted lens). It is **disclosure, not discount**: it MUST NOT
    feed ``effective_confidence``, ranking, ``min_confidence`` filtering, or
    §6.6 conflict resolution. Computed at read time, never stored. Present only
    when the viewer has two or more policies (§3) — a one-policy store mints no
    contestedness, since absence of measurement is not measured invariance.
    """

    spread: float  # max − min across the policy set, [0, 1]
    renderings: list[PolicyRendering]  # per-policy, attributed (local first)


class ContestedBadge(BaseModel):
    """The composed per-claim contested badge.

    A claim renders *contested* iff at least one of three named bases fires;
    the badge is a basis-carrying disjunction — a set of fired basis labels,
    never a blended scalar — so every badge names which instrument(s) produced
    it (§1). The three gates (§2): ``stance`` — ≥1 ``DISPUTES`` position in the
    claim's query-time stance distribution; ``divergence`` — the
    claim's :class:`ContestednessReading` spread is at least
    ``contestedness.callout_threshold``; ``inconsistency`` — an open
    INCONSISTENCY particle references the claim (subsumed as a
    basis). A claim with no available basis fired carries **no** badge (None in
    the parallel list), never an explicit "uncontested" (§3).

    Invariants (§4): computed at read time, never stored; MUST NOT feed
    ``effective_confidence``, ranking, ``min_confidence`` filtering, or §6.6
    conflict resolution — disclosure, not discount. The divergence reading and
    stance distribution are not duplicated here; they remain the existing
    envelope blocks (the drill-downs).
    """

    # Fired basis labels, non-empty (a bare "contested" with no basis is
    # non-conforming, §4). Ordered stance, divergence, inconsistency.
    bases: list[Literal["stance", "divergence", "inconsistency"]] = Field(min_length=1)
    # Drill-down when the "inconsistency" basis fired: the open INCONSISTENCY
    # particle's id (exactly the marker this badge subsumes).
    inconsistency_id: str | None = None
    # (M6): set whenever "stance" is among the bases — holders are
    # unverified raw keys; the caveat MUST travel with the badge.
    caveat: str | None = None


class AsOfSuccessor(BaseModel):
    """The belief that replaced a retired as-of hit.

    Attached to an :class:`AsOfNote` when a successor particle exists (some
    particle's ``supersedes`` points at the hit). ``particle_show`` on ``id``
    is the drill-down for the full record.
    """

    id: str
    content: str
    asserted_at: datetime


class AsOfNote(BaseModel):
    """The supersession crossing for a query hit retired after the as-of instant.

    A **response model, not particle substrate** — computed at read time,
    never stored. Annotates a hit that was believed at the reference instant T
    but has since been retired: its current status + reason, the retirement
    instant, the ladder rung that dated it (so the instant is itself
    auditable), and the replacing belief when one exists.
    """

    status: Status
    status_reason: StatusReason | None = None
    # The reconstructed transaction-time end of the belief. Always known for a
    # visible retired hit — a retirement the §2b ladder cannot date is excluded
    # fail-closed and never surfaces as a hit.
    retired_at: datetime
    # Which §2b ladder rung answered: the stored ``retired_at`` column, the
    # successor's ``asserted_at``, an operator event, or ``valid_until``.
    basis: Literal["stored", "successor", "event", "valid_until"]
    successor: AsOfSuccessor | None = None


class RelevanceNote(BaseModel):
    """The question-level relevance disclosure for a semantic query.

    A **response model, not particle substrate** — computed at read time from
    the rendered top-k, never stored, and never an input to ranking or
    ``effective_confidence`` (the two-quantity discipline).
    ``max_similarity`` is the maximum raw cosine similarity over the rendered
    hits on the normalized scale; when it falls below ``floor``
    (``config.query.relevance_floor``) the answer is a deterministic
    no-LLM refusal and the hits are presented as nearest-but-likely-unrelated.
    Absent (``None`` on the response) for structural modes, when no embedding
    model is available, and when the result is empty.
    """

    max_similarity: float
    floor: float
    below_floor: bool


class PredicateInfo(BaseModel):
    """One row of the predicate-vocabulary listing."""

    value: str
    kind: TermKind
    claim_count: int


class ClaimCoverage(BaseModel):
    """Coverage data for a structural-filter result.

    Rendered as the footer line "matched against the N of M ACTIVE particles
    carrying a structured claim (store coverage P%)" — absence of a hit must
    never be mistaken for absence of a belief. ``not_normalizable_excluded``
    is the §2.2 disclosure: claims excluded from a gt/lt comparison because
    their object would not normalize to a comparable type.
    """

    active_total: int
    with_claims: int
    matched: int
    not_normalizable_excluded: int = 0
    # rows explicitly excluded by the caller's
    # min_effective_confidence floor (aggregate modes only; there is no
    # default floor).
    below_min_effective_confidence: int = 0


class AggregateBucket(BaseModel):
    """One ``--group-by`` bucket: matching claims with their
    effective-confidence distribution."""

    key: str
    # Human label when the key is an id — the resolved Subject name for
    # group_by=subject; None when the key is its own label.
    label: str | None = None
    claim_count: int
    min_effective_confidence: float
    median_effective_confidence: float
    max_effective_confidence: float


class StructuralAggregate(BaseModel):
    """Deterministic aggregate result.

    Counts **claims**, never entities — duplicates per subject exist. The
    effective-confidence distribution (min/median/max) is disclosed beside
    every count; the distribution fields are None only when zero claims match.
    """

    claim_count: int
    min_effective_confidence: float | None = None
    median_effective_confidence: float | None = None
    max_effective_confidence: float | None = None
    group_by: StructuralGroupBy | None = None
    buckets: list[AggregateBucket] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    particles: list[Particle]
    effective_confidences: list[float]
    content_published_ats: list[datetime | None] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(
        default_factory=list
    )  # deprecated: entry_ids with PENDING/FAILED status
    subject_coverage_gaps: list[SubjectCoverageGap] = Field(default_factory=list)
    truncation_warning: str | None = (
        None  # set when top_k cutoff may have excluded relevant particles
    )
    # per-result agreement distribution, query-time only. Parallel
    # to ``particles`` (``agreement_distributions[i]`` is the positions on
    # ``particles[i]``); empty unless ``QueryRequest.include_agreement``. The
    # aggregate MUST NOT feed ``effective_confidences`` — it is surfaced beside
    # confidence, never multiplied into it.
    agreement_distributions: list[list[StancePosition]] = Field(default_factory=list)
    # (M6): holder distinctness is unverified raw-key grouping; set
    # whenever any distribution is non-empty so consumers do not read the holder
    # set as a count of verified agents.
    agreement_caveat: str | None = None
    # per-result contestedness readings, query-time only. Parallel
    # to ``particles`` (``contestedness[i]`` is the reading for ``particles[i]``);
    # empty unless ``QueryRequest.include_contestedness`` AND the viewer has ≥2
    # policies (§3). The spread MUST NOT feed ``effective_confidences`` — it is
    # disclosure surfaced beside confidence, never a discount on it (§5).
    contestedness: list[ContestednessReading] = Field(default_factory=list)
    # the composed contested badge, query-time only. Parallel to
    # ``particles`` (``contested[i]`` is the badge on ``particles[i]``; None for
    # a claim with no available basis fired — never an explicit "uncontested",
    # §3). Populated by default, gated by ``contestedness.badge_enabled`` (§7);
    # empty when the badge is disabled. MUST NOT feed ``effective_confidences``
    # or ranking (§4) — the cheap always-on summary whose drill-downs are the
    # ``agreement_distributions`` / ``contestedness`` blocks above.
    contested: list[ContestedBadge | None] = Field(default_factory=list)
    # when a NARRATIVE particle is a hit, its SEQUENCE_IN constituents
    # (the narrative's claims, in order), keyed by the narrative particle id — so a
    # consumer gets the memory's content without a second round-trip. Populated
    # only for NARRATIVE hits; the narrative particle still appears in
    # ``particles`` at its rank position. Empty on the federated path.
    narrative_constituents: dict[str, list[Particle]] = Field(default_factory=dict)
    # echo of the request's as-of reference instant; None on a
    # normal (present-time) query.
    as_of: datetime | None = None
    # per-result supersession crossings, query-time only. Parallel
    # to ``particles`` (``as_of_notes[i]`` annotates ``particles[i]``); the entry
    # is None for a hit that is still ACTIVE today. Empty unless the request set
    # ``as_of``.
    as_of_notes: list[AsOfNote | None] = Field(default_factory=list)
    # how many once-believed retired particles were excluded
    # fail-closed from this as-of view because their retirement instant is not
    # reconstructible (rung 4). Born-retired rows are never counted. Always 0
    # on a normal query.
    as_of_excluded_undatable: int = 0
    # present on every structural-filter result (prefilter,
    # deterministic listing, and aggregate modes); None on a plain semantic
    # query. Carries the coverage-footer data and the §2.2 gt/lt
    # non-normalizable disclosure count.
    claim_coverage: ClaimCoverage | None = None
    # the deterministic aggregate result; None outside the
    # count / group_by modes. MUST NOT feed ranking or confidence — it is the
    # result itself, not a signal.
    structural_aggregate: StructuralAggregate | None = None
    # the predicate-vocabulary listing; empty outside the
    # list_predicates mode.
    predicate_vocabulary: list[PredicateInfo] = Field(default_factory=list)
    # the question-level relevance disclosure — max raw cosine over
    # the rendered top-k vs. ``config.query.relevance_floor``. When
    # ``below_floor``, ``answer`` is a deterministic no-LLM refusal and the
    # hits are nearest-but-likely-unrelated. None on structural modes, when no
    # embedding model is available, and on an empty result. MUST NOT feed
    # ranking or ``effective_confidences`` — disclosure only.
    relevance: RelevanceNote | None = None
    # set when the semantic ranking ran WITHOUT an embedding model.
    # The retrieval is then not semantic at all: similarity is aliased to
    # effective confidence, so the top-k is the store's most-confident beliefs
    # irrespective of the question — and because ``relevance`` is None in that
    # state, the floor cannot refuse the answer either. Both facts have
    # to be said out loud, or a confidently-worded answer over unrelated
    # material reads exactly like a good one. None whenever ranking was
    # genuinely semantic, and on structural modes that never embed.
    ranking_degraded: str | None = None
    # Honesty disclosure for the NL answer path (the posture —
    # degradation is disclosed, never quiet): set when answer generation
    # failed (billing, network, provider error) and ``answer`` is therefore
    # the deterministic fallback listing of the retrieved beliefs, not
    # generated prose. None on success AND on the paths that make no LLM
    # call by design (structural modes, the below-floor refusal).
    answer_generation_error: str | None = None
    # refusal flag, both flavours: True when ``answer`` is a
    # no-relevant-knowledge refusal — the §2 deterministic below-floor answer,
    # or the §4 responder-declared one (the stripped NO_RELEVANT_KNOWLEDGE
    # marker). Consumers relabel the hit list ("nearest beliefs — likely
    # unrelated") and the truncation warning is suppressed server-side (advice
    # to widen top_k under a refusal is incoherent). Disclosure only — never
    # feeds ranking or ``effective_confidences``.
    answer_refused: bool = False


class GraphParticleInfo(BaseModel):
    """One particle's epistemics payload in a graph render.

    A **view model, not particle substrate** — every derived quantity here
    (``effective_confidence``, the contested badge, the as-of note, the
    utility score) is computed at render time and never stored.
    Rendered as an edge when the particle spans ≥2 in-scope subjects,
    as a node-panel cargo row when it has one.
    """

    id: str
    content: str
    status: Status
    status_reason: StatusReason | None = None
    # The stored, immutable confidence.value.
    confidence: float
    # Computed at render time (never stored); drives the opacity encoding.
    effective_confidence: float
    subject_ids: list[str]
    asserted_at: datetime
    valid_until: datetime | None = None
    # Forward pointer to the predecessor this particle replaced.
    supersedes: str | None = None
    contested: ContestedBadge | None = None
    # Recency-weighted reinforcement score — utility *evidence*
    # display (drives node size only, never opacity); 0.0 = no evidence.
    utility_score: float = 0.0
    source_uri: str | None = None
    # The scope's foreground set (vs incidental cargo), highlighted in renders:
    # query scope's retrieval hits; inconsistency scope's anchor INCONSISTENCY
    # record + its disputants; projection scope's selected particles. The field
    # name is the query-scope original, kept for wire compatibility.
    retrieval_hit: bool = False
    # True for a particle rendered only because history was requested — not
    # part of the current (or as-of-current) belief surface; drives the
    # ghost/tombstone form encoding.
    ghost: bool = False
    # supersession crossing for an as-of render; None otherwise.
    as_of_note: AsOfNote | None = None


class GraphNode(BaseModel):
    """One Subject node in a graph render.

    Subjects are entities, not beliefs: the per-particle epistemics live on
    :class:`GraphParticleInfo`; a node carries only display aggregates that
    are labeled as such in the render.
    """

    subject_id: str
    # disambiguated display name.
    label: str
    subject_class: str | None = None
    # Hop distance from the scope anchor (0 = anchor / retrieval-hit subject).
    hop: int = 0
    # Max effective confidence over the node's in-scope ACTIVE particles —
    # the "best-supported claim" display aggregate.
    max_effective_confidence: float = 0.0
    # Summed recency-weighted utility evidence over in-scope particles;
    # drives node size (log-scaled), neutral at 0.0 (cold start).
    utility_score: float = 0.0
    # True iff any incident in-scope particle carries a contested badge.
    contested: bool = False
    # Single-subject particle ids listed in the node's detail panel, by
    # descending effective confidence, capped at graph.max_particles_per_subject.
    cargo: list[str] = Field(default_factory=list)
    # How many cargo particles the per-subject cap dropped (0 = uncapped).
    cargo_truncated: int = 0


class GraphEdge(BaseModel):
    """One rendered subject-pair segment of a multi-subject particle.

    A particle spanning 3+ in-scope subjects renders as a pairwise clique —
    one segment per pair, all sharing ``particle_id`` (clicking any segment
    opens the same particle; the panel discloses the full subject list).
    """

    particle_id: str
    source: str
    target: str


class GraphSupersession(BaseModel):
    """One directed supersession crossing included in a history render."""

    predecessor_id: str
    successor_id: str


class GraphCensus(BaseModel):
    """Machine-readable render census (disclosure discipline).

    Candidate counts are what an uncapped render would have shown; rendered
    counts are what this render shows. When they differ the human-readable
    disclosure line names the binding knob — a capped render is a disclosed
    lower bound, never a silent truncation.
    """

    scope: str
    candidate_subjects: int
    rendered_subjects: int
    candidate_particles: int
    rendered_particles: int
    # cross-exporter contract: particles dropped below the
    # min_particle_confidence floor (a filter, reported separately from caps).
    dropped_below_threshold: int = 0
    # rung-4 fail-closed exclusions in an as-of render; always 0 otherwise.
    excluded_undatable: int = 0


class GraphData(BaseModel):
    """One scoped subgraph render — the contract shared by the static HTML
    exporter (embedded JSON) and the future ``GET /graph`` endpoint.

    Never a whole store: ``scope_type`` + ``scope_ref`` are mandatory, and the
    census + ``disclosures`` carry the anti-hairball cap accounting.
    """

    scope_type: Literal["subject", "query", "inconsistency", "projection"]
    # The anchor subject id, the query text, the INCONSISTENCY particle id, or
    # the "manifest#section" address of a projection selection.
    scope_ref: str
    # reference instant for an as-of render; None = present time.
    as_of: datetime | None = None
    # True when retired supersession-chain ancestors are included.
    history: bool = False
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    supersessions: list[GraphSupersession] = Field(default_factory=list)
    # Every rendered particle's epistemics payload, keyed by particle id.
    particles: dict[str, GraphParticleInfo] = Field(default_factory=dict)
    census: GraphCensus
    # Human-readable disclosure lines (cap bindings, threshold drops,
    # undatable exclusions) rendered as the page banner.
    disclosures: list[str] = Field(default_factory=list)


class CalibrationBucket(BaseModel):
    """One row in the calibration_source breakdown of a QualityReport."""

    source: str
    count: int
    fraction: float


class QualityReport(BaseModel):
    """Extraction quality dashboard snapshot (Appendix B §8).

    All metrics are computed from live DB queries with no LLM involvement.
    For full structural/semantic diagnostics use the Lint operation instead.
    """

    generated_at: datetime = Field(default_factory=_utcnow)
    # Particles
    active_particles: int
    inconsistency_particles: int
    calibration: list[CalibrationBucket]
    extractor_direct_fraction: float
    # Corpus
    total_entries: int
    snapshots_pending: int
    snapshots_in_progress: int
    snapshots_complete: int
    snapshots_failed: int
    # Subjects
    total_subjects: int
    subjects_without_particles: int
    # Structured-claim coverage. Reported, never enforced:
    # absence of an annotation is a legal permanent state, so this is a count
    # and never a finding. ``structured_claims_by_structurizer`` is keyed
    # ``"<structurizer_id>@<version>"``.
    structured_claims: int = 0
    structured_claims_by_structurizer: dict[str, int] = Field(default_factory=dict)


class LintFinding(BaseModel):
    particle_id: str | None = None
    subject_id: str | None = None  # set for subject-level findings (e.g. PHANTOM_SUBJECT)
    corpus_entry_id: str | None = None
    finding_type: str  # e.g. "STALENESS", "RETRACTION_CASCADE", "PHANTOM_SUBJECT"
    severity: str  # "ERROR" | "WARNING" | "INFO"
    detail: str
    recommended_action: str | None = None
    # The claim text (``Particle.content``) of the particle this finding
    # references — set whenever ``particle_id`` is present, ``None`` otherwise
    #. Lets a curation client (the Obsidian plugin, the
    # PWA) show WHAT a flagged particle says without a second
    # ``particles particle show <id>`` round-trip. A particle's ``content`` is already a
    # one-sentence atomic claim, so the full text is carried — no excerpt field.
    # Server-side enrichment: each finder already holds the ``Particle`` at the
    # point of creation, so this is populated inline with no extra store lookup.
    particle_content: str | None = None
    # The bases that fired on a ``CONTESTED`` finding, in the badge's
    # canonical order (stance, divergence, inconsistency); ``None`` on every
    # other finding type. Carried structurally, in the same spirit as
    # ``particle_content`` above, so the downstream hygiene surfaces — the
    # card, the census bucket, the run record — can
    # report the class *by basis* instead of re-deriving it from ``detail`` or
    # collapsing three signals into one unattributed total.
    contested_bases: list[str] | None = None
    # The full id of the open INCONSISTENCY particle behind a ``CONTESTED``
    # finding whose ``inconsistency`` basis fired; ``None`` otherwise. Carried
    # structurally (the ``detail`` prose truncates it to 8 chars) so a client
    # can link straight to the contradiction's evidence — the
    # ``scope=inconsistency`` graph render — without parsing prose.
    inconsistency_id: str | None = None


class LintReport(BaseModel):
    run_at: datetime = Field(default_factory=_utcnow)
    findings: list[LintFinding] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)  # finding_type → count
    # finding_type → count of findings whose underlying particle was auto-transitioned
    # (PROVENANCE_STALE / RETRACTED_DEPENDENCY / CORPUS_ENTRY_MISSING) by this run.
    # Empty when fix=False; categories considered are listed in FIX_CAPABLE_CATEGORIES.
    fixed_counts: dict[str, int] = Field(default_factory=dict)
    # True when semantic checks were requested but skipped because the LLM was
    # unavailable (account-level failure, circuit breaker open). A client
    # surfaces this rather than mistaking the missing semantic findings for a clean bill.
    semantic_skipped: bool = False


# Categories whose findings the orchestrator can auto-resolve via status transition
# when fix=True. Surfaced in LintReport.fixed_counts and in CLI "Auto-fixed: …" output.
FIX_CAPABLE_CATEGORIES: tuple[str, ...] = (
    "STALENESS",
    "RETRACTION_CASCADE",
    "CORPUS_LINK_INTEGRITY",
)


# ---------------------------------------------------------------------------
# Co-evidential candidate suggestion (`links suggest`)
# ---------------------------------------------------------------------------


class CoEvidentialCandidate(BaseModel):
    """One candidate co-evidential pair within a Subject.

    Surfaced when two ACTIVE particles in the same Subject have cosine
    similarity at or above ``links_suggest.candidate_threshold`` and are not
    already linked CO_EVIDENTIAL (transitively). The ``verdict`` is populated
    only in ``LLM_JUDGE`` / ``APPLY`` mode; it is ``None`` in ``REPORT`` mode.
    ``applied`` is set ``True`` when ``APPLY`` mode created the link.
    """

    particle_a: str
    particle_b: str
    similarity: float = Field(ge=0.0, le=1.0)
    verdict: JudgeVerdictKind | None = None
    applied: bool = False


class CandidateCluster(BaseModel):
    """All candidate pairs proposed within a single Subject."""

    subject_id: str
    subject_name: str | None = None
    candidates: list[CoEvidentialCandidate] = Field(default_factory=list)


class JudgeVerdict(BaseModel):
    """An LLM verdict on one candidate pair, keyed by the short id-pair.

    ``pair_key`` is ``"<a8>+<b8>"`` — the two particles' 8-char id prefixes
    joined by ``+``, matching the key the batch prompt asks the LLM to return.
    """

    pair_key: str
    verdict: JudgeVerdictKind
    rationale: str | None = None


class SuggestReport(BaseModel):
    """Result of a ``links suggest`` run.

    ``clusters`` carries the per-Subject candidate lists. ``summary`` counts
    total candidates / judged / applied. ``warnings`` records non-fatal
    conditions (e.g. token-budget fan-out, missing embeddings).
    """

    run_at: datetime = Field(default_factory=_utcnow)
    mode: SuggestMode
    clusters: list[CandidateCluster] = Field(default_factory=list)
    total_candidates: int = 0
    judged_pairs: int = 0
    applied_pairs: int = 0
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Exact-duplicate auto-merge (`links dedup`)
# ---------------------------------------------------------------------------


class DuplicateGroup(BaseModel):
    """One identical-content ACTIVE group — Tier A.

    Membership is decided by **exact content equality** (``content_hash`` is
    :func:`particles.core.duplicate_key.content_hash` over the shared
    ``content`` string — the §6.10 normalized key, whitespace runs collapsed
    and sentence-final punctuation trimmed, case and wording preserved), not by
    a similarity threshold: cosine is symptom, the hash is the mechanism, so
    the tier stays decidable and model-independent. The mop and the extract-time suppression rung share that one key, so prevention
    and cleanup reach exactly the same pairs. Every member is ACTIVE, truth-apt,
    asserted, and shares the group's ``stance:holder`` (``None`` for
    non-stances). Members need **not** carry a Subject — it was
    demoted from a membership gate to an election preference, since
    at exact identity the safety comes from the content key, not the Subject.

    ``survivor_id`` is elected deterministically — subject-linked before
    subject-less, then earliest ``asserted_at``, ties broken by lexicographically
    smallest id — which is what makes the merge idempotent and the revert
    scriptable. ``redundant_ids`` are the copies a merge would supersede; the
    survivor is never mutated.
    """

    content_hash: str
    content_excerpt: str
    subject_ids: list[str] = Field(default_factory=list)
    survivor_id: str
    redundant_ids: list[str] = Field(default_factory=list)
    # Subject composition of this group: ``linked`` = every member
    # carries a Subject (the only class it could reach), ``orphan`` = none
    # do, ``mixed`` = both. The split is what makes the widened reach visible in
    # a dry run before ``--apply`` is typed.
    subject_class: Literal["linked", "mixed", "orphan"] = "linked"
    # True once this group has actually been merged (APPLY only; never in a dry run).
    merged: bool = False


class DedupReport(BaseModel):
    """Result of a ``links dedup`` run.

    ``dry_run=True`` is the read-only census: ``groups`` is populated and
    nothing in the store changed. ``dry_run=False`` additionally reports what
    was merged; ``deferred_groups`` / ``deferred_redundant`` disclose the
    remainder when ``links_suggest.auto_merge.max_per_run`` binds, so a capped
    run never reads as a complete cleanup.
    """

    run_at: datetime = Field(default_factory=_utcnow)
    dry_run: bool = True
    groups: list[DuplicateGroup] = Field(default_factory=list)
    total_groups: int = 0
    total_redundant: int = 0
    merged_groups: int = 0
    merged_particles: int = 0
    links_created: int = 0
    deferred_groups: int = 0
    deferred_redundant: int = 0
    warnings: list[str] = Field(default_factory=list)


class UnmergeSkipReason(StrEnum):
    """Why one listed copy could not be restored by an unmerge.

    Drift is the normal case, not the exception — the live store had a survivor
    move on within 24 hours of the merge — so every skip is named rather than
    silently dropped, and none of them is fatal to the rest of the group.
    """

    #: Already ACTIVE — a previous unmerge restored it. The idempotent case.
    ALREADY_ACTIVE = "ALREADY_ACTIVE"
    #: Moved to some other status since the merge (a retraction cascade, an
    #: operator retraction). Restoring it would overturn a later decision.
    NOT_SUPERSEDED = "NOT_SUPERSEDED"
    #: Still SUPERSEDED, but by something other than the auto-merge — so the
    #: §6.6 reason gate refuses it and this revert has no standing over it.
    NOT_MERGE_SUPERSEDED = "NOT_MERGE_SUPERSEDED"
    #: Listed in the event but no longer in the store.
    MISSING = "MISSING"


class UnmergeSkip(BaseModel):
    """One copy the revert declined to restore, with the reason it declined."""

    particle_id: str
    reason: UnmergeSkipReason
    #: The status found, for the operator's disclosure line. ``None`` if MISSING.
    found_status: str | None = None
    found_status_reason: str | None = None


class UnmergeGroup(BaseModel):
    """The revert plan (or outcome) for one ``DUPLICATES_MERGED`` event.

    The survivor is **never** touched — symmetric with the
    never-mutate-the-survivor rule, and it holds even when the survivor itself
    has drifted, which is why ``survivor_status`` is reported rather than acted
    on: the survivor's later staleness is an independent fact with its own
    cause, and an unmerge has no standing to overturn it.
    """

    merge_event_id: str
    survivor_id: str
    survivor_status: str | None = None
    content_hash: str | None = None
    restored_ids: list[str] = Field(default_factory=list)
    skipped: list[UnmergeSkip] = Field(default_factory=list)
    relations_deleted: int = 0
    #: True once the writes actually landed (never in a dry run).
    reverted: bool = False


class UnmergeReport(BaseModel):
    """Result of a ``links unmerge`` run.

    ``dry_run=True`` is the plan: ``groups`` is fully populated, including
    every skip, and nothing in the store changed. The counts are always
    complete even when markdown rendering truncates the group list, so a
    partial revert never reads as a whole one.
    """

    run_at: datetime = Field(default_factory=_utcnow)
    dry_run: bool = True
    #: Human-readable description of what selected these events, echoed back so
    #: a `--since` window is visible in the JSON artifact.
    selector: str = ""
    groups: list[UnmergeGroup] = Field(default_factory=list)
    total_events: int = 0
    restored_particles: int = 0
    skipped_particles: int = 0
    relations_deleted: int = 0
    warnings: list[str] = Field(default_factory=list)
