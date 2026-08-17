"""The conformance contract — which fields are REQUIRED / RECOMMENDED / OPTIONAL.

This is the single source of truth operators consult when asking *"why did my
extractor get a warning on field X?"*. Each entry carries a one-line rationale
so a reader can decide whether to dispute the tier rather than fix the field.

the contract is intentionally one small Python file. Adding a
new field means updating this list; reclassifying a field means changing the
tier here and accepting whatever new failures fall out of the next CI run.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from particles.conformance.types import (
    DiversityRule,
    DiversitySeverity,
    FieldContract,
    FieldTier,
)
from particles.extraction.subject_scope import subject_expected

CONTRACT: list[FieldContract] = [
    # ------------------------------------------------------------------
    # REQUIRED — schema invariants depend on these
    # ------------------------------------------------------------------
    FieldContract("id", FieldTier.REQUIRED, "Primary key; absence breaks every store"),
    FieldContract(
        "content",
        FieldTier.REQUIRED,
        "The claim itself; min_length=1 already enforced by Pydantic",
    ),
    FieldContract(
        "confidence.value",
        FieldTier.REQUIRED,
        "Without confidence, ranking is meaningless",
    ),
    FieldContract(
        "provenance",
        FieldTier.REQUIRED,
        "Particle without provenance is an orphan. PARTICLE-typed "
        "refs carry the referenced particle id in corpus_entry_id",
    ),
    FieldContract(
        "asserted_by",
        FieldTier.REQUIRED,
        "Audit trail; required for trust attribution",
    ),
    FieldContract(
        "status",
        FieldTier.REQUIRED,
        "Lifecycle state (defaults to ACTIVE; required to be set)",
    ),
    FieldContract("schema_version", FieldTier.REQUIRED, "Forward-compatibility marker"),
    FieldContract("particle_type", FieldTier.REQUIRED, "CLAIM vs REVIEW vs ANNOTATION"),
    FieldContract(
        "subject_ids",
        FieldTier.REQUIRED,
        "Subjects are first-class; orphan particles undermine the knowledge graph",
    ),
    FieldContract(
        "uncertainty_nature",
        FieldTier.REQUIRED,
        "Required by the schema, but value diversity is enforced — see DIVERSITY",
    ),
    # ------------------------------------------------------------------
    # RECOMMENDED — extractors expected to populate
    # ------------------------------------------------------------------
    FieldContract(
        "confidence.calibration_source",
        FieldTier.RECOMMENDED,
        "Without it, the stored value's provenance is unauditable; "
        "consumers must assume extractor-direct",
    ),
    FieldContract(
        "provenance[].snapshot_id",
        FieldTier.RECOMMENDED,
        "Snapshot-level provenance is the spec target; entry-level is the v0.2 floor",
    ),
    FieldContract(
        "extractor_ref",
        FieldTier.RECOMMENDED,
        "Required for trust weighting; "
        "without it queries fall back to general-extractor trust",
    ),
    # ------------------------------------------------------------------
    # OPTIONAL — informational only
    # ------------------------------------------------------------------
    FieldContract(
        "confidence.variance",
        FieldTier.OPTIONAL,
        "Useful when the extractor can model output uncertainty; many cannot",
    ),
    FieldContract(
        "confidence.calibration_method",
        FieldTier.OPTIONAL,
        "Populated when a calibration pass has been applied",
    ),
    FieldContract("uncertainty_kind", FieldTier.OPTIONAL, "Extension A free-text label"),
    FieldContract("valid_until", FieldTier.OPTIONAL, "Time-bounded claims only"),
    FieldContract("supersedes", FieldTier.OPTIONAL, "Set by reindex, not by extractors"),
    FieldContract(
        "tags",
        FieldTier.OPTIONAL,
        "Extension C — populated by tag-aware extractors",
    ),
    FieldContract(
        "context_fingerprint",
        FieldTier.OPTIONAL,
        "Extension C.1 — populated where fingerprinting applies "
        "(stamped by the pipeline, "
        "not by individual extractors)",
    ),
    FieldContract(
        "properties",
        FieldTier.OPTIONAL,
        "Structured ontology-keyed data for structured extractors only",
    ),
    FieldContract(
        "structured_claim",
        FieldTier.OPTIONAL,
        "S-P-O annotation. OPTIONAL rather than RECOMMENDED because "
        "makes absence a legal permanent state with no coverage "
        "floor — a tier carrying a threshold would contradict it. The rate is "
        "the per-extractor coverage probe",
    ),
    FieldContract(
        "canonical_form",
        FieldTier.OPTIONAL,
        "marker for which of the prose/structured pair is the "
        "assertion. Always populated (it has a default), so the useful signal "
        "is its distribution, not its rate",
    ),
    FieldContract(
        "sequence_context",
        FieldTier.OPTIONAL,
        "Reserved for ordered-extraction extractors",
    ),
    FieldContract("basis", FieldTier.OPTIONAL, "Free-form supporting evidence dict"),
    FieldContract(
        "contributors",
        FieldTier.OPTIONAL,
        "Extension D/E contributor attribution — populated by "
        "import / contribution / review, not by extractors; OPTIONAL so the "
        "per-extractor fill-rate is not spuriously flagged",
    ),
]


#: Per-field measurement exemptions — "which particles does this
#: field's rate even *apply* to?", keyed by contract field path. A field absent
#: from this map is measured over every particle the run produced, which is what
#: every field did before.
#:
#: Only ``subject_ids`` has one, and it is not a conformance invention: techspec
#: §9's subject-count table already enumerates the legitimate zero-subject cases,
#: and ``L-STR-09`` already honoured most of them. §14.5 measured a 100 % floor
#: over *everything*, so the two sections contradicted each other — a latent
#: defect that any DOCUMENT_META emitter had and that journal prose was
#: merely the first genre to expose at scale.
#:
#: The predicate is deliberately **spec-derived, not extractor-declared**. An
#: extractor cannot widen its own exemption: adding a class means changing
#: techspec §9 and this map. That is the distinction drawn when it
#: rejected a self-declared extractor attribute as a legal key for conditional
#: rule application.
FIELD_EXEMPTIONS: dict[str, Callable[[Any], bool]] = {
    "subject_ids": lambda p: not subject_expected(p.particle_type, p.properties),
}


DIVERSITY: list[DiversityRule] = [
    DiversityRule(
        field="uncertainty_nature",
        min_distinct_values=2,
        severity=DiversitySeverity.ADVISORY,
        rationale=(
            "ALEATORY vs EPISTEMIC is the core PSUM distinction; an extractor that "
            "always emits EPISTEMIC is technically populating the field but giving "
            "queries no signal. ADVISORY, not FAIL: the outcome tracks "
            "whether the *source vocabulary* carries a distinguishable "
            "stochastic-quantity signal — English prose does, generic RDF and "
            "catalogue records do not — so a structured extractor reporting one "
            "distinct value is honest, and an LLM extractor reporting two is a "
            "sampled result that may not repeat. Read the value_counts margin, "
            "not the pass/fail."
        ),
    ),
]


# Enum-typed fields where 'populated' means 'non-default value across the
# fixture run' rather than 'any value'. Used by the validator to count
# distinct_values; matters for DIVERSITY rules.
ENUM_FIELDS: frozenset[str] = frozenset(
    {
        "uncertainty_nature",
        "particle_type",
        "status",
        "confidence.calibration_source",
        "canonical_form",
    }
)


def field_value(particle: Any, dotted_path: str) -> Any:
    """Resolve a dotted field path (``"confidence.value"``) on a particle.

    Supports a ``[]`` token to descend into list elements:
    ``"provenance[].snapshot_id"`` returns a list of ``snapshot_id`` values,
    one per ``ProvenanceRef`` in the particle's provenance chain.

    Returns ``None`` if any step of the path resolves to ``None`` or a missing
    attribute. The validator interprets ``None`` (and empty list / empty
    string / empty dict for non-enum fields) as *unpopulated*.
    """
    parts = dotted_path.split(".")
    current: Any = particle
    for part in parts:
        if part.endswith("[]"):
            attr = part[:-2]
            seq = getattr(current, attr, None)
            if seq is None or not isinstance(seq, list):
                return None
            # The remaining parts apply to each element. Recurse.
            remainder = ".".join(parts[parts.index(part) + 1 :])
            if not remainder:
                return seq
            return [field_value(item, remainder) for item in seq]
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def is_populated(value: Any) -> bool:
    """A value counts as 'populated' if it is non-None and non-empty.

    Empty list, empty string, empty dict all count as *not* populated. For
    list-of-values returned by ``field_value`` with a ``[]`` path, every
    element must itself be populated.
    """
    if value is None:
        return False
    if isinstance(value, (str, list, dict)) and len(value) == 0:
        return False
    if isinstance(value, list):
        # provenance[].snapshot_id case: all entries must be populated
        return all(is_populated(v) for v in value)
    return True
