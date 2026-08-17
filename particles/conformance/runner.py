"""Conformance Profile runner.

Self-certifies an implementation against the Conformance Profile:

- **L2 (deterministic-compute)** recomputes every ``profile.yaml`` test vector
  via the SDK's own pure functions and asserts the published expectation. Two
  kinds of family:

  - the §4 **formulas** (:mod:`particles.core.scoring.confidence`,
    :mod:`particles.core.scoring.decay`, :mod:`particles.extraction.calibration`)
    are numeric and must match within ``float_tolerance``;
  - the §5 **algorithms** — the §6.4 conflict-ladder ordering
    (:mod:`particles.core.conflict_resolution`), the §16.1 fingerprint
    (:mod:`particles.core.fingerprint`), and §15.1 cascade gating
    (:mod:`particles.core.cascade_gate`) — produce categorical outputs
    (a verdict, a hex digest, a boolean and a count) and must match exactly.

  Reproducible exactly.
- **L3 (profile-similarity)** embeds the ``similarity_vectors.json`` pairs under
  the live embedding profile and checks each pair lands in its band and each
  top-k case's expected members appear in the computed top-k. SKIPPED when the
  embedding model is unavailable (no network / weights).

L1 (structural) is delegated to the existing schema/SHACL artifacts and their
validators (:mod:`particles.conformance.jsonschema`,
:mod:`particles.conformance.shacl`); the runner reports it as such rather than
re-deriving it.

The runner is the reference SDK's self-certification harness; an independent
implementation ports the same vectors against its own functions to make the
identical yes/no claim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from particles.conformance.profile import (
    ConformanceProfile,
    LadderParticleStub,
    load_profile,
    profile_path,
)
from particles.core.schema import (
    AssertionModality,
    Confidence,
    Particle,
    PolicyProvenance,
    UncertaintyNature,
)
from particles.core.status import Status

# A fixed reference instant so the recency vectors are reproducible — the real
# decay function takes ``now`` as a parameter, so no wall-clock dependency.
_REF_NOW = datetime(2026, 1, 1, tzinfo=UTC)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class LevelReport:
    level: str
    status: str  # "PASS" | "FAIL" | "SKIPPED"
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


def _cmp(name: str, got: float, expected: float, tol: float) -> CheckResult:
    ok = abs(got - expected) <= tol
    return CheckResult(name, ok, f"got={got!r} expected={expected!r} tol={tol:g}")


def _eq(name: str, got: object, expected: object) -> CheckResult:
    """Exact comparison — the §5 algorithm families are categorical, not numeric."""
    ok = got == expected
    return CheckResult(name, ok, f"got={got!r} expected={expected!r}")


def _ladder_particle(stub: LadderParticleStub) -> Particle:
    """Materialise the minimal Particle the §6.4 ladder reads.

    The vector carries only ``assertion_modality`` (the truth-apt gate) and
    ``uncertainty_nature`` (the ALEATORY exclusion); every other field here is
    inert filler required by the schema, chosen so it cannot influence the
    verdict. Keeping the vector to those two fields is what makes it portable
    to an implementation whose particle type differs.
    """
    return Particle(
        content="conformance vector",
        confidence=Confidence(value=0.5),
        uncertainty_nature=UncertaintyNature(stub.uncertainty_nature),
        assertion_modality=AssertionModality(stub.assertion_modality),
        asserted_by="conformance-runner",
    )


def run_l2(profile: ConformanceProfile) -> LevelReport:
    """Recompute the deterministic test vectors via the SDK's real functions."""
    from particles.core.cascade_gate import apply_cascade_cap, cascade_gate_passes
    from particles.core.conflict_resolution import resolve_conflict
    from particles.core.fingerprint import context_fingerprint
    from particles.core.scoring.confidence import (
        compute_effective_confidence,
        merge_co_evidential_confidence,
    )
    from particles.core.scoring.decay import recency_factor_from_params
    from particles.extraction.calibration import TemperatureScaler

    tol = profile.float_tolerance
    checks: list[CheckResult] = []

    for v in profile.test_vectors.effective_confidence:
        got = compute_effective_confidence(
            v.value, v.extractor_trust_weight, v.source_trust_rank, v.recency_factor
        )
        checks.append(_cmp(f"effective_confidence/{v.id}", got, v.expected, tol))

    for rv in profile.test_vectors.recency_factor:
        content_published_at = _REF_NOW - timedelta(days=rv.age_days)
        got = recency_factor_from_params(
            content_published_at, rv.half_life_days, rv.floor, now=_REF_NOW
        )
        checks.append(_cmp(f"recency_factor/{rv.id}", got, rv.expected, tol))

    for cv in profile.test_vectors.calibration_apply:
        scaler = TemperatureScaler()
        scaler.temperature = cv.T
        got = scaler.calibrate(cv.raw)
        checks.append(_cmp(f"calibration_apply/{cv.id}", got, cv.expected, tol))

    for nv in profile.test_vectors.noisy_or_merge:
        got = merge_co_evidential_confidence([(e[0], e[1]) for e in nv.entries])
        checks.append(_cmp(f"noisy_or_merge/{nv.id}", got, nv.expected, tol))

    # §6.4 ladder ordering — the verdict is a categorical
    # outcome, so this is exact equality, not a tolerance comparison.
    for lv in profile.test_vectors.conflict_ladder:
        verdict = resolve_conflict(
            _ladder_particle(lv.existing),
            _ladder_particle(lv.new),
            has_contradiction_signal=lv.has_contradiction_signal,
            new_supersedes_existing=lv.new_supersedes_existing,
            existing_supersedes_new=lv.existing_supersedes_new,
            trust_score_existing=lv.trust_score_existing,
            trust_score_new=lv.trust_score_new,
            trust_differential_threshold=lv.trust_differential_threshold,
            single_trust_order=lv.single_trust_order,
        )
        checks.append(_eq(f"conflict_ladder/{lv.id}", verdict.value, lv.expected))

    # §16.1 fingerprint. Step 1 — the ACTIVE filter — is a store
    # query in the SDK, so the runner applies it to the vector's plain rows;
    # steps 2–3 (sort + SHA-256) are the shared pure function.
    for fv in profile.test_vectors.context_fingerprint:
        active = [p.id for p in fv.particles if p.status == Status.ACTIVE.value]
        checks.append(_eq(f"context_fingerprint/{fv.id}", context_fingerprint(active), fv.expected))

    # §15.1 cascade gating — the policy gate and the per-run cap.
    for gv in profile.test_vectors.cascade_gate:
        got_gate = cascade_gate_passes(
            PolicyProvenance(gv.policy_provenance),
            reviewer_confirmations=gv.reviewer_confirmations,
            min_reviewer_confirmations=gv.min_reviewer_confirmations,
        )
        checks.append(_eq(f"cascade_gate/{gv.id}", got_gate, gv.expected))

    for cv2 in profile.test_vectors.cascade_cap:
        got_cap = apply_cascade_cap(cv2.candidate_count, cv2.max_per_run)
        checks.append(
            _eq(f"cascade_cap/{cv2.id}", got_cap, (cv2.expected_processed, cv2.expected_capped))
        )

    status = "PASS" if all(c.passed for c in checks) else "FAIL"
    return LevelReport("L2", status, checks)


def run_l3(profile: ConformanceProfile, vectors_dir: Path | None = None) -> LevelReport:
    """Embed the similarity vectors under the live profile; check bands + top-k.

    SKIPPED (not FAIL) when the embedding model cannot be loaded — the L3 claim
    is about the *embedding backend*, which an offline / no-weights environment
    cannot exercise.
    """
    import numpy as np

    from particles.embeddings import (
        cosine_similarity,
        get_embedding_model,
        get_embedding_profile,
    )

    model = get_embedding_model()
    if model is None:
        return LevelReport(
            "L3",
            "SKIPPED",
            [CheckResult("embedding_model", False, "embedding model unavailable")],
        )

    vec_dir = vectors_dir or profile_path().parent
    data = json.loads((vec_dir / profile.similarity_vectors_ref).read_text(encoding="utf-8"))

    checks: list[CheckResult] = []

    # Disclose the live profile so a non-reference encoder's verdict is legible.
    live = get_embedding_profile()
    ref = profile.embedding_profile.reference
    if (live.model, live.dim, live.normalization) != (ref.model, ref.dim, ref.normalization):
        checks.append(
            CheckResult(
                "embedding_profile",
                True,
                f"non-reference profile {live.model}/{live.dim}/{live.normalization} "
                f"(reference {ref.model}/{ref.dim}/{ref.normalization}); bands still apply",
            )
        )

    def _embed(text: str) -> np.ndarray:
        vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        return np.asarray(vec, dtype=float)

    for pair in data["pairs"]:
        sim = cosine_similarity(_embed(pair["text_a"]), _embed(pair["text_b"]))
        lo, hi = pair["band"]
        passed = lo <= sim <= hi
        checks.append(CheckResult(f"pair/{pair['id']}", passed, f"sim={sim:.4f} band=[{lo}, {hi}]"))

    for tk in data["topk"]:
        q = _embed(tk["query"])
        scored = [(c["id"], cosine_similarity(q, _embed(c["text"]))) for c in tk["corpus"]]
        ranked = [cid for cid, _ in sorted(scored, key=lambda s: -s[1])][: tk["k"]]
        expected = set(tk["expected_topk"])
        passed = expected.issubset(set(ranked))
        checks.append(
            CheckResult(
                f"topk/{tk['id']}",
                passed,
                f"top{tk['k']}={ranked} expected⊇{sorted(expected)}",
            )
        )

    status = "PASS" if all(c.passed for c in checks) else "FAIL"
    return LevelReport("L3", status, checks)


def run_check(
    levels: tuple[str, ...] = ("L2", "L3"),
    profile: ConformanceProfile | None = None,
    vectors_dir: Path | None = None,
) -> list[LevelReport]:
    """Run the requested conformance levels and return one report per level."""
    prof = profile or load_profile()
    reports: list[LevelReport] = []
    if "L2" in levels:
        reports.append(run_l2(prof))
    if "L3" in levels:
        reports.append(run_l3(prof, vectors_dir))
    return reports
