# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Conformance Profile loader & validator.

Loads ``artifacts/conformance/profile.yaml`` — the machine-readable companion to
``docs/spec/conformance-profile.md`` — into typed Pydantic models, and resolves
a constant's declared ``config_path`` against the live :func:`get_config`
singleton so the drift-guard test
(``tests/conformance/test_profile_sync.py``) can assert the published value
still equals the SDK default. The profile is the single source of truth for the
behavioural/quantitative conformance surface; this module is its parser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


def _profile_file() -> Path:
    """Locate ``profile.yaml``: wheel-packaged copy first, source tree second.

    The same ladder the schema artifacts use. ``parents[2]`` is the
    repo root only in a source checkout; for an installed distribution it is
    ``site-packages``, so without the packaged copy every installed caller —
    ``particles conformance check``, the conformance runner, the calibration
    fitter — raised ``FileNotFoundError`` on a path that never existed. The
    force-include lives on the Client distribution, which owns
    ``particles/conformance``.
    """
    packaged = Path(__file__).parent.parent / "_artifacts" / "conformance" / "profile.yaml"
    if packaged.exists():
        return packaged
    return Path(__file__).parents[2] / "artifacts" / "conformance" / "profile.yaml"


class EmbeddingProfileRef(BaseModel):
    """The reference ``{model, dim, normalization}`` an L3 backend declares."""

    model: str
    dim: int
    normalization: str


class EmbeddingProfileBlock(BaseModel):
    reference: EmbeddingProfileRef


class SimilarityScale(BaseModel):
    metric: str
    range: list[float]
    clamp_negative: bool


class ConstantEntry(BaseModel):
    """One §2 constant: its published value plus the config attribute it mirrors."""

    name: str
    value: float
    config_path: str | None = None
    level: str
    spec: str


class RecencyDecaySource(BaseModel):
    half_life_days: float
    floor: float


class RecencyDecay(BaseModel):
    config_path_root: str
    level: str
    sources: dict[str, RecencyDecaySource]


class EffConfVector(BaseModel):
    id: str
    value: float
    extractor_trust_weight: float
    source_trust_rank: float
    recency_factor: float
    expected: float


class RecencyVector(BaseModel):
    id: str
    age_days: float
    half_life_days: float
    floor: float
    expected: float


class CalibrationVector(BaseModel):
    id: str
    raw: float
    T: float
    expected: float


class NoisyOrVector(BaseModel):
    id: str
    entries: list[tuple[float, str]]
    expected: float


class LadderParticleStub(BaseModel):
    """The plain-data particle a §6.4 ladder vector carries.

    Deliberately *not* a :class:`~particles.core.schema.Particle`: the ladder
    reads exactly two attributes, so a vector that carried a whole particle
    would publish incidental fields a second implementation must not have to
    reproduce. The runner materialises a minimal Particle from these two.
    """

    assertion_modality: str
    uncertainty_nature: str


class ConflictLadderVector(BaseModel):
    """One (existing, new) pair plus the caller-resolved ladder inputs.

    The defaults mirror :func:`particles.core.conflict_resolution.resolve_conflict`
    so a vector states only the inputs its rung actually turns on.
    """

    id: str
    existing: LadderParticleStub
    new: LadderParticleStub
    has_contradiction_signal: bool = True
    new_supersedes_existing: bool = False
    existing_supersedes_new: bool = False
    trust_score_existing: float | None = None
    trust_score_new: float | None = None
    trust_differential_threshold: float = 0.15
    single_trust_order: bool = True
    expected: str


class FingerprintParticleStub(BaseModel):
    """An ``(id, status)`` row of the §16.1 baseline the digest is taken over."""

    id: str
    status: str


class FingerprintVector(BaseModel):
    id: str
    particles: list[FingerprintParticleStub]
    expected: str


class CascadeGateVector(BaseModel):
    id: str
    policy_provenance: str
    reviewer_confirmations: int
    min_reviewer_confirmations: int
    expected: bool


class CascadeCapVector(BaseModel):
    id: str
    candidate_count: int
    max_per_run: int
    expected_processed: int
    expected_capped: bool


class TestVectors(BaseModel):
    """The published L2 vector set — §4's formulas and §5's algorithms.

    The four formula families reproduce within ``float_tolerance``; the three
    algorithm families are categorical (verdict string, hex digest, boolean +
    count) and reproduce exactly.
    """

    effective_confidence: list[EffConfVector]
    recency_factor: list[RecencyVector]
    calibration_apply: list[CalibrationVector]
    noisy_or_merge: list[NoisyOrVector]
    conflict_ladder: list[ConflictLadderVector]
    context_fingerprint: list[FingerprintVector]
    cascade_gate: list[CascadeGateVector]
    cascade_cap: list[CascadeCapVector]


class CalibrationBlock(BaseModel):
    temperature_bounds: list[float]
    fit_objective: str
    #: The functional form ``temperature`` parameterises — ``"logit"``
    #: for ``sigmoid(logit(raw) / T)``. Modelled and published so
    #: the form would be *machine-readable rather than inferred from prose*; left
    #: off this block it was parsed and silently dropped, which made that claim
    #: false and left ``formulas.calibration_apply`` free to go on naming the
    #: retired ``clamp(raw / T, 0, 1)`` form with nothing able to notice (it did,
    #: through 1.115.0).
    transform: str


class ConformanceProfile(BaseModel):
    """The parsed ``profile.yaml`` — machine-readable companion."""

    profile_version: str
    float_tolerance: float
    embedding_profile: EmbeddingProfileBlock
    similarity_scale: SimilarityScale
    similarity_vectors_ref: str
    constants: list[ConstantEntry]
    similarity_thresholds: list[ConstantEntry]
    recency_decay: RecencyDecay
    formulas: dict[str, str]
    calibration: CalibrationBlock
    test_vectors: TestVectors

    def all_constants(self) -> list[ConstantEntry]:
        """Every config-bound constant (§2 scalar thresholds + §2.4 similarity)."""
        return [*self.constants, *self.similarity_thresholds]


def load_profile(path: Path | None = None) -> ConformanceProfile:
    """Load and validate the Conformance Profile artifact.

    Args:
        path: Override for the profile location (default:
            ``artifacts/conformance/profile.yaml`` relative to the repo root).
    """
    src = path or _profile_file()
    data = yaml.safe_load(src.read_text(encoding="utf-8"))
    return ConformanceProfile.model_validate(data)


def profile_path() -> Path:
    """Absolute path to the shipped ``profile.yaml`` artifact."""
    return _profile_file()


def resolve_config_value(config_path: str) -> Any:
    """Read a dotted attribute path (e.g. ``trust.differential_threshold``) off
    the live :func:`get_config` singleton — the drift-guard introspection seam."""
    from particles.config import get_config

    obj: Any = get_config()
    for attr in config_path.split("."):
        obj = getattr(obj, attr)
    return obj
