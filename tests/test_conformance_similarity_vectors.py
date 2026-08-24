# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Conformance instrument for the embedding-similarity contract.

Loads the frozen test-vector set at ``artifacts/conformance/similarity_vectors.json``
and asserts that the reference embedding profile lands inside every declared
band and reproduces every top-k membership. This is the runnable portability
target a second implementation reproduces in its own language; here it doubles
as a regression guard that the reference profile still satisfies its own
published contract.

Marked ``integration`` because it needs the real ``all-MiniLM-L6-v2`` encoder
(network/model download on first run); a key-less / model-less unit run skips it
per ``tests/AGENTS.md``. The band semantics and metric are pinned in
technical-specification.md §8.5.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from particles.embeddings import cosine_similarity, get_embedding_model

pytestmark = pytest.mark.integration

VECTORS_PATH = (
    Path(__file__).resolve().parent.parent / "artifacts" / "conformance" / "similarity_vectors.json"
)


def _load() -> dict[str, object]:
    return json.loads(VECTORS_PATH.read_text())


def test_vector_file_exists_and_declares_reference_profile() -> None:
    data = _load()
    assert data["profile"] == {"model": "all-MiniLM-L6-v2", "dim": 384, "normalization": "l2"}
    assert len(data["pairs"]) >= 20  # question 3 resolved: ~20-40 pairs
    assert len(data["topk"]) >= 1


def test_reference_profile_lands_in_every_band() -> None:
    data = _load()
    model = get_embedding_model()
    if model is None:  # pragma: no cover - integration guard
        pytest.skip("embedding model unavailable")

    failures: list[str] = []
    for pair in data["pairs"]:
        emb = model.encode(
            [pair["text_a"], pair["text_b"]],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        sim = cosine_similarity(emb[0], emb[1])
        lo, hi = pair["band"]
        # The normative invariant: similarity is always on the [0, 1] scale.
        assert 0.0 <= sim <= 1.0, f"{pair['id']}: {sim} outside [0, 1]"
        if not (lo <= sim <= hi):
            failures.append(f"{pair['id']}: sim={sim:.4f} outside band [{lo}, {hi}]")
    assert not failures, "out-of-band vectors:\n" + "\n".join(failures)


def test_reference_profile_reproduces_topk_membership() -> None:
    data = _load()
    model = get_embedding_model()
    if model is None:  # pragma: no cover - integration guard
        pytest.skip("embedding model unavailable")

    failures: list[str] = []
    for case in data["topk"]:
        q_emb = model.encode([case["query"]], convert_to_numpy=True, normalize_embeddings=True)[0]
        scored: list[tuple[str, float]] = []
        for doc in case["corpus"]:
            d_emb = model.encode([doc["text"]], convert_to_numpy=True, normalize_embeddings=True)[0]
            scored.append((doc["id"], cosine_similarity(q_emb, d_emb)))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = {doc_id for doc_id, _ in scored[: case["k"]]}
        expected = set(case["expected_topk"])
        # Membership (⊇), not exact ordering — ordering is non-normative (§8.5).
        if not expected <= top:
            failures.append(
                f"{case['id']}: expected {sorted(expected)} subset of top-{case['k']} {sorted(top)}"
            )
    assert not failures, "top-k membership failures:\n" + "\n".join(failures)
