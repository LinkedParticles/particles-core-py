"""Tests for ``particles/embeddings.py`` — the progress-bar config wiring.

The encoder singleton + load path are exercised across the suite (and the
tracing proxy in ``tests/test_observability.py``); these pin the
``embeddings.progress_bars`` knob: the per-encode ``show_progress_bar``
injection and the enable/disable selection in :func:`_toggle_progress_bars`.
"""

from __future__ import annotations

import importlib
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from particles.config import reset_config
from particles.embeddings import (
    EMBEDDING_MODEL_ID,
    EmbeddingProfile,
    _toggle_progress_bars,
    _TracingEmbeddingModel,
    cosine_similarity,
    get_embedding_profile,
)


class _RecordingInner:
    """Stand-in encoder that records the kwargs it was called with."""

    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] = {}

    def encode(
        self,
        texts: list[str],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        **kwargs: object,
    ) -> list[list[float]]:
        self.last_kwargs = kwargs
        return [[0.0]] * len(texts)


class TestEncodeProgressBarInjection:
    def test_show_progress_bar_false_by_default(self) -> None:
        reset_config()
        inner = _RecordingInner()
        _TracingEmbeddingModel(inner).encode(["a", "b"])
        assert inner.last_kwargs["show_progress_bar"] is False

    def test_show_progress_bar_true_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARTICLES_EMBEDDINGS_PROGRESS_BARS", "1")
        reset_config()
        inner = _RecordingInner()
        _TracingEmbeddingModel(inner).encode(["a"])
        assert inner.last_kwargs["show_progress_bar"] is True

    def test_caller_override_wins(self) -> None:
        # setdefault: an explicit caller value is not clobbered by the config knob.
        reset_config()  # config default is False
        inner = _RecordingInner()
        _TracingEmbeddingModel(inner).encode(["a"], show_progress_bar=True)
        assert inner.last_kwargs["show_progress_bar"] is True


class TestEncodeSerialisedAcrossThreads:
    def test_concurrent_encodes_never_overlap(self) -> None:
        """Two threads calling ``encode`` at once are serialised by the wrapper.

        The extract pipeline offloads ``encode`` to worker threads and a pooled
        / concurrent run has several in flight; two threads inside the same
        torch model at once is a native crash on the MPS backend. The wrapper
        must hold one call until the other returns.
        """
        import threading
        import time

        class _SlowInner:
            def __init__(self) -> None:
                self.active = 0
                self.max_active = 0
                self._guard = threading.Lock()

            def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
                with self._guard:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.05)
                with self._guard:
                    self.active -= 1
                return [[0.0]] * len(texts)

        reset_config()
        inner = _SlowInner()
        model = _TracingEmbeddingModel(inner)
        threads = [threading.Thread(target=model.encode, args=(["x"],)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert inner.max_active == 1


class _FakeBarModule:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def enable_x(self) -> None:
        self.calls.append("enable")

    def disable_x(self) -> None:
        self.calls.append("disable")


class TestToggleProgressBars:
    def test_disables_when_off_enables_when_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeBarModule()
        monkeypatch.setattr(importlib, "import_module", lambda name: fake)
        _toggle_progress_bars("anything", "enable_x", "disable_x", show=False)
        _toggle_progress_bars("anything", "enable_x", "disable_x", show=True)
        assert fake.calls == ["disable", "enable"]

    def test_missing_module_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(name: str) -> object:
            raise ImportError(name)

        monkeypatch.setattr(importlib, "import_module", _boom)
        # Must not raise — an absent sub-dependency degrades to a no-op.
        _toggle_progress_bars("missing.mod", "enable_x", "disable_x", show=False)

    def test_missing_attribute_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(importlib, "import_module", lambda name: _FakeBarModule())
        # A renamed/absent toggle symbol degrades to a no-op rather than raising.
        _toggle_progress_bars("anything", "nonexistent_enable", "nonexistent_disable", show=True)


class TestCosineSimilarity:
    """The single normative similarity primitive: normalized cosine clamped to [0, 1]."""

    def test_identical_vectors_score_one(self) -> None:
        v = np.array([0.6, 0.8], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_anti_correlated_is_clamped_to_zero(self) -> None:
        # Raw cosine of opposite vectors is -1; the clamp pins it to 0 so an
        # anti-correlated pair never sorts below an unrelated 0-similarity pair.
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0

    def test_result_always_in_unit_interval(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(200):
            a = rng.standard_normal(8).astype(np.float32)
            b = rng.standard_normal(8).astype(np.float32)
            sim = cosine_similarity(a, b)
            assert 0.0 <= sim <= 1.0

    def test_in_range_value_is_unchanged_by_clamp(self) -> None:
        # A positive cosine (the regime every threshold compares against) passes
        # through untouched — evidence the clamp does not shift existing semantics.
        a = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        raw = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert raw > 0.0
        assert cosine_similarity(a, b) == pytest.approx(raw)

    def test_zero_vector_does_not_raise(self) -> None:
        a = np.zeros(4, dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        sim = cosine_similarity(a, b)
        assert 0.0 <= sim <= 1.0


class TestEmbeddingProfile:
    """Structured {model, dim, normalization} profile recorded in store metadata."""

    def test_default_profile_is_the_reference(self) -> None:
        reset_config()
        profile = get_embedding_profile()
        assert profile == EmbeddingProfile(model=EMBEDDING_MODEL_ID, dim=384, normalization="l2")
        assert profile.model == "all-MiniLM-L6-v2"

    def test_as_dict_shape(self) -> None:
        reset_config()
        assert get_embedding_profile().as_dict() == {
            "model": "all-MiniLM-L6-v2",
            "dim": 384,
            "normalization": "l2",
        }

    def test_profile_reads_config_at_call_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # dim / normalization come from config — a swap is honoured after reset.
        monkeypatch.setenv("PARTICLES_EMBEDDINGS_DIM", "768")
        monkeypatch.setenv("PARTICLES_EMBEDDINGS_NORMALIZATION", "none")
        reset_config()
        profile = get_embedding_profile()
        assert (profile.dim, profile.normalization) == (768, "none")

    def test_profile_is_frozen(self) -> None:
        profile = EmbeddingProfile(model="m", dim=1, normalization="l2")
        with pytest.raises(FrozenInstanceError):
            profile.dim = 2  # type: ignore[misc]
