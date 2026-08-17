"""Shared sentence-transformer embedding singleton.

The embedding model is a cross-cutting concern: extraction near-dup
detection (``particles/extraction/pipeline.py``), query semantic search
(``particles/operations/query.py``), benchmark equivalence scoring
(``particles/benchmark/equivalence.py``), and subject-resolver alias
matching (``particles/extraction/subject_resolver.py``) all rely on the
same encoder. This module owns the lazy-loaded singleton and the test
seam (``set_embedding_model``); callers should import
:func:`get_embedding_model` from here rather than reaching into any
consumer's private member.

The model is the ``all-MiniLM-L6-v2`` SentenceTransformer.
First call loads from disk (or downloads on first run); subsequent calls
return the cached instance. If ``sentence-transformers`` is unavailable
the loader logs a warning and returns ``None``, which every caller treats
as "semantic search disabled — fall back to a conservative path".
"""

from __future__ import annotations

import contextlib
import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from opentelemetry import trace

from particles.config import get_config

log = logging.getLogger(__name__)

# Hand-rolled telemetry. The OTel API is used directly (no-op until a
# provider is installed), keeping this Client-layer module free of the Engine
# SDK. The model load (multi-second on first call) and every encode() — on any
# call path — are otherwise invisible to the trace.
_tracer = trace.get_tracer("particles.embeddings")

# The id stamped onto every vector this process writes. Cosine
# similarity across two embedding spaces is meaningless, so stored vectors carry
# the id of the model that produced them and the query path guards on a match —
# the same shape as the schema_version guard. A future embedding swap
# changes this constant, which makes every previously-stored vector read as
# stale (prompting a re-embed) instead of silently corrupting query results.
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"


def get_embedding_model_id() -> str:
    """Return the id of the embedding model this process writes."""
    return EMBEDDING_MODEL_ID


@dataclass(frozen=True)
class EmbeddingProfile:
    """The structured embedding profile recorded in store metadata.

    A *profile* is the triple that fully determines the embedding space a stored
    vector lives in: the encoder ``model``, its output ``dim``ensionality, and
    the ``normalization`` applied to each vector. Two stores can compare cosine
    similarities — and therefore reproduce the standard's similarity thresholds —
    only when they share a profile; a profile change requires re-embedding
    .

    The reference profile published by the standard is
    ``{model: all-MiniLM-L6-v2, dim: 384, normalization: l2}``. ``model`` is the
    compiled-in :data:`EMBEDDING_MODEL_ID`; ``dim`` and ``normalization`` come
    from ``config.embeddings`` so an operator who swaps the encoder declares the
    matching profile in one place.
    """

    model: str
    dim: int
    normalization: str

    def as_dict(self) -> dict[str, object]:
        """Return the profile as a plain ``{model, dim, normalization}`` dict.

        This is the shape persisted to store metadata and emitted in
        interchange — a structured object, never a free string.
        """
        return {"model": self.model, "dim": self.dim, "normalization": self.normalization}


def get_embedding_profile() -> EmbeddingProfile:
    """Return the structured embedding profile this process writes.

    The profile triple ``{model, dim, normalization}`` is the metadata an
    implementation MUST record so a reader can tell which embedding space stored
    vectors occupy. ``model`` is :data:`EMBEDDING_MODEL_ID`; ``dim`` and
    ``normalization`` are read from ``config.embeddings`` at call time (never
    captured at import, so ``reset_config()`` is honoured).
    """
    cfg = get_config().embeddings
    return EmbeddingProfile(model=EMBEDDING_MODEL_ID, dim=cfg.dim, normalization=cfg.normalization)


def cosine_similarity(
    a: np.ndarray[Any, np.dtype[np.floating[Any]]],
    b: np.ndarray[Any, np.dtype[np.floating[Any]]],
) -> float:
    """Cosine similarity of two embedding vectors, clamped to ``[0, 1]``.

    This is the **single normative similarity primitive** for the standard:
    cosine over the (L2-normalized) embedding vectors, with negative values
    clamped to ``0``. Every similarity threshold in the standard is expressed on
    this ``[0, 1]`` scale, so all engine call sites that compare against a
    threshold MUST route through here rather than computing a raw cosine.

    The clamp only ever touches the ``[-1, 0)`` half of the cosine range, which
    for the near-paraphrase text pairs the thresholds gate on does not occur in
    practice — so clamping leaves every in-range (non-negative) similarity, and
    therefore every existing threshold comparison, unchanged. It removes one
    real failure mode: an anti-correlated pair scoring below ``0`` and silently
    sorting *below* an unrelated ``0``-similarity pair.

    A ``1e-10`` epsilon guards the zero-vector degenerate case in the norm
    division. Vectors are assumed (but not required) to be unit-norm — the
    explicit norm division makes the result correct either way.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        The cosine similarity in ``[0.0, 1.0]``.
    """
    raw = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    return max(0.0, min(1.0, raw))


class EmbeddingModel:
    """Protocol-compatible wrapper around sentence-transformers SentenceTransformer."""

    def encode(
        self,
        texts: list[str],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> list[Any]:
        return []  # pragma: no cover


#: One encoder, one caller at a time. ``model.encode`` is offloaded to worker
#: threads (``asyncio.to_thread`` in the extract pipeline), and a
#: pooled or ``--concurrency`` run has several tasks in flight — without this
#: lock two threads drive the same torch model at once, and on the Metal (MPS)
#: backend that is a native crash (intermittent SIGSEGV in ``libtorch_cpu``,
#: reproduced 2026-08-16 by the memory benchmark's pooled extraction). The
#: encoder saturates its device on one call anyway, so serialising costs no
#: throughput; it just makes concurrency safe by construction.
_encode_lock = threading.Lock()


class _TracingEmbeddingModel(EmbeddingModel):
    """Wrap the real encoder so every ``encode()`` is an ``embed.encode`` span.

    Centralising the span here covers every call path — extraction, query,
    subject resolution, benchmark — with no change to the call sites (they only
    ever use ``.encode()``; any other attribute delegates to the wrapped model).
    No-op when observability is off. The same choke point holds
    :data:`_encode_lock`, so every path is also serialised across threads.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def encode(
        self,
        texts: list[str],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ) -> list[Any]:
        # Suppress the per-encode "Batches: 100%|…" tqdm bar unless the operator
        # opted back in (embeddings.progress_bars). setdefault so an explicit
        # caller override still wins.
        kwargs.setdefault("show_progress_bar", get_config().embeddings.progress_bars)
        with _tracer.start_as_current_span("embed.encode") as span, _encode_lock:
            span.set_attribute("embed.count", len(texts))
            result: Any = self._inner.encode(
                texts,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize_embeddings,
                **kwargs,
            )
            return result  # type: ignore[no-any-return]

    def __getattr__(self, name: str) -> Any:
        # Delegate anything other than encode to the wrapped model. ``_inner`` is
        # set in __init__ and lives in __dict__, so this never recurses for it.
        return getattr(self._inner, name)


_embedding_model: EmbeddingModel | None = None


def _toggle_progress_bars(module_name: str, enable: str, disable: str, show: bool) -> None:
    """Call ``module_name.{enable|disable}()`` best-effort.

    Routed through ``importlib`` + an ``Any`` module handle on purpose: the tqdm
    toggles live in optional sub-dependencies of sentence-transformers whose
    typing and exact symbol names drift across versions (``huggingface_hub`` ships
    types, ``transformers`` does not), so a missing module or attribute degrades
    to a no-op rather than a strict-typing or import error.
    """
    import importlib

    try:
        mod: Any = importlib.import_module(module_name)
    except Exception:  # pragma: no cover - module absent
        return
    fn = getattr(mod, enable if show else disable, None)
    if callable(fn):
        with contextlib.suppress(Exception):  # pragma: no cover - defensive across versions
            fn()


def _apply_progress_bar_config() -> None:
    """Toggle the HuggingFace/transformers tqdm progress bars per config.

    Governs the one-time ``Loading weights: 100%|…`` model-load bar; the
    per-encode ``Batches`` bar is handled separately via the ``show_progress_bar``
    kwarg in :meth:`_TracingEmbeddingModel.encode`.
    """
    show = get_config().embeddings.progress_bars
    _toggle_progress_bars(
        "huggingface_hub.utils", "enable_progress_bars", "disable_progress_bars", show
    )
    _toggle_progress_bars(
        "transformers.utils.logging", "enable_progress_bar", "disable_progress_bar", show
    )


def get_embedding_model() -> EmbeddingModel | None:
    """Return the shared SentenceTransformer instance (lazy-loaded).

    Returns ``None`` if ``sentence-transformers`` cannot be imported or
    the model fails to load; callers must handle this case. The one-time load
    runs under an ``embed.load`` span — it is multi-second on first
    call and was previously invisible to the trace.
    """
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import (
                SentenceTransformer,  # type: ignore[import-untyped,unused-ignore]
            )

            _apply_progress_bar_config()
            with _tracer.start_as_current_span("embed.load") as span:
                span.set_attribute("embed.model_id", EMBEDDING_MODEL_ID)
                inner = SentenceTransformer(EMBEDDING_MODEL_ID)  # type: ignore[no-untyped-call,unused-ignore]
            _embedding_model = _TracingEmbeddingModel(inner)
            log.info("Loaded embedding model %s", EMBEDDING_MODEL_ID)
        except Exception as exc:
            log.warning("Could not load embedding model: %s — semantic search disabled", exc)
            return None
    return _embedding_model


def set_embedding_model(model: EmbeddingModel | None) -> None:
    """Override the embedding model (for testing)."""
    global _embedding_model
    _embedding_model = model
