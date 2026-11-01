# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from opentelemetry import trace

from particles import egress
from particles.config import get_config
from particles.http import hub_client

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

# The hub commit of that model this release loads. Pinned, so a change on the
# hub cannot change which weights an install, or the published image, loads
# (security review P38). Move it deliberately: new weights can move vectors
# while the id above stays the same.
EMBEDDING_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


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
    only when they share a profile; a profile change requires re-embedding.

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


def cosine_similarities(
    query: np.ndarray[Any, np.dtype[np.floating[Any]]],
    vectors: Sequence[np.ndarray[Any, np.dtype[np.floating[Any]]]],
) -> list[float]:
    """:func:`cosine_similarity` of ``query`` against each vector, in one pass.

    The same formula, the same ``1e-10`` epsilon and the same ``[0, 1]`` clamp,
    computed over a stacked matrix instead of once per pair: on the
    query-latency baseline the per-pair loop cost 81 ms at 56k candidates and
    the matrix form 6 ms. A value can differ from the per-pair one in its last
    bit, because the dot products are summed in another order. Each row is
    reduced by the same loop wherever it sits in the matrix (``einsum``, not a
    BLAS kernel that blocks rows by position), so two identical vectors always
    get identical scores and keep their tie.
    """
    if not vectors:
        return []
    matrix = np.stack(vectors)
    dots = np.einsum("ij,j->i", matrix, query)
    norms = np.sqrt(np.einsum("ij,ij->i", matrix, matrix))
    raw = dots / (norms * np.linalg.norm(query) + 1e-10)
    clamped: list[float] = np.clip(raw, 0.0, 1.0).tolist()
    return clamped


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
        # The exception's type alone, never its message (security review
        # 2026-10-26-1, R7).
        with (
            _tracer.start_as_current_span(
                "embed.encode", record_exception=False, set_status_on_exception=False
            ) as span,
            _encode_lock,
        ):
            span.set_attribute("embed.count", len(texts))
            try:
                result: Any = self._inner.encode(
                    texts,
                    convert_to_numpy=convert_to_numpy,
                    normalize_embeddings=normalize_embeddings,
                    **kwargs,
                )
            except Exception as exc:
                span.set_attribute("error.type", type(exc).__name__)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                raise
            return result  # type: ignore[no-any-return]

    def __getattr__(self, name: str) -> Any:
        # Delegate anything other than encode to the wrapped model. ``_inner`` is
        # set in __init__ and lives in __dict__, so this never recurses for it.
        return getattr(self._inner, name)


_embedding_model: EmbeddingModel | None = None
# Set when an offline load failed: the model is not cached and the
# posture refuses the Hub, so retrying on every call would only repeat it.
_offline_load_failed = False


def _hub_refused() -> bool:
    """Whether ``network.egress`` refuses the Hugging Face Hub."""
    policy = egress.current_policy()
    return policy.refusing and not policy.allows(*egress.HUGGINGFACE_HOST)


def _use_hub_client_writing_no_key_log() -> None:
    """Have ``huggingface_hub`` download through :func:`particles.http.hub_client`.

    Its default client verifies with :func:`ssl.create_default_context` under
    ``SSL_CERT_FILE`` or ``SSL_CERT_DIR``, which honours an inherited
    ``SSLKEYLOGFILE`` on the session with the Hub and on the hop to an
    ``https://`` proxy (security review 2026-10-30-1). The factory is the
    process's, so an embedding application's own Hub calls get it too: the
    same client, with the key log off.
    """
    try:
        from huggingface_hub import set_client_factory
    except ImportError:  # an older hub: it downloads through requests, not httpx2
        return
    set_client_factory(hub_client)


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


def _load_cached(loader: Any) -> Any:
    """Load the pinned revision of the encoder from the local cache only.

    Never the cache's ``main`` ref: that names whatever another tool last
    fetched from the hub, so it would decide which weights load (security
    review Q58). A cache holding only another revision loads nothing, and the
    caller disables semantic features with the reason.
    """
    try:
        return loader(EMBEDDING_MODEL_ID, revision=EMBEDDING_MODEL_REVISION, local_files_only=True)
    except Exception as exc:
        raise OSError(
            f"revision {EMBEDDING_MODEL_REVISION} of {EMBEDDING_MODEL_ID} is not in the local "
            "cache; fetch it once with huggingface.co allowed (`hf download "
            f"sentence-transformers/{EMBEDDING_MODEL_ID} --revision {EMBEDDING_MODEL_REVISION}`)"
        ) from exc


def get_embedding_model() -> EmbeddingModel | None:
    """Return the shared SentenceTransformer instance (lazy-loaded).

    Returns ``None`` if ``sentence-transformers`` cannot be imported or
    the model fails to load; callers must handle this case. The one-time load
    runs under an ``embed.load`` span — it is multi-second on first
    call and was previously invisible to the trace.
    """
    global _embedding_model, _offline_load_failed
    if _embedding_model is None:
        # Under a posture that refuses the Hub, load from the local cache only:
        # no download, and no per-file probe of the Hub on every process start.
        offline = _hub_refused()
        if offline and _offline_load_failed:
            return None
        try:
            from sentence_transformers import (
                SentenceTransformer,  # type: ignore[import-untyped,unused-ignore]
            )

            _apply_progress_bar_config()
            with _tracer.start_as_current_span(
                "embed.load", record_exception=False, set_status_on_exception=False
            ) as span:
                span.set_attribute("embed.model_id", EMBEDDING_MODEL_ID)
                try:
                    if offline:
                        inner = _load_cached(SentenceTransformer)
                    else:
                        _use_hub_client_writing_no_key_log()
                        inner = SentenceTransformer(  # type: ignore[no-untyped-call,unused-ignore]
                            EMBEDDING_MODEL_ID, revision=EMBEDDING_MODEL_REVISION
                        )
                except Exception as exc:
                    span.set_attribute("error.type", type(exc).__name__)
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    raise
            _embedding_model = _TracingEmbeddingModel(inner)
            log.info("Loaded embedding model %s", EMBEDDING_MODEL_ID)
        except Exception as exc:
            if offline:
                _offline_load_failed = True
                log.warning(
                    "Could not load embedding model %s from the local cache, and "
                    "network.egress refuses huggingface.co: semantic features are "
                    "disabled for this process (%s)",
                    EMBEDDING_MODEL_ID,
                    exc,
                )
                return None
            log.warning("Could not load embedding model: %s — semantic search disabled", exc)
            return None
    return _embedding_model


def set_embedding_model(model: EmbeddingModel | None) -> None:
    """Override the embedding model (for testing)."""
    global _embedding_model, _offline_load_failed
    _embedding_model = model
    _offline_load_failed = False
