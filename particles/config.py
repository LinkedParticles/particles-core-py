# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Consolidated configuration.

Precedence: env var > config.yaml > compiled default.
Secrets (ANTHROPIC_API_KEY, NUMISTA_API_KEY) are never read from config.yaml.

Call get_config() anywhere to access the current configuration.
Call reset_config() in tests to reload from scratch.
"""

from __future__ import annotations

import ipaddress
import logging
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from particles import __version__

log = logging.getLogger(__name__)


def _default_user_agent() -> str:
    # Points at the public Engine repo (the thing users install and run), not
    # the private development upstream — a courtesy contact URL a server
    # operator can resolve. Kept a real runtime default (not an export-time
    # rewrite) so the private and public trees send the same, resolvable UA.
    return f"particles-sdk/{__version__} (+https://github.com/LinkedParticles/particles-engine-py)"


class WriteLockConfig(BaseModel):
    """Cross-process single-writer discipline for the canonical store.

    SQLite has exactly one writer, but the always-on engine host invites a
    second writer process (a direct-I/O CLI verb on the host vs. the engine).
    When ``enabled``, every write transaction acquires a per-store advisory lock
    (an in-process ``asyncio.Lock`` + a cross-process ``filelock``) — never held
    across the LLM/extract phase — so writers serialize fairly across processes
    instead of racing SQLite's ``busy_timeout``. Active only for **file-based
    SQLite**; in-memory SQLite and PostgreSQL are a no-op (no second process /
    genuine concurrent writers). The ``particles.sqlite.busy`` counter
    (Phase 2) measures the residual contention.
    """

    # Master switch. False ⇒ today's busy_timeout-only behaviour.
    enabled: bool = True
    # How long a writer waits for the lock before raising WriteLockTimeout —
    # sized to the 30 s busy_timeout it backs up (particles/db.py).
    timeout_seconds: float = 30.0
    # Lockfile path; None ⇒ derived as ``<db_file>.writelock`` beside the store DB.
    path: str | None = None


class StorageConfig(BaseModel):
    blob_dir: str = "./corpus_blobs"
    database_url: str = "sqlite+aiosqlite:///./particles.db"
    # Additional named stores for multi-store / federation. Maps a
    # StoreHandle -> database URL; the implicit "default" store always resolves
    # to `database_url` above. Static config; dynamic per-tenant provisioning is
    # handled by a separate layer above this one.
    stores: dict[str, str] = Field(default_factory=dict)
    # Cross-process single-writer discipline.
    write_lock: WriteLockConfig = Field(default_factory=WriteLockConfig)
    # Hard upper bound on the ``limit`` query param of the read/list endpoints
    # (security review F5). A larger ``limit`` — or, via SQLite's negative-limit
    # = all-rows quirk, a negative one — would force a full row scan + full
    # Pydantic-list materialization in memory. The API boundary clamps every
    # caller-supplied ``limit`` to this value (negatives/zero are rejected up
    # front by the ``Query(ge=1)`` bound). Enforced in ``particles/api/app.py``.
    max_page_size: int = Field(default=1000, ge=1)
    # How many distinct snapshot content-hashes the blob-reachability probe
    # stats against the resolved blob_dir. The probe is the cheap
    # detection half of the scattering story — it runs in
    # `config validate` and `hook doctor`, and a total miss is the signature of
    # blobs written under a different working directory. Bounded because a
    # large store would otherwise stat every snapshot for a diagnostic.
    blob_health_sample: int = Field(default=50, ge=1)


class HttpConfig(BaseModel):
    user_agent: str = Field(default_factory=_default_user_agent)
    timeout_seconds: float = 30.0
    # Hard cap on the size of any single fetched HTTP body, in bytes. httpx
    # transparently decompresses gzip/deflate/br/zstd, so a small compressed
    # payload can expand without bound — a "gzip bomb". Fetches stream the
    # decompressed body with a running total and abort once it exceeds this
    # cap; a declared Content-Length over the cap is rejected before any body
    # is read. Default 100 MiB is well above any legitimate page / PDF / API
    # response this SDK fetches. Also passed to the Reddit curl subprocess as
    # --max-filesize.
    max_bytes: int = 100 * 1024 * 1024


class ApiConfig(BaseModel):
    """FastAPI app tunables (defense-in-depth limits for the HTTP surface).

    ``max_request_body_bytes`` caps the *inbound* request body the app will
    accept before returning ``413 Request Entity Too Large`` — a backstop
    against memory-exhaustion from an unbounded upload (uvicorn/Starlette
    impose no default limit). Distinct from ``http.max_bytes``, which caps the
    *outbound* bodies this SDK fetches. Enforced by the ASGI middleware in
    ``particles/api/_middleware.py``. Set to ``0`` to disable the in-app check
    (e.g. when a reverse proxy already enforces a smaller cap). The default is
    generous enough for ordinary file deposits; operators exposing the API
    publicly should tune it to their largest legitimate upload. This is a
    process-local guard, distinct from the fail-closed deployment gate
    (``bind_host``, below).

    ``bind_host`` is the interface the operator binds the API to; it must
    match uvicorn's ``--host``, since the ASGI app cannot read uvicorn's
    bind address itself. It is the boundary the fail-closed startup check
    enforces: when ``PARTICLES_API_KEY`` is unset — so bearer
    auth is disabled (the ``"dev-key"`` local-dev affordance) — and
    ``bind_host`` is **not** a loopback address (``127.0.0.0/8`` / ``::1`` /
    ``localhost``), the app refuses to start rather than silently serve
    unauthenticated traffic beyond loopback. Two ways out: set
    ``PARTICLES_API_KEY`` to a real secret, or keep the bind on loopback.
    The default ``"127.0.0.1"`` keeps the local-dev loop friction-free.

    ``trusted_proxies`` makes the per-request loopback gate (
    mechanism (ii)) proxy-aware. Behind a reverse proxy the network
    peer is the proxy's address, so without this a *remote* client would read as
    loopback and the dev-key skip would wrongly apply to it. List the proxy IPs
    / CIDRs you trust; when the immediate peer is one of them, the gate honours
    ``X-Forwarded-For`` (the nearest untrusted hop) to identify the real client.
    **Default empty** — ``X-Forwarded-For`` is ignored entirely and the gate
    reads the raw peer exactly as before, so a spoofable header is never trusted
    unless you opt in.

    ``rate_limit_per_minute`` is an in-app token-bucket cap on the
    LLM/embedding-driving endpoints (``/query``, ``/extract``, ``/reindex``, the
    semantic ``/lint`` path), keyed on the real client host (security review F6).
    Each of those endpoints drives a paid Anthropic completion and/or an
    embedding per request, so an unauthenticated or compromised caller could
    otherwise burn tokens unbounded. ``0`` (or any value ``≤ 0``) disables the
    in-app limiter — appropriate for the default loopback-bound single-operator
    engine, and for deployments where a reverse proxy already rate-limits. The
    limiter is a *second* line of defense, not a substitute for the
    reverse-proxy / auth posture. Because the local CLI / MCP tools
    run in-process (they never cross the HTTP boundary unless a remote engine is
    configured), turning this on does not throttle local single-process use.

    ``require_auth_for_reads`` extends the bearer gate to the **read** surface
    (security review F2). The bearer gates the *write* verbs only;
    the read routes (``/particles``, ``/corpus``, ``/subjects``, ``/quality``,
    ``/lint/report``, ``/taxonomies``, …) carry no auth by default, so once a
    real ``PARTICLES_API_KEY`` is set the reads remain open. That is safe on a
    loopback bind, but exposes the full belief store on a non-loopback bind
    (``engine serve 0.0.0.0:8000``). Set this ``true`` to require the
    same bearer on every read route. **Default ``False``** preserves the
    historical read posture. Note: the three highest-value reads — ``/query``
    (bills the operator's ``ANTHROPIC_API_KEY``), ``/events`` (the operator
    audit log), and ``/digest`` (the provenance-ranked belief digest) — are
    gated by the bearer **regardless** of this flag; under the ``dev-key``
    loopback skip they stay open for local development.
    """

    max_request_body_bytes: int = 25 * 1024 * 1024  # 25 MiB
    bind_host: str = "127.0.0.1"
    trusted_proxies: list[str] = Field(default_factory=list)
    rate_limit_per_minute: int = 60
    require_auth_for_reads: bool = False

    @field_validator("trusted_proxies")
    @classmethod
    def _validate_trusted_proxies(cls, value: list[str]) -> list[str]:
        """Reject overly-broad or loopback proxy ranges (security review F19).

        Each entry must parse as an IP or CIDR. A reverse proxy is never
        legitimately the everything-range (``0.0.0.0/0`` / ``::/0``) or a
        loopback range (``127.0.0.0/8`` / ``::1/128``): trusting either
        re-enables the ``X-Forwarded-For`` spoof the per-request loopback gate
        (``particles/api/auth.py``) defends against — an attacker whose hop is
        "trusted" can forge the real-client address. An empty list (the
        default) is valid: ``X-Forwarded-For`` is ignored entirely. ``strict``
        is ``False`` so a host-bit-set CIDR like ``"10.0.0.5/8"`` is accepted,
        matching the consuming ``_ip_in_networks`` semantics.
        """
        for entry in value:
            try:
                network = ipaddress.ip_network(entry.strip(), strict=False)
            except ValueError as exc:
                raise ValueError(
                    f"api.trusted_proxies entry {entry!r} is not a valid IP or CIDR: {exc}"
                ) from exc
            if int(network.prefixlen) == 0:
                raise ValueError(
                    f"api.trusted_proxies entry {entry!r} is the everything-range "
                    f"(prefix /0); a reverse proxy is never the entire internet — "
                    f"trusting it re-enables X-Forwarded-For spoofing. List the "
                    f"specific proxy IPs / CIDRs instead."
                )
            if network.is_loopback:
                raise ValueError(
                    f"api.trusted_proxies entry {entry!r} is a loopback range; "
                    f"a reverse proxy is never loopback, and trusting it re-enables "
                    f"X-Forwarded-For spoofing. List the specific proxy IPs / CIDRs."
                )
        return value


class BuildConfig(BaseModel):
    """Provenance of the artifact this process is running from.

    ``date`` is the build timestamp the container image stamps into itself
    (``deploy/Dockerfile`` takes it as ``--build-arg BUILD_DATE`` and exports
    it as ``PARTICLES_BUILD_DATE``), disclosed by ``GET /health`` so a client
    can show how old the engine it is talking to actually is. A long-running
    container is the case this exists for: the version alone says which
    release the code is, and pairing it with a date says whether that release
    is the one you think you deployed.

    ``None`` (the default) whenever nothing stamped it — running from source,
    or an image built without the build arg — and ``/health`` then simply
    omits the field rather than guessing. Not a tunable: an operator has no
    reason to set this in ``config.yaml``, and a value written there is a
    claim about the artifact that the artifact did not make.
    """

    date: str | None = None

    @field_validator("date", mode="before")
    @classmethod
    def _blank_is_unstamped(cls, value: object) -> object:
        """Treat an empty ``date`` as absent.

        The image always exports ``PARTICLES_BUILD_DATE``, and it exports the
        empty string when built without ``--build-arg BUILD_DATE`` — which the
        env-override pass would otherwise read as a *present* value, so
        ``/health`` would carry ``built_at: ""``. An empty stamp is no stamp.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value


class EngineConfig(BaseModel):
    """Thin-client → remote-engine connection settings.

    When ``base_url`` is ``None`` (the default), the CLI runs every verb
    in-process against the local store (``LocalBackend`` — today's behaviour,
    byte-for-byte unchanged). When it is set to an engine's URL (e.g.
    ``http://mac-mini:8000``), the CLI becomes a thin HTTP client
    (``HttpBackend``) and the engine runs extraction / embedding / query / lint
    server-side. ``base_url`` is the *client* side of the picture: the
    server's own bind + fail-closed gate live under ``api`` (above).

    Both fields are **non-secret** and may live in ``config.yaml`` or come from
    ``PARTICLES_ENGINE_BASE_URL`` / ``PARTICLES_ENGINE_TIMEOUT_SECONDS``. The
    bearer token the client presents is a *secret* and is read via
    ``particles.secrets.get_engine_token_optional`` (``PARTICLES_ENGINE_TOKEN``)
    — never a field here, never in ``config.yaml``.
    """

    base_url: str | None = None
    timeout_seconds: float = 60.0


class ObservabilityConfig(BaseModel):
    """OpenTelemetry observability settings.

    Off by default. When ``enabled`` is true **and** the optional ``otel`` extra
    is installed (``pip install particles[otel]``), ``setup_observability()``
    (``particles/observability/``) installs a tracer/meter provider plus the
    FastAPI / httpx / SQLAlchemy / logging auto-instrumentation, so a request's
    time is visible as a ``traceparent``-propagated span tree across the
    client → engine → store boundary the split created. With the extra
    absent **or** ``enabled`` false, no provider is installed and every span /
    metric call is a cheap no-op (the base ``opentelemetry-api`` no-op).

    All fields here are **non-secret** and may live in ``config.yaml`` or come
    from the ``PARTICLES_OBSERVABILITY_*`` env overrides. The exporter auth
    credential (a SaaS / authenticated-collector token) is a *secret*, read via
    ``particles.secrets.get_otel_exporter_headers_optional`` — never a field here,
    never in ``config.yaml``, mirroring the endpoint/token
    split (the non-secret ``endpoint`` URL lives here; the token does not).
    """

    # Master switch. False ⇒ setup installs nothing; the API no-op covers all
    # instrumentation call sites at zero cost.
    enabled: bool = False
    # OTel resource ``service.name`` — how this process labels its spans/metrics.
    service_name: str = "particles"
    # Exporter selection. ``console`` (default) prints spans/metrics to the log —
    # zero-infra diagnose-now mode; ``otlp`` ships to ``endpoint`` (a local
    # collector or a SaaS backend — same code, different endpoint); ``none``
    # installs a provider with no exporter (tests / pure no-op).
    exporter: Literal["none", "console", "otlp"] = "console"
    # OTLP target when ``exporter == "otlp"`` (e.g. http://localhost:4318 for a
    # local collector, or a SaaS ingest URL). Non-secret; the credential is the
    # PARTICLES_OTEL_EXPORTER_HEADERS secret. Unset ⇒ the OTLP SDK default endpoint.
    endpoint: str | None = None
    # Per-signal enables — turn off a signal without disabling the whole layer.
    traces: bool = True
    metrics: bool = True
    logs: bool = True
    # Head-sampling ratio for traces (1.0 = always-on, correct for single-operator
    # volume; ratio sampling matters only at shared-store scale).
    sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)


class EmbeddingsConfig(BaseModel):
    """Sentence-transformer encoder settings.

    ``progress_bars`` toggles the tqdm progress bars the embedding stack prints
    to stderr — the ``Loading weights: 100%|…`` bar on the one-time model load
    and the ``Batches: 100%|…`` bar on each ``encode()``. They are noise for a
    CLI verb like ``query`` (which always loads the model), so they default
    **off**; flip to ``true`` (or set ``PARTICLES_EMBEDDINGS_PROGRESS_BARS=1``)
    when you want the load/encode progress feedback back.

    ``dim`` and ``normalization`` are the other two components of the structured
    ``embedding_profile`` recorded in store metadata (the third
    component, the model name, is :data:`particles.embeddings.EMBEDDING_MODEL_ID`).
    They default to the reference profile (``384`` / ``l2``) and should be
    changed only in lockstep with the encoder, since a profile change requires
    re-embedding the store. The similarity contract is **cosine over
    L2-normalized vectors clamped to ``[0, 1]``**; ``normalization`` records how
    the stored vectors are produced, not whether the clamp applies (the clamp is
    unconditional — see :func:`particles.embeddings.cosine_similarity`).
    """

    progress_bars: bool = False
    dim: int = 384
    normalization: str = "l2"


class ProviderSelection(BaseModel):
    """A (provider, model) pairing for one completion purpose.

    ``provider`` is ``"anthropic"`` (the native-SDK adapter, the hosted
    default) or the name of an entry in ``llm.providers`` — an operator-named
    OpenAI-compatible endpoint: ``"local"`` (the compiled-in Ollama
    entry), or any name the operator defines (``"openai"``,
    ``"deepseek"``, …). Membership is cross-validated on :class:`LLMConfig`,
    so a dangling name fails config load. Only the per-purpose ``model``
    string lives here. ``max_tokens`` is deliberately *not* here — it is
    per-call, supplied by the call site (8192 for an extraction pass, 16 for
    the benchmark judge).
    """

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"


class OpenAICompatProviderConfig(BaseModel):
    """One named OpenAI-compatible provider entry in ``llm.providers``.

    Endpoint + resilience + dialect policy for a single named provider —
    hosted (api.openai.com, api.deepseek.com, a gateway) or local (Ollama,
    llama.cpp, vLLM, LM Studio). The model string stays per-purpose on
    :class:`ProviderSelection`. The API key is a *secret* read via
    :func:`particles.secrets.get_llm_api_key_optional` for the entry's name
    (``PARTICLES_LLM_API_KEY_<NAME>``) — never a field here.
    """

    # OpenAI-compatible base URL; the adapter appends ``/chat/completions``.
    # Default targets a local Ollama server (the case).
    base_url: str = "http://localhost:11434/v1"
    # Per-request wall-clock timeout (seconds). Local models on modest hardware
    # can be slow, so the default is generous.
    timeout_seconds: float = 120.0
    # Bounded retry on transient failures (connection error, timeout, 429/5xx);
    # raw httpx has no built-in retry, unlike the Anthropic SDK.
    max_retries: int = 2
    # Base for exponential backoff between retries (seconds): backoff * 2**attempt.
    retry_backoff_seconds: float = 1.0
    # JSON-schema structured-output enforcement.
    # "auto": when a call site supplies a ``response_schema``, send OpenAI-style
    # ``response_format: {"type": "json_schema", …, strict: true}`` (array-root
    # schemas are transparently wrapped for object-root dialects), retrying once
    # without it — with a logged downgrade — if the endpoint rejects the
    # parameter. "strict": additionally transform the schema to the
    # OpenAI-strict dialect (every property key in ``required``, optionality as
    # a union with ``null``) — required for api.openai.com and other
    # strict-mode endpoints, whose validators reject ``required`` ⊂
    # ``properties``. "off" disables enforcement entirely (the tolerant
    # call-site parsers remain the only line, the pre-0194 behaviour).
    structured_output: Literal["strict", "auto", "off"] = "auto"
    # Dialect knob: which body member carries the completion-length cap.
    # OpenAI's reasoning models reject "max_tokens" in favour of
    # "max_completion_tokens"; local runtimes accept the classic name.
    max_tokens_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    # Dialect knob: when False the adapter omits ``temperature`` entirely, for
    # models that reject non-default values (OpenAI reasoning models). Call
    # sites keep passing what they pass; the adapter drops it at the wire.
    send_temperature: bool = True
    # The registered adapter kind that instantiates this entry.
    # Validated against the adapter registry at first resolution (a config →
    # llm import would mint a subpackage cycle; see LLMConfig's validator).
    # Today the only kind a named entry meaningfully selects is
    # "openai_compat"; a future non-OpenAI-dialect adapter (Bedrock, native
    # Gemini) registers a new kind.
    adapter: str = "openai_compat"


# Back-compat alias: the entry schema was named after its one
# ``local`` instance; the schema is provider-agnostic now.
LocalProviderConfig = OpenAICompatProviderConfig


class BatchCompletionConfig(BaseModel):
    """Asynchronous batch-completion policy for ``complete_many``.

    Provider-level, like :class:`LocalProviderConfig`: one policy covers every
    purpose, because the trade being made is the same everywhere — half-price
    tokens in exchange for a completion time measured in minutes-to-hours.
    Only a call site that has declared itself **latency-tolerant** can take
    that trade; the knobs here bound it once it has.

    ``enabled`` is the operator kill switch: off, every ``complete_many`` runs
    the sequential ``complete()`` fallback and the nightly cycle behaves
    exactly as it did before.
    """

    # Master switch. Off ⇒ every complete_many() degrades to sequential
    # complete() calls, whatever the caller's latency tolerance.
    enabled: bool = True
    # Below this many requests, batch anyway? No — a handful of probes is not
    # worth a submit + poll round trip whose floor is one poll interval, and
    # the serial cost of many tiny batches is what makes an overnight run
    # overrun. Under the floor, complete_many() runs them sequentially.
    min_requests: int = Field(default=4, ge=1)
    # Hard ceiling on one submitted batch. The API's own limits are far higher
    # (100k requests / 256 MB); this bounds the blast radius of a single
    # submission and is what a >max run is chunked on.
    max_requests_per_batch: int = Field(default=1000, ge=1)
    # Seconds between processing_status polls. Batches "usually" finish within
    # an hour, so a sub-minute poll is pure API chatter.
    poll_interval_seconds: float = Field(default=30.0, gt=0.0)
    # Wall-clock ceiling per submitted batch. On expiry the batch is cancelled
    # and its unfinished requests come back as None (the call site's existing
    # probe-unavailable degradation) rather than stalling the run; the ones it
    # finished are kept (see cancel_grace_seconds). The API's
    # own expiry is 24h; this default keeps a nightly cycle inside its night.
    max_wait_seconds: float = Field(default=3600.0, gt=0.0)
    # After cancelling an over-running batch, keep polling up to this long for
    # it to reach ``ended`` so the requests that did finish can be read back.
    # Measured: cancellation ended 15 of 16 observed batches
    # within 180 s, median ~40 s. 0 restores the pre-0290 discard.
    cancel_grace_seconds: float = Field(default=180.0, ge=0.0)
    # Inside a batch-wait budget scope (the consolidation cycle), a cancelled
    # batch's ``canceled`` remainder is re-run sequentially at full price when
    # ``len(remainder) * max_tokens`` is at most this: the
    # matcher's and census's short judgements qualify, extraction never does.
    # 0 disables the re-run.
    cancel_rerun_max_output_tokens: int = Field(default=15000, ge=0)


class PromptCacheConfig(BaseModel):
    """Prompt-cache policy for the completion port.

    Provider-level like :class:`BatchCompletionConfig`: one switch covers every
    purpose. Only the Anthropic adapter acts on it — it renders a request's
    ``cache_prefix`` as a cached system block (``cache_control: ephemeral``),
    billing a repeated prefix as a ~10% cache read. Off ⇒ the adapter folds the
    prefix into the plain system string and sends no ``cache_control``, so
    billing and behaviour are exactly pre-0252. The operator A/B / cost-debug
    switch, mirroring ``llm.batch.enabled``.
    """

    enabled: bool = True


class TokenPrice(BaseModel):
    """One model's list price in US$ per million tokens (``llm.price_per_mtok``)."""

    input: float = Field(ge=0.0)
    output: float = Field(ge=0.0)


# Anthropic first-party list prices (US$ per MTok) that ship as the
# ``llm.price_per_mtok`` defaults, checked against
# https://platform.claude.com/docs/en/about-claude/pricing on 2026-09-17
# (``claude-sonnet-5-5`` added 2026-10-01 at the same price as Sonnet 5). A
# cost estimate is the first thing a new user reads before spending, so the
# default models are priced out of the box; every rendered estimate prints the
# per-MTok figure it used, which keeps a stale entry visible rather than
# quiet. An operator entry for the same key replaces the shipped one.
_LIST_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


_T = TypeVar("_T")


def lookup_by_model(
    mapping: Mapping[str, _T], provider: str, model: str
) -> tuple[_T | None, str | None]:
    """``(entry, key)`` for a resolved selection: ``provider:model`` first, then the bare id.

    The one key-resolution convention every per-model mapping shares
    (``llm.price_per_mtok`` and the estimate assumptions keyed like it), so a
    key that prices a model is the key that selects its token assumptions.
    Membership, not truthiness: a legitimate ``0`` entry is an entry.
    """
    for key in (f"{provider}:{model}", model):
        if key in mapping:
            return mapping[key], key
    return None, None


def _default_prices() -> dict[str, TokenPrice]:
    return {
        model: TokenPrice(input=inp, output=out)
        for model, (inp, out) in _LIST_PRICES_PER_MTOK.items()
    }


class LLMConfig(BaseModel):
    """Per-purpose completion-provider selection.

    ``default`` is the fallback pairing; each purpose may override it. The
    operator can thus route high-volume ``extraction`` to a cheap model while
    keeping low-volume ``synthesis`` on a hosted one. All compiled defaults are
    ``claude-sonnet-4-6`` so the out-of-the-box behaviour is unchanged from
    before the port existed.

    The legacy ``extraction.model`` / ``wiki.model`` keys are migrated into
    this section by ``_migrate_legacy_keys`` (``extraction.model`` →
    ``llm.default.model``, ``wiki.model`` → ``llm.synthesis.model``).
    """

    default: ProviderSelection = Field(default_factory=ProviderSelection)
    extraction: ProviderSelection | None = None
    semantic_lint: ProviderSelection | None = None
    query_response: ProviderSelection | None = None
    synthesis: ProviderSelection | None = None
    benchmark: ProviderSelection | None = None
    # the memory-benchmark *answering* model (QA conditions ii–iv).
    # Separate from ``benchmark`` (the judge) so answerer and judge can be
    # pinned independently; the runner refuses a QA condition set whose
    # resolved answer-model ids differ.
    benchmark_answer: ProviderSelection | None = None
    # the abstraction-promotion pass (synthesis + entailment/dedup
    # judges). Unset ⇒ falls back to ``default``, i.e. the same routing the
    # dream cycle's other semantic passes use.
    abstraction: ProviderSelection | None = None
    # the second reading of a contradiction the ``semantic_lint``
    # probe flagged, given each claim's source passage, note and date. Unset ⇒
    # ``default``: a store that routes ``semantic_lint`` to a small model for
    # cost reads each flag again on the default model before the audit counts
    # it.
    verification: ProviderSelection | None = None
    # the judge that picks among a name's Wikidata candidates when
    # ``subjects.wikidata_candidate_selection`` is ``llm_judge``. Unset ⇒
    # ``default``.
    subject_resolution: ProviderSelection | None = None
    # the use judge, which credits a shown belief only when it rules
    # the session's actions applied it. Its own purpose, defaulting to Sonnet,
    # because the activation gate measured a Haiku-class judge short of the
    # bar (a 5-run mean of 78 and 72 of 100) and Sonnet well over it (94 on
    # both samples); ``semantic_lint`` stays free to run on a small model.
    use_judge: ProviderSelection | None = Field(
        default_factory=lambda: ProviderSelection(provider="anthropic", model="claude-sonnet-5-5")
    )
    # Named OpenAI-compatible providers. Keys are operator-chosen
    # provider names — the calibration/disclosure key is "<name>:<model>",
    # so treat a rename as a recalibration event. "anthropic" is
    # reserved for the native adapter and never appears here. The compiled-in
    # "local" entry (an Ollama endpoint) is inserted by the
    # validator when absent, so `provider: local` always resolves; precedence
    # is explicit providers.local > deprecated llm.local > compiled default.
    providers: dict[str, OpenAICompatProviderConfig] = Field(default_factory=dict)
    # DEPRECATED: the pre-registry home of the single local
    # endpoint. Honoured as ``providers["local"]`` when that key is not set
    # explicitly; remove after a deprecation cycle.
    local: OpenAICompatProviderConfig | None = None
    # Asynchronous batch completion for latency-tolerant call sites.
    # Provider-level like ``local``: one policy, applied wherever a caller has
    # declared it can wait (today: the nightly consolidation cycle).
    batch: BatchCompletionConfig = Field(default_factory=BatchCompletionConfig)
    # Prompt caching for a repeated system prefix. Provider-level
    # like ``batch``; only the Anthropic adapter acts on a request's
    # ``cache_prefix``. Off restores exact pre-0252 billing in one knob.
    prompt_cache: PromptCacheConfig = Field(default_factory=PromptCacheConfig)
    # Cool-off (seconds) after an account-level LLM failure — bad/missing key,
    # no permission, or out-of-credits — before the semantic seam probes the API
    # again (circuit breaker). 0 disables the breaker.
    unavailable_backoff_seconds: int = 60
    # List prices every cost estimate turns tokens into dollars with (the
    # audit and the benchmark harnesses alike), keyed by resolved model id
    # (``claude-sonnet-5``) or, to pin one provider's route,
    # ``"<provider>:<model>"``. Ships with the Anthropic list prices in
    # ``_LIST_PRICES_PER_MTOK``; entries set here merge over those, so adding
    # a price for an OpenAI-compatible model keeps the shipped ones. A model
    # with no entry is reported as unpriced, never as free.
    price_per_mtok: dict[str, TokenPrice] = Field(default_factory=_default_prices)
    # Fraction of list price removed when a call rides a batch API
    # (``llm.batch``). Anthropic's batch discount is 50 % on input and output.
    # Applied by every estimate that projects batched calls (the benchmark
    # harnesses) and when a run's measured usage is priced (``particles
    # audit``, ``particles memory consolidate``).
    # (Formerly ``benchmark_memory.batch_discount``; the old key is migrated.)
    batch_discount: float = Field(default=0.5, ge=0.0, le=1.0)
    # Prompt-cache pricing as multiples of a model's input price, applied when
    # a run's measured usage is priced. Anthropic bills a 5-minute cache write
    # at 1.25x input and a cache read at 0.1x.
    cache_write_price_multiplier: float = Field(default=1.25, ge=0.0)
    cache_read_price_multiplier: float = Field(default=0.1, ge=0.0)

    @field_validator("price_per_mtok", mode="before")
    @classmethod
    def _merge_over_list_prices(cls, value: Any) -> Any:
        """Operator entries replace the shipped ones key by key; the rest stay."""
        if isinstance(value, dict):
            return {**_default_prices(), **value}
        return value

    def price_for(self, selection: ProviderSelection) -> TokenPrice | None:
        """The price entry for ``selection``: ``provider:model`` first, then the bare id."""
        price, _ = lookup_by_model(self.price_per_mtok, selection.provider, selection.model)
        return price

    def for_purpose(self, purpose: str) -> ProviderSelection:
        """Return the selection for ``purpose``, falling back to ``default``.

        ``purpose`` is one of the :data:`particles.llm.LLMPurpose` values,
        which match this model's field names by construction.
        """
        override = getattr(self, purpose, None)
        return override if isinstance(override, ProviderSelection) else self.default

    @model_validator(mode="after")
    def _validate_provider_wiring(self) -> LLMConfig:
        """Fail dangling provider names / adapter kinds at config load.

        Also folds the deprecated ``llm.local`` block into ``providers`` and
        guarantees the compiled-in ``local`` entry exists, preserving
        out-of-the-box behaviour.
        """
        if self.local is not None:
            log.warning(
                "config: 'llm.local' is deprecated and will be "
                "removed in a future release. Move the block to "
                "'llm.providers.local'."
            )
            self.providers.setdefault("local", self.local)
        self.providers.setdefault("local", OpenAICompatProviderConfig())
        if "anthropic" in self.providers:
            raise ValueError(
                "'anthropic' is a reserved provider name (the native SDK "
                "adapter) and cannot appear in llm.providers; pick another "
                "name for an OpenAI-compatible endpoint"
            )
        # The `adapter` kind is deliberately NOT validated here: the kind
        # registry lives in particles.llm, and a config → llm import would
        # mint a new subpackage cycle (llm reads config at call time), which
        # the acyclic-siblings contract forbids. A dangling kind
        # fails loudly at first resolution in `get_provider` instead.
        for field_name in type(self).model_fields:
            selection = getattr(self, field_name, None)
            if not isinstance(selection, ProviderSelection):
                continue
            if selection.provider != "anthropic" and selection.provider not in self.providers:
                raise ValueError(
                    f"llm.{field_name}.provider {selection.provider!r} is "
                    "neither 'anthropic' nor a key of llm.providers"
                )
        return self


class DuplicateSuppressionConfig(BaseModel):
    """Extract-time exact-duplicate suppression.

    Declines to mint a particle whose claim is already held verbatim by an
    ACTIVE particle with the same subjects and stance holder, recording the new
    source on the existing particle instead.

    **Default ON**, unlike the ``links_suggest.auto_merge`` flag it is
    the prevention-side twin of. The two differ categorically: auto-merge
    supersedes existing ACTIVE particles and needs a revert path, whereas this
    only declines to *create* — and because the predicate is exact content
    identity with the same subjects and holder, a suppression cannot mean a
    distinct fact was dropped (the claim is on the ACTIVE surface verbatim, and
    the new source's evidence is appended to it). With the leak measured at
    15.8 % of mint, a default-OFF prevention would not stop the regrowth
    cleanup exists to undo.
    """

    enabled: bool = True


class RetiredValueQuarantineConfig(BaseModel):
    """Retired-value quarantine on the write paths.

    When a candidate claim is the exact twin (same normalized content, subject
    set and stance holder; the exact-duplicate key) of a particle an operator or
    reviewer retired by judgment — ``EXPLICIT_RETRACTION``,
    ``EXPLICIT_SUPERSESSION``, or a review / cascade ``CONFLICT_RESOLVED``
    loser — the candidate is stored **quarantined** (``PROVENANCE_STALE`` /
    ``CONFLICT_PENDING``) behind an INCONSISTENCY record for review instead of
    re-entering ACTIVE. The source still says it, and the ledger records that;
    the value does not walk back onto the answer surface without a person.

    Retirements that encode no judgment about the value — ``SOURCE_RETRACTED``,
    ``SUPERSEDED_BY_REINDEX``, ``DUPLICATE_MERGED``, ``DOCUMENT_SUPERSEDED``,
    ``VALIDITY_EXPIRED``, the trust-differential demotions — never fire it.

    **Default ON.** Like exact-duplicate suppression, it only
    declines to *activate*: the candidate is stored in full and one review
    lifts it, so a wrong hold costs a review
    round-trip, while the failure it prevents (a reindex of an unchanged source
    silently re-minting a retracted claim as ACTIVE) leaves no trace at all.
    """

    enabled: bool = True


class ExtractionConfig(BaseModel):
    # Total output allowance per extraction call. The reply itself runs to 2 to
    # 3.5 tokens per source character (measured 2026-09-25 on claude-sonnet-5),
    # and an adaptive-thinking model (claude-sonnet-5 thinks whenever
    # ``thinking`` is omitted; about 17% of its output on those files) spends
    # its thinking from this same budget. The Messages API offers no separate
    # thinking cap on that model (``budget_tokens`` is rejected). 8192 left 17
    # of 96 memory files cut short or empty in the 2026-09-25 audit run. At
    # 16384, with the retry at 20000, the owner's 2026-09-26 run of
    # ``extract --all-pending --tag memory-file`` still cut many memory files
    # twice and lost every claim past the cut (one kept 79 claims and lost the
    # rest), with both attempts billed. 32000 covers a memory file at the
    # measured p90 (~8,000 characters, 19k to 34k output tokens); a larger
    # file falls to the retry. The Anthropic adapter streams any call above
    # the SDK's ~21333-token non-streaming ceiling, so the budget is no longer
    # bounded by it. A budget is a cap, not a charge: only tokens produced are
    # billed.
    max_tokens: int = Field(default=32000, gt=0)
    # One retry of an extraction call at this larger budget when its reply
    # came back with no text, cut at the budget, or unparseable. The same call
    # at the same budget tends to reproduce those, while a larger budget is a
    # different call. Never fires on a billing, network, or refusal failure,
    # and costs nothing on a reply that parsed whole. A value at or below
    # ``max_tokens`` disables the retry. 64000 covers a whole
    # ``html_chunk_size`` chunk (15,000 characters) at the worst measured rate
    # of 3.5 answer tokens a character plus thinking, and is the output limit
    # of claude-haiku-4-5, the smallest of the Claude models routed here.
    retry_max_tokens: int = Field(default=64000, ge=0)
    similarity_threshold: float = 0.80
    pdf_page_overlap_lines: int = 5
    # PDF hardening (security): a malicious PDF can carry an enormous page
    # count, one page that extracts to gigabytes of text, or content that
    # makes pypdf spin. ``max_pdf_pages`` caps how many pages are processed
    # (the rest are skipped with a quality note); ``max_pdf_page_chars``
    # truncates a single page's extracted text; ``max_pdf_seconds`` is a
    # wall-clock budget for the whole paged-extraction loop. Defaults are far
    # above any legitimate document this SDK ingests.
    max_pdf_pages: int = 2000
    max_pdf_page_chars: int = 1_000_000
    max_pdf_seconds: float = 1800.0
    # a standalone image deposit (IMAGE source type) is sent to the
    # vision-capable provider in one multimodal call. Cap the bytes a single
    # image can carry (hosted vision APIs reject very large images anyway); an
    # oversized image is skipped with an IMAGE_BYTES_CAP quality note. ~5 MB.
    max_image_bytes: int = 5_000_000
    html_chunk_size: int = 15000
    html_chunk_overlap_lines: int = 5
    # Shared chunked-extraction knobs. Used by any extractor that
    # routes through extract_with_carry_forward (gist, reddit, …). When the
    # rendered comment / discussion text exceeds the single-call threshold,
    # the extractor splits into chunks of comment_chunk_chars each and makes
    # one LLM call per chunk. Total LLM calls per source are capped at
    # max_llm_calls_per_source.
    single_call_threshold_chars: int = 30000
    comment_chunk_chars: int = 10000
    max_llm_calls_per_source: int = 8
    # ``extract --all-pending`` (0.42.2) treats an IN_PROGRESS snapshot
    # whose ``extraction_started_at`` is older than this threshold as
    # orphaned and resets it to PENDING. Catches snapshots stranded by
    # SIGKILL / segfault / oom whose try/except cleanup didn't run. The
    # default (30 min) is well above the worst-case extraction runtime
    # for any single snapshot (~10-15 min for a 100-page PDF with
    # max_llm_calls_per_source=8) but short enough that operator
    # recovery is quick.
    stale_in_progress_minutes: float = 30.0
    # Cost-estimate assumption, not a runtime knob: the general extractor's
    # fixed system prompt (rules + modality / polarity / stance / structure /
    # validity addenda, ~8k chars) is re-sent on every call, so an estimate
    # adds this many input tokens per call rather than pricing source text
    # alone. On a short source it rivals the source itself. Read by the audit
    # estimate and the memory-rot benchmark.
    estimate_prompt_overhead_tokens: int = Field(default=2500, ge=0)
    # The bulk extraction paths skip an unextracted snapshot of a MUTABLE
    # entry once a newer extractable snapshot of the same entry exists: the
    # generation cascade would retire the older generation's beliefs anyway,
    # so extracting it buys nothing and, out of order, retires the current
    # generation. The skipped snapshot is marked COMPLETE and recorded in
    # ``snapshots.superseded_by_snapshot_id``. ``False`` restores extracting
    # every generation, for an operator who wants intermediate generations
    # as PROVENANCE_STALE belief history and accepts the bill.
    collapse_superseded_pending: bool = True
    # an APPEND_ONLY entry's snapshot is extracted as a delta, only
    # the text it adds to the last snapshot the store extracted, with the end
    # of the earlier text shown to the model as context it extracts nothing
    # from. ``False`` restores reading every snapshot as a whole.
    append_only_delta: bool = True
    # Characters of already-extracted text before the delta shown as context
    # (cut back to a paragraph boundary). 0 reads the delta with no context.
    append_context_chars: int = Field(default=4000, ge=0)
    # The largest delta chunk sent in one call. Each chunk after the first
    # takes the end of the chunk before it as its context.
    append_chunk_chars: int = Field(default=7500, ge=1000)
    # a pair the §6.6 contradiction probe confirms is read a second
    # time, before the write loop, with each claim's source kind, name, time
    # and passage (on ``llm.verification``). A pair the reading does not
    # confirm is treated as the probe's NO. ``False`` restores the write path
    # before it: the first probe alone decides.
    verify_conflicts: bool = True
    # don't re-mint a claim the store already holds verbatim.
    duplicate_suppression: DuplicateSuppressionConfig = Field(
        default_factory=DuplicateSuppressionConfig
    )
    # a claim retired by judgment is quarantined for review when
    # re-asserted, not re-minted ACTIVE.
    retired_value_quarantine: RetiredValueQuarantineConfig = Field(
        default_factory=RetiredValueQuarantineConfig
    )
    # source types whose prose is scanned for tool turns before
    # extraction, so a tool's words are never extracted as the speaker's.
    # Deliberately the conversational set: a transcript is where a tool turn
    # appears beside a human one. Empty disables the marker.
    tool_turn_source_types: list[str] = Field(default_factory=lambda: ["CONVERSATION", "JOURNAL"])
    # Source types whose extracted claims are screened for an agent session's
    # momentary working state (current worktree, cwd, branch), which is true
    # for minutes and never a durable belief. CONVERSATION only: a coding-agent
    # transcript is where that narration appears. Empty disables the filter.
    session_state_source_types: list[str] = Field(default_factory=lambda: ["CONVERSATION"])


class ExtractionScopeConfig(BaseModel):
    """LLM-semantic document-scope labelling of extraction candidates.

    When ``enabled``, the general extractor classifies each candidate as
    ``WORLD`` or ``DOCUMENT_META`` (a claim about the source document's own
    structure / editorial apparatus). ``mode`` governs what happens to a
    ``DOCUMENT_META`` candidate:

    * ``label`` (default) — tag it; downstream excludes it from §6.6
      contradiction-checking and the default query surface, but it stays in
      the store.
    * ``suppress`` — drop it before persisting.
    * ``passthrough`` — tag it for inspection but apply no exclusion (for
      evaluating the classifier before trusting it to shape results).

    ``exempt_source_tags`` lifts the exclusion for whole *sources*:
    a corpus entry carrying one of these tags stamps ``scope_action =
    source_exempt`` on its flagged claims, so a rules document's prescriptions
    reach the default query + projection surfaces. ``rule-file`` is the
    tag; empty the list to disable the exemption.
    """

    enabled: bool = True
    mode: Literal["label", "suppress", "passthrough"] = "label"
    exempt_source_tags: list[str] = ["rule-file"]


class ExtractionModalityConfig(BaseModel):
    """LLM-semantic ``assertion_modality`` classification of candidates.

    When ``enabled`` (default), the general extractor classifies each candidate
    as ``FALSIFIABLE`` (default / truth-apt), ``EVALUATIVE``, ``EXPERIENTIAL``,
    or ``CONSTITUTIVE``, populating the first-class assertion-modality field. The
    engine then applies truth-semantics (§6.6 / L-SEM-01 / L-IDX-01) only to
    ``FALSIFIABLE`` particles. There is no ``mode`` knob (unlike
    :class:`ExtractionScopeConfig`): the field *is* the effect, so there is
    nothing to suppress and no tag-without-effect mode. ``enabled: false``
    reproduces the pre-0125 all-``FALSIFIABLE`` behaviour exactly.
    """

    enabled: bool = True


class ModalityRegenerationConfig(BaseModel):
    """Defaults for ``particles modality``, the adjudicability-default regeneration.

    The pass reclassifies ACTIVE claims whose modality stamp names a classifier
    rule other than today's, one LLM call per claim on the ``extraction``
    purpose, so it carries the ``structure`` backfill's three knobs: a call-aware
    rate cap, a resumable per-run cap (``0`` means the whole backlog), and a
    commit interval so an interrupt loses seconds, not hours. It writes only the
    modality record; content, confidence and provenance never move.
    """

    rate_limit_per_minute: int = Field(default=60, ge=0)
    batch_limit: int = Field(default=200, ge=0)
    commit_interval: int = Field(default=25, ge=1)


class ExtractionPolarityConfig(BaseModel):
    """LLM-semantic claim-polarity classification of candidates (cap. 1).

    When ``enabled`` (default), the general extractor classifies how the source
    document *presents* each candidate proposition — ``ASSERTED`` (default /
    held), ``DECLINED`` (rejected / superseded / deferred / out-of-scope), or
    ``HYPOTHETICAL`` (counterfactual / conditional / future projection / worked
    example) — recording the two non-asserted values on
    ``properties["extraction:polarity"]`` (the key was the bare ``polarity``
    before).
    The operation layer then keeps non-asserted particles off the default
    factual surface (query / projection / export / §6.6 / L-SEM-01 / L-IDX-01),
    overridable via the ``include_non_asserted`` opt-in. Like
    :class:`ExtractionModalityConfig` there is no ``mode`` knob: the label *is*
    the effect. Default-safe — unknown / missing values fall back to
    ``ASSERTED``. ``enabled: false`` reproduces the pre-0145 behaviour exactly
    (every candidate ``ASSERTED``, nothing excluded).
    """

    enabled: bool = True


class ExtractionValidityConfig(BaseModel):
    """Event-anchored validity extraction.

    When ``enabled`` (default), the general extractor emits ``valid_until`` on a
    candidate — and thence on the persisted ``Particle`` — only for a claim
    carrying a genuine, resolvable, future-dated validity boundary ("the
    contract runs through 2026", "the exam is tomorrow"), biased hard toward
    **under-emission** so a durable fact that merely *mentions* a date ("I met
    her in 2019") is never wrongly assigned a boundary and later retired as
    ``VALIDITY_EXPIRED`` by the §9.3 staleness lint. Emission is gated by three
    conjunctive conditions: an explicit boundary cue (the categorical LLM
    judgment), ``validity_confidence >= min_boundary_confidence``, and a resolved
    date in the future (a born-expired ``valid_until <= now`` is dropped).

    ``min_boundary_confidence`` is the emission floor on the model's
    self-assessed *boundary* confidence — a distinct quantity from the
    candidate's ``confidence_value`` (which scores how clearly the claim is
    stated). It is the operator's lever for trading recall against the
    over-eager-expiry rate the ``benchmark/validity`` harness measures; it gates
    a structural decision (does a boundary exist?), never the stored
    ``confidence.value``. ``enabled: false`` reproduces the pre-0197
    behaviour exactly (no candidate ever carries ``valid_until`` from extraction).
    """

    enabled: bool = True
    min_boundary_confidence: float = 0.75


class StructuredClaimConfig(BaseModel):
    """Derived S-P-O annotation beside the prose claim.

    When ``enabled`` (default), the general extractor's prompt and response
    schema gain a ``structured_claim`` field and the ingest pipeline stamps the
    resulting triple onto the particle. This costs **no extra LLM call** — the
    triple rides the extraction reply already being paid for — and it never
    touches ``content``, ``confidence`` or provenance. A malformed or
    missing triple simply drops the annotation and keeps the claim: absence is a
    legal *permanent* state, because prose that does not
    triple-ize cleanly is better left unannotated than annotated falsely.
    ``enabled: false`` reproduces the pre-0218 prompt byte-for-byte.

    The ``backfill_*`` knobs are the defaults for ``particles structure``, the
    verb that annotates particles extracted before this landed (or stamped by a
    superseded structurizer version). That pass *does* pay one LLM call per
    particle, hence the rate limit. ``backfill_batch_limit: 0`` (or
    ``--limit 0``) means the whole backlog in one run;
    ``backfill_commit_interval`` is how often that long run commits, so an
    interrupt costs seconds of work rather than hours.
    """

    enabled: bool = True
    backfill_rate_limit_per_minute: int = 60
    backfill_batch_limit: int = 200
    backfill_commit_interval: int = 25


class RdfConfig(BaseModel):
    """RDF deposit — the structure-canonical parsing extractor.

    The extractor is deterministic: no LLM call and no network call, ever. A
    structure-canonical particle's ``content`` is *derived* from its triple, and
    a derivation that depended on a remote label service or a model would not be
    reproducible across two extractions of the same snapshot.

    ``default_confidence`` states the extractor's confidence in its *reading*,
    not in the source — a parse is exact, so it is high. How much the source is
    believed is the separate trust quantity (``DEFAULT_TRUST_WEIGHT`` on the
    extractor plus the operator's ``SourceTrustStatement``s), per the
    two-quantity separation. It is overridden per-triple only when the document
    itself annotates a confidence with one of ``confidence_predicates``.

    ``skip_predicates`` are the triples that are *about the document* rather than
    about the world: label predicates (consumed by verbalization, and preserved
    as the Subject's canonical name), collection plumbing, and ontology headers.
    ``uri_namespaces`` maps an absolute-IRI prefix onto a Subject Authority
    namespace slug; CURIE prefixes are absent because the parser has already
    expanded them by the time a term is inspected.
    """

    max_triples: int = 5000
    max_bytes: int = 16 * 1024 * 1024
    default_confidence: float = 0.95
    include_blank_node_subjects: bool = False
    skip_predicates: list[str] = Field(
        default_factory=lambda: [
            # Label predicates — consumed by the verbalization ladder.
            "http://www.w3.org/2000/01/rdf-schema#label",
            "http://www.w3.org/2004/02/skos/core#prefLabel",
            "http://purl.org/dc/terms/title",
            "http://purl.org/dc/elements/1.1/title",
            "http://xmlns.com/foaf/0.1/name",
            # Collection plumbing — syntax, not assertion.
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#first",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest",
            # Ontology headers — metadata about the file.
            "http://www.w3.org/2002/07/owl#imports",
            "http://www.w3.org/2002/07/owl#versionInfo",
        ]
    )
    confidence_predicates: list[str] = Field(
        default_factory=lambda: [
            # The term this SDK's own context.jsonld publishes, so a subsequent
            # export re-deposits at the confidence it was exported with. There is
            # no *standard* confidence predicate in RDF — not in PROV-O, not in
            # the nanopublication vocabularies — so a publisher's own predicate
            # is named here by the operator rather than guessed by the parser.
            "https://linkedparticles.org/vocab#confidenceValue",
        ]
    )
    uri_namespaces: dict[str, str] = Field(
        default_factory=lambda: {
            "http://www.wikidata.org/entity/": "wikidata",
            "https://www.wikidata.org/wiki/": "wikidata",
            "http://nomisma.org/id/": "nomisma",
        }
    )


class DocumentSupersessionConfig(BaseModel):
    """Lift document supersedes-metadata into the §6.6 rung-1.5 prior (cap. 2).

    When ``enabled`` (default), the ADR genre adapter records each ADR's
    ``supersedes:`` / ``superseded_by:`` frontmatter as a corpus-entry
    supersession relation at deposit, and §6.6 conflict resolution gains a new
    rung **1.5**, above the trust rung: when two truth-apt claims conflict and
    one's provenance document (transitively) supersedes the other's, the
    superseded claim is demoted ``ACTIVE → PROVENANCE_STALE`` with
    ``status_reason = DOCUMENT_SUPERSEDED`` and no ``INCONSISTENCY`` is surfaced.
    The relation is document-level but the prior is **conflict-gated** — a
    still-true, non-conflicting claim from the superseded document is never
    touched. Single-trust-order stores only in v1 (matching the trust rung).
    ``enabled: false`` reproduces the pre-cap-2 behaviour exactly
    (no supersession prior; a superseded decision falls through to the trust
    rung / INCONSISTENCY).
    """

    enabled: bool = True


class DocumentPrecedenceConfig(BaseModel):
    """Latest-decision-wins tie-break among detected conflicts.

    When ``enabled`` (default), the query/projection ranker breaks a tie
    **between two ACTIVE particles a contradiction probe has flagged as
    conflicting** (and only those) in favour of the one whose provenance
    document is the **later authored decision** — the ADR ``date`` + id ordinal
    via the genre-adapter seam, falling back to the snapshot's
    ``content_published_at``. The recency-loser's combined score is multiplied
    by ``rank_penalty`` at sort time only (the ``narrative_rank_weight``
    shape); the reported ``effective_confidence`` and the stored
    ``confidence.value`` are **untouched**, and no status changes.
    It is the rank-time, no-authored-edge superset-filler for the
    store-mutating authored-edge supersession: an authored ``supersedes:`` edge
    already demotes its loser off ACTIVE before ranking, so this tie-break only
    sees the residual conflicts with no edge. ``enabled: false`` reproduces the
    pre-0166 behaviour byte-for-byte (no precedence reorder); the tie-break is
    also inert outside a detected conflict and when neither side exposes a
    comparable precedence key (default-safe — do nothing rather than guess).
    """

    enabled: bool = True
    # The rank-time multiplier applied to the recency-loser of a detected
    # conflict (cf. query.narrative_rank_weight). < 1.0 demotes the older
    # decision below the newer; 1.0 is inert (no reorder).
    rank_penalty: float = Field(default=0.6, ge=0.0, le=1.0)


class JournalExtractorConfig(BaseModel):
    """Journal-aware extractor for ``JOURNAL``-typed entries.

    When ``enabled`` (default), a ``JOURNAL`` corpus entry (set by
    ``particles deposit --journal`` / ``--source-type JOURNAL``) is routed to
    the journal extractor, which reifies first-person prose into
    ``EXPERIENTIAL`` particles, tags opinions ``EVALUATIVE``, and emits the
    ``NARRATIVE`` graph for the entry. ``enabled: false`` makes the
    extractor decline, so ``JOURNAL`` entries fall through to the general
    extractor unchanged.

    ``synthesize_merged_narrative``: when an over-length entry is
    extracted in multiple chunks, the Engine narrative-merge post-pass makes one
    extra LLM call to synthesize a single whole-entry NARRATIVE label from the
    per-chunk labels. ``false`` skips that call and uses the first chunk's label
    (deterministic, no extra call); the same first-label fallback also fires
    automatically if the synthesis call fails.
    """

    enabled: bool = True
    synthesize_merged_narrative: bool = True


class ImportProjectConfig(BaseModel):
    """Recursive multi-file structured-source deposit.

    ``particles import project <dir>`` walks a software-project tree and
    deposits one corpus entry per source file. ``extensions`` is the set of file
    suffixes deposited as ``PYTHON_SOURCE`` (the first registered glob instance);
    ``ignore_dirs`` are directory names pruned during the walk
    (dot-prefixed components are pruned regardless, so ``.git`` / ``.venv`` need
    not be listed — they are kept here for explicitness). Underscore-prefixed
    module files (``__init__.py`` / ``_shared.py``) are **kept**, unlike the
    vault walker's ``_``-component skip.
    """

    extensions: list[str] = Field(default_factory=lambda: [".py"])
    ignore_dirs: list[str] = Field(
        default_factory=lambda: [
            "__pycache__",
            "node_modules",
            "build",
            "dist",
            ".git",
            ".venv",
            "venv",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
        ]
    )


class MigrationConfig(BaseModel):
    """Inbound migration from another memory store.

    ``particles import mcp-memory <path>`` deposits an incumbent store's export
    verbatim and a structured, no-LLM extractor turns each record into a
    particle. Every such particle carries ``CalibrationSource.IMPORTED`` and the
    single ``import_confidence`` below — **never** the incumbent's own score,
    which is preserved as a tag and structurally kept out of the ranking
    arithmetic (§5). One number, one meaning: *this was believed by a system we
    cannot interrogate.*

    Raising the floor here is not how you come to trust a migrated store. The
    lever for that is ``particles trust set`` against the export's source type,
    which is revisable, auditable, and demotion-safe — where ``confidence.value``
    is immutable at creation. Read via ``get_config()`` inside the
    extractor, never at import.
    """

    # Deliberately low. A migrated belief is second-hand: this store never saw
    # the claim made, cannot check it, and did not calibrate the number.
    import_confidence: float = Field(default=0.35, ge=0.0, le=1.0)


class WebClipperConfig(BaseModel):
    """Frontmatter-Markdown captures intake.

    ``particles import web-clipper <dir>`` walks a folder of frontmatter-Markdown
    captures (the Obsidian Web Clipper is the first and only shipped profile),
    maps each capture's leading YAML header onto deposit fields, strips the
    header, and deposits the **body** as a ``WEB_PAGE`` corpus entry keyed on the
    clipped ``source:`` URL — restoring the provenance (real URL, publication
    date, per-file tags, source type) that ``import vault`` discards.

    This is a plain **config-table profile**, not a typed protocol (
    Decision §5): one producer does not justify a ``FrontmatterProfile`` protocol,
    and verb / profile generalisation is deferred. A second producer is
    an operator edit of these keys, not a code change. Read via ``get_config()``
    inside the walker, never at import.
    """

    # Frontmatter keys tried in order for the entry's ``uri_r`` (the clipped page
    # URL). The first present, non-empty value wins; fragment-stripped, not fetched.
    url_keys: list[str] = Field(default_factory=lambda: ["source", "url"])
    # Frontmatter keys tried in order for ``content_published_at``,
    # parsed against ``deposit_date.formats``. Below an explicit ``--date``.
    date_keys: list[str] = Field(default_factory=lambda: ["published"])
    # Frontmatter keys whose list (or scalar) values become entry tags, merged
    # with any run-wide ``--tags``.
    tag_keys: list[str] = Field(default_factory=lambda: ["tags"])
    # The source-type stamp for a clipped entry. ``WEB_PAGE`` because the entry
    # genuinely is a web page archived locally — trustable, decayable,
    # and queryable as the page it clipped.
    source_type: str = "WEB_PAGE"


class ExtractionStanceConfig(BaseModel):
    """Extraction-time endorsement-stance detection.

    When ``enabled`` (default), the general extractor flags a candidate that
    explicitly endorses / disputes another claim *co-extracted from the same
    source* (endorsing, disputing, rebutting, concurring). The pipeline then
    reifies it into a stance particle bound to its target by an ``ENDORSES`` /
    ``DISPUTES`` edge, stamping ``stance:holder`` (the source author) and the
    optional ``stance:magnitude``. Default-safe toward *under*-emission
    (M3): a candidate is a stance only when the LLM names an in-batch
    target and the source author is derivable — a spurious stance is permanent
    substrate that distorts the query-time agreement view. ``enabled: false``
    reproduces the pre-0119 behaviour exactly (no stance fields, no edges).
    """

    enabled: bool = True


class ExtractionVisionConfig(BaseModel):
    """Vision / multimodal extraction of image-bearing PDF pages.

    When ``enabled`` (off by default — opt-in cost; also requires the
    ``[vision]`` extra), the general extractor's per-page PDF loop becomes
    modality-aware: a **visual** page is sent to the vision-capable provider as
    one multimodal call (its ``pypdf`` text *plus* a rendered image of the
    page), while a text page keeps the cheap text-only path. Vision tokens are
    paid only on visual pages.

    * ``trigger`` — ``image_bearing`` (default): a page is visual when it has
      embedded raster images or its extracted text is below
      ``low_text_threshold`` (a scanned / figure-only page). ``always``: every
      page takes the multimodal path — the escape hatch for documents whose
      diagrams are vector art on text-rich pages, at the cost of vision tokens
      on every page.
    * ``low_text_threshold`` — char count below which a page is treated as
      visual (scanned-page heuristic).
    * ``render_dpi`` — page-image resolution; 150 keeps diagrams legible within
      the model's per-image token cap.
    * ``max_pages`` — cap on how many pages per document take the vision path;
      pages past it fall back to text-only with a ``VISION_PAGE_CAP`` note.

    ``enabled: false`` reproduces the pre-0171 text-only PDF behaviour exactly.
    """

    enabled: bool = False
    trigger: Literal["image_bearing", "always"] = "image_bearing"
    low_text_threshold: int = 200
    render_dpi: int = 150
    max_pages: int = 50


class TrustConfig(BaseModel):
    differential_threshold: float = 0.15
    cascade_max_per_run: int = 500
    cascade_min_reviewer_confirmations: int = 3
    # trust_rank written on the SourceTrustStatement that a §9.6 Review
    # PREFER_A/PREFER_B resolution derives from the reviewer's judgment.
    reviewer_trust_rank: float = Field(default=0.8, ge=0.0, le=1.0)
    # source_type -> knowledge-domain label, consulted by
    # infer_domain() as a fallback for source types no extractor MUST-claims.
    # This is what makes the AUTHOR-scoped trust tier reachable for directly
    # asserted (CONVERSATION-sourced) content so the agent_trust_rank seed binds.
    # Additive: a source type absent here AND from every extractor clause still
    # resolves to no domain (neutral trust), exactly as before.
    source_type_domains: dict[str, str] = Field(
        default_factory=lambda: {"CONVERSATION": "agent-memory"}
    )


class UpdateSupersessionConfig(BaseModel):
    """Same-subject update supersession at extraction.

    Two halves behind one switch. **Candidacy:** each extracted claim with an
    about-subject key (its structured-claim subject, else its sole subject) is
    also compared with ACTIVE claims about that subject in *any* corpus entry,
    at a subject-scoped content-cosine floor well below
    ``extraction.similarity_threshold`` — value updates differ in exactly the
    word that carries the value, so they rarely clear 0.80. Every pair still
    goes through the contradiction probe; the floor only finds candidates.
    **Rung 2.5:** a confirmed contradiction between two extractor-asserted
    claims that no trust key distinguishes resolves to the strictly newer one
    by source date, in ``single`` and ``multi`` stores alike (a lineage
    updating itself is not cross-contributor arbitration).
    """

    enabled: bool = True
    # Subject-scoped §6.6 candidacy floor on the normalized cosine scale (ADR
    # 0179). 0.45 is the measured operating point on the live stores:
    # 76 % of real same-subject update pairs found, 1.7 % of cross-attribute
    # same-subject pairs let through (each costs one probe that answers NO).
    subject_floor: float = Field(default=0.45, ge=0.0, le=1.0)
    # Probes per extracted claim from the subject-scoped search, at most.
    max_candidates: int = Field(default=3, ge=1)


class ReconciliationConfig(BaseModel):
    """Cross-entry §6.6 reconciliation policy.

    ``store_mode`` selects the trust-resolution regime applied when a
    contradicting pair clears the contradiction-signal gate:

    * ``single`` (default) — a single global trust order. §6.6 rung 2
      auto-supersede fires: the higher-trust claim wins and the lower-trust
      one is demoted (today's behavior — unchanged; preserves the
      invariant §1 that single-store solo behavior is byte-for-byte the same).
    * ``multi`` — a multi-contributor / consensus store. There is no global
      trust order (trust is per-viewer at query time), so
      auto-supersede is suppressed and the contradiction surfaces as an
      INCONSISTENCY (both claims stay ACTIVE), ranked per-viewer downstream.
      The consensus invariant: a contributor's claim is never dropped by
      another contributor's trust.

    ``store_mode`` is the global default; ``per_store`` overrides it per store
    handle. Resolve the effective mode via
    :meth:`ParticlesConfig.reconciliation_mode_for`.
    """

    store_mode: Literal["single", "multi"] = "single"
    # per-store override of store_mode, keyed
    # by store handle. An MCP-write-enabled store (mcp.write.enabled_stores)
    # defaults to "multi" even without an entry here; an explicit "single" on a
    # write store is rejected by ParticlesConfig's validator.
    per_store: dict[str, Literal["single", "multi"]] = Field(default_factory=dict)
    # Same-subject update supersession: cross-entry candidacy keyed
    # on the claim's subject, and the rung 2.5 same-lineage recency rung.
    update_supersession: UpdateSupersessionConfig = Field(default_factory=UpdateSupersessionConfig)


class SourceDecayConfig(BaseModel):
    half_life_days: float
    floor: float = 0.10


class ContentAgeDecayConfig(BaseModel):
    sources: dict[str, SourceDecayConfig] = Field(
        default_factory=lambda: {
            "REDDIT_POST": SourceDecayConfig(half_life_days=60.0),
            "GITHUB_REPO": SourceDecayConfig(half_life_days=365.0, floor=0.40),
            "GITHUB_GIST": SourceDecayConfig(half_life_days=180.0, floor=0.20),
            "GITHUB_PAGES": SourceDecayConfig(half_life_days=365.0, floor=0.25),
        }
    )


class UtilityRuleConfig(BaseModel):
    """Local base for the usefulness policy — the store's ``default`` utility rule.

    The analogue of a ``content_age_decay`` source entry: the store-local base a
    lens ``utility_rules`` layer overlays, most-skeptical-wins.

    ``rank_lift`` is the ``λ`` in ``rank_score = effective_confidence +
    λ·ln(1 + R)`` — the single knob that replaced ``weight`` /
    ``floor`` / ``cap`` triple when the bounded multiplier was superseded.
    ``0.0`` disables the lift (projection ranks by effective confidence alone).

    The default ``0.015`` is **empirically calibrated** (re-centred,
    re-measured post-dedup, re-centred
    again), not derived. The admissible band is a property of the
    **surface**, not of the store, because a larger head has more room to expose
    duplicate clusters — so the three head sizes this SDK renders disagree.
    Measured on the dogfood store (27,048 ACTIVE beliefs) after the
    subject-agnostic exact-duplicate merge; the rows below cover the
    projection surfaces and the digest:

    ==========================================  =====  ====================
    surface                                     ``N``  band (≥95% distinct)
    ==========================================  =====  ====================
    projection ``top_k``                        60     0.011 – (no ceiling)
    projection ``max_lines``                    120    0.006 – (no ceiling)
    digest ``digest_max_beliefs``               200    0.004 – (no ceiling)
    ==========================================  =====  ====================

    Note what those bands lack: an **upper edge**. Every prior calibration was
    squeezed between a floor (the target must reach the head) and a
    duplicate-cluster ceiling, and ``0.011`` was the log-midpoint of the
    resulting ``[0.0075, 0.0165]`` intersection. The clusters that set that
    ceiling were drained, so the largest in-head duplicate cluster is now
    **2 at every λ up to 0.6** and the log-midpoint rule no longer yields a
    finite answer.

    The selection rule is therefore now **margin above the binding floor**:
    ``0.015`` sits 1.36× above the ``N = 60`` floor of ``0.011`` (one grid step
    below it, at ``0.010``, the target drops to rank 61 — outside the
    head), holds that target at rank 24 rather than 48, and keeps all 60
    ``N = 60`` head slots distinct and 120/120 at ``N = 120``. In-band choice
    above that is otherwise inconsequential.

    **What bounds λ from above is now the owner lens, not duplicates.**
    Its ``ω`` floor tracks λ, because a larger utility term holds the head
    harder against a flat-step cohort. Measured here:

    ====== ==========  ==================================
    λ      ω floor     shipped ``ω = 0.04``
    ====== ==========  ==================================
    0.011  0.018       admissible, 2.2× margin
    0.015  0.024       admissible, 1.67× margin
    0.020  0.032       admissible, 1.25× margin
    0.025  0.040       on its floor
    0.030  0.048       out of band — cohort leaves the head
    ====== ==========  ==================================

    So raising λ past ≈0.02 is a **joint λ/ω re-calibration**, not a one-line
    change to this value.

    ``λ`` is deliberately **not** auto-fitted — no label says
    which belief *should* occupy a head slot, and the confidence spread a fit
    would key on is flattened to exactly zero by the cap. Re-calibrate
    against your own store with ``particles memory sweep-rank-lift`` rather than
    porting this number; it is a property of one store's confidence spread and
    event volume.

    If the sweep's *ceiling* is what binds for you, the fix is deduplication,
    not a smaller ``λ`` — but check that your dedup pass can
    actually *reach* the clusters setting the ceiling. On the dogfood store it
    could not at first: merge cut near-duplicate mass from 16.0% to
    3.1% of ACTIVE, yet the ceiling *fell* (0.0190 → 0.0165 at ``N = 200``)
    rather than rising, because the two clusters that set it were 21
    **byte-identical** copies each carrying 1/21 and 0/21 subject links, and
    ``suggest_co_evidential`` iterates Subjects so it never saw them—
    reporting zero groups while 211 exact-duplicate groups / 534
    redundant ACTIVE copies remained. The grouping was made subject-agnostic
    and the ceiling then left the measurable range entirely:
    305 groups / 350 redundant copies remain (1.29% of ACTIVE), largest cluster
    7, none of them in the head. That is the payoff predicted,
    arriving one dedup pass later than it expected.
    """

    half_life_uses_days: float = Field(default=30.0, gt=0.0)
    rank_lift: float = Field(default=0.015, ge=0.0)


class UtilityMiningConfig(BaseModel):
    """The transcript-mining pass that produces per-belief utility evidence.

    Only beliefs the session was shown are candidates, and only an LLM judge's
    "applied" ruling credits one. A literal token match nominates a
    candidate; the behavioural route nominates the shown beliefs with no token
    hit. ``behavioural_matching`` turns the judge on; off, nothing is recorded.
    Every judge call, of either route, counts against
    ``max_behavioural_calls`` per run (cost-discipline). The default
    of 150 covers a night of about four sessions at a median of 11 calls each.

    ``behavioural_candidate_limit`` bounds how many shown beliefs with no token
    hit go to the judge: the pre-filter ranks them by embedding similarity
    between the session's action lines and each belief (the local model; no
    LLM cost) and keeps the top ``behavioural_candidate_limit``. ``0`` disables
    the filter (every such belief competes).
    """

    enabled: bool = True
    behavioural_matching: bool = True
    max_behavioural_calls: int = Field(default=150, ge=0)
    behavioural_candidate_limit: int = Field(default=200, ge=0)


class UtilityConfig(BaseModel):
    """Usefulness (outcome-learning) lens config (composition).

    ``enabled`` gates whether the utility rank-lift is applied to projection /
    digest ranking at all (off ⇒ byte-for-byte the pre-0190 ranking; on with no
    utility evidence ⇒ also identical, since the bonus is ``+0`` at cold
    start). ``default`` is the store's local base utility rule; adopted lenses'
    ``utility_rules`` overlay it, most-skeptical-wins. ``mining`` configures the
    pass that produces the evidence.

    ``explicit_weight`` is what one operator gesture
    (``particles memory useful <id>``) is worth relative to one mined event. It
    lives here rather than on :class:`UtilityRuleConfig` because it parameterises
    *evidence production* in the local store, not the portable policy for
    *interpreting* another store's utility evidence — so it is deliberately not
    part of the lens ``utility_rules`` vocabulary.

    It must be well above 1.0 or the explicit channel cannot function: the miner
    emits one event per (belief, session) and accumulates tens unattended, while
    a gesture fires once and is capped at one credit per belief per principal per
    day. At ``1.0`` a deliberate press buys ``λ·ln 2`` of rank-lift against the
    ``λ·ln(1+R)`` a well-used belief earns for free — roughly a fifth of head
    entry on the dogfood store — so the verb would exist and change nothing.
    """

    enabled: bool = True
    default: UtilityRuleConfig = Field(default_factory=UtilityRuleConfig)
    mining: UtilityMiningConfig = Field(default_factory=UtilityMiningConfig)
    explicit_weight: float = Field(default=25.0, ge=0)


class OwnerLensConfig(BaseModel):
    """Read-time owner-relevance lens — the *aboutness* axis.

    The third read-time axis on the recall surfaces, alongside truth
    (``confidence.value`` × trust × decay) and use
    (``λ·ln(1+R)``). It adds ``ω · A(p)`` to the projection / digest
    **ranking** score, where ``A(p)`` is 1 when the belief is about the viewer
    and 0 otherwise. Promotion-only (``ω ≥ 0``), never folded into
    ``confidence.value`` or the displayed ``effective_confidence``, and never
    stored.

    ``subjects`` identifies **the viewer** — the party whose lenses are in
    effect for this read. It lives in the reader's config rather than in the
    store because viewer identity is reader-local: three contributors sharing
    one store each need their own, which a store-resident field would defeat.
    This is the single-viewer binding of the viewer
    seam, valid up to multi-tenant line.

    A **list**, not a scalar, because a viewer's Subject fragments in practice
    ("Jeff" / "Jeff Gage") until the N→1 merge lands. Entries are
    canonical names or Subject ids and resolve **locally only** — never a live
    authority lookup, since the digest is a zero-LLM, zero-network surface.
    Resolve-or-inert: if nothing resolves the
    lens is inert and the ordering is byte-identical to ``enabled: false``.

    ``rank_lift`` (``ω``) is **store-specific and must be calibrated** against
    the deployment's own confidence spread and cohort size — see
    ``particles quality rank-lift-sweep``. It ships ``0.0`` (inert) so the lens
    changes nothing until an operator sets both ``subjects`` and a swept ``ω``.
    """

    enabled: bool = True
    subjects: list[str] = Field(default_factory=list)
    rank_lift: float = Field(default=0.0, ge=0.0)


class ObserverScopeConfig(BaseModel):
    """Observer scope — the project as an observer.

    A belief's observer scope is derived at read time from the ``project:<key>``
    tags on the corpus entries its sources name; nothing is stored on the claim.
    This section holds the one piece of static policy that derivation needs.

    ``harness_tags`` are the tags a harness adapter puts on everything it
    harvests. A keyless corpus entry that carries one is **unattributed** —
    in view store-wide and for no project observer — rather than global: a
    stamping gap must fail closed, not make a harness's deposits visible
    everywhere. A keyless entry without one (a hand deposit, a web page, a
    user-level rule file) is global. Add your adapter's tag here when you wire
    a second harness.

    Which surfaces read through a project observer is the adapter's setting
    (``claude_code.observer_scope``), not this section's.
    """

    harness_tags: list[str] = Field(default_factory=lambda: ["claude-code"])


_CALIBRATION_SOURCE_VALUES = frozenset(
    {"EXTRACTOR_DIRECT", "AGENT_ASSERTED", "CALIBRATED_BENCHMARK", "HUMAN_REVIEW"}
)


class UncalibratedCapConfig(BaseModel):
    """Read-side cap on uncalibrated confidence values (opt-in).

    When ``enabled``, the ``confidence.value`` factor entering the
    ``effective_confidence`` formula is clamped to ``cap_value`` for any
    particle whose ``confidence.calibration_source`` is listed in ``sources``.
    This is a **read-side** ``min`` on the value fed into the formula — the
    stored, immutable ``confidence.value`` is never mutated. It
    composes with the trust-weight cap: a single particle can be
    subject to both (the value is capped, *then* multiplied by the
    extractor-trust / source-trust / recency factors).

    The default targets ``EXTRACTOR_DIRECT`` (raw, uncalibrated model output).
    Calibrated or human-assigned values (``CALIBRATED_BENCHMARK`` /
    ``HUMAN_REVIEW``) are absent from the default ``sources``, so they are never
    capped unless an operator explicitly opts them in. Default off, so behaviour
    is byte-for-byte unchanged until adopted.
    """

    enabled: bool = False
    cap_value: float = Field(default=0.7, ge=0.0, le=1.0)
    # Calibration sources the cap applies to; values must be members of
    # ``particles.core.scoring.confidence.CalibrationSource`` (typed as ``list[str]`` to
    # avoid a config → core import cycle — ``core.scoring.confidence`` imports this
    # module's ``get_config``). The default targets raw extractor output only.
    sources: list[str] = Field(default_factory=lambda: ["EXTRACTOR_DIRECT"])

    @field_validator("sources")
    @classmethod
    def _validate_sources(cls, value: list[str]) -> list[str]:
        unknown = [s for s in value if s not in _CALIBRATION_SOURCE_VALUES]
        if unknown:
            raise ValueError(
                f"confidence.uncalibrated_cap.sources contains unknown "
                f"calibration source(s) {unknown!r}; valid values are "
                f"{sorted(_CALIBRATION_SOURCE_VALUES)}"
            )
        return value


class ConfidenceConfig(BaseModel):
    """Read-side confidence-modulation operator policy."""

    uncalibrated_cap: UncalibratedCapConfig = Field(default_factory=UncalibratedCapConfig)


class ConformanceTrustCapConfig(BaseModel):
    """Read-side conformance → extractor-trust cap (opt-in).

    When ``enabled``, an extractor whose last persisted conformance run showed a
    *genuinely evaluable* REQUIRED failure (fixtures produced particles **and** a
    REQUIRED field fell short — never the zero-fixture "unknown" case) has its
    **effective** trust weight clamped to ``cap_value`` at query time. The stored
    ``ExtractorRow.trust_weight`` is never mutated, and conformance remains
    report-only as a *gate* (no CI / registration block) — this is purely an
    operator policy that reads the conformance status the validator persists.
    Default off, so behaviour is byte-for-byte unchanged until adopted.
    """

    enabled: bool = False
    cap_value: float = Field(default=0.5, ge=0.0, le=1.0)
    # Extractor ids the cap never applies to (an auditable operator override).
    exempt: list[str] = Field(default_factory=list)


class ConformanceConfig(BaseModel):
    """Conformance-validator operator policy."""

    trust_cap: ConformanceTrustCapConfig = Field(default_factory=ConformanceTrustCapConfig)


class DepositDateConfig(BaseModel):
    """Deposit-time content-date capture for local-file deposits.

    Populates ``content_published_at`` on the local-file / archival deposit path
    (``deposit_file`` / ``deposit_vault``) so an old document's particles are not
    all stamped at import time. Resolution precedence (highest wins): an explicit
    operator ``--date`` > a leading date line in the content > the file mtime.
    The URL / importer deposit paths set the field from source metadata and are
    unaffected by these knobs.
    """

    # Scan the head of the content for a standalone date line (e.g. a journal's
    # leading `2026-03-15`).
    detect_leading_date: bool = True
    # How many leading non-blank lines to scan for that date line.
    leading_date_scan_lines: int = 5
    # Fall back to the file's modification time when there is no `--date` and no
    # leading date. mtime is reset to "now" by copy / git-checkout / download, so
    # disable this when mining freshly-materialized trees.
    mtime_fallback: bool = True
    # `strptime` patterns tried in order against a candidate date line.
    formats: list[str] = Field(default_factory=lambda: ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"])


class SubjectsConfig(BaseModel):
    wikidata_rate_limit_rps: float = 2.0
    wikidata_cache_ttl_seconds: float = 86400.0
    wikidata_link_suppress_threshold: float = 0.25
    # external-authority candidates scored below this floor are not
    # attached at resolution time — the resolver cascade falls through to a
    # bare-local Subject (Step 3 → Step 4). The floor sits strictly below the
    # 0.5 "unscoreable" sentinel, so a link that could not be scored
    # still attaches as before; only scored-and-low links (the plausible-but-
    # wrong mislinks) are dropped. Generic over authorities — exact-identifier
    # authorities resolve at confidence 1.0 and are never abstained. Must be
    # ≤ wikidata_link_suppress_threshold (abstain ≤ suppress < trust).
    external_link_abstain_threshold: float = 0.15
    # which Wikidata search candidate an unqualified name takes.
    # `top_hit` takes the first usable hit, as the resolver did before 1.168.16
    # and as `llm_judge` still does for a name that is not ambiguous.
    # `best_description` scores every candidate the one search returns
    # (up to five; descriptions ride the response, so no extra call) against
    # the claim text with the local encoder, adopts the best only at or above
    # `wikidata_link_suppress_threshold`, and below it keeps the extracted name
    # with the candidate recorded as a low-confidence ref. Measured 2026-09-30
    # on prose-article-seed-001 v0.3.0, it scored worse than `top_hit` (subject
    # resolution accuracy 0.800 against 0.886, wrong links 6 against 3): a
    # longer, more specific description of the wrong entity outscores the right
    # one. Kept so that measurement is reproducible and a later selector can be
    # compared against both (docs/benchmarks/subject-resolution-2026-09-30.md).
    #
    # `llm_judge` (the default since 1.168.16) asks the model only
    # when the name is ambiguous:
    # several usable candidates, or a single one whose description scores below
    # `wikidata_link_suppress_threshold` against the claim (the pure test is
    # `ingest.authorities.wikidata_judge.is_ambiguous`). One call per such name,
    # never per candidate, on the `llm.subject_resolution` purpose: the claim
    # and each candidate's label and description go in, a QID or "none of
    # these" comes out. "None" leaves a bare local Subject. Every verdict is kept
    # in the probe-verdict ledger, keyed by name and claim, candidate set,
    # prompt version and model, so the same input resolves the same way. Every
    # other name takes the top hit, as `top_hit` does. Measured 2026-10-01 on the
    # same suite: accuracy 0.971 against 0.914 for `top_hit`, no correct link
    # lost, 14 of 35 names judged at about US$0.08 per 100 names on
    # claude-sonnet-4-6; on an ordinary-prose gold set, 0.898 to 0.918 against
    # 0.735, 45 of 49 names judged at about US$0.20 per 100 names, no correct
    # link lost on either (docs/benchmarks/subject-resolution-judge-2026-10-01.md).
    # With no LLM key, an open breaker, or an unusable reply it takes the top
    # hit, so `top_hit` is also what an offline store gets. Set `top_hit` to
    # resolve with no model call at all.
    wikidata_candidate_selection: Literal["top_hit", "best_description", "llm_judge"] = "llm_judge"
    # Output budget of one `llm_judge` call. The reply is one short
    # JSON object; a model that thinks before answering needs more.
    wikidata_judge_max_tokens: int = Field(default=200, gt=0)
    # How many Wikidata search hits the `llm_judge` is shown. The
    # ambiguity gate and every other selection value read the first five, as
    # before, so which names reach the model and how a name resolves without it
    # are unchanged; the extra hits ride the same one search request and go
    # through the prefix-expansion filter and its alias read like the first
    # five. Measured 2026-10-02 on ordinary-prose-001: the right item sat at
    # rank 6 for "Jest" (the test framework) and "Chelsea" (the club), which
    # five hits never showed the judge; at 7 hits both link in 10 of 10 runs,
    # for about 20 more input tokens per judged name (10 hits link the same
    # two for about 70 more). A deeper list also led the judge to link an
    # invented person to a given-name item, so given-name, family-name and
    # disambiguation items are no longer shown to it at any depth
    # (`wikidata_judge.judgeable_hits`;
    # docs/benchmarks/subject-resolution-judge-2026-10-02.md). The hits
    # are part of the verdict key, so changing this asks once more about each
    # judged name whose search returns more than five hits. Wikidata caps the
    # request at 50.
    wikidata_judge_search_limit: int = Field(default=7, ge=5, le=50)
    # `subjects find-duplicates`: two subjects are reported
    # as candidate duplicates when the max cosine similarity across their
    # {canonical_name} ∪ aliases embeddings is at or above this threshold.
    find_duplicates_similarity_threshold: float = 0.88
    # Source types that skip live-ontology authorities (Wikidata) during
    # resolution. Two reasons, both ending in the same treatment:
    #   - private referents by construction (chat-transcript harvests, personal
    #     journals — "the user's hamster", "Luna"): ~none of the names resolve,
    #     and the fruitless, rate-limited calls serialise the whole process;
    #   - bulk migration intake: a per-entity live lookup over a
    #     multi-thousand-entity export is slow and network-dependent, and — the
    #     decisive part — can rewrite ``canonical_name`` into something the
    #     migrating user never chose. Enrichment stays available through every
    #     other surface afterwards; it is deferred, not skipped.
    # Exact-identifier authorities (Numista / ISBN / DOI) are recognize-only and
    # unaffected. Override to widen or narrow the set.
    skip_live_authorities_source_types: list[str] = Field(
        default_factory=lambda: ["CONVERSATION", "JOURNAL", "MCP_MEMORY_EXPORT"]
    )
    # one persona per store for conversational sources. An
    # extractor names the speaker differently from session to session ("user",
    # "the user", "the speaker"), which minted a Subject per surface form and
    # split the persona's claims across them — so an update about "the user"
    # was never compared with a belief about "User" (measured: 13 of 31
    # residual stale answers in the live run). Any alias below
    # resolves to `persona_canonical_name` for these source types. A resolution
    # rule only: Subjects already split stay split (`subjects merge` is the
    # operator's tool), and an empty alias list disables it.
    #
    # The fold is only as good as the agreement between the two paths that use
    # it. It shipped off for one release (1.146.3) because it measurably cost
    # update supersessions — 1 / 3 / 0 against 13 / 10 / 13 unfolded, over three
    # fixed live extraction samples — and back on in 1.146.4 once that was
    # traced to its cause: the §6.6 precompute looked a candidate's subject up
    # by the extractor's *raw* surface form while the write path stored it
    # folded, found nothing, and skipped the rung entirely. With both paths
    # folding, the same three samples give 12 / 15 / 14 supersessions —
    # better than not folding at all, and better on recall too (0.963 against
    # 0.944 mean).
    #
    # The fold records each surface form it folds as an alias of the canonical
    # Subject, so a lookup by name resolves "user" or "the speaker"
    # without knowing about the fold, and `subjects show` lists what the
    # persona has been called. Those aliases are scoped the way the fold is:
    # a path that binds a particle to a Subject outside a persona source
    # (`persona_source_types`, or an entry tagged with `persona_source_tags`)
    # (extraction of a web page, an interchange import) never resolves a
    # persona form through them, so "I" or "User" from such a source still
    # becomes its own Subject. A caller that must agree with the write path
    # about a name uses `ingest.subject_resolver.find_existing_subject` rather
    # than folding by hand. Renaming `persona_canonical_name` on a folded store
    # wants a `subjects merge` of the old persona Subject into the new one.
    persona_aliases: list[str] = Field(
        default_factory=lambda: [
            "user",
            "the user",
            "speaker",
            "the speaker",
            "i",
            "me",
            "myself",
        ]
    )
    persona_canonical_name: str = "the user"
    persona_source_types: list[str] = Field(default_factory=lambda: ["CONVERSATION", "JOURNAL"])
    # a corpus entry carrying any of these tags is a persona source
    # whatever its source type. A Claude Code memory file is deposited as
    # LOCAL_MARKDOWN, since that is what it is on disk, but its
    # "I" and "the user" are the store's speaker exactly as a transcript's are.
    # Outside the fold the speaker resolved per file: "user" went to a live
    # Wikidata lookup and came back as "user account", "the user" did not, and
    # the two sessions of one person never met on a subject. A folded persona
    # form never goes to a live authority, whichever route reached the fold.
    # An empty list keys the fold on source type alone, as before 1.159.2.
    persona_source_tags: list[str] = Field(default_factory=lambda: ["memory-file"])

    @model_validator(mode="after")
    def _abstain_at_most_suppress(self) -> SubjectsConfig:
        """enforce abstain ≤ suppress.

        A suppress floor below the abstain floor would describe an empty
        "attach-but-flag" band — a misconfiguration (everything below suppress
        would already have been abstained at resolution time).
        """
        if self.external_link_abstain_threshold > self.wikidata_link_suppress_threshold:
            raise ValueError(
                "external_link_abstain_threshold must be <= "
                "wikidata_link_suppress_threshold (abstain <= suppress): "
                f"{self.external_link_abstain_threshold} > "
                f"{self.wikidata_link_suppress_threshold}"
            )
        return self


_DEFAULT_GATE_DISPOSITIONS: dict[str, Literal["suppress", "qualify"]] = {
    "self_vocabulary": "suppress",
    "version_number": "suppress",
    "reference_code": "qualify",
    "filename": "qualify",
    "snake_case": "qualify",
    "cli_command": "qualify",
}
_DEFAULT_RELINK_TIERS: tuple[Literal[1, 2, 3], ...] = (1, 2, 3)


class SubjectGateConfig(BaseModel):
    """Extraction-time non-entity subject gate.

    A general, deterministic lexical gate over non-entity token classes
    (self-vocabulary enums, version numbers, reference / doc-ID codes,
    filenames, CLI command strings, snake_case identifiers), applied to every
    extractor's candidates in the Extract pipeline before subject resolution.
    Since each class is either suppressed or qualified by the
    source's project (``dispositions``).
    """

    enabled: bool = True
    # Names that always pass the gate (operator override for false positives).
    allowlist: list[str] = Field(default_factory=list)
    # Class-D anchor: the leading token of a CLI command string (matched
    # case-sensitively), e.g. "particles subjects merge".
    cli_binaries: list[str] = Field(default_factory=lambda: ["particles"])
    # Source types whose candidates skip the gate entirely (§ binding
    # constraint). Code-domain extractors legitimately mint snake_case / dotted
    # code-symbol subjects (e.g. ``particles.core.scoring.confidence.effective_confidence``),
    # which the lexical gate would otherwise strip — keying the exemption on the
    # source type is what keeps the docstring extractor's subjects intact.
    exempt_source_types: list[str] = Field(default_factory=lambda: ["PYTHON_SOURCE"])
    # what the gate does with each token class. ``qualify`` keeps
    # the name as a Subject scoped by the source's project (the artifact
    # authority), and only when a project key is known; ``suppress`` drops it
    # as the original gate did. A class missing from the map is
    # suppressed, so setting every class to ``suppress`` restores the original
    # gate's behaviour exactly.
    dispositions: dict[str, Literal["suppress", "qualify"]] = Field(
        default_factory=lambda: dict(_DEFAULT_GATE_DISPOSITIONS)
    )
    # the recovery tiers ``subjects relink-gated`` applies by
    # default and the ``gated_subjects`` curation card covers — 1 the
    # ``extraction:gated_subjects`` record, 2 the structured claim's subject
    # term, 3 the claim's backtick spans. A tier joins this list only after its
    # 100-claim, >= 90% precision check on the store it will run on.
    relink_tiers: list[Literal[1, 2, 3]] = Field(
        default_factory=lambda: list(_DEFAULT_RELINK_TIERS)
    )


class AuthorityConfig(BaseModel):
    """Per-authority resolution policy.

    Keyed by ``namespace`` in the top-level ``authorities`` map. ``enabled``
    turns an authority on/off without code; ``priority`` overrides its built-in
    arbitration rank (lower wins). Wikidata's rate limit / cache TTL
    stay on ``SubjectsConfig`` for back-compat.
    """

    enabled: bool = True
    priority: int | None = None


class WikidataRankConfig(BaseModel):
    preferred: float = 0.99
    normal: float = 0.85
    deprecated: float = 0.30


class WikidataConfig(BaseModel):
    rank_confidence: WikidataRankConfig = Field(default_factory=WikidataRankConfig)


class RedditConfig(BaseModel):
    min_comment_score: int = 2
    # Raised 30→200; reddit now routes through the shared
    # chunked-extraction helper so larger comment counts no longer flood
    # a single LLM call.
    top_comment_count: int = 200
    # Raised 500→1000 for parity with the gist comment body limit.
    comment_body_limit: int = 1000


class HackerNewsConfig(BaseModel):
    # Maximum number of comments the importer walks per thread. Mirrors
    # ``reddit.top_comment_count`` in intent but enforced at fetch time
    # (each comment is a separate Firebase API call) rather than at
    # extraction time. Default 200 matches Reddit's cap.
    max_comments: int = 200
    # Threshold for including a comment in the prose handed to the LLM.
    # HN's Firebase API rarely exposes per-comment scores (only stories
    # carry ``score``), so this filter typically only fires for the
    # occasional graded item — kept for symmetry with reddit and to give
    # operators a knob when Firebase starts populating the field.
    min_comment_score: int = 1
    # Number of spaces inserted per depth level when rendering nested
    # comments. Two spaces matches HN's own UI indent and keeps the
    # rendered prose compact enough to fit large threads in one LLM call.
    comment_indent: int = 2


class MastodonConfig(BaseModel):
    # Total cap on context items (ancestors + descendants) the importer
    # keeps per thread. Ancestors are retained first (the reply chain UP
    # is typically short); remaining budget goes to descendants. Deep,
    # high-engagement threads are truncated and the extractor surfaces a
    # MASTODON_REPLY_LIMIT_HIT quality note. Mirrors
    # ``hackernews.max_comments`` in intent.
    max_replies: int = 200
    # Threshold for including a reply in the prose handed to the LLM,
    # based on the reply's ``favourites_count``. Mastodon is less
    # engagement-driven than HN — most replies legitimately have 0
    # favourites — so the default keeps everything. Raise to filter
    # low-signal noise on viral threads.
    min_reply_favourites: int = 0
    # Number of spaces inserted per depth level when rendering nested
    # replies. Two spaces keeps long threads compact enough for one LLM
    # call.
    reply_indent: int = 2


class GithubConfig(BaseModel):
    # Maximum number of gist comments (oldest first) passed to the LLM.
    gist_top_comment_count: int = 50
    # Maximum characters per gist comment body in the LLM prompt.
    gist_comment_body_limit: int = 1000
    # Maximum number of gist comments fetched at deposit time (paginated).
    # Raised 500→5000 since pagination now follows Link headers
    # and the cap exists only as an abuse-stop on mis-authenticated runs.
    # Set to 0 to disable the cap entirely.
    gist_max_comments: int = 5000
    # Minimum content-token count for a comment to be considered substantive
    # (drives the synthesis fallback so pleasantries don't generate subjects).
    gist_substantive_min_tokens: int = 5
    # If True, synthesize one CandidateParticle per substantive commenter not
    # already covered by LLM + overlap attribution. Off by default since
    # chunked LLM extraction now mines each comment for topical claims;
    # commenters who *still* aren't represented after extraction tend to be
    # ones whose comments contained no extractable claim (vague experience
    # reports, generic encouragement). Enable for archival / completeness
    # use cases where every substantive commenter must have a vault page.
    gist_synthesize_commenter_particles: bool = False
    # Chunked-extraction thresholds (gist_single_call_threshold_chars,
    # gist_comment_chunk_chars, gist_max_llm_calls) moved to
    # ``extraction.*`` — they are now shared with the reddit
    # extractor. Use config.extraction.single_call_threshold_chars,
    # config.extraction.comment_chunk_chars, and
    # config.extraction.max_llm_calls_per_source instead.


class QueryConfig(BaseModel):
    default_top_k: int = 40
    default_min_confidence: float = 0.0
    # minimum effective_equivalence for a CO_EVIDENTIAL edge to
    # collapse a pair at query time. 0.0 reproduces pre-0106 behaviour (collapse
    # on any link); raise it to require stronger same-claim evidence before
    # merging (e.g. discount weak AUTO_CLUSTER_V1 cosine links).
    equivalence_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    # Combined-score weights for §9.3 ranking:
    #   combined = similarity_weight × cosine_sim + confidence_weight × eff_conf
    # The two need not sum to 1.0, but keeping them normalized makes the
    # combined score comparable across configs.
    similarity_weight: float = Field(default=0.6, ge=0.0)
    confidence_weight: float = Field(default=0.4, ge=0.0)
    # rank-time demotion for NARRATIVE particles. A narrative's
    # combined score is multiplied by this weight at sort time (NOT a discount on
    # its reported effective_confidence) so a richly-linked NARRATIVE label
    # doesn't dominate top-k (§Harder). 1.0 = no demotion (old behaviour).
    narrative_rank_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    # Truncation-warning heuristic: warn that top_k may be too low when the
    # combined-score gap between the last rendered result and the first
    # excluded one is below truncation_min_gap, OR when more than
    # truncation_near_count excluded particles scored within
    # truncation_near_margin of the cutoff.
    truncation_min_gap: float = Field(default=0.05, ge=0.0)
    truncation_near_margin: float = Field(default=0.15, ge=0.0)
    truncation_near_count: int = Field(default=5, ge=0)
    # when the max raw cosine similarity over the rendered top-k is
    # below this floor, the semantic query answers deterministically that the
    # store holds nothing relevant (no LLM call; hits still returned, labelled
    # as nearest-but-likely-unrelated). Raw similarity only — never the
    # combined score, whose confidence term is the pollution being detected.
    # The default is calibrated to the reference embedding profile
    # (all-MiniLM-L6-v2; measured off-topic band ≤ 0.15, on-topic ≥ 0.6);
    # re-examine on a non-reference profile. 0.0 disables the gate.
    relevance_floor: float = Field(default=0.25, ge=0.0, le=1.0)
    # Output budget for the §9.3 NL answer call. This is a *total output*
    # allowance on the wire, not a response-length cap: an extended-thinking
    # model spends its thinking tokens from the same budget, so a value sized
    # to the prose alone returns a reply with no text block at all and the
    # answer degrades to the deterministic listing. Moved here from the
    # misfiled ``extraction.query_max_tokens`` (whose 1024 default did exactly
    # that on `claude-sonnet-5`); the old key is still honoured for a
    # deprecation cycle. 4096 matches the memory benchmark's answer budget,
    # which was raised for this same failure class.
    answer_max_tokens: int = Field(default=4096, gt=0)
    # One retry of the answer call at this larger budget when the first came
    # back with no text block — the deterministic budget failure, which is the
    # only failure worth re-issuing: an identical call at an identical budget
    # reproduces it, while a bigger budget is a different call. Never fires on
    # a billing/network/refusal failure, which degrades immediately as before.
    # 0 disables the retry. Costs nothing on a query that answered.
    answer_retry_max_tokens: int = Field(default=16384, ge=0)
    # compose grounded answers by default. In grounded mode the
    # composer cites retrieved particle ids for every sentence and tags an
    # uncited one as its own inference or background; the parser validates
    # each cited id against the retrieved set and records the labels on
    # ``QueryResponse.answer_attribution``. Nothing is dropped and nothing is
    # stored. A request's ``grounded`` field overrides this per call. The
    # default follows the leakage measurement recorded in
    # docs/benchmarks/leakage.md.
    grounded_answers: bool = False


class ContestednessConfig(BaseModel):
    """Per-claim contestedness tuneables.

    Contestedness is the max−min spread of a claim's ``effective_confidence``
    across the viewer's policy set (local policy + each adopted lens). The metric
    itself is render-threshold-free in the query envelope; these knobs gate only
    the *rendered* surfaces — the prose ``[!contested]`` callout and the lint
    store-level distribution — so a faint spread does not clutter every page.
    """

    # prose exporters render a [!contested] callout, and the lint
    # distribution counts a claim as "highly contested", only when its spread is
    # at least this value. The envelope always carries the raw reading regardless
    # — thresholds are a renderer concern, not an envelope concern (§4). The
    # composed badge reuses this same value as its divergence-basis gate
    # (deliberately not a second threshold).
    callout_threshold: float = Field(default=0.2, ge=0.0, le=1.0)

    # master switch for composing and attaching the contested badge
    # on the read surfaces (query envelope, digest, MEMORY.md projection, MCP,
    # CLI). Default on — a disclosure surface that is off by default is not
    # surfacing. Off restores the pre-badge per-instrument behavior exactly.
    badge_enabled: bool = True


class LintConfig(BaseModel):
    """Lint detector tuneables (§9.4).

    The legacy ``lint.co_evidential_candidate_threshold`` key migrated to
    ``links_suggest.candidate_threshold`` and is handled by
    ``_migrate_legacy_keys`` — it is not a field here.
    """

    # CONFIDENCE_DECAY: flag an ACTIVE EPISTEMIC particle whose
    # confidence.variance has grown past this threshold (read-only finding;
    # no status change).
    variance_threshold: float = Field(default=0.15, ge=0.0)

    # L-SEM-01 cross-source contradiction detection: a candidate
    # pair is sent to the LLM contradiction probe only when the two particles'
    # embeddings are cosine-close at or above this threshold. The gate bounds
    # the store-wide candidate set so the check does not pay an O(n²) LLM cost
    # (only the cosine comparison is O(n²)). 0.6 sits in the gap between the
    # llm_wiki_vault planted-conflict pairs (C1–C3 ≈ 0.79–0.87) and unrelated
    # controls (≈ 0.14–0.37); lower catches subtler conflicts at higher LLM
    # cost. The S1 staleness pair (≈ 0.40) is below the gate by design — it is
    # a recency problem, not an embedding-near contradiction (§
    # Deferred).
    contradiction_candidate_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    # RECENCY_DECAY: flag an ACTIVE particle whose effective_confidence is
    # materially reduced by content age alone (decay; surfaced in lint).
    # A particle fires when 1 - recency_factor >= this threshold —
    # i.e. age alone has discounted its confidence by at least this fraction.
    # Default 0.5 = flag once age has at least halved the recency multiplier.
    # Read-only WARNING; never flips status. Source types with no decay config
    # (recency_factor == 1.0) or whose floor exceeds 1 - threshold never fire.
    recency_decay_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class ExporterCommonConfig(BaseModel):
    """Options that apply to every shipped exporter.

    The cross-exporter contract: every shipped exporter MUST honor
    ``min_particle_confidence`` by dropping particles whose
    ``effective_confidence`` falls below the threshold *before* any per-
    exporter downstream step (prompt input, cache hash, count-based
    ``min_particles`` check, rendered output, references). Default 0.0
    keeps every existing invocation backwards-compatible.
    """

    # Drop particles with effective_confidence below this threshold from
    # every shipped exporter's output. 0.0 = no filter (default). See
    # (the cross-exporter contract) for semantics.
    min_particle_confidence: float = 0.0

    # Minimum particle count for which the LLM synthesis path
    # (``--with-synthesis``) runs in the prose exporters (Obsidian,
    # Logseq). Subjects with fewer particles still get a rendered
    # page but the synthesis step is skipped — paraphrasing a
    # single claim adds no value over the structural audit trail
    # and burns an LLM call per subject. Set to 1 to synthesise
    # every subject regardless of count. Hoisted from
    # ``obsidian.synthesis_min_particles`` in 0.42.1 so the Logseq
    # exporter honours the same gate.
    synthesis_min_particles: int = 3


class GraphConfig(BaseModel):
    """Scoped epistemic graph view.

    The anti-hairball caps: every graph render is a scoped subgraph, and these
    bounds are enforced exporter/server-side. When a cap binds, the render
    carries a bounded-view disclosure naming the knob — a capped view is a
    disclosed lower bound, never a silent truncation.
    """

    # Maximum Subject nodes per render. Truncation drops lowest-rank nodes
    # first (hop distance then support for subject scope; retrieval rank for
    # query scope) and is disclosed in the rendered banner + census.
    max_nodes: int = Field(default=150, ge=1)
    # Maximum single-subject particles listed in one node's detail panel,
    # by descending effective confidence (mirrors the MCP hot-subject cap).
    max_particles_per_subject: int = Field(default=50, ge=1)
    # Upper bound on the --hops neighbourhood radius for subject scope.
    max_hops: int = Field(default=2, ge=1)
    # Retrieval-set size for query scope (`export graph --query`), bounded by
    # the query pipeline's own top_k ceiling.
    query_top_k: int = Field(default=25, ge=1, le=200)


class ObsidianConfig(BaseModel):
    # Minimum number of extracted particles a subject must have to appear in the export.
    min_particles: int = 0
    # Minimum number of graph links (incoming + outgoing) a subject must have.
    # Subjects below this threshold are suppressed as isolated nodes.
    min_links: int = 1
    # Default output directory for `particles export obsidian` when the
    # operator omits the path argument. `~` is expanded to the home
    # directory. None (the unset default) means the CLI requires an
    # explicit path. Typical operator value, a folder of its own inside the
    # vault (exporting into a vault root that holds your own notes needs
    # `--force` on the first run):
    #   ~/Library/Mobile Documents/iCloud~md~obsidian/Documents/MyVault/Particles
    default_output_path: str | None = None
    # ``synthesis_min_particles`` moved to ``exporter_common`` in 0.42.1
    # so the Logseq exporter honours the same gate. Read it via
    # ``get_config().exporter_common.synthesis_min_particles``.
    # when true (default), `export obsidian --with-synthesis` also emits
    # one note per ACTIVE NARRATIVE under `Narratives/`, rendered as cited prose.
    # Set false to keep only per-subject articles.
    emit_narrative_notes: bool = True


class InboxConfig(BaseModel):
    """URL inbox for the iOS-Share-Sheet → iCloud → Mac workflow.

    Operators share URLs from Safari / Reddit / etc. via an iOS Shortcut
    that appends each one to a plain-text file in iCloud Drive. The Mac
    polls that file (``particles inbox watch`` or ``inbox process``)
    and deposits each pending URL through the regular
    :func:`particles.corpus.deposit.deposit_url` flow. See
    ``docs/cli.md`` § Inbox for the Shortcut setup steps.
    """

    # Absolute path (``~`` expanded) to the inbox file. None means the
    # `particles inbox` commands will refuse to run until configured.
    # The file is auto-created on first append by the iOS Shortcut and
    # auto-rewritten by the processor (atomic write-then-rename so
    # iCloud sync doesn't see a half-written file).
    file_path: str | None = None
    # `inbox watch` poll cadence in seconds. The processor is cheap
    # (one mtime check + a file read if changed) so the default leans
    # toward fresh.
    poll_interval_seconds: int = 30


class WikiConfig(BaseModel):
    """Wiki-article exporter tuneables.

    A subject is rendered as a standalone article only when it has at least
    ``min_particles`` ACTIVE particles — single-claim subjects add no value
    over the Obsidian listing export.
    """

    # Minimum ACTIVE particles a subject must have to qualify for an article.
    min_particles: int = 3
    # Per-article output budget (LLM max_tokens). 4096 fits a typical wiki
    # article comfortably; longer subjects spill into multi-paragraph form.
    max_tokens: int = 4096
    # Encyclopedic tone vs conversational. Encyclopedic is the spec's default
    # and what reviewers expect; flip for domain-tailored runs.
    encyclopedic_tone: bool = True
    # When False, the semantic-alignment LLM-judge (Layer B)
    # is skipped — Layer A's regex ID-membership check is the only safety
    # net and the article frontmatter records this. Operators trading
    # cost for safety can set this False.
    layer_b_enabled: bool = True
    # Maximum fraction of an article's citations that may be flagged
    # `unrelated` by the Layer B judge before the article fails.
    # 0.0 means a single ornamental citation fails the article (strict);
    # 1.0 means only `contradicts` verdicts ever fail (most lenient).
    # 0.30 is the default — calibrated to encyclopedic prose where some
    # citation stuffing is expected from the LLM and the judge itself
    # is noisy. Lowering tightens the contract; raising loosens it.
    # Any `contradicts` verdict still hard-fails the article regardless.
    layer_b_unrelated_tolerance: float = 0.30
    # Whether to attempt a second LLM-synthesis pass after a Layer B
    # failure (amendment). Default False because operator
    # dry-runs after shipped showed 0% recovery rate on the
    # retry path: the strict Layer-B-specific prompt either produced
    # output equivalent to attempt 1 (same misalignments) or regressed
    # to zero-citation output that Layer A then rejected — strictly
    # worse than falling back to the structured listing immediately.
    # Layer A retries (correcting invented IDs / zero-citation bodies)
    # work fine and remain unconditionally enabled. Operators who want
    # to spend the LLM budget on the long-shot Layer B retry can set
    # this to True.
    layer_b_retry_enabled: bool = False
    # when true (default), the wiki export also emits one cited
    # article per ACTIVE NARRATIVE under `Narratives/`, rendered by the same
    # path the Obsidian narrative notes use. Set false to
    # keep only per-subject articles. Suppressed automatically when the run is
    # narrowed with `--subjects` (narratives are subject-less).
    emit_narrative_notes: bool = True


class LogseqConfig(BaseModel):
    """Logseq exporter tuneables.

    The Logseq exporter otherwise shares the cross-exporter knobs
    (``exporter_common.min_particle_confidence`` /
    ``exporter_common.synthesis_min_particles``) and reads the article budget
    from ``wiki`` (``max_tokens`` / ``layer_b_enabled``), so this section holds
    only what is genuinely Logseq-specific.
    """

    # when true (default), `export logseq --with-synthesis` also emits
    # one page per ACTIVE NARRATIVE in Logseq's `Narratives/` page namespace
    # (on disk: `pages/Narratives___<slug>.md`), rendered as cited prose via the
    # path. Mirrors `obsidian.emit_narrative_notes`.
    emit_narrative_notes: bool = True


class NotionConfig(BaseModel):
    """Notion exporter tuneables — **non-secret only**.

    The integration token is a SECRET and lives in ``secrets.py``
    (``NOTION_API_KEY``), read via :func:`particles.secrets.get_notion_api_key`,
    **never here** and never in ``config.yaml``. Everything on this
    model is an ordinary, non-secret operational parameter — which database to
    sync into and what to name its properties. A Notion database id is not a
    secret: it is a workspace-scoped identifier, useless without the shared
    integration token.
    """

    # The Notion database id the exporter syncs subjects into (one row per
    # subject). None means ``export notion`` refuses to run until
    # configured, unless ``--database-id`` is passed per-invocation.
    database_id: str | None = None
    # Property name on the target database that stores the Particles subject id
    # (the idempotent-upsert key). Re-sync queries the database
    # for an existing row carrying this id before creating one, so re-running
    # updates rather than duplicates. Must be a property that already exists on
    # the operator's database (a rich-text or title property).
    subject_id_property: str = "Particle Subject ID"
    # Sentinel heading text the exporter writes at the top of the managed block
    # range in each subject page. On re-sync the exporter owns —
    # and overwrites — every block from this heading to the end of the page
    # (default), so a re-sync drops stale particles and adds new ones. The
    # ``--no-update-blocks`` opt-out creates a page's blocks once and never
    # rewrites them, preserving any hand-edits below the heading.
    managed_block_heading: str = "Particles (managed — do not edit below)"


class AutoMergeConfig(BaseModel):
    """Exact-duplicate auto-merge — the one store-mutating curation path.

    Applies **only** to Tier A: byte-identical ACTIVE content within a Subject,
    decided by content hash with no LLM verdict and no similarity threshold.
    Every near-duplicate stays advisory exactly as it is left today.
    """

    # Default OFF, permanently: a stock install never auto-mutates a store.
    # Turning this on is an operator decision made against the
    # § Context measurement, and it is the only way `links dedup --apply` is
    # permitted to write.
    enabled: bool = False
    # Cap on **groups** merged per run (one group = one content hash = one
    # event). When the cap binds, the report discloses the remaining
    # group / redundant-copy counts so a capped run never reads as a complete
    # cleanup. Re-run to continue.
    max_per_run: int = 500


class LinksSuggestConfig(BaseModel):
    # cosine-similarity threshold for proposing CO_EVIDENTIAL link
    # candidates between same-Subject particles via `particles links suggest`.
    # A higher threshold (closer to 1.0) is more conservative — only obvious
    # paraphrases are surfaced. A lower threshold catches looser matches at the
    # cost of more candidates to review. 0.92 is the default informed by the
    # ADR-0033 embedding model's typical paraphrase-detection floor.
    # (Renamed from lint.co_evidential_candidate_threshold in 0.46.0; the old
    # key is still accepted for one minor cycle with a deprecation warning.)
    candidate_threshold: float = 0.92
    # per-Subject candidate clusters larger than this fan out across
    # multiple LLM-judge calls so a single prompt never exceeds the token
    # budget. The fan-out heuristic chunks to fit and never splits a transitive
    # cluster; operators get a WARNING in the SuggestReport when it fires.
    max_cluster_size: int = 50
    # `--apply` targeting more than this many pairs requires an
    # explicit `--yes` so a stray invocation can't link thousands at once.
    apply_confirm_threshold: int = 10
    # exact-duplicate auto-merge. Default OFF.
    auto_merge: AutoMergeConfig = Field(default_factory=AutoMergeConfig)


class VocabularyConfig(BaseModel):
    """The vocabulary document's proposal step (§4)."""

    # cosine similarity at which `vocab propose` clusters the
    # normalised forms of one subject class into an alias candidate (average
    # linkage). 0.85 is the measured setting of the 2026-10-01 canonicalisation
    # page; lower finds more candidates and more wrong ones for a reviewer.
    alias_similarity: float = Field(default=0.85, gt=0.0, le=1.0)
    # How many candidates of each kind (alias, profile) one `vocab propose`
    # run records as proposals, ranked by the claims each would cover.
    propose_limit: int = Field(default=30, ge=1)


class CitationSignalConfig(BaseModel):
    """Citation-signal deposit suggestions.

    Track URLs mentioned across the corpus (including undeposited ones) and
    rank the undeposited ones — by trust-weighted distinct-source diversity ×
    recency — as operator deposit suggestions. Suggestion-only, never
    auto-deposit.
    """

    # Master switch for harvesting URL mentions at extraction time. Off means
    # no new mentions are captured (existing rows + suggestions stay readable).
    capture_enabled: bool = True
    # A URL must be cited by at least this many *distinct* sources to surface
    # as a suggestion — raw frequency is gameable (one spammer), so diversity
    # is the floor.
    min_distinct_sources: int = 2
    # The lint finding (L-CITE-01) is deliberately more conservative than the
    # verb: it only fires for URLs cited by at least this many distinct sources.
    lint_min_distinct_sources: int = 3
    # Maximum suggestions returned by `corpus links suggest` / the lint check
    # by default — the backlog is rank-capped.
    rank_cap: int = 20
    # Exponential decay half-life (days) for a citation's recency weight, and
    # the floor it never decays below. Recent citations rank above stale ones
    # without a single old high-trust citation ever vanishing.
    recency_half_life_days: float = Field(default=180.0, gt=0.0)
    recency_floor: float = Field(default=0.10, ge=0.0, le=1.0)
    # Boilerplate guard: drop a URL whose only citing sources are from the URL's
    # own host (site-internal nav / footer links — "about", "privacy", the
    # site's own other articles), keeping cross-site citation signals. A
    # genuinely viral primary source is cited from *other* domains, so this
    # never filters the signal the feature exists to surface.
    filter_site_internal: bool = True


_DEFAULT_REFETCH_FLOORS: dict[str, int] = {
    "WEB_PAGE": 3600,
    "FORUM": 3600,
    "BLOG": 3600,
    "ACADEMIC_PAPER": 604800,
    "PDF": 604800,
    "DATA_EXPORT": 86400,
    "CSV": 86400,
    "CONVERSATION": 0,
    "LOCAL_FILE": 0,
    "LOCAL_MARKDOWN": 0,
    "GITHUB_REPO": 3600,
    "GITHUB_GIST": 3600,
    "GITHUB_PAGES": 3600,
}


class LocalRefreshConfig(BaseModel):
    """The local-source refresh tier — change detection for ``file://`` entries.

    The gate on *which* entries are refreshed is not here: it is the
    ``fetch_policy = LAZY`` flag on the entry itself, so refreshing stays an
    operator promise made per source at deposit time rather than a global
    switch. These knobs bound the sweep, not its membership.
    """

    # The consolidation pass on/off. The pass is zero-LLM, so unlike
    # every other semantic pass it also runs on a --structural-only night.
    enabled: bool = True
    # Per-run cap on entries checked, oldest-entry-first so a capped run makes
    # round-robin progress instead of re-checking the same head every night.
    max_entries: int = Field(default=200, ge=0)
    # Follow symlinked sources. ``deposit_file`` records
    # ``path.resolve().as_uri()``, so a path that was *already* a symlink at
    # deposit time is stored as its target and is unaffected by this knob. What
    # it governs is the path swapped for a symlink afterwards: a change of
    # *identity* rather than of content, which an unattended pass should decline
    # to follow rather than silently ingest.
    follow_symlinks: bool = False


class RuleSourcesConfig(BaseModel):
    """The rule-source set — which local documents the store tracks.

    The sibling of :class:`LocalRefreshConfig`, and the pairing is the point:
    **this section is membership, ``local_refresh`` is cadence.** A file
    registered here is deposited ``MUTABLE`` + ``LAZY``, which is the whole
    integration with the loop — change detection, the generation
    cascade and consolidation pass 0.5 are all 0206's and are not duplicated.

    Motivating measurement: the store held 34 particles *about*
    the never-prepend-``export PATH`` rule, mined from conversations that
    discussed it, and not one stating the rule. Conversations about rules yield
    claims about rules; only the rule document yields the rule.
    """

    enabled: bool = True
    # Files or directories to track. ``~`` and ``$VAR`` are expanded; a
    # directory is walked for ``filenames``. EMPTY ⇒ discover (the nearest
    # ancestor of the working directory containing a ``.git`` entry, plus
    # ``~/.claude``). A non-empty list disables discovery and is taken
    # literally, so an operator who pins the set gets exactly the set.
    paths: list[str] = Field(default_factory=list)
    # Basenames that count as a rule document when walking a directory.
    filenames: list[str] = Field(default_factory=lambda: ["AGENTS.md", "CLAUDE.md"])
    # Walk depth relative to each registered root (0 = the root itself only).
    max_depth: int = Field(default=4, ge=0)
    # Hard cap on one resolution. Truncation is always disclosed, never silent.
    max_files: int = Field(default=200, ge=0)
    # Directory names skipped anywhere in the walk. ``worktrees`` is here for a
    # measured reason: an agent worktree is a full checkout carrying its own
    # copy of every rule document, so walking one registers transient copies
    # whose files vanish with the worktree and whose content is either
    # byte-identical to the canonical file (noise) or a branch's uncommitted
    # draft (wrong). The canonical checkout is the source; copies are not.
    exclude_dirs: list[str] = Field(
        default_factory=lambda: [
            ".git",
            ".venv",
            "node_modules",
            "site-packages",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            "worktrees",
            "site",
            "build",
            "dist",
        ]
    )


class McpWriteConfig(BaseModel):
    """Write-surface policy for the MCP server (§5/§6).

    Default-deny: ``enabled_stores`` is empty, so a stock install accepts no
    MCP writes at all. The asserting identity is server-bound (not a per-call
    argument), agent-asserted confidence is clamped, and cross-asserter
    mutation is off by default.
    """

    # Allowlist of store handles writable over MCP. Empty = no MCP writes.
    enabled_stores: list[str] = Field(default_factory=list)
    # Seeded AUTHOR-scoped trust_rank for a new asserter identity (§6/§6a).
    agent_trust_rank: float = Field(default=0.8, ge=0.0, le=1.0)
    # Whether retract/supersede may target another identity's particles (§6).
    allow_cross_asserter: bool = False
    # The server-bound asserting principal stamped onto asserted_by / author_id
    # / event actor — NOT a per-call client argument (§4a/§6, M3).
    asserter_identity: str = "mcp:claude-code"
    # Ceiling on agent self-reported confidence.value, clamped at construction
    # (§4a). An agent cannot self-report full certainty.
    max_asserted_confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    # Claim-granularity soft-gate (§3.3): reject a compound /
    # multi-claim assertion before it is constructed. A size proxy, not a
    # semantic check — interim. The COMPOUND_ASSERTION lint reads
    # the same knobs so the gate and the lint cannot drift. 0 disables a check.
    max_assertion_chars: int = Field(default=320, ge=0)
    max_assertion_sentences: int = Field(default=3, ge=0)


class McpRecallConfig(BaseModel):
    """Session-start recall surface for the MCP server.

    The compiled memory digest (``particles://digest/<store>``) is rendered on
    demand — no cache (it has no LLM cost, so the synthesis-cache
    pattern would only add staleness). ``digest_stores`` lists *additional*
    stores whose digest is enumerated in ``resources/list`` beyond the
    write-enabled memory stores; the resource *template* addresses any store
    regardless. ``digest_max_beliefs`` caps the rendered index (top-N by
    effective confidence) so the artifact stays within a client's context
    budget; truncation is disclosed in the footer (no silent cap).
    """

    # Extra store handles whose digest is listed beyond the write-enabled set.
    digest_stores: list[str] = Field(default_factory=list)
    # Max beliefs rendered per digest (top-N by effective confidence). 0 = no cap.
    digest_max_beliefs: int = Field(default=200, ge=0)


class McpMemoryCompatConfig(BaseModel):
    """Reference memory-server compatibility façade.

    The façade mirrors ``@modelcontextprotocol/server-memory``'s tool surface
    so an existing client works unmodified. These knobs govern the three places
    where a faithful mirror would be harmful on a real store: the uncapped
    ``read_graph`` dump, the 320-char agent-write granularity ceiling (a
    reference observation has no length limit), and the reference's
    purely-substring search.
    """

    # The server-bound asserting principal for façade writes, distinct from the
    # native surface's ``mcp.write.asserter_identity`` so façade-origin claims
    # stay separately attributable.
    asserter_identity: str = "mcp:memory-compat"
    # Confidence stamped on façade-asserted observations and relations. Clamped
    # by ``mcp.write.max_asserted_confidence`` like any other agent write.
    asserted_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    # `read_graph` caps. The reference server dumps the whole
    # graph; on a real store that is megabytes of JSON into the model's context.
    # Truncation is always disclosed in an appended content block, never silent.
    # 0 = no cap.
    read_graph_max_entities: int = Field(default=250, ge=0)
    read_graph_max_observations_per_entity: int = Field(default=25, ge=0)
    # `search_nodes` / `open_nodes` result cap (same disclosure rule). 0 = no cap.
    search_max_entities: int = Field(default=100, ge=0)
    # Granularity ceiling for a façade observation. Deliberately far above
    # ``mcp.write.max_assertion_chars`` (320): the reference contract has no
    # length limit and `read_graph` must return the exact string, so an
    # observation is never truncated or split. 0 disables.
    max_observation_chars: int = Field(default=4000, ge=0)
    # Union semantic recall into `search_nodes` alongside the reference's
    # substring match. Default off: turning it on changes result sets a
    # reference client never asked to change, and costs an embedding model
    # where the reference needs nothing.
    semantic_augmentation: bool = False


class McpConfig(BaseModel):
    write: McpWriteConfig = Field(default_factory=McpWriteConfig)
    recall: McpRecallConfig = Field(default_factory=McpRecallConfig)
    memory_compat: McpMemoryCompatConfig = Field(default_factory=McpMemoryCompatConfig)


class ClaudeCodeHarvestConfig(BaseModel):
    """Write-side (SessionEnd harvest) knobs for the Claude Code hooks (§3/§4/§7)."""

    # Harvest distilled session transcripts. False ⇒ memory-file harvest only
    # (the "transcript-free beliefs" posture).
    transcripts: bool = True
    # Allow the SessionEnd harvest to ship material to a remote engine
    # (``engine.base_url``). Off by default: transcripts are the most sensitive
    # payload the SDK touches, so leaving the machine is an explicit opt-in.
    # The refusal is logged; the catch-up sweep back-fills once enabled.
    allow_remote: bool = False
    # Extract deposited entries inside the hook (LLM-priced). Default deferred:
    # the hook only deposits; extraction runs on the store's schedule.
    extract_inline: bool = False
    # Ceiling on inline extractions per SessionEnd run (current session first).
    max_extract_entries_per_session: int = Field(default=3, ge=0)
    # How many recent transcripts the level-triggered catch-up sweep re-checks
    # after handling the current session. 0 disables the sweep.
    catchup_limit: int = Field(default=5, ge=0)


class ClaudeCodeConfig(BaseModel):
    """Claude Code hook integration.

    Read by the ``particles hook session-start`` / ``session-end`` verbs that
    ``particles init claude-code`` installs into Claude Code's settings. The
    read side pushes the digest into context at session start; the
    write side harvests the session's transcript + memory files at session end.
    """

    # Per-machine state directory for the integration: the hook
    # log lives here, and the projection keeps its manifest/snapshot
    # here. ``~`` expands to the user's home directory.
    state_dir: str = "~/.particles/claude-code"
    # Hook invocation log (JSONL, one line per hook run).
    # None ⇒ ``<state_dir>/hooks.jsonl``. Deliberately a local file, not the
    # operator event log: its most important entries are written when
    # the store is unreachable.
    hook_log_path: str | None = None
    # Byte-level guard on the injected digest, truncating on a line boundary
    # with a disclosed footer. Complements
    # ``mcp.recall.digest_max_beliefs`` (a belief line has no fixed width).
    # 0 = no byte cap.
    digest_max_bytes: int = Field(default=24_000, ge=0)
    # Internal deadline for a hook run, far under Claude Code's own 600 s hook
    # timeout. On expiry the hook logs and exits 0 with no output — a memory
    # outage must cost an empty digest, never a hung session start.
    hook_deadline_seconds: float = Field(default=10.0, gt=0.0)
    # What a session's digest and MEMORY.md region see. ``store`` is
    # the whole store, as before. ``project`` reads through the session's
    # project observer: global beliefs plus those observed in this project.
    # Engages only once ``particles memory rescope`` has run on the store.
    observer_scope: Literal["store", "project"] = "store"
    harvest: ClaudeCodeHarvestConfig = Field(default_factory=ClaudeCodeHarvestConfig)


class AgentMemoryProjectionGitConfig(BaseModel):
    """Optional git-versioned history of the projected ``MEMORY.md`` view.

    When ``enabled`` **and** the projection target is inside a git repo, each
    render that changes files under the memory directory is committed with a
    structured message (run id + ranking-delta summary), giving operators a
    diffable, rollback-able history of the *view* while the store stays truth.
    Off by default: committing into an operator's repo is opt-in. Every git
    failure degrades silently — the commit is a bonus, the projection is the
    product.
    """

    # Master switch. Off ⇒ the projection never touches git.
    enabled: bool = False
    # GPG signing. False (default) passes ``--no-gpg-sign`` so an unattended
    # SessionEnd-hook commit never blocks on a signing agent; True drops the
    # override and lets the operator's own ``commit.gpgsign`` config decide.
    # This SDK's GPG requirement is never imposed on the operator's repo, and
    # a signing failure never fails the projection.
    sign: bool = False
    # Per-commit author identity, passed via ``-c user.name/-c user.email``
    # (never written into the operator's config). None ⇒ use the operator's
    # own git identity; when that is absent the commit degrades silently.
    author_name: str | None = None
    author_email: str | None = None
    # Upper bound on the added/removed excerpt lines in the commit message; the
    # count line always states the true totals so a large delta isn't silently
    # truncated.
    max_delta_excerpts: int = Field(default=6, ge=0)


class AgentMemoryProjectionConfig(BaseModel):
    """The MEMORY.md projection — a drift-gated cited view of the memory store.

    Namespace shared with the hook integration: the hooks move the
    bytes; this decides what the projected ``memory-index`` region says. The
    manifest wins wherever both speak — ``max_lines`` / ``min_confidence``
    here are **init-time defaults** baked into the ``memory.yaml`` that
    ``particles init claude-code`` writes; editing the manifest afterwards is
    the supported tuning path.
    """

    # Master switch: render + splice the memory-index region during the
    # SessionEnd harvest cycle, and run the SessionStart trailer freshness
    # check. Off ⇒ the behaviour exactly (full digest push, no
    # region writes). The sentinel strip at harvest stays active either way —
    # the corpus must never contain the store's own rendered output.
    enabled: bool = True
    # Path of the projection manifest. None ⇒ `<claude_code.state_dir>/memory.yaml`
    # (written by `particles init claude-code` when absent).
    manifest: str | None = None
    # Init-time manifest defaults (§3/§5) — mirrored into the
    # generated memory.yaml, not read at render time (the manifest wins).
    max_lines: int = Field(default=120, ge=1)
    min_confidence: float = Field(default=0.30, ge=0.0, le=1.0)
    # Fold-and-archive (default-on): after a cycle's harvest of
    # MEMORY.md succeeded, agent-authored lines outside the projected region
    # are *moved* — never destroyed — to `<state_dir>/MEMORY.archive.md`
    # (append-only, itself harvested next cycle). False keeps authored lines
    # in place; the projected region is still rendered.
    fold_authored_lines: bool = True
    # Optional git-versioned history of the projected view (opt-in).
    git: AgentMemoryProjectionGitConfig = Field(default_factory=AgentMemoryProjectionGitConfig)


class AgentMemoryConfig(BaseModel):
    """Agent-memory product surface — projection knobs live here."""

    projection: AgentMemoryProjectionConfig = Field(default_factory=AgentMemoryProjectionConfig)


class CurationLeverageWeights(BaseModel):
    """Weights for the curation leverage score.

    The score is a weighted sum of four normalized (0–1) signals, all read
    from data the store already holds. The set is intentionally small and
    additive; the spreading-activation variant is deferred.
    """

    # How many ACTIVE particles depend on this card's belief(s) through the
    # provenance DAG — a wrong belief 8 particles rest on outranks an isolated one.
    dependency_count: float = Field(default=1.0, ge=0.0)
    # Whether the belief carries the composed contested badge. Widened
    # from the inconsistency basis alone to all three bases (stance /
    # divergence / inconsistency), so "contested" means the same thing here as it
    # does at recall. Gated by ``contestedness.badge_enabled`` like every other
    # badge surface: off, this reverts to the open-INCONSISTENCY reading.
    contestedness: float = Field(default=1.0, ge=0.0)
    # How long the flagged belief has gone untended (age, normalized).
    staleness_age: float = Field(default=0.5, ge=0.0)
    # Whether the card blocks a clean documentation projection.
    # Live since wired the projection-manifest hook: a belief that
    # feeds a projected doc section (listed in ``curation.projection_manifests``)
    # gets this weight. Contributes nothing when no manifests are configured
    # (the hook stays inert), so the default is safe to ship at 1.0.
    projection_blocking: float = Field(default=1.0, ge=0.0)


class CurationStakesConfig(BaseModel):
    """Stakes weighting of the curation leverage score.

    ``leverage = stakes · (base + urgency)``, where urgency is the
    weighted sum and stakes is how much the card's beliefs are relied on: the
    larger of their use (the reinforcement score, normalized by
    ``use_norm_cap``) and their dependents, lifted by ``floor``. A card about
    beliefs nobody uses ranks low however urgent its kind.
    """

    # false restores the score exactly.
    enabled: bool = True
    # Constant urgency every card gets, so stakes can lift a card whose urgency
    # sum is near zero (a young belief on a store with no dependents).
    base: float = Field(default=0.25, ge=0.0)
    # Stakes of a card about beliefs with no use and no dependents, and of a
    # belief-free card. > 0 keeps the order on a store with no utility
    # evidence; 1.0 with base 0 reproduces the scores.
    floor: float = Field(default=0.05, gt=0.0, le=1.0)
    # Reinforcement score at which use saturates: ln(1+R) / ln(1+cap).
    use_norm_cap: int = Field(default=20, ge=1)


class CurationConfig(BaseModel):
    """Curation surface — the curation queue + session model.

    A finite, leverage-ranked worklist that unions the existing read
    diagnostics into one card list. The session is finite by design
    (``session_size``) so curation stays a habit, not an infinite backlog.
    """

    # "Today's N" — the finite per-session cap (the top-N by leverage).
    session_size: int = Field(default=7, ge=1)
    # How long a snoozed card drops out of the queue before resurfacing.
    snooze_days: int = Field(default=14, ge=1)
    # Run the LLM-assisted finders (semantic contradiction). Off by default so
    # the queue is cheap; --semantic / this knob opts in.
    semantic: bool = False
    # the principal an operator-authored belief is attributed to
    # (``asserted_by`` and the excerpt's author) on the operator supersede paths,
    # ``curate apply supersede`` and ``POST /particles/{id}/supersede``. One
    # operator using two surfaces is one principal; the event actor records the
    # surface. A ``platform:identifier`` string (spec §6.5).
    operator_identity: str = Field(default="operator:local", min_length=1)
    # Soft cap for the dependency-count normalizer: log1p(n) / log1p(cap).
    dependency_norm_cap: int = Field(default=20, ge=1)
    # Age (days) at which staleness_age saturates to 1.0.
    staleness_norm_days: float = Field(default=365.0, gt=0.0)
    leverage_weights: CurationLeverageWeights = Field(default_factory=CurationLeverageWeights)
    # scale leverage by the stakes of the card's beliefs.
    stakes: CurationStakesConfig = Field(default_factory=CurationStakesConfig)
    # leverage multiplier for a DUPLICATE_PAIR card the LLM judge
    # marked DISTINCT (not the same claim). < 1.0 sinks the cleared pair toward
    # the bottom of the queue without hiding it — preserving recall against a
    # wrong LLM clear (hard-suppression is deferred). 1.0 disables the
    # demotion; PARAPHRASE / UNSURE / absent verdicts are never demoted.
    duplicate_distinct_demotion: float = Field(default=0.1, ge=0.0, le=1.0)
    # Projection manifests: paths to ``operations.projection`` doc
    # manifests whose selected particles get ``projection_blocking`` leverage —
    # a belief that feeds a generated doc is worth tending first. Empty leaves
    # the projection-blocking signal inert (the default). Paths are
    # resolved relative to the process working directory.
    projection_manifests: list[str] = Field(default_factory=list)
    # serve the queue from a persisted card collection instead of
    # running every finder per request. `false` restores the pre-0238
    # build-every-request behaviour verbatim (correct, just minutes slow on a
    # large store: 172 s measured on the 2026-08-02 dogfood store).
    snapshot_enabled: bool = True
    # Age past which the queue response is stamped `stale: true`. The default
    # gives a nightly build a full day of slack, so a single skipped
    # run does not cry wolf. The cycle rebuilds only when its census runs, so
    # while consolidation.census is enabled the threshold is at least the
    # census interval plus a day.
    snapshot_max_age_hours: float = Field(default=36.0, gt=0.0)
    # Collections kept per store — a small ring so a bad build is one row from
    # a rollback. Snapshots are pure cache; nothing references them.
    snapshot_retain: int = Field(default=3, ge=1)
    # Eviction horizon for delta-scoped cards carried across builds (
    # §4). A carried card whose beliefs are never touched is dropped after this
    # many days rather than accreting forever; re-probing the tail is future work.
    snapshot_carry_forward_days: int = Field(default=30, ge=1)
    # the default window, in days back from now, over which
    # `curate --precision` and the quality report's `curation_precision` block
    # read the gesture log. `curate --precision --since DATE` overrides it.
    precision_window_days: int = Field(default=30, ge=1)


class ReindexConfig(BaseModel):
    """``particles reindex --estimate``.

    The estimate re-extracts a seeded sample of a version-scoped reindex's
    snapshots, judges each sample's new claims against its stored ones with the
    equivalence judge, and projects the changed share and the cost of
    the full sweep. It spends only on the sample.

    * ``estimate_sample_size`` — snapshots re-extracted per estimate. The
      reported share carries a 95% interval, which is wide on a small sample;
      raise this for a tighter one at proportionally higher cost.
    * ``estimate_seed`` — the sample's random seed, so two estimates over the
      same scope sample the same snapshots (``--seed`` overrides per run).
    * ``estimate_judge_batch_pairs`` — claim pairs per equivalence-judge call.
    """

    estimate_sample_size: int = Field(default=12, ge=1)
    estimate_seed: int = 0
    estimate_judge_batch_pairs: int = Field(default=25, ge=1)


class AuditConfig(BaseModel):
    """The first-run memory audit — presentation + cost gating only.

    The audit composes the existing finders and inherits their thresholds
    (``links_suggest.candidate_threshold``, ``lint.contradiction_candidate_threshold``,
    the decay floor); nothing here tunes detection.
    """

    # How many exemplar cards each headline class shows, ranked by leverage.
    exemplars_per_class: int = Field(default=3, ge=0)
    # Cap on session transcripts harvested by `particles audit --transcripts`
    # (newest first). Transcripts are large and LLM-extraction-priced; they
    # never ride the first run silently.
    transcript_max_entries: int = Field(default=20, ge=0)
    # Estimated-extraction-call count above which the CLI requires confirmation
    # (`--yes` pre-confirms; non-interactive runs without it abort with the
    # estimate printed).
    confirm_call_threshold: int = Field(default=50, ge=0)
    # Cap on contradiction LLM probes per audit run,
    # spent on the highest-similarity candidate pairs first. The
    # store-wide candidate set scales with near-duplicate density, so on an
    # already-populated store an uncapped probe blows the cost
    # envelope (owner dogfood 2026-07-11: 61+ min, 50+ probes on a ~1,000-
    # particle store). When the cap binds, the report discloses "probed X of
    # Y candidate pairs" (§6 honesty stance — never a silent partial census).
    # 0 probes nothing (disclosed the same way). `particles lint` is not
    # affected: cost gating is the audit's concern, lint stays exhaustive.
    max_contradiction_probes: int = Field(default=50, ge=0)
    # read each pair the probe flagged a second time, on
    # ``llm.verification``, with each claim's source passage, note name and
    # note date, and count only the confirmed pairs in the headline. The report
    # discloses "N flagged, M confirmed". Measured 2026-09-26 on a 96-file
    # memory store, three runs: 18 to 23 flagged, the same 4 pairs confirmed
    # each time, among them the one known real contradiction and no false
    # positive. false counts every flag, the pre-0275 behaviour. The nightly
    # census and every semantic card collection (curate --semantic, GET
    # /curation) read this too; `particles lint` never verifies.
    verify_contradictions: bool = True
    # Cap on second readings per run, spent in probe order (cross-source and
    # most similar first). A flag past the cap is disclosed as unverified and
    # is not counted.
    max_contradiction_verifications: int = Field(default=25, ge=0)
    # Output budget of one second reading. The verification purpose defaults to
    # a model that may think before it answers, and its thinking spends from
    # this budget. The measured runs averaged ~130 output tokens, but one
    # reading on 2026-09-26 (claude-sonnet-5, 96-note store) was cut at 1024
    # and its flag went unread.
    verify_max_tokens: int = Field(default=1024, ge=1)
    # A second reading cut at ``verify_max_tokens`` (or empty, the budget spent
    # on thinking) is re-issued once at this larger budget, since the same call
    # at the same budget tends to repeat the cut. A reply still cut is counted
    # as a failed reading and its flag stays unverified. A value at or below
    # ``verify_max_tokens`` disables the retry.
    verify_retry_max_tokens: int = Field(default=4096, ge=0)
    # Cost- and time-estimate assumptions (disclosed in the printed estimate),
    # re-based on the 2026-09-25 owner audit: 96 Claude Code memory files
    # (~53k source tokens) into a fresh store, claude-sonnet-5 extraction at
    # ``extraction.max_tokens`` 16384 with the retry at 20000 (since raised to
    # 32000 and 64000), claude-haiku-4-5 probes, 200-probe cap. Measured: 104
    # extraction calls averaging ~7,000 output tokens each (725k in all, mostly
    # adaptive thinking, ~13.6 per source token), so output per call is nearly
    # flat in the source size and is modelled per call, not per source token.
    # 8 of the 96 calls were retried at ``extraction.retry_max_tokens``. 374
    # semantic-lint calls: the 200
    # capped audit probes plus ~174 §6.6 probes the extraction pipeline makes
    # while it reconciles each new belief against a near-identical one. $7.97
    # billed; 96 minutes wall time. The pre-fix estimate printed $1.92-3.57.
    #
    # Expected output tokens per extraction call, averaged over a run with its
    # retries included. The per-model mapping (keyed like
    # ``llm.price_per_mtok``: the resolved model id, or ``"<provider>:<model>"``)
    # is consulted first; the scalar is the fallback. The 2026-09 provider
    # survey measured ~6.7k on claude-sonnet-5 and ~3.6k on claude-haiku-4-5, so
    # routing ``llm.extraction`` to haiku warrants an entry of about 3600.
    estimate_output_tokens_per_extraction_call: int = Field(default=7000, ge=0)
    estimate_output_tokens_per_extraction_call_by_model: dict[str, int] = Field(
        default_factory=dict
    )
    # The expected output range is the per-call figure times (1 - spread) with no
    # retries, up to (1 + spread) with the expected retries. A spread, not a
    # measurement: output per call varies with how much a source says.
    estimate_output_spread: float = Field(default=0.25, ge=0.0, lt=1.0)
    # Fraction of extraction calls expected to be retried at the larger
    # ``extraction.retry_max_tokens`` budget (8 of 96 in the measured run).
    # Ignored when that retry is disabled. Measured at the old 16384 first
    # budget; the 32000 budget should retry fewer calls, so the figure is kept
    # as a conservative bound until a run re-measures it.
    estimate_extraction_retry_rate: float = Field(default=0.083, ge=0.0, le=1.0)
    # §6.6 contradiction probes the extraction pipeline makes per extraction
    # call, on top of the capped audit probes (~174 over 96 calls in the
    # measured run). Density-dependent and uncapped: a store already holding
    # near-duplicates of the harvest makes more.
    estimate_reconcile_probes_per_extraction_call: float = Field(default=1.8, ge=0.0)
    # One contradiction probe (audit or §6.6): instruction plus two claims in,
    # a one-sentence reason and a verdict line out. The output figure is the
    # probe call's own ``max_tokens`` (250), an upper bound; the measured run
    # averaged ~305 in and ~56 out.
    estimate_probe_input_tokens: int = Field(default=350, ge=0)
    estimate_probe_output_tokens: int = Field(default=250, ge=0)
    # Wall time per call, for the time the estimate prints. Calls run one at a
    # time. The measured run's 96 minutes over 104 extraction calls and 374
    # probes fits ~50 s per extraction call once the probes (short haiku
    # replies) are taken at ~1.5 s each; the split is an assumption, the total
    # is measured.
    estimate_seconds_per_extraction_call: float = Field(default=50.0, ge=0.0)
    estimate_seconds_per_probe: float = Field(default=1.5, ge=0.0)
    # One second reading: the instruction plus two claims with their
    # passages in (measured ~1,450 tokens on memory notes), and the call's own
    # ``max_tokens`` (``verify_max_tokens``, 1024, room for a model that thinks)
    # out, an upper bound that leaves out the rare retry.
    # The measured runs averaged ~1,500 in and ~130 out on claude-sonnet-5.
    estimate_verify_input_tokens: int = Field(default=1500, ge=0)
    estimate_verify_output_tokens: int = Field(default=1024, ge=0)
    # Wall time per second reading: the measured runs read 22 flags in ~41 s.
    estimate_seconds_per_verification: float = Field(default=2.0, ge=0.0)

    @field_validator("estimate_output_tokens_per_extraction_call_by_model")
    @classmethod
    def _non_negative_output_tokens(cls, value: dict[str, int]) -> dict[str, int]:
        """Every per-model output figure is a token count: zero or more."""
        for key, tokens in value.items():
            if tokens < 0:
                raise ValueError(
                    f"audit.estimate_output_tokens_per_extraction_call_by_model[{key!r}] "
                    f"must be >= 0, got {tokens}"
                )
        return value


class AbstractionConfig(BaseModel):
    """Abstraction-promotion pass — cluster settled specifics into
    semantic beliefs carrying premise links.

    Deliberately *not* exposed: the derived particle's stored-confidence rule
    (min-of-premises — tunable confidence invites gaming the
    projection cut) and the provider choice (rides ``llm.abstraction``).
    """

    # Pass runs at all. Off until the evaluation supports it.
    enabled: bool = False
    # ``propose``: candidates surface as curation cards for operator review.
    # ``auto``: candidates are asserted directly (still entailment-gated,
    # still §6.6-reconciled). Graduating the default to ``auto`` is gated by
    # evidence.
    mode: Literal["propose", "auto"] = "propose"
    # Minimum premises per cluster; below this co-evidential merge covers it.
    min_cluster_size: int = Field(default=3, ge=2)
    # Only consolidate settled beliefs: every premise older than this.
    min_source_age_days: int = Field(default=14, ge=0)
    # Per-cycle cap on promotion-shaped LLM spend (candidates synthesized +
    # revalidations run), cap discipline.
    max_promotions_per_run: int = Field(default=5, ge=0)
    # Premises must be non-derived below this depth. 1 = premises are never
    # themselves derived. The §5 invalidation contract handles the general
    # DAG, so raising this is a config change, not a design change.
    max_depth: int = Field(default=1, ge=1)
    # Discard candidates the entailment judge cannot confirm are supported by
    # the conjunction of their premises (faithfulness gate).
    require_entailment: bool = True
    # ``suppress_in_projection``: the projection ranker skips premises of an
    # ACTIVE derived particle (ranking-side only — no status change).
    # ``none`` disables the suppression.
    source_demotion: Literal["suppress_in_projection", "none"] = "suppress_in_projection"
    # Read-time effective-confidence multiplier for a derived particle while
    # any premise is non-ACTIVE (pending revalidation).
    stale_support_discount: float = Field(default=0.5, ge=0.0, le=1.0)
    # Pairwise cosine floor for cluster membership (sweep axis).
    # Lower than links_suggest.candidate_threshold by design: clusters gather
    # *related but distinct* claims, not near-duplicates.
    cluster_similarity_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    # Exclude time-anchored claims from cluster eligibility: any ``valid_until``
    # bearer or a date/relative-time mention in the content. The
    # first measurement (oracle-variant 50-question A/B,
    # 2026-07-18) localized the entire QA-at-budget regression to
    # temporal-reasoning questions (0.769 → 0.615), every other question type
    # unchanged: a faithful generalization over date-anchored premises blurs
    # the date the question needs — §7 vague-but-true landing on dates. The
    # detector is deliberately over-inclusive; a missed abstraction is cheap,
    # a blurred date is the regression.
    exclude_time_anchored: bool = True


class ContradictionDisclosureConfig(BaseModel):
    """The nightly contradiction disclosure pass.

    A contradiction the census's second reading confirms, between claims from
    two sources, opens an INCONSISTENCY record the next session's digest
    flags. Disclosure only: no claim's status or confidence changes.
    """

    # Master switch. Off, the cycle behaves as before this pass existed, apart
    # from its report line saying so.
    enabled: bool = True
    # New records opened per run; regroup replacements do not count. Past the
    # cap a disagreement waits in the run record for the next run.
    max_per_run: int = Field(default=10, ge=0)
    # Second readings per run spent re-reading the pairs of open records that
    # were confirmed under an instruction that has since changed. A
    # pair the new reading rejects is withdrawn, and a record left with none
    # closes as ``withdrawn``. Past the cap a pair is re-read on a later night.
    max_rereadings_per_run: int = Field(default=25, ge=0)


class ConsolidationBatchWaitConfig(BaseModel):
    """The run-level batch-wait budget.

    ``llm.batch.max_wait_seconds`` bounds one batch; this bounds the sum of
    every batch one consolidation run waits on. Each batch waits at most the
    balance left, and once less than ``min_remaining_seconds`` is left the
    remaining sets run sequentially at full price. Nothing is skipped, and the
    run report and ``CONSOLIDATION_RUN`` record say what was moved.
    """

    # Total seconds one run may spend waiting on submitted batches.
    budget_seconds: float = Field(default=3600.0, gt=0.0)
    # Below this balance a batch would almost surely be cancelled after its
    # finished requests were billed, so the next set runs sequentially instead.
    min_remaining_seconds: float = Field(default=300.0, ge=0.0)


class ReanchorConfig(BaseModel):
    """Re-anchoring claims that relied on a superseded state.

    When an update retires a state claim, the claims cut from the same passage
    that relied on it being current are replaced by a dated restatement
    anchored to that state. Nothing is retired without a checked restatement
    to replace it, and a restatement never changes another belief.
    """

    # Master switch. Off, the cycle skips the pass and says so; rung 2.5 is
    # unaffected.
    enabled: bool = True
    # Update retirements examined per run, oldest first: one probe call each.
    # A retirement past the cap waits for the next run.
    max_retirements_per_run: int = Field(default=20, ge=0)
    # Candidates sent in one probe call, highest similarity to the retired
    # claim first.
    max_candidates_per_retirement: int = Field(default=8, ge=1)
    # Second readings per run, and so writes. Each restatement also spends a
    # duplicate check and up to reconciliation.update_supersession.max_candidates
    # contradiction probes.
    max_restatements_per_run: int = Field(default=20, ge=0)


class ConsolidationCensusConfig(BaseModel):
    """How often the cycle runs its census, pass 3.

    The census is the store-wide contradiction and duplicate sweep whose cards
    feed the curation queue. It is report-only, so it cannot move retrieval,
    and it was most of a cycle's LLM spend on the measured LongMemEval run. It
    therefore runs on its own, slower cadence beside the nightly cycle; on the
    nights between, pass 4 serves the last census's stored cards. The probe cap
    stays where it was, ``audit.max_contradiction_probes``.
    ``particles audit`` is unaffected and still runs its census on first contact.
    """

    # Off: the cycle never runs the census, and pass 4 serves whatever card
    # collection is already stored. The report says the census is off.
    enabled: bool = True
    # The census runs when the last one on this store (read from the run
    # records) started at least this many hours ago. 168 = weekly. A run that
    # starts up to an hour early still counts as due, so a scheduler's drift
    # does not slip the census a whole night. 0 runs it on every cycle, which
    # was the behaviour before this knob existed.
    interval_hours: int = Field(default=168, ge=0)


class ConsolidationConfig(BaseModel):
    """The scheduled consolidation cycle — cadence + cost gating only.

    The cycle composes the existing passes and inherits every existing cap
    (``audit.max_contradiction_probes``, ``utility.mining.max_behavioural_calls``,
    ``extraction.max_llm_calls_per_source``); nothing here re-tunes detection —
    the rule applied to the cycle.
    """

    # ``--if-due`` threshold: skip when the last successful CONSOLIDATION_RUN is
    # younger than this. Default 20 h = daily scheduling with headroom for clock
    # drift, so an hourly catch-up retry is harmless.
    min_interval_hours: int = 20
    # Pass 1 (extract catch-up) on/off. Extraction is LLM-priced; off leaves the
    # PENDING backlog to interactive verbs.
    extract_pending: bool = True
    # Pass 1 per-run cap: PENDING snapshots extracted per cycle, oldest first.
    # A capped run discloses the remainder; the next run continues.
    max_pending_entries: int = Field(default=20, ge=0)
    # Pass 1 pooled batching: run the capped set as concurrent
    # per-snapshot tasks whose LLM requests merge into one Message
    # Batches job (50% price; batch mechanics reuse the llm.batch.* knobs).
    # False restores the serial per-snapshot loop exactly.
    extract_batching: bool = True
    # Pooled pass: how many snapshot tasks may use the store at once. A task
    # gives its slot (and its DB connection) back while it waits on the batch,
    # so every snapshot still joins the one batch; this bounds the database
    # phases before and after it. Keep it well under the SQLAlchemy pool
    # (5 + 10 overflow): a task can hold two connections while it fails.
    extract_db_concurrency: int = Field(default=4, ge=1)
    # Run the LLM passes (extract catch-up, reconcile probes, contradiction
    # probe, behavioural utility matching) on scheduled runs. False ships
    # structural-only-until-enabled — the §11 demotion path (owner-resolved
    # true, 2026-07-12).
    semantic: bool = True
    # Pass 2 (reconcile sweep) per-run cap on replacement-signal probes — one
    # LLM call per candidate pair, spent highest-similarity-first; truncation
    # is disclosed ("probed X of Y candidate pairs"), in the spirit of the
    # census cap. Correction rider on the consolidation cadence (v1.74.1).
    max_reconcile_probes: int = Field(default=50, ge=0)
    # per-run probe budget for the same-subject update sweep.
    # Higher than the document-supersession cap because the backlog it clears
    # is a store's whole history of changed facts, and because its pre-filter
    # (update_order) means every probe is spent on a pair rung 2.5 can act on.
    max_update_probes: int = Field(default=100, ge=0)
    # Cycle lock. The kernel holds the lock, so a live cycle is never
    # reclaimed and a crashed one frees it at once. Past this age a skipped
    # caller warns on stderr that the holder may be hung. It is still the
    # reclaim age where the lock falls back to pid-and-age: no advisory file
    # locks, or a pre-change holder.
    lock_timeout_minutes: int = Field(default=120, ge=1)
    # How often a holder refreshes the lock's heartbeat_at, from a thread.
    lock_heartbeat_seconds: float = Field(default=60.0, gt=0)
    # A holder on another host (a container on a shared mount, whose kernel
    # lock this host cannot see) is live while its heartbeat is younger than this.
    lock_heartbeat_stale_minutes: int = Field(default=10, ge=1)
    # Abstraction-promotion pass, between utility mining and the
    # projection re-render.
    abstraction: AbstractionConfig = Field(default_factory=AbstractionConfig)
    # Pass 3b, the nightly contradiction disclosure.
    contradiction_disclosure: ContradictionDisclosureConfig = Field(
        default_factory=ContradictionDisclosureConfig
    )
    # The re-anchor pass, between the update sweep and the census.
    reanchor: ReanchorConfig = Field(default_factory=ReanchorConfig)
    # The run-level batch-wait budget.
    batch_wait: ConsolidationBatchWaitConfig = Field(default_factory=ConsolidationBatchWaitConfig)
    # The run's dollar budget, in US$ at list price over the token
    # counts the provider reports (``llm.price_per_mtok``). Checked before each
    # LLM-priced pass against the run's spend so far plus the pass's estimate,
    # and before each Message Batches chunk: a pass that would exceed it is
    # skipped and disclosed ("spent US$X of US$Y; pass N skipped"), a chunk is
    # not submitted. Zero-LLM passes always run. ``None`` is unbounded.
    budget_usd: float | None = Field(default=None, ge=0)
    # The census cadence: pass 3 runs weekly by default, and the
    # nights between serve its stored cards.
    census: ConsolidationCensusConfig = Field(default_factory=ConsolidationCensusConfig)


class DaemonConfig(BaseModel):
    """Resident daemon mode — in-process scheduling inside ``engine serve``.

    Opt-in and off by default: without ``--daemon`` or ``enabled: true`` here,
    ``engine serve`` behaves exactly as it always has. When on,
    the FastAPI lifespan starts background asyncio tasks — a consolidation tick
    and the intake watchers — so a container needs no launchd/cron alongside it.

    This is a **rider**, not a supersession: for deployments that
    opt in, the in-process scheduler replaces the external one; everywhere else
    the external-scheduler contract stands.
    """

    # Master switch. ``engine serve --daemon`` sets it for one process via the
    # registered ``PARTICLES_DAEMON_ENABLED`` override (the flag wins).
    enabled: bool = False
    # Store handle every daemon task operates on. The engine serves one store;
    # a host operator consolidating a differently-named store (e.g. ``memory``)
    # points this at it. Literal rather than ``particles.db.DEFAULT_STORE``:
    # config is Client-layer and must not import the Engine.
    store: str = "default"
    # Consolidation-tick period. The tick calls the operation with ``--if-due``
    # semantics, so ``consolidation.min_interval_hours`` (default 20) is the real
    # cadence and over-ticking is harmless — this only bounds how soon after
    # becoming due a run starts.
    consolidation_tick_minutes: int = Field(default=60, ge=1)
    # Web-clipper watcher (intake): the captures directory to poll.
    # Unset (the default) leaves the watcher inactive. ``~`` is expanded. The
    # inbox watcher has no switch of its own — it is active whenever
    # ``inbox.file_path`` is set.
    web_clipper_dir: str | None = None
    # Web-clipper poll period. mtime-poll only: the tree is stat-walked and the
    # one-shot scan runs only when something changed (no watchdog /
    # FSEvents dependency — rejection of filesystem-event watchers
    # stands untouched).
    web_clipper_poll_minutes: int = Field(default=5, ge=1)


class MemoryBenchmarkConfig(BaseModel):
    """Agent-memory benchmark evaluation — LongMemEval.

    Run knobs + cost gating only: the pipeline under test runs the shipped
    defaults (default thresholds, decay and the cap active) — the
    published number must describe the product, not a lab build, so nothing
    here re-tunes detection or ranking.
    """

    # HuggingFace revision (commit sha) of the pinned LongMemEval v1 cleaned
    # dataset (xiaowu0162/longmemeval-cleaned); bumping it is a deliberate
    # diff. Finalized 2026-07-18 (the repo's only commit at pin time; the
    # loader's per-file SHA-256 pins were recorded against it).
    dataset_revision: str = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
    # Dataset variant: oracle (evidence sessions only) | s (~40-session
    # haystacks) | m (~500-session haystacks). The published table is the
    # ``s`` variant (owner-resolved 2026-07-12).
    variant: str = "s"
    # Top-k for the retrieval stage and the qa_particles context. Mirrors the
    # product's query default (``QueryRequest.top_k``, the CLI ``--top-k``
    # and the MCP ``query`` tool all default to 40) because the published
    # number must describe the product, not a lab build. The
    # value is recorded on the run tuple, so a lower k is a disclosed
    # ablation rather than the product. ``tests/test_config.py`` pins the two
    # defaults equal so they cannot drift.
    top_k: int = Field(default=40, ge=1)
    # Version of the answer scaffold shared by the three QA conditions
    # (``_ANSWER_SYSTEMS`` in the runner). 1 is the inaugural 2026-08-16
    # table's text; 2 adds question-type-blind reader guidance (conditional
    # abstention, preference grounding, an enumerate-merge-count protocol,
    # date arithmetic, latest-wins). Recorded on the run tuple and in the
    # checkpoint key: runs under different versions are not comparable.
    answer_scaffold: int = Field(default=2, ge=1, le=2)
    # Version of the judge-prompt protocol (``judge_prompt`` in the runner).
    # 1 is the 1.74.0 paraphrase of the dataset's autoeval prompts, which the
    # inaugural 2026-08-16 table was scored under; 2 is the official
    # ``get_anscheck_prompt`` templates verbatim (the judge *model* stays the
    # configured Anthropic one). Recorded on the run tuple and in the
    # checkpoint key: verdicts under different protocols are not comparable.
    judge_protocol: int = Field(default=2, ge=1, le=2)
    # How the qa_particles context renders each claim's subjects (
    # specifies that context as "claim text, subjects, dates").
    #   uuids - the resolved subject_ids verbatim, which is what every
    #           published table through 1.146.5 was measured under
    #   names - each subject's canonical_name
    #   none  - no subject field at all
    # Measured 2026-09-20 over the kept s150 store set: `names` costs 33.6%
    # fewer context tokens than `uuids` at top_k 40 and `none` 52.2% fewer,
    # because a UUID tokenizes far worse than the name it stands for.
    # Recorded on the run tuple and in the checkpoint key.
    #
    # The default moved to `names` in 1.148.2, after the nine-run ablation:
    # three repeats per rendering showed two samples of ONE rendering disagree
    # on 4-8 questions, `uuids` returned 0.820 three times while disagreeing
    # with itself each time, and only one question in each direction is one a
    # rendering always decides. So the accuracy ordering between renderings is
    # noise and the 33.6% token saving is not.
    #
    # `names` rather than the cheaper `none` (52.2%), deliberately. The
    # qa_particles context is specified as "claim text, subjects, dates",
    # so rendering the subject readably is a CORRECTION while
    # removing the field is a redefinition of what the benchmark measures.
    # And `none` scoring
    # above `names` is precisely the noise the repeats established: choosing on
    # that 0.7-point gap would be selecting on sampling. `none` stays available
    # for anyone who wants to re-open the spec question with data.
    #
    # `uuids` reproduces every published figure through 1.146.16.
    subject_rendering: Literal["uuids", "names", "none"] = "names"
    # Dev-loop default question count; a full run requires --all.
    default_question_limit: int = Field(default=10, ge=1)
    # Seed for the stratified-by-question-type subset selection; part of the
    # recorded run tuple, so two subset runs under one seed are comparable.
    sample_seed: int = 13
    # Estimated-LLM-call count above which the CLI requires confirmation
    # (mirrors audit.confirm_call_threshold; --yes pre-confirms,
    # non-interactive runs without it abort with the estimate printed).
    confirm_call_threshold: int = Field(default=50, ge=0)
    # Retries for a *transient* answer/judge call failure before the call is
    # reported as an infra failure and excluded from the accuracy denominator.
    # A no-text-block reply is never retried — it is deterministic
    # at a fixed budget — and is excluded under the separate budget count.
    call_retries: int = Field(default=2, ge=0)
    # Backoff before each retry, multiplied by the attempt number. 0 disables
    # the wait (what the unit tier sets).
    call_retry_backoff_seconds: float = Field(default=2.0, ge=0.0)
    # The answering model's context window, in tokens — the budget the
    # pre-flight check holds the qa_full_context baseline's prompt against
    # before any LLM call. The default is the Claude window; set it
    # to match whatever llm.benchmark_answer resolves to when routing
    # elsewhere. The check is what makes the ~500-session ``m`` variant refuse
    # up front instead of silently crushing the baseline via overflow.
    answer_context_window_tokens: int = Field(default=200_000, ge=1)
    # Per-call *output*-token assumptions the pre-run estimate multiplies the
    # projected call counts by. Output dominates an extraction run's bill —
    # the 2026-09 provider survey measured ~6.7k output tokens per
    # claude-sonnet-5 extraction call against ~3.9k input, and the inaugural
    # LongMemEval run came in at nearly twice an input-only projection
    # (Phase 0 costing) — so an estimate that ignores them is not a
    # cost preview. The answer scaffold says "answer concisely" and the judge
    # returns a bare yes/no, hence the small QA-side defaults. Every figure is
    # disclosed in the rendered estimate as the assumption it is.
    estimate_output_tokens_per_extraction_call: int = Field(default=6_700, ge=0)
    estimate_output_tokens_per_answer_call: int = Field(default=300, ge=0)
    # The judge figure predates adaptive thinking: thinking is billed as
    # output and the judge call's budget is 1,024, but no run has recorded
    # the judge's usage, so the default stays at the bare-verdict size and
    # EXCLUDES thinking tokens. Raise it from measured usage, not by guess.
    estimate_output_tokens_per_judge_call: int = Field(default=32, ge=0)
    # Per-model override of the extraction figure, keyed like ``llm.price_per_mtok``
    # (resolved model id, or ``"<provider>:<model>"`` to pin one route), and
    # resolved by the same lookup so the two can never disagree on a key.
    # Consulted before the scalar above; the scalar is the fallback. Output
    # per call is a property of the model — the 2026-09 survey measured ~6.7k
    # on claude-sonnet-5 but ~3.6k on claude-haiku-4-5 — so an arm routing
    # ``llm.extraction`` elsewhere is mis-estimated ~2x on the write side
    # without an entry here. The rendered estimate says which source it used.
    estimate_output_tokens_per_extraction_call_by_model: dict[str, int] = Field(
        default_factory=dict
    )
    # Characters per input token — the factor the estimate and the pre-flight
    # context-window check divide prompt bytes by. The scalar is the classic
    # ~4-chars/token rule of thumb, correct for the pre-4.7 tokenizer
    # (claude-haiku-4-5, claude-sonnet-4-6). Models on the newer tokenizer
    # (Claude 4.7 and later, claude-sonnet-5 included) produce ~1.3-1.4x the
    # tokens for the same text, so the per-model mapping — same key shape and
    # lookup as ``llm.price_per_mtok`` — is what keeps their input projection and
    # their window verdict honest. The write side is keyed on the extraction
    # model, the answer side and the window check on the answer model.
    chars_per_token: float = Field(default=4.0, gt=0.0)
    chars_per_token_by_model: dict[str, float] = Field(default_factory=dict)

    @field_validator("estimate_output_tokens_per_extraction_call_by_model")
    @classmethod
    def _non_negative_output_tokens(cls, value: dict[str, int]) -> dict[str, int]:
        """Every per-model output figure is a token count: zero or more."""
        for key, tokens in value.items():
            if tokens < 0:
                raise ValueError(
                    f"estimate_output_tokens_per_extraction_call_by_model[{key!r}] "
                    f"must be >= 0, got {tokens}"
                )
        return value

    @field_validator("chars_per_token_by_model")
    @classmethod
    def _positive_chars_per_token(cls, value: dict[str, float]) -> dict[str, float]:
        """A chars-per-token factor is a divisor: strictly positive."""
        for key, factor in value.items():
            if factor <= 0:
                raise ValueError(f"chars_per_token_by_model[{key!r}] must be > 0, got {factor}")
        return value

    # The batch discount lives in ``llm.batch_discount`` (``--pooled`` batches
    # the write side, ``--batch-qa`` the answerer/judge, both only while
    # ``llm.batch.enabled``); the old key here is migrated there.
    # Prices live in ``llm.price_per_mtok``, shared with the audit estimate;
    # the old ``benchmark_memory.price_per_mtok`` key is migrated there.


class RotBenchmarkConfig(BaseModel):
    """Memory-rot benchmark — currency / supersession / source trust.

    Run knobs, world size, and cost-projection assumptions only. The pipeline
    under test runs the shipped defaults: nothing here re-tunes detection,
    reconciliation, or ranking (the harness measures the product
    as configured and must never make the number look better). Prices come
    from ``llm.price_per_mtok`` so the harnesses and the audit cannot
    disagree about what a model costs.
    """

    # Seeds a run covers when none is given on the CLI. A seed *is* the
    # fixture: the world is a pure function of (seed, days), so three seeds
    # are three independent worlds and the report shows their spread.
    seeds: list[int] = Field(default_factory=lambda: [42, 43, 44])
    # Simulated world length in days and the probe checkpoints within it.
    days: int = Field(default=90, ge=30)
    checkpoints: list[int] = Field(default_factory=lambda: [15, 30, 45, 60, 75, 90])
    # Probe top-k. RotBench scores the context block a system returns, which
    # is a short list; 10 is that surface. Recorded on the run tuple.
    top_k: int = Field(default=10, ge=1, le=200)
    # The domain trust score the `source` poison channel's untrusted domain is
    # given in each scratch store — the operator's policy. Ignored
    # under --no-trust-policy, which measures the neutral default.
    untrusted_domain_trust: float = Field(default=0.2, ge=0.0, le=1.0)
    # Relevance floors the offline sweep evaluates over the recorded top-1
    # cosines. Pure arithmetic; no call is made.
    floor_sweep: list[float] = Field(
        default_factory=lambda: [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    )
    # Cost-projection assumptions for the `live` arm (the estimate discloses
    # them). A rot session is a short templated chat, far smaller than a
    # LongMemEval haystack session, so it gets its own output assumption.
    estimate_output_tokens_per_extraction_call: int = Field(default=1500, ge=0)
    # The per-call prompt overhead is ``extraction.estimate_prompt_overhead_tokens``,
    # shared with the audit estimate (the old key here is migrated there).
    # Contradiction probes per world, and their size. Bounded above by the
    # candidate pairs over the similarity threshold; the world's value-change
    # and decoy events are what pair, so this scales with them.
    estimate_probe_calls_per_update: float = Field(default=4.0, ge=0.0)
    estimate_probe_input_tokens: int = Field(default=120, ge=0)
    estimate_probe_output_tokens: int = Field(default=200, ge=0)
    # Above this many projected LLM calls the CLI asks before spending
    # (--yes pre-confirms). The `oracle` arm projects zero and never asks.
    confirm_call_threshold: int = Field(default=100, ge=0)

    @field_validator("checkpoints")
    @classmethod
    def _checkpoints_sorted_positive(cls, value: list[int]) -> list[int]:
        """Checkpoints are strictly increasing positive days."""
        if not value or any(d <= 0 for d in value) or value != sorted(set(value)):
            raise ValueError("checkpoints must be non-empty, positive, strictly increasing")
        return value


class RelevanceFloorBenchmarkConfig(BaseModel):
    """Relevance-floor benchmark — the gate's error rates.

    Run knobs and cost-projection assumptions only. The query op under test
    runs as configured; the single thing the judged stage changes is the gate
    itself, which it disables for the run so the response step can be scored
    over the top-k the floor would have suppressed. Prices come from
    ``llm.price_per_mtok``.
    """

    # Floors the sweep evaluates (the range, so the synthetic and
    # the real-question curves line up setting for setting).
    floors: list[float] = Field(default_factory=lambda: [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    # Retrieval depth of the replay — the query surfaces' own default, since
    # the floor reads the maximum cosine over the *rendered* top-k.
    top_k: int = Field(default=40, ge=1, le=200)
    # Where `harvest` reads agent transcripts and writes the held-out set. The
    # set is real questions from a real person: it lives outside any
    # repository by default and is never vendored.
    transcripts_dir: str = "~/.claude/projects"
    heldout_path: str = "~/.particles/benchmark/relevance-floor/heldout.jsonl"
    # Length bounds of a question-shaped sentence harvested from a typed prompt.
    min_question_chars: int = Field(default=15, ge=1)
    max_question_chars: int = Field(default=300, ge=1)
    # Seed of the `--limit` sample (stratified by source).
    sample_seed: int = 0
    # Version of the reference-free grounded-and-useful judge prompt. Never
    # edit a protocol's text in place; add a version.
    judge_protocol: int = Field(default=1, ge=1)
    # Concurrent questions in the judged stage (each holds its own session).
    concurrency: int = Field(default=4, ge=1)
    # A transient answer/judge failure is retried this many times with a
    # linear backoff before the question is excluded as `infra`.
    call_retries: int = Field(default=2, ge=0)
    call_retry_backoff_seconds: float = Field(default=2.0, ge=0.0)
    # Cost-projection assumptions (disclosed by --estimate). Input is measured
    # from each question's real top-k, so only output needs assuming.
    estimate_answer_prompt_overhead_tokens: int = Field(default=700, ge=0)
    estimate_answer_output_tokens: int = Field(default=500, ge=0)
    estimate_judge_prompt_overhead_tokens: int = Field(default=350, ge=0)
    estimate_judge_output_tokens: int = Field(default=20, ge=0)
    # Above this many projected LLM calls the CLI asks before spending
    # (--yes pre-confirms). The unjudged replay projects zero and never asks.
    confirm_call_threshold: int = Field(default=50, ge=0)

    @field_validator("floors")
    @classmethod
    def _floors_sorted_unit(cls, value: list[float]) -> list[float]:
        """Floors are strictly increasing and inside the cosine scale."""
        if not value or any(not 0.0 <= f <= 1.0 for f in value) or value != sorted(set(value)):
            raise ValueError("floors must be non-empty, within [0, 1], strictly increasing")
        return value


class LeakageBenchmarkConfig(BaseModel):
    """Model-prior leakage benchmark — unsupported answer sentences.

    Every sentence of a query answer is judged against the particles the
    answer was composed from, by the shared entailment judge on the
    ``llm.benchmark`` purpose. Run knobs and cost-projection assumptions only:
    the query op under test runs exactly as configured. The question set is
    the relevance-floor benchmark's private held-out set
    (``benchmark_relevance_floor.heldout_path``). Prices come from
    ``llm.price_per_mtok``.
    """

    # Retrieval depth the query op answers over — the query surfaces' default.
    top_k: int = Field(default=40, ge=1, le=200)
    # Seed of the `--limit` sample (stratified by source).
    sample_seed: int = 0
    # Version of the sentence-attribution rubric. Never edit a protocol's text
    # in place; add a version. 1 is verdict-first (the baseline); 2
    # asks for the reason first. Compare numbers only within one.
    judge_protocol: int = Field(default=2, ge=1)
    # Refuse a run whose judge resolves to the same model as the composer. A
    # model judging its own output shares its parametric background, so it
    # tends to find its own leaked facts supported; a different judge blunts
    # that circularity. Off only to measure the effect of turning it off.
    require_distinct_judge: bool = True
    # Preceding answer sentences shown to the judge with each sentence, so an
    # "it" or "this" can be resolved. Context only: it is never a premise.
    context_sentences: int = Field(default=2, ge=0)
    # Token budget of one judge call (room for an adaptive-thinking judge).
    judge_max_tokens: int = Field(default=1024, ge=16)
    # Concurrent questions (each holds its own session while it is answered).
    # Keep it under the store's connection pool (SQLAlchemy's default, 5 + 10
    # overflow): above that a question times out waiting for a connection and
    # is excluded as `infra`.
    concurrency: int = Field(default=4, ge=1)
    # Judge calls in flight at once, across the whole run. An answer's
    # sentences are judged concurrently under this run-wide bound.
    judge_concurrency: int = Field(default=8, ge=1)
    # A transient answer/judge failure is retried this many times with a
    # linear backoff before it is excluded as `infra`.
    call_retries: int = Field(default=2, ge=0)
    call_retry_backoff_seconds: float = Field(default=2.0, ge=0.0)
    # Cost-projection assumptions (disclosed by --estimate). Nothing is
    # retrieved before the projection, so the top-k's size and the answer's
    # length (and so its sentence count) are assumed; the judge count is an
    # upper bound, since a refused answer is never judged.
    estimate_answer_prompt_overhead_tokens: int = Field(default=700, ge=0)
    estimate_answer_output_tokens: int = Field(default=500, ge=0)
    # Tokens one rendered particle adds to a prompt (the top-k is not
    # retrieved before the projection, so its size is assumed too).
    estimate_particle_tokens: int = Field(default=45, ge=0)
    estimate_sentences_per_answer: float = Field(default=8.0, ge=0.0)
    estimate_judge_prompt_overhead_tokens: int = Field(default=600, ge=0)
    estimate_judge_output_tokens: int = Field(default=120, ge=0)
    # Above this many projected LLM calls the CLI asks before spending
    # (--yes pre-confirms).
    confirm_call_threshold: int = Field(default=50, ge=0)


class BenchmarkConfig(BaseModel):
    """Benchmark-harness run persistence (family).

    ``runs_dir`` is where ``particles extractor benchmark`` persists each
    run's report as one JSON file (envelope: schema-format stamp + the
    resolved extraction provider:model pairing + the full §13.3 report).
    Persistence is a CLI concern — the harness itself stays report-only.
    The per-run flag ``--no-save`` skips it for throwaway experiments.
    Sibling harnesses (modality / polarity / validity / compare) may adopt
    the same directory later; filenames carry the harness kind.

    ``confirm_call_threshold`` gates the repeat-runs mode (``--runs N``),
    whose cost scales linearly with N: above this many projected
    extraction calls the CLI requires confirmation (mirrors
    ``audit.confirm_call_threshold``; ``--yes`` pre-confirms).
    A single run is never gated — the estimate only prints when N > 1.

    ``record_claim_text`` controls whether each report carries the *text* of
    the claims the extractor emitted, beside their ids. On (the default) is
    what makes a saved report auditable: a benchmark run never persists to the
    store, so an emitted particle's uuid resolves to nothing once the process
    exits, and a precision figure cannot be inspected after the fact. Turn it
    off when pointing the harness at a corpus whose content must not land in a
    run file — the ids, counts, and every metric are unaffected, and gold text
    from the suite YAML is unaffected too (the report has always carried it).

    ``subject_aware_matching`` embeds each emitted claim as the
    particle actually asserts it — subject prepended when the ``content``
    string does not already name it — instead of comparing a subject-elided
    claim against subject-bearing gold prose. On by default because the
    un-qualified comparison measurably misreports: it charges an extractor
    twice, once to precision and once to recall, for putting the subject in
    the field the schema provides.

    Set it **false to reproduce a pre-0262 number** — the provider-survey
    pages and every run file written before 1.140.0 were measured under the
    old semantics, and are not comparable to a run made under the new ones.
    That is the knob's purpose; it is not a tuning dial.

    ``record_demotion_rulings`` appends one labelled pair to
    ``<runs_dir>/demotion-rulings.jsonl`` each time the operator affirms or
    dismisses a ``demotion`` curation card: both claims' texts and content
    hashes, the subject, the demotion reason, the probe verdicts that produced
    it, the ruling, who and when. The memory-rot benchmark reads that file as
    its real-pairs section. On by default because the file is local to the
    operator's home directory; it does hold claim text from their store, so
    turn it off when that text must not land outside the database. The
    gesture's meaning is the same either way.
    """

    runs_dir: str = "~/.particles/benchmark/runs"
    confirm_call_threshold: int = Field(default=50, ge=0)
    record_claim_text: bool = True
    subject_aware_matching: bool = True
    record_demotion_rulings: bool = True


class MetricsConfig(BaseModel):
    """Publication-metrics capture and deposit — the two halves of one pipeline.

    The pipeline is deliberately **split in two, and the split is the point**.
    One of its sources expires: the GitHub traffic endpoints
    (``/traffic/views``, ``/traffic/clones``) serve a trailing **14-day**
    window and nothing older, so a figure not captured inside that window is
    gone permanently. Every other source here is backfillable (pypistats keeps
    ~180 days; GoatCounter and the ``starred_at`` stargazer stream keep full
    history). So *capture* runs unattended on a schedule and only writes raw
    JSON, while *deposit* runs on the operator's laptop whenever it happens to
    be up and batches whatever has accumulated. Neither half may be made to
    depend on the other being available.

    The fields below are read by the deposit half (``scripts/deposit_metrics.py``)
    via ``get_config()``. The capture half (``scripts/capture_metrics.py``) is
    stdlib-only by design — it must not be able to fail because this SDK's
    dependency closure did — so it carries its own copy of the identity fields
    as module constants, and ``tests/test_capture_metrics.py`` asserts the two
    copies are equal. Change a value here and that test tells you to change it
    there.

    Nothing in this section is a secret. The GitHub and GoatCounter tokens are
    read through ``particles.secrets``.
    """

    # Where snapshots live, relative to the repository root. The scheduled
    # capture writes here on a dedicated data branch; the deposit half reads
    # the same layout, whether from that branch or from a local directory.
    snapshot_dir: str = "metrics/snapshots"
    # The published repositories whose traffic / stars / forks are captured.
    repos: list[str] = Field(
        default_factory=lambda: [
            "LinkedParticles/particles-standard",
            "LinkedParticles/particles-engine-py",
            "LinkedParticles/particles-core-py",
        ]
    )
    # The PyPI distributions whose download counts are captured.
    pypi_distributions: list[str] = Field(
        default_factory=lambda: ["linkedparticles", "linkedparticles-core"]
    )
    # The GoatCounter site code (``<code>.goatcounter.com``). The analytics
    # tag has been live on both published sites since 1.139.6. Without a token
    # the source is recorded as unavailable; it never fails a capture run.
    goatcounter_site: str = "linkedparticles"
    # Tags applied to every corpus entry the deposit half writes, so the
    # deposited series is addressable as one body of material.
    deposit_tags: list[str] = Field(default_factory=lambda: ["metrics", "publication"])
    # Per-request timeout for the capture half's HTTP calls.
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    # Hard cap on stargazer pages fetched per repository (100 stars per page).
    # An abuse-stop, not a tuning knob: the full history is wanted.
    stargazer_max_pages: int = Field(default=50, ge=1)


class CliConfig(BaseModel):
    """Interactive CLI output behaviour.

    ``heartbeat_seconds`` is the silence threshold after which a long-running
    verb prints a liveness line to stderr (see
    ``particles/api/cli/_progress.py``). It is a *threshold*, so it lives in
    config rather than behind a flag; set to 0 to disable entirely. The
    heartbeat is already suppressed whenever stderr is not a TTY, so this knob
    only matters for interactive use.
    """

    heartbeat_seconds: float = Field(default=20.0, ge=0)


class SourcePassageConfig(BaseModel):
    """Source-passage hydration (``particles.ingest.source_passage``).

    Display-only knobs: nothing here reaches ranking. ``locate_min_overlap``
    is the share of a particle's distinct terms a paragraph must contain to be
    offered as the *located* passage when no chunk hash can be matched; below
    it the whole snapshot text is shown instead of a guess.
    ``max_passage_chars`` caps the text returned for any one passage, and
    ``query_show_limit`` how many of a query's top hits ``--show-source``
    hydrates (one blob read each).
    """

    locate_min_overlap: float = Field(default=0.5, gt=0.0, le=1.0)
    max_passage_chars: int = Field(default=6000, ge=200)
    query_show_limit: int = Field(default=5, ge=1)


class ParticlesConfig(BaseModel):
    storage: StorageConfig = Field(default_factory=StorageConfig)
    cli: CliConfig = Field(default_factory=CliConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    http: HttpConfig = Field(default_factory=HttpConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    build: BuildConfig = Field(default_factory=BuildConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)
    claude_code: ClaudeCodeConfig = Field(default_factory=ClaudeCodeConfig)
    agent_memory: AgentMemoryConfig = Field(default_factory=AgentMemoryConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    extraction_scope: ExtractionScopeConfig = Field(default_factory=ExtractionScopeConfig)
    extraction_modality: ExtractionModalityConfig = Field(default_factory=ExtractionModalityConfig)
    modality_regeneration: ModalityRegenerationConfig = Field(
        default_factory=ModalityRegenerationConfig
    )
    extraction_stance: ExtractionStanceConfig = Field(default_factory=ExtractionStanceConfig)
    extraction_polarity: ExtractionPolarityConfig = Field(default_factory=ExtractionPolarityConfig)
    extraction_validity: ExtractionValidityConfig = Field(default_factory=ExtractionValidityConfig)
    extraction_vision: ExtractionVisionConfig = Field(default_factory=ExtractionVisionConfig)
    structured_claim: StructuredClaimConfig = Field(default_factory=StructuredClaimConfig)
    rdf: RdfConfig = Field(default_factory=RdfConfig)
    document_supersession: DocumentSupersessionConfig = Field(
        default_factory=DocumentSupersessionConfig
    )
    document_precedence: DocumentPrecedenceConfig = Field(default_factory=DocumentPrecedenceConfig)
    journal_extractor: JournalExtractorConfig = Field(default_factory=JournalExtractorConfig)
    import_project: ImportProjectConfig = Field(default_factory=ImportProjectConfig)
    web_clipper: WebClipperConfig = Field(default_factory=WebClipperConfig)
    migration: MigrationConfig = Field(default_factory=MigrationConfig)
    trust: TrustConfig = Field(default_factory=TrustConfig)
    reconciliation: ReconciliationConfig = Field(default_factory=ReconciliationConfig)
    content_age_decay: ContentAgeDecayConfig = Field(default_factory=ContentAgeDecayConfig)
    utility: UtilityConfig = Field(default_factory=UtilityConfig)
    owner_lens: OwnerLensConfig = Field(default_factory=OwnerLensConfig)
    observer_scope: ObserverScopeConfig = Field(default_factory=ObserverScopeConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    conformance: ConformanceConfig = Field(default_factory=ConformanceConfig)
    deposit_date: DepositDateConfig = Field(default_factory=DepositDateConfig)
    subjects: SubjectsConfig = Field(default_factory=SubjectsConfig)
    subject_gate: SubjectGateConfig = Field(default_factory=SubjectGateConfig)
    authorities: dict[str, AuthorityConfig] = Field(default_factory=dict)
    wikidata: WikidataConfig = Field(default_factory=WikidataConfig)
    reddit: RedditConfig = Field(default_factory=RedditConfig)
    hackernews: HackerNewsConfig = Field(default_factory=HackerNewsConfig)
    mastodon: MastodonConfig = Field(default_factory=MastodonConfig)
    github: GithubConfig = Field(default_factory=GithubConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    source_passage: SourcePassageConfig = Field(default_factory=SourcePassageConfig)
    contestedness: ContestednessConfig = Field(default_factory=ContestednessConfig)
    lint: LintConfig = Field(default_factory=LintConfig)
    obsidian: ObsidianConfig = Field(default_factory=ObsidianConfig)
    wiki: WikiConfig = Field(default_factory=WikiConfig)
    logseq: LogseqConfig = Field(default_factory=LogseqConfig)
    notion: NotionConfig = Field(default_factory=NotionConfig)
    exporter_common: ExporterCommonConfig = Field(default_factory=ExporterCommonConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    inbox: InboxConfig = Field(default_factory=InboxConfig)
    links_suggest: LinksSuggestConfig = Field(default_factory=LinksSuggestConfig)
    vocabulary: VocabularyConfig = Field(default_factory=VocabularyConfig)
    citation_signal: CitationSignalConfig = Field(default_factory=CitationSignalConfig)
    curation: CurationConfig = Field(default_factory=CurationConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    reindex: ReindexConfig = Field(default_factory=ReindexConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)
    benchmark_memory: MemoryBenchmarkConfig = Field(default_factory=MemoryBenchmarkConfig)
    benchmark_rot: RotBenchmarkConfig = Field(default_factory=RotBenchmarkConfig)
    benchmark_relevance_floor: RelevanceFloorBenchmarkConfig = Field(
        default_factory=RelevanceFloorBenchmarkConfig
    )
    benchmark_leakage: LeakageBenchmarkConfig = Field(default_factory=LeakageBenchmarkConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    refetch_floors: dict[str, int] = Field(default_factory=lambda: dict(_DEFAULT_REFETCH_FLOORS))
    local_refresh: LocalRefreshConfig = Field(default_factory=LocalRefreshConfig)
    rule_sources: RuleSourcesConfig = Field(default_factory=RuleSourcesConfig)

    def reconciliation_mode_for(self, store: str | None) -> str:
        """Effective §6.6 reconciliation mode for a store handle.

        Precedence: an explicit ``reconciliation.per_store`` entry wins; failing
        that, an MCP-write-enabled store defaults to ``"multi"`` (consensus, so
        a confirmed contradiction surfaces as an INCONSISTENCY rather than
        auto-superseding); otherwise the global ``reconciliation.store_mode``.
        """
        handle = store or "default"
        explicit = self.reconciliation.per_store.get(handle)
        if explicit is not None:
            return explicit
        if handle in self.mcp.write.enabled_stores:
            return "multi"
        return self.reconciliation.store_mode

    def digest_listed_stores(self) -> list[str]:
        """Store handles whose memory digest is enumerated in MCP ``resources/list``.

        The write-enabled memory stores (``mcp.write.enabled_stores``) — whose
        standing context an agent loads at session start — plus any extra
        ``mcp.recall.digest_stores``, de-duplicated and order-stable (enabled
        first). The resource *template* ``particles://digest/{store}`` addresses
        any store on demand regardless of this list; this is only the
        auto-listed (discoverable) subset.
        """
        seen: dict[str, None] = {}
        for handle in (*self.mcp.write.enabled_stores, *self.mcp.recall.digest_stores):
            seen.setdefault(handle, None)
        return list(seen)

    @model_validator(mode="after")
    def _validate_write_store_modes(self) -> ParticlesConfig:
        """Reject a write-enabled store that resolves to ``single``.

        Write stores default to ``multi``; the only way to reach ``single`` is an
        explicit ``reconciliation.per_store`` override, which is a configuration
        error — it would re-open the rung-2 auto-supersede the §6 defence closes.
        """
        for handle in self.mcp.write.enabled_stores:
            if self.reconciliation_mode_for(handle) == "single":
                raise ValueError(
                    f"MCP write-enabled store {handle!r} resolves to reconciliation "
                    f"store_mode 'single'; write stores must reconcile in 'multi' "
                    f". Remove the reconciliation.per_store override or "
                    f"set reconciliation.per_store[{handle!r}] = 'multi'."
                )
        return self


# ---------------------------------------------------------------------------
# Layer declaration
# ---------------------------------------------------------------------------

#: Top-level :class:`ParticlesConfig` sections that a **Client-layer** module
#: reads.
#:
#: Both distributions ship one config model, because they share one import
#: package — `particles/config.py` rides the Client distribution and
#: the Engine has none of its own. The consequence is that a
#: ``linkedparticles-core``-only install carries every section, including the
#: two-thirds of them nothing in that install can act on. This frozenset is how
#: that surface is made legible instead of carved: it names the sections a
#: core-alone consumer can actually set to effect, and it is what tags each
#: section in ``config.yaml.sample`` ``[client]`` or ``[engine]``.
#:
#: It is **documentation with a test behind it, not a runtime gate.** Nothing
#: filters, rejects, or warns on an Engine section at load time; a core-alone
#: install still validates and holds all 64. Two checks in
#: ``tests/test_config_client_sections.py`` keep it honest — the sections a
#: Client module actually reads must be a **subset** of this set (an undeclared
#: read fails; a section that stops being read does not churn it), and the
#: sample's tags must agree with it.
#:
#: It is also the seam a future Client/Engine config carve would cut along, kept
#: measured so that carve discovers nothing new about where the line is. The
#: carve itself is reserved, not taken.
CLIENT_SECTIONS: frozenset[str] = frozenset(
    {
        "confidence",
        "content_age_decay",
        "contestedness",
        "embeddings",
        "extraction",
        "extraction_modality",
        "extraction_polarity",
        "extraction_scope",
        "extraction_stance",
        "extraction_validity",
        "extraction_vision",
        "github",
        "hackernews",
        "http",
        "journal_extractor",
        "llm",
        "mastodon",
        "migration",
        "rdf",
        "reddit",
        "storage",
        "structured_claim",
        "trust",
        "wikidata",
    }
)

# ---------------------------------------------------------------------------
# Env var overrides (backward-compatible names)
# ---------------------------------------------------------------------------

# (env_var, section, field) — string value; Pydantic coerces types on validation
_ENV_OVERRIDES: list[tuple[str, str, str]] = [
    ("DATABASE_URL", "storage", "database_url"),
    ("PARTICLES_BLOB_DIR", "storage", "blob_dir"),
    ("PARTICLES_USER_AGENT", "http", "user_agent"),
    ("PARTICLES_MAX_REQUEST_BODY_BYTES", "api", "max_request_body_bytes"),
    ("PARTICLES_API_BIND_HOST", "api", "bind_host"),
    ("PARTICLES_API_RATE_LIMIT_PER_MINUTE", "api", "rate_limit_per_minute"),
    ("PARTICLES_API_REQUIRE_AUTH_FOR_READS", "api", "require_auth_for_reads"),
    ("PARTICLES_MAX_PAGE_SIZE", "storage", "max_page_size"),
    # Stamped by the image, not set by an operator (deploy/Dockerfile).
    ("PARTICLES_BUILD_DATE", "build", "date"),
    ("PARTICLES_ENGINE_BASE_URL", "engine", "base_url"),
    ("PARTICLES_ENGINE_TIMEOUT_SECONDS", "engine", "timeout_seconds"),
    ("PARTICLES_OBSERVABILITY_ENABLED", "observability", "enabled"),
    ("PARTICLES_OBSERVABILITY_EXPORTER", "observability", "exporter"),
    ("PARTICLES_OBSERVABILITY_ENDPOINT", "observability", "endpoint"),
    ("PARTICLES_OBSERVABILITY_SERVICE_NAME", "observability", "service_name"),
    ("PARTICLES_OBSERVABILITY_SAMPLE_RATIO", "observability", "sample_ratio"),
    # Dotted field ⇒ one level of nesting: storage.write_lock.*.
    ("PARTICLES_WRITE_LOCK_ENABLED", "storage", "write_lock.enabled"),
    ("PARTICLES_WRITE_LOCK_TIMEOUT_SECONDS", "storage", "write_lock.timeout_seconds"),
    ("PARTICLES_EMBEDDINGS_PROGRESS_BARS", "embeddings", "progress_bars"),
    ("PARTICLES_EMBEDDINGS_DIM", "embeddings", "dim"),
    ("PARTICLES_EMBEDDINGS_NORMALIZATION", "embeddings", "normalization"),
    ("TRUST_DIFFERENTIAL_THRESHOLD", "trust", "differential_threshold"),
    ("RECONCILIATION_STORE_MODE", "reconciliation", "store_mode"),
    ("TRUST_CASCADE_MAX_PER_RUN", "trust", "cascade_max_per_run"),
    ("TRUST_CASCADE_MIN_REVIEWER_CONFIRMATIONS", "trust", "cascade_min_reviewer_confirmations"),
    ("PDF_PAGE_OVERLAP_LINES", "extraction", "pdf_page_overlap_lines"),
    ("HTML_CHUNK_SIZE", "extraction", "html_chunk_size"),
    ("HTML_CHUNK_OVERLAP_LINES", "extraction", "html_chunk_overlap_lines"),
    ("OBSIDIAN_DEFAULT_OUTPUT_PATH", "obsidian", "default_output_path"),
    # Moved to exporter_common in 0.42.1 so Logseq honours the same gate.
    # Env var name unchanged for backwards compat.
    ("OBSIDIAN_SYNTHESIS_MIN_PARTICLES", "exporter_common", "synthesis_min_particles"),
    ("INBOX_FILE_PATH", "inbox", "file_path"),
    ("INBOX_POLL_INTERVAL_SECONDS", "inbox", "poll_interval_seconds"),
    ("WIKI_LAYER_B_UNRELATED_TOLERANCE", "wiki", "layer_b_unrelated_tolerance"),
    ("WIKI_LAYER_B_RETRY_ENABLED", "wiki", "layer_b_retry_enabled"),
    ("AUDIT_CONFIRM_CALL_THRESHOLD", "audit", "confirm_call_threshold"),
    ("BENCHMARK_MEMORY_CONFIRM_CALL_THRESHOLD", "benchmark_memory", "confirm_call_threshold"),
    ("BENCHMARK_RUNS_DIR", "benchmark", "runs_dir"),
    ("BENCHMARK_CONFIRM_CALL_THRESHOLD", "benchmark", "confirm_call_threshold"),
    ("BENCHMARK_RECORD_CLAIM_TEXT", "benchmark", "record_claim_text"),
    ("BENCHMARK_SUBJECT_AWARE_MATCHING", "benchmark", "subject_aware_matching"),
    ("AUDIT_MAX_CONTRADICTION_PROBES", "audit", "max_contradiction_probes"),
    ("AUDIT_VERIFY_CONTRADICTIONS", "audit", "verify_contradictions"),
    ("AUDIT_MAX_CONTRADICTION_VERIFICATIONS", "audit", "max_contradiction_verifications"),
    # Resident daemon mode. ``engine serve --daemon`` sets
    # PARTICLES_DAEMON_ENABLED for its own process before reset_config() — the
    # same launcher-configures-itself bootstrap the bind-host override uses.
    ("PARTICLES_DAEMON_ENABLED", "daemon", "enabled"),
    ("PARTICLES_DAEMON_STORE", "daemon", "store"),
    ("PARTICLES_DAEMON_WEB_CLIPPER_DIR", "daemon", "web_clipper_dir"),
]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _find_config_file() -> Path | None:
    """Resolve the ``config.yaml`` this process should load.

    Order: ``PARTICLES_CONFIG`` (absolute authority — a named file that does
    not exist resolves to ``None`` rather than falling through), then the
    working directory, then the nearest ancestor holding a ``config.yaml``,
    first match wins.

    The upward walk exists because falling back to compiled defaults is
    *silent*: a process launched from a subdirectory — a git worktree, a hook
    spawn, a ``cd`` into ``scripts/`` — used to lose every knob the operator
    set at the repo root, and the divergence surfaced weeks later as blobs
    written somewhere nobody looks (the 2026-07-18 sharding incident).

    The walk stops after examining the directory that holds a ``.git`` entry,
    so a ``config.yaml`` in ``$HOME`` never becomes the ambient config for
    every project beneath it: a config at a repo root is a deliberate project
    setting, one above it is an accident waiting to be inherited.
    """
    explicit = os.environ.get("PARTICLES_CONFIG")
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    start = Path.cwd()
    for directory in (start, *start.parents):
        candidate = directory / "config.yaml"
        if candidate.exists():
            return candidate
        # `exists()`, not `is_dir()`: in a git worktree (and in a submodule)
        # `.git` is a *file* containing a `gitdir:` pointer. That is precisely
        # the case this bound has to catch, since worktrees are where the
        # subdirectory-launch problem showed up.
        if (directory / ".git").exists():
            return None
    return None


def sqlite_file_path(url: str) -> str | None:
    """Return a file-based SQLite URL's on-disk path, or ``None`` for memory/non-SQLite.

    ``sqlite+aiosqlite:///./particles.db`` → ``./particles.db``; an in-memory URL
    (``:memory:`` or a path-less ``sqlite://``) and any non-SQLite URL → ``None``.

    Client-layer (pure string parsing, no store access) so the config resolver
    below and the Engine's write-lock derivation share one rule instead of two.
    """
    if not url.startswith("sqlite") or ":///" not in url:
        return None
    path = url.split(":///", 1)[1]
    if not path or path == ":memory:":
        return None
    return path


def resolve_store_adjacent_path(value: str) -> Path:
    """Resolve a store-adjacent path, anchoring relative values to the store.

    A store and its content must travel together. The write lock already obeys
    this — it derives ``<db_file>.writelock`` *beside the store DB* — while
    ``storage.blob_dir`` resolved against the process working directory. That
    inconsistency is the bug: an absolute ``DATABASE_URL`` (env var or config)
    combined with the relative ``./corpus_blobs`` default silently decouples a
    store's rows from its content, scattering blobs across whichever directory
    each process happened to run in. This extends the write-lock rule to every
    store-adjacent path.

    Resolution order:

    1. An **absolute** value is returned unchanged (``~`` expanded).
    2. A relative value anchors to the **store's own directory** — the parent of
       the default store's SQLite file — whenever that path is absolute.
    3. Otherwise the **loaded config file's directory** (Postgres and other
       non-file DSNs have no store directory).
    4. Otherwise the working directory, as before.

    Step 2 deliberately does nothing when the DSN is itself relative: the store
    and its blobs are then both cwd-relative and already travel together. Only
    the *mixed* case — absolute store, relative sidecar — is corrected.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    db_file = sqlite_file_path(get_config().storage.database_url)
    if db_file is not None:
        anchor = Path(db_file).expanduser()
        if anchor.is_absolute():
            return anchor.parent / path

    config_file = _find_config_file()
    if config_file is not None:
        return config_file.parent.resolve() / path

    return path


def _load_config() -> ParticlesConfig:
    raw: dict[str, Any] = {}

    config_path = _find_config_file()
    if config_path is not None:
        with open(config_path) as f:
            loaded = yaml.safe_load(f)
        if isinstance(loaded, dict):
            raw = loaded

    _migrate_legacy_keys(raw)

    for env_var, section, field in _ENV_OVERRIDES:
        value = os.environ.get(env_var)
        if value is not None:
            if section not in raw:
                raw[section] = {}
            # A dotted ``field`` (e.g. ``write_lock.enabled``) overrides one
            # level of nesting under the section.
            if "." in field:
                outer, inner = field.split(".", 1)
                raw[section].setdefault(outer, {})[inner] = value
            else:
                raw[section][field] = value

    return ParticlesConfig.model_validate(raw)


def validate_config() -> tuple[Path | None, ParticlesConfig]:
    """Resolve and validate the active configuration without touching the singleton.

    The seam behind ``particles config validate``. Returns the config
    file that would be loaded (``None`` when no ``config.yaml`` is found and only
    compiled-in defaults + env overrides apply) and the validated
    :class:`ParticlesConfig`.

    Raises:
        pydantic.ValidationError: a field failed validation.
        yaml.YAMLError: the config file is not parseable YAML.

    Unlike :func:`get_config`, this never reads or writes the cached singleton,
    so it always reflects the file on disk right now.
    """
    return _find_config_file(), _load_config()


def _migrate_legacy_keys(raw: dict[str, Any]) -> None:
    """Migrate deprecated config keys in-place, logging a one-time warning.

    Four migrations are active:

    * ``lint.co_evidential_candidate_threshold`` was renamed to
      ``links_suggest.candidate_threshold``.
    * per-purpose model selection moved into the ``llm`` section. The
      old ``extraction.model`` key migrates to ``llm.default.model`` (it drove
      extraction, query-response, and the reconcile-ladder contradiction check
      — the "general" purposes), and ``wiki.model`` migrates to
      ``llm.synthesis.model``.
    * ``extraction.query_max_tokens`` — the query op's answer budget, never an
      extraction knob — moved to ``query.answer_max_tokens``.

    * ``benchmark_memory.price_per_mtok`` moved to ``llm.price_per_mtok``,
      ``benchmark_memory.batch_discount`` to ``llm.batch_discount``, and
      ``benchmark_rot.estimate_extraction_prompt_overhead_tokens`` to
      ``extraction.estimate_prompt_overhead_tokens``, shared with the audit
      estimate and the measured usage line.

    Each old key is honoured only when its new home is unset, so an operator
    who has already moved to the new key wins. The shims may be removed once
    their deprecation cycle ends.
    """
    _migrate_co_evidential_threshold(raw)
    _migrate_llm_model_keys(raw)
    _migrate_query_answer_max_tokens(raw)
    _migrate_shared_estimate_keys(raw)


def _migrate_co_evidential_threshold(raw: dict[str, Any]) -> None:
    lint_section = raw.get("lint")
    if not isinstance(lint_section, dict):
        return
    legacy = lint_section.pop("co_evidential_candidate_threshold", None)
    if legacy is None:
        return
    links = raw.setdefault("links_suggest", {})
    if isinstance(links, dict) and "candidate_threshold" not in links:
        links["candidate_threshold"] = legacy
    log.warning(
        "config: 'lint.co_evidential_candidate_threshold' is deprecated "
        " and will be removed in a future release. Use "
        "'links_suggest.candidate_threshold' instead."
    )


def _migrate_llm_model_keys(raw: dict[str, Any]) -> None:
    """extraction.model → llm.default.model; wiki.model → llm.synthesis.model."""
    for section_name, purpose in (("extraction", "default"), ("wiki", "synthesis")):
        section = raw.get(section_name)
        if not isinstance(section, dict):
            continue
        legacy_model = section.pop("model", None)
        if legacy_model is None:
            continue
        llm = raw.setdefault("llm", {})
        if not isinstance(llm, dict):
            continue
        target = llm.setdefault(purpose, {})
        if isinstance(target, dict) and "model" not in target:
            target["model"] = legacy_model
        log.warning(
            "config: '%s.model' is deprecated and will be removed in "
            "a future release. Use 'llm.%s.model' instead.",
            section_name,
            purpose,
        )


def _migrate_query_answer_max_tokens(raw: dict[str, Any]) -> None:
    """``extraction.query_max_tokens`` → ``query.answer_max_tokens``.

    The knob only ever drove the §9.3 answer call; living under ``extraction``
    made it read as an extraction cap and hid it from the operator looking for
    why an answer degraded to a listing. The value is carried over verbatim —
    an operator who had deliberately raised it keeps their setting — but the
    *default* changed (1024 → 4096), so an operator who had merely pinned the
    old default in their file is warned and should drop the key.
    """
    section = raw.get("extraction")
    if not isinstance(section, dict):
        return
    legacy = section.pop("query_max_tokens", None)
    if legacy is None:
        return
    query_section = raw.setdefault("query", {})
    if isinstance(query_section, dict) and "answer_max_tokens" not in query_section:
        query_section["answer_max_tokens"] = legacy
    log.warning(
        "config: 'extraction.query_max_tokens' is deprecated and will be removed "
        "in a future release. Use 'query.answer_max_tokens' instead — note the "
        "default rose from 1024 to 4096, because an extended-thinking model "
        "spends its thinking tokens from this same budget and 1024 left it with "
        "none for the answer."
    )


def _migrate_shared_estimate_keys(raw: dict[str, Any]) -> None:
    """Cost-estimate keys lifted out of the benchmark sections so the audit shares them.

    ``benchmark_memory.price_per_mtok`` → ``llm.price_per_mtok`` (entries
    already under the new key win, key by key),
    ``benchmark_memory.batch_discount`` → ``llm.batch_discount``, and
    ``benchmark_rot.estimate_extraction_prompt_overhead_tokens`` →
    ``extraction.estimate_prompt_overhead_tokens``.
    """
    moves = (
        ("benchmark_memory", "price_per_mtok", "llm", "price_per_mtok"),
        ("benchmark_memory", "batch_discount", "llm", "batch_discount"),
        (
            "benchmark_rot",
            "estimate_extraction_prompt_overhead_tokens",
            "extraction",
            "estimate_prompt_overhead_tokens",
        ),
    )
    for old_section, old_key, new_section, new_key in moves:
        section = raw.get(old_section)
        if not isinstance(section, dict) or old_key not in section:
            continue
        legacy = section.pop(old_key)
        target = raw.setdefault(new_section, {})
        if isinstance(target, dict):
            current = target.get(new_key)
            if isinstance(legacy, dict) and isinstance(current, dict):
                target[new_key] = {**legacy, **current}
            elif current is None:
                target[new_key] = legacy
        log.warning(
            "config: '%s.%s' is deprecated and will be removed in a future "
            "release. Use '%s.%s' instead.",
            old_section,
            old_key,
            new_section,
            new_key,
        )


_config: ParticlesConfig | None = None
_reset_hooks: list[Callable[[], None]] = []


def get_config() -> ParticlesConfig:
    """Return the process-level config singleton."""
    global _config
    if _config is None:
        _config = _load_config()
    return _config


def register_reset_hook(hook: Callable[[], None]) -> None:
    """Register a callback to run on every :func:`reset_config`.

    This inverts the former ``config`` → ``db`` coupling: the
    Engine layer (:mod:`particles.db`) registers its ``reset_engine`` here at
    import time, so ``config`` — a Client-layer module — need not import any
    Engine module. The Client/Engine import boundary is enforced by
    ``import-linter``.
    """
    _reset_hooks.append(hook)


def reset_config() -> None:
    """Reset the config singleton — for testing and CLI reloads.

    Also runs every registered reset hook (see :func:`register_reset_hook`).
    The Engine registers ``reset_engine`` so the cached SQLAlchemy engine is
    discarded and a subsequent ``get_engine()`` rebuilds against the new
    ``storage.database_url``.
    """
    global _config
    _config = None
    for hook in _reset_hooks:
        hook()
