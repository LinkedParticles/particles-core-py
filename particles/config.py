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
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

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
    # and its requests come back as None — the call site's existing
    # probe-unavailable degradation — rather than stalling the run. The API's
    # own expiry is 24h; this default keeps a nightly cycle inside its night.
    max_wait_seconds: float = Field(default=3600.0, gt=0.0)


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
    # Named OpenAI-compatible providers. Keys are operator-chosen
    # provider names — the calibration/disclosure key is "<name>:<model>"
    #, so treat a rename as a recalibration event. "anthropic" is
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


class ExtractionConfig(BaseModel):
    max_tokens: int = 8192
    query_max_tokens: int = 1024
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
    # don't re-mint a claim the store already holds verbatim.
    duplicate_suppression: DuplicateSuppressionConfig = Field(
        default_factory=DuplicateSuppressionConfig
    )


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
    extractor plus the operator's ``SourceTrustStatement``s), per the two-quantity separation. It is overridden per-triple only when the document
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
    touched. Single-trust-order stores only in v1 (matching the trust rung). ``enabled: false`` reproduces the pre-cap-2 behaviour exactly
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
    ``EXPERIENTIAL`` particles, tags opinions ``EVALUATIVE``, and emits the ``NARRATIVE`` graph for the entry. ``enabled: false`` makes the
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
    suffixes deposited as ``PYTHON_SOURCE`` (the first registered glob instance); ``ignore_dirs`` are directory names pruned during the walk
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


class ReconciliationConfig(BaseModel):
    """Cross-entry §6.6 reconciliation policy.

    ``store_mode`` selects the trust-resolution regime applied when a
    contradicting pair clears the contradiction-signal gate:

    * ``single`` (default) — a single global trust order. §6.6 rung 2
      auto-supersede fires: the higher-trust claim wins and the lower-trust
      one is demoted (today's behavior — unchanged; preserves the invariant §1 that single-store solo behavior is byte-for-byte the same).
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

    The default ``0.015`` is **empirically calibrated** (re-centred
, re-measured post-dedup, re-centred
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

    If the sweep's *ceiling* is what binds for you, the fix is deduplication
    , not a smaller ``λ`` — but check that your dedup pass can
    actually *reach* the clusters setting the ceiling. On the dogfood store it
    could not at first: merge cut near-duplicate mass from 16.0% to
    3.1% of ACTIVE, yet the ceiling *fell* (0.0190 → 0.0165 at ``N = 200``)
    rather than rising, because the two clusters that set it were 21
    **byte-identical** copies each carrying 1/21 and 0/21 subject links, and
    ``suggest_co_evidential`` iterates Subjects so it never saw them
     — reporting zero groups while 211 exact-duplicate groups / 534
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

    The literal matcher (deterministic, zero-cost) is always on when mining
    runs; ``behavioural_matching`` adds the bounded LLM soft-guideline matcher
    , capped at ``max_behavioural_calls`` per run (
    cost-discipline).

    ``behavioural_candidate_limit`` bounds *which* beliefs compete for that
    call budget: the behavioural tier's candidate set is every ACTIVE belief
    the literal tier did not match — nearly the whole store — so without a
    relevance filter the budget is spent on the first N beliefs in list order.
    The pre-filter ranks candidates by embedding similarity between the
    session's action lines and each belief (the local model; no LLM cost) and
    keeps the top ``behavioural_candidate_limit``. ``0`` disables the filter
    (every unmatched belief competes, pre-filter-free legacy behaviour).
    """

    enabled: bool = True
    behavioural_matching: bool = True
    max_behavioural_calls: int = Field(default=50, ge=0)
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
    one store each need their own, which a store-resident field would defeat
    . This is the single-viewer binding of the viewer
    seam, valid up to multi-tenant line.

    A **list**, not a scalar, because a viewer's Subject fragments in practice
    ("Jeff" / "Jeff Gage") until the N→1 merge lands. Entries are
    canonical names or Subject ids and resolve **locally only** — never a live
    authority lookup, since the digest is a zero-LLM, zero-network surface
    . Resolve-or-inert: if nothing resolves the
    lens is inert and the ordering is byte-identical to ``enabled: false``.

    ``rank_lift`` (``ω``) is **store-specific and must be calibrated** against
    the deployment's own confidence spread and cohort size — see
    ``particles quality rank-lift-sweep``. It ships ``0.0`` (inert) so the lens
    changes nothing until an operator sets both ``subjects`` and a swept ``ω``.
    """

    enabled: bool = True
    subjects: list[str] = Field(default_factory=list)
    rank_lift: float = Field(default=0.0, ge=0.0)


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


class SubjectGateConfig(BaseModel):
    """Extraction-time non-entity subject gate.

    A general, deterministic lexical gate that suppresses promotion of
    non-entity token classes (self-vocabulary enums, reference / doc-ID codes,
    filenames, CLI command strings, snake_case identifiers) to Subjects, applied
    to every extractor's candidates in the Extract pipeline before subject
    resolution.
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
    # materially reduced by content age alone (decay; surfaced in lint
    #). A particle fires when 1 - recency_factor >= this threshold —
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
    # explicit path. Typical operator value:
    #   ~/Library/Mobile Documents/iCloud~md~obsidian/Documents/MyVault
    default_output_path: str | None = None
    # ``synthesis_min_particles`` moved to ``exporter_common`` in 0.42.1
    # so the Logseq exporter honours the same gate. Read it via
    # ``get_config().exporter_common.synthesis_min_particles``.
    # when true (default), `export obsidian --with-synthesis` also emits
    # one note per ACTIVE NARRATIVE under `Narratives/`, rendered as cited prose
    #. Set false to keep only per-subject articles.
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

    The gate on *which* entries are refreshed is not here: it is the ``fetch_policy = LAZY`` flag on the entry itself, so refreshing stays an
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
    # the hook only deposits; extraction runs on the store's schedule
    #.
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


class CurationConfig(BaseModel):
    """Curation surface — the bus-stop-editing queue + session model.

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
    # Soft cap for the dependency-count normalizer: log1p(n) / log1p(cap).
    dependency_norm_cap: int = Field(default=20, ge=1)
    # Age (days) at which staleness_age saturates to 1.0.
    staleness_norm_days: float = Field(default=365.0, gt=0.0)
    leverage_weights: CurationLeverageWeights = Field(default_factory=CurationLeverageWeights)
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
    # gives the nightly consolidation cadence a full day of slack, so a single
    # skipped run does not cry wolf.
    snapshot_max_age_hours: float = Field(default=36.0, gt=0.0)
    # Collections kept per store — a small ring so a bad build is one row from
    # a rollback. Snapshots are pure cache; nothing references them.
    snapshot_retain: int = Field(default=3, ge=1)
    # Eviction horizon for delta-scoped cards carried across builds (
    # §4). A carried card whose beliefs are never touched is dropped after this
    # many days rather than accreting forever; re-probing the tail is future work.
    snapshot_carry_forward_days: int = Field(default=30, ge=1)


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
    # Stale cycle-lock reclaim: a consolidate.lock whose pid is dead or whose
    # age exceeds this is reclaimed, so a crashed run cannot wedge the cadence.
    lock_timeout_minutes: int = Field(default=120, ge=1)
    # Abstraction-promotion pass, between utility mining and the
    # projection re-render.
    abstraction: AbstractionConfig = Field(default_factory=AbstractionConfig)


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
    # Top-k for the retrieval stage and the qa_particles context.
    top_k: int = Field(default=10, ge=1)
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
    # reported as an infra failure and excluded from the accuracy denominator
    #. A no-text-block reply is never retried — it is deterministic
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


class BenchmarkConfig(BaseModel):
    """Benchmark-harness run persistence (family).

    ``runs_dir`` is where ``particles extractor benchmark`` persists each
    run's report as one JSON file (envelope: schema-format stamp + the
    resolved extraction provider:model pairing + the full §13.3 report).
    Persistence is a CLI concern — the harness itself stays report-only.
    The per-run flag ``--no-save`` skips it for throwaway experiments.
    Sibling harnesses (modality / polarity / validity / compare) may adopt
    the same directory later; filenames carry the harness kind.

    ``confirm_call_threshold`` gates the repeat-runs mode (``--runs N``), whose cost scales linearly with N: above this many projected
    extraction calls the CLI requires confirmation (mirrors
    ``audit.confirm_call_threshold``; ``--yes`` pre-confirms).
    A single run is never gated — the estimate only prints when N > 1.
    """

    runs_dir: str = "~/.particles/benchmark/runs"
    confirm_call_threshold: int = Field(default=50, ge=0)


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


class ParticlesConfig(BaseModel):
    storage: StorageConfig = Field(default_factory=StorageConfig)
    cli: CliConfig = Field(default_factory=CliConfig)
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
    citation_signal: CitationSignalConfig = Field(default_factory=CitationSignalConfig)
    curation: CurationConfig = Field(default_factory=CurationConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)
    benchmark_memory: MemoryBenchmarkConfig = Field(default_factory=MemoryBenchmarkConfig)
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
    ("AUDIT_MAX_CONTRADICTION_PROBES", "audit", "max_contradiction_probes"),
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

    Two migrations are active:

    * ``lint.co_evidential_candidate_threshold`` was renamed to
      ``links_suggest.candidate_threshold``.
    * per-purpose model selection moved into the ``llm`` section. The
      old ``extraction.model`` key migrates to ``llm.default.model`` (it drove
      extraction, query-response, and the reconcile-ladder contradiction check
      — the "general" purposes), and ``wiki.model`` migrates to
      ``llm.synthesis.model``.

    Each old key is honoured only when its new home is unset, so an operator
    who has already moved to the new key wins. The shims may be removed once
    their deprecation cycle ends.
    """
    _migrate_co_evidential_threshold(raw)
    _migrate_llm_model_keys(raw)


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
