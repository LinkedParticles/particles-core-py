"""Centralised reads for secret environment variables.

Secrets must never appear in ``config.yaml``: they are read from
the process environment only. This module is the single seam through which
application code touches ``os.environ`` for those values, so a future
migration to a secrets manager (1Password, AWS Secrets Manager, sops, …)
only has to edit one file.

One function per secret. Each function reads ``os.environ.get(...)`` exactly
once per call and preserves the original call-site behaviour (some raise on
missing values, some return ``None``, some default to a sentinel).

``PARTICLES_CONFIG`` is the bootstrap path override for ``particles/config.py``
itself; the helper here is exposed for symmetry but ``config.py`` continues
to read the variable directly to avoid an import cycle. The generic
``_ENV_OVERRIDES`` mechanism in ``config.py`` is structural and intentionally
not consolidated here — it overrides operational config fields, not secrets.
"""

from __future__ import annotations

import os

_PARTICLES_API_KEY_DEFAULT = "dev-key"


def get_anthropic_api_key() -> str:
    """Return ``ANTHROPIC_API_KEY`` or raise ``ValueError`` if unset.

    Used by the shared Anthropic client (``particles.llm.get_client``). The
    error message points the operator at the console where they can mint a
    key — the Anthropic SDK's own "Could not resolve authentication method"
    message is too generic to act on.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it before running extract / query / lint: "
            "  export ANTHROPIC_API_KEY=sk-ant-…  "
            "(get a key at https://console.anthropic.com/settings/keys)"
        )
    return api_key


def get_anthropic_api_key_optional() -> str | None:
    """Return ``ANTHROPIC_API_KEY`` or ``None`` if unset.

    For callers that gracefully degrade when the key is absent (e.g.
    optional LLM-synthesis passes in the Obsidian / wiki exporters).
    """
    return os.environ.get("ANTHROPIC_API_KEY") or None


def get_github_api_key_optional() -> str | None:
    """Return ``GITHUB_API_KEY`` or ``None`` if unset.

    This secret is optional — anonymous requests are allowed
    (60 req/hr) and a key only raises the limit to 5000 req/hr.
    """
    return os.environ.get("GITHUB_API_KEY") or None


def get_numista_api_key() -> str:
    """Return ``NUMISTA_API_KEY`` or raise ``ValueError`` if unset.

    Required for Numista deposits — the Numista API has no anonymous mode.
    """
    api_key = os.environ.get("NUMISTA_API_KEY")
    if not api_key:
        raise ValueError(
            "NUMISTA_API_KEY environment variable is required for Numista deposits. "
            "Get a key at https://en.numista.com/api/doc/"
        )
    return api_key


def get_notion_api_key() -> str:
    """Return ``NOTION_API_KEY`` or raise ``ValueError`` if unset.

    Required for the Notion exporter (``export notion``) — Notion's API has no
    anonymous mode, so the exporter cannot write to a workspace without a token.
    Shaped after :func:`get_numista_api_key` (raise-on-missing); the error points
    the operator at where to mint an internal integration token. The token is a
    *secret*: it never appears in ``config.yaml``, never on
    ``ParticlesConfig``, never in the CLI argv, and never in a summary dict —
    this getter is the only seam through which it is read.
    """
    api_key = os.environ.get("NOTION_API_KEY")
    if not api_key:
        raise ValueError(
            "NOTION_API_KEY environment variable is required for `export notion`. "
            "Create an internal integration and copy its token at "
            "https://www.notion.so/my-integrations , then share the target "
            "database with that integration."
        )
    return api_key


def get_particles_api_key() -> str:
    """Return ``PARTICLES_API_KEY`` or the ``"dev-key"`` sentinel.

    The FastAPI bearer-auth dependency treats ``"dev-key"`` as a signal to
    skip authentication entirely (local development). Production deployments
    set this to a real secret; the startup warning in
    ``warn_if_dev_auth_in_use`` shouts if it is left unset.
    """
    return os.environ.get("PARTICLES_API_KEY", _PARTICLES_API_KEY_DEFAULT)


def get_engine_token_optional() -> str | None:
    """Return ``PARTICLES_ENGINE_TOKEN`` or ``None`` if unset.

    The bearer token the **thin client** presents to a remote engine when
    ``engine.base_url`` is configured (``HttpBackend``). It is deliberately
    *distinct* from ``PARTICLES_API_KEY`` (which is the key the *server* expects,
    read by the FastAPI auth dependency): the engine reads its copy, the client
    reads the token it sends. For a single operator they may hold the same value,
    but they are separate roles, so they are separate secrets.

    Optional: a loopback dev engine runs with the ``dev-key`` skip and needs no
    token, so ``HttpBackend`` omits the ``Authorization`` header when this is
    ``None``. A non-loopback engine will reject the unauthenticated request
    (401), which is the intended, legible failure.
    """
    return os.environ.get("PARTICLES_ENGINE_TOKEN") or None


def get_llm_api_key_optional(provider_name: str) -> str | None:
    """Return the API key for a named completion provider, or ``None``.

    Reads ``PARTICLES_LLM_API_KEY_<NAME>`` where ``<NAME>`` is the
    ``llm.providers`` entry name uppercased with non-alphanumerics mapped to
    ``_`` (``openai`` → ``PARTICLES_LLM_API_KEY_OPENAI``). For the compiled-in
    ``local`` entry the legacy ``PARTICLES_LOCAL_LLM_API_KEY`` is
    honoured as a fallback. Optional, mirroring
    :func:`get_local_llm_api_key_optional`: bare local runtimes need no key,
    and the adapter omits the ``Authorization`` header when this is ``None``.
    A *secret*: never in ``config.yaml``, never a field on
    ``ParticlesConfig``.
    """
    suffix = "".join(c if c.isalnum() else "_" for c in provider_name).upper()
    key = os.environ.get(f"PARTICLES_LLM_API_KEY_{suffix}") or None
    if key is None and provider_name == "local":
        return get_local_llm_api_key_optional()
    return key


def get_local_llm_api_key_optional() -> str | None:
    """Return ``PARTICLES_LOCAL_LLM_API_KEY`` or ``None`` if unset.

    The bearer token the ``local`` completion adapter
    (``particles/llm/adapters/local.py``) presents to an OpenAI-compatible
    endpoint. Optional: bare local runtimes (Ollama, llama.cpp) need no key, so
    the adapter omits the ``Authorization`` header when this is ``None``; only a
    gateway that enforces auth requires it. A *secret*: it never
    appears in ``config.yaml`` and is never a field on ``ParticlesConfig`` — the
    non-secret endpoint URL lives in ``config.llm.local.base_url``, the token
    here, mirroring the ``engine.base_url`` / ``PARTICLES_ENGINE_TOKEN``
    split.
    """
    return os.environ.get("PARTICLES_LOCAL_LLM_API_KEY") or None


def get_otel_exporter_headers_optional() -> str | None:
    """Return ``PARTICLES_OTEL_EXPORTER_HEADERS`` or ``None`` if unset.

    The credential the OTLP exporter presents to an authenticated collector or a
    SaaS observability backend, in the W3C OTLP-headers format — a comma-separated
    ``key=value`` list, e.g. ``"authorization=Bearer abc123"``. It is a *secret*:
    it never appears in ``config.yaml`` and is never a field on
    ``ParticlesConfig``. The non-secret ``observability.endpoint`` URL lives in
    config; the token lives here — mirroring the ``engine.base_url`` /
    ``PARTICLES_ENGINE_TOKEN`` split.

    Optional: the console exporter and an unauthenticated local collector need no
    credential, so ``setup_observability`` omits exporter headers when this is
    ``None``.
    """
    return os.environ.get("PARTICLES_OTEL_EXPORTER_HEADERS") or None


def get_particles_config_path() -> str | None:
    """Return ``PARTICLES_CONFIG`` (config-file path override) or ``None``.

    Exposed for symmetry. ``particles.config`` itself continues to read the
    env var directly to avoid an import cycle during bootstrap.
    """
    return os.environ.get("PARTICLES_CONFIG") or None
