# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

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

import bisect
import os
import re
from collections.abc import Iterable

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


def get_anthropic_base_url_optional() -> str | None:
    """Return ``ANTHROPIC_BASE_URL`` or ``None`` if unset.

    Not a secret: the Anthropic SDK reads this variable itself to override its
    endpoint. It is read here, at the module that owns the environment
    boundary, so the no-egress pre-check and posture report name the
    host the SDK will really connect to.
    """
    return os.environ.get("ANTHROPIC_BASE_URL") or None


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

    An empty or whitespace-only value is no key, so it reads as the sentinel
    too: the engine is keyless, serves reads only, and the loopback startup
    gate applies. Before this an empty key counted as a key, so the gate let a
    non-loopback bind serve its reads to the network (security review
    2026-10-25, the empty-key residual). Every reader of the key
    goes through this function, so they all agree.
    """
    value = os.environ.get("PARTICLES_API_KEY")
    if value is None or not value.strip():
        return _PARTICLES_API_KEY_DEFAULT
    return value


def get_metrics_github_token() -> str:
    """Return ``PARTICLES_METRICS_GITHUB_TOKEN`` or raise ``ValueError`` if unset.

    The publication-metrics capture reads the **traffic** endpoints
    (``/traffic/views``, ``/traffic/clones``) of repositories it does not run
    in. A workflow's ambient ``GITHUB_TOKEN`` is scoped to its own repository
    and cannot read another's traffic, so this is a separate fine-grained
    personal access token with read access to those repositories' metrics.

    Deliberately raise-on-missing rather than optional, unlike
    :func:`get_github_api_key_optional`. The traffic window is 14 days and
    non-backfillable: a capture that silently skipped it would look exactly
    like a repository with no traffic, for as long as it took anyone to
    notice — and by then the data would be gone. Failing loudly is the only
    safe behaviour.
    """
    token = os.environ.get("PARTICLES_METRICS_GITHUB_TOKEN")
    if not token:
        raise ValueError(
            "PARTICLES_METRICS_GITHUB_TOKEN is not set. The publication-metrics "
            "capture needs a fine-grained personal access token with "
            "'Repository permissions -> Administration: Read-only' (which is "
            "what gates the traffic endpoints) on the published repositories. "
            "A workflow's default GITHUB_TOKEN cannot read another repository's "
            "traffic. Store it as the repository secret of the same name."
        )
    return token


def get_metrics_goatcounter_token_optional() -> str | None:
    """Return ``PARTICLES_METRICS_GOATCOUNTER_TOKEN`` or ``None`` if unset.

    Optional, and the asymmetry with :func:`get_metrics_github_token` is
    deliberate: GoatCounter keeps full history, so a run that collects nothing
    from it loses nothing — the figures can be fetched later. The capture
    records the source as unavailable and carries on.
    """
    return os.environ.get("PARTICLES_METRICS_GOATCOUNTER_TOKEN") or None


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
    token = os.environ.get("PARTICLES_ENGINE_TOKEN")
    return token if token and token.strip() else None


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


# The variables :func:`redact_known_secrets` treats as secret values. A secret added
# above joins this tuple (or, for a per-name family, the prefix scan below).
_SECRET_VARIABLES = (
    "ANTHROPIC_API_KEY",
    "GITHUB_API_KEY",
    "NUMISTA_API_KEY",
    "NOTION_API_KEY",
    "PARTICLES_API_KEY",
    "PARTICLES_METRICS_GITHUB_TOKEN",
    "PARTICLES_METRICS_GOATCOUNTER_TOKEN",
    "PARTICLES_ENGINE_TOKEN",
    "PARTICLES_LOCAL_LLM_API_KEY",
    # Read by the Anthropic SDK itself, never by a getter here; listed so an
    # error or log line that echoes them is scrubbed (security review
    # 2026-10-25-2, R9).
    "ANTHROPIC_AUTH_TOKEN",
)
_LLM_API_KEY_PREFIX = "PARTICLES_LLM_API_KEY_"
_OTLP_HEADER_VARIABLES = (
    "PARTICLES_OTEL_EXPORTER_HEADERS",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    "OTEL_EXPORTER_OTLP_METRICS_HEADERS",
    "OTEL_EXPORTER_OTLP_LOGS_HEADERS",
)

#: A value shorter than this is never treated as a secret to redact: replacing
#: every occurrence of a short string would garble ordinary text, and no real
#: credential is that short (the ``dev-key`` sentinel is seven characters).
_MIN_REDACTED_LENGTH = 8

REDACTED = "[REDACTED]"


def _header_secret_values(raw: str) -> list[str]:
    """The values in a ``name=value,…`` header list, and each word of each value.

    A value such as ``Bearer abc123`` yields both the whole value and
    ``abc123``, so an echo of the bare token is caught too.
    """
    values: list[str] = []
    for item in raw.split(","):
        _name, sep, value = item.partition("=")
        if not sep:
            # A malformed item (``name: value``) may still hold the secret.
            value = item.partition(":")[2]
        value = value.strip()
        values.append(value)
        values.extend(value.split())
    return values


def _custom_header_secret_values(raw: str) -> list[str]:
    """The values in ``ANTHROPIC_CUSTOM_HEADERS`` (``name: value`` lines), and each word.

    Parsed the way the Anthropic SDK parses it: one header per line, split at
    the first colon.
    """
    values: list[str] = []
    for line in raw.split("\n"):
        _name, sep, value = line.partition(":")
        if sep:
            value = value.strip()
            values.append(value)
            values.extend(value.split())
    return values


def known_secret_values() -> list[str]:
    """Every configured secret value, longest first, for :func:`redact_known_secrets`.

    Read from the environment at call time, so a key set or rotated after
    import is still covered. The ``dev-key`` sentinel is not a secret and is
    never listed.
    """
    values: list[str | None] = [os.environ.get(name) for name in _SECRET_VARIABLES]
    values.extend(v for k, v in os.environ.items() if k.startswith(_LLM_API_KEY_PREFIX))
    # The Particles OTLP header secret, and the OpenTelemetry SDK's own header
    # variables in the same ``name=value,…`` form, which it sends on its own
    # (security review 2026-10-26-1, Q13).
    for name in _OTLP_HEADER_VARIABLES:
        headers = os.environ.get(name)
        if headers:
            values.extend(_header_secret_values(headers))
    custom = os.environ.get("ANTHROPIC_CUSTOM_HEADERS")
    if custom:
        values.extend(_custom_header_secret_values(custom))
    kept = {
        v
        for v in values
        if v and len(v) >= _MIN_REDACTED_LENGTH and v != _PARTICLES_API_KEY_DEFAULT
    }
    return sorted(kept, key=len, reverse=True)


def redact_known_secrets(text: str, extra: Iterable[str | None] = ()) -> str:
    """Replace every known secret value in ``text`` with ``[REDACTED]``.

    Applied to text that came from outside the process (a provider's error
    body, an exception message built from one) before it is raised, logged or
    stored, because an endpoint can echo the credential it was sent in any
    spelling (security review 2026-10-25, Q13). ``extra`` names values the
    caller sent that the environment may not hold, such as a header it built.
    """
    secrets = set(known_secret_values())
    secrets.update(v for v in extra if v and len(v) >= _MIN_REDACTED_LENGTH)
    for value in sorted(secrets, key=len, reverse=True):
        text = text.replace(value, REDACTED)
    # A per-client engine key is masked by its shape too, whether or not this
    # process holds it (security review 2026-10-26-1, W1).
    return mask_engine_keys(text, REDACTED)


#: The shape of a per-client engine key, ``ptk_<role>_<id>_<secret>``,
#: anywhere in a text. ``particles.api._keys`` parses the same shape; a test
#: ties the two. No boundary is required before the key: it may follow any
#: escape whose last character is a letter or digit (a percent escape, a shell
#: backslash escape, quoted-printable ``=3D``, a JSON unicode escape), and the
#: fixed ``ptk_<role>_<12 hex>_`` prefix is distinctive on its own (security
#: review 2026-10-27-1, W31; security review 2026-10-28-1, O25).
ENGINE_KEY_SHAPE = re.compile(r"ptk_(?:read|agent|op)_[0-9a-f]{12}_[A-Za-z0-9_-]{43,}")

#: What can sit inside a per-client key without changing what a reader copies
#: out of it: a Unicode default-ignorable code point (a zero-width space, a
#: soft hyphen, a joiner, a variation selector, a bidi control), and an escape
#: of the ``_`` separator (percent, JSON or C hex, HTML character reference).
#: Each ignorable is dropped and each escape read as ``_`` before the key shape
#: is matched (security review 2026-10-30-1, E33).
_KEY_NOISE = re.compile(
    "[­͏؜ᅟᅠ឴឵᠋-᠏​-‏"
    "‪-‮⁠-⁯ㅤ︀-️﻿ﾠ￰-￸"
    "\U0001bca0-\U0001bca3\U0001d173-\U0001d17a\U000e0000-\U000e0fff]"
    r"|%5[fF]|\\u005[fF]|\\x5[fF]|&#(?:95|x5[fF]);"
)


def _engine_key_spans(text: str) -> list[tuple[int, int]]:
    """The spans of ``text`` that read as a per-client key once noise is removed.

    The key shape is matched against a shadow of ``text`` with every
    :data:`_KEY_NOISE` match dropped (an ignorable) or read as ``_`` (an
    escape), and each match is mapped back to the span of ``text`` it came
    from, noise inside it included.
    """
    shadow: list[str] = []
    # (shadow offset, original offset, original end of an escape or -1)
    marks: list[tuple[int, int, int]] = []
    length = 0
    pos = 0
    for noise in _KEY_NOISE.finditer(text):
        if noise.start() > pos:
            marks.append((length, pos, -1))
            shadow.append(text[pos : noise.start()])
            length += noise.start() - pos
        if len(noise.group()) > 1:
            marks.append((length, noise.start(), noise.end()))
            shadow.append("_")
            length += 1
        pos = noise.end()
    if not marks and pos == 0:
        return [m.span() for m in ENGINE_KEY_SHAPE.finditer(text)]
    if pos < len(text):
        marks.append((length, pos, -1))
        shadow.append(text[pos:])
    starts = [mark[0] for mark in marks]

    def _origin(index: int) -> tuple[int, int]:
        at, origin, escape_end = marks[bisect.bisect_right(starts, index) - 1]
        if escape_end >= 0:
            return origin, escape_end
        return origin + index - at, origin + index - at + 1

    return [
        (_origin(m.start())[0], _origin(m.end() - 1)[1])
        for m in ENGINE_KEY_SHAPE.finditer("".join(shadow))
    ]


def mask_engine_keys(text: str, replacement: str) -> str:
    """Replace every per-client engine key in ``text`` with ``replacement``.

    A key is found whole or split by an invisible character or an escaped
    separator (:data:`_KEY_NOISE`), and the whole span it occupies is replaced.
    """
    spans = _engine_key_spans(text)
    if not spans:
        return text
    parts: list[str] = []
    pos = 0
    for start, end in spans:
        parts.append(text[pos:start])
        parts.append(replacement)
        pos = end
    parts.append(text[pos:])
    return "".join(parts)
