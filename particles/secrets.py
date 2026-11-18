# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Centralised reads for secret environment variables.

Secrets must never appear in ``config.yaml``: they are read from
the process environment, except for the explicitly scoped scheduled
consolidation Keychain fallback and the engine bearer's checked token file
(``PARTICLES_ENGINE_TOKEN_FILE``). Every other secret is
environment-only.
This module is the single seam through which application code touches
``os.environ`` for those values, so a future migration to a secrets manager
(1Password, AWS Secrets Manager, sops, …)
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
import hashlib
import json
import os
import re
import stat
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduledCredential:
    """Non-secret credential locator, bound to one store/provider/endpoint."""

    profile: str
    store_id: str
    provider: str
    endpoint: str

    @property
    def service(self) -> str:
        """A deterministic locator; a changed endpoint never inherits a key."""
        identity = json.dumps([self.profile, self.store_id, self.provider, self.endpoint])
        return "dev.particles.extract." + hashlib.sha256(identity.encode()).hexdigest()


_scheduled_credential: ContextVar[ScheduledCredential | None] = ContextVar(
    "particles_scheduled_credential", default=None
)


_scheduled_clients: ContextVar[dict[str, object] | None] = ContextVar(
    "scheduled_clients", default=None
)
_scheduled_values: ContextVar[set[str] | None] = ContextVar("scheduled_values", default=None)


def scheduled_client_cache(provider: str) -> dict[str, object] | None:
    """A native transport cache confined to this explicit credential scope."""
    binding = _scheduled_credential.get()
    return (
        _scheduled_clients.get() if binding is not None and binding.provider == provider else None
    )


@contextmanager
def scheduled_credential_scope(binding: ScheduledCredential | None) -> Iterator[None]:
    """Enable the OS fallback for this consolidation call only, resetting on exit."""
    token = _scheduled_credential.set(binding)
    clients: dict[str, object] | None = {} if binding is not None else None
    values: set[str] | None = set() if binding is not None else None
    client_token = _scheduled_clients.set(clients)
    value_token = _scheduled_values.set(values)
    try:
        yield
    finally:
        if clients is not None:
            for client in clients.values():
                close = getattr(client, "close", None)
                if callable(close):
                    with suppress(Exception):
                        close()  # cleanup must not replace a consolidation failure
            clients.clear()
        if values is not None:
            values.clear()
        _scheduled_values.reset(value_token)
        _scheduled_clients.reset(client_token)
        _scheduled_credential.reset(token)


def _scheduled_key(provider: str, endpoint: str | None = None) -> str | None:
    binding = _scheduled_credential.get()
    if binding is None or binding.provider != provider:
        return None
    if endpoint is None:
        if provider == "anthropic":
            endpoint = get_anthropic_base_url_optional() or "https://api.anthropic.com"
        else:
            from particles.config import get_config

            configured = get_config().llm.providers.get(provider)
            endpoint = configured.base_url if configured is not None else None
    if endpoint != binding.endpoint:
        return None  # an old profile never silently supplies a different endpoint
    from particles.keychain import read_generic_password

    value = read_generic_password(binding.service, "extraction")
    values = _scheduled_values.get()
    if value and values is not None:
        values.add(value)
    return value


_PARTICLES_API_KEY_DEFAULT = "dev-key"


def get_anthropic_api_key(*, endpoint: str | None = None) -> str:
    """Return ``ANTHROPIC_API_KEY`` or raise ``ValueError`` if unset.

    Used by the shared Anthropic client (``particles.llm.get_client``). The
    error message points the operator at the console where they can mint a
    key — the Anthropic SDK's own "Could not resolve authentication method"
    message is too generic to act on.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY") or _scheduled_key("anthropic", endpoint)
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
    return os.environ.get("ANTHROPIC_API_KEY") or _scheduled_key("anthropic")


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


class EngineTokenFileError(ValueError):
    """``PARTICLES_ENGINE_TOKEN_FILE`` names a file that fails a check.

    The message names the path and the check, and never any byte of the file.
    ``check`` is a short machine name (``missing``, ``mode``, ``owner`` …) that
    the hook log records as ``token-file-refused``.
    """

    def __init__(self, path: str, check: str, detail: str) -> None:
        self.path = path
        self.check = check
        super().__init__(f"PARTICLES_ENGINE_TOKEN_FILE {path}: refused ({check}): {detail}")


#: The largest token file read; a per-client key is about sixty bytes.
_TOKEN_FILE_MAX_BYTES = 4096

#: Values read from token files in this process, by path (read
#: once per process, so a rotation takes effect at the next process start).
_token_file_values: dict[str, str] = {}


def engine_token_file_path() -> str | None:
    """The token file this process's engine bearer comes from, or ``None``.

    ``None`` when ``PARTICLES_ENGINE_TOKEN`` supplies the token or neither
    variable is set. Reads nothing but the environment and never raises.
    """
    if (os.environ.get("PARTICLES_ENGINE_TOKEN") or "").strip():
        return None
    path = (os.environ.get("PARTICLES_ENGINE_TOKEN_FILE") or "").strip()
    return path or None


def _read_token_file(path: str) -> str:
    """Read and check one token file; raise ``EngineTokenFileError``."""
    if os.name == "nt" or not hasattr(os, "O_NOFOLLOW"):
        raise EngineTokenFileError(
            path, "platform", "file modes cannot be checked here; set PARTICLES_ENGINE_TOKEN"
        )
    if not os.path.isabs(path):
        # Used as given: `~` is never expanded, so a shell and a hook cannot disagree.
        raise EngineTokenFileError(path, "relative-path", "the path must be absolute")
    euid = os.geteuid()
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0))
    except FileNotFoundError:
        raise EngineTokenFileError(path, "missing", "no such file") from None
    except OSError as exc:
        if os.path.islink(path):
            raise EngineTokenFileError(path, "symlink", "the file is a symbolic link") from None
        raise EngineTokenFileError(path, "unreadable", exc.strerror or "cannot open") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise EngineTokenFileError(path, "not-regular", "not a regular file")
        if st.st_nlink != 1:
            raise EngineTokenFileError(path, "hard-link", "the file has more than one link")
        if st.st_uid != euid:
            raise EngineTokenFileError(path, "owner", "the file belongs to another account")
        if st.st_mode & 0o077:
            raise EngineTokenFileError(
                path, "mode", f"mode {stat.S_IMODE(st.st_mode):04o} allows group or other access"
            )
        parent = os.stat(os.path.dirname(path))
        if parent.st_uid != euid:
            raise EngineTokenFileError(
                path, "parent-owner", "the directory holding it belongs to another account"
            )
        if parent.st_mode & 0o022:
            raise EngineTokenFileError(
                path, "parent-writable", "another account can write the directory holding it"
            )
        _check_token_file_ancestors(path, euid)
        if st.st_size > _TOKEN_FILE_MAX_BYTES:
            raise EngineTokenFileError(path, "too-large", "the file is over 4 KiB")
        raw = os.read(fd, _TOKEN_FILE_MAX_BYTES + 1)
    finally:
        os.close(fd)
    if len(raw) > _TOKEN_FILE_MAX_BYTES:
        raise EngineTokenFileError(path, "too-large", "the file is over 4 KiB")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise EngineTokenFileError(path, "not-utf8", "the file is not UTF-8") from None
    if text.endswith("\r\n"):
        text = text[:-2]
    elif text.endswith("\n"):
        text = text[:-1]
    if not text:
        raise EngineTokenFileError(path, "empty", "the file holds no token")
    if "\n" in text or "\r" in text:
        raise EngineTokenFileError(path, "not-one-line", "the file holds more than one line")
    if any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in text):
        raise EngineTokenFileError(
            path, "malformed", "the token holds whitespace or a control character"
        )
    return text


def _check_token_file_ancestors(path: str, euid: int) -> None:
    """Refuse a token file whose directory another account could replace.

    Above the immediate parent, every directory up to the root must belong to
    this user or to root, and be writable by no one else unless it carries the
    sticky bit (``/tmp``), which stops another account renaming what is ours.
    Otherwise that account could move the parent aside and put its own in place.
    """
    current = os.path.dirname(os.path.dirname(path))
    while True:
        try:
            st = os.stat(current)
        except OSError:
            raise EngineTokenFileError(
                path, "ancestor-unreadable", f"{current} cannot be examined"
            ) from None
        if st.st_uid not in (euid, 0):
            raise EngineTokenFileError(
                path, "ancestor-owner", f"{current} belongs to another account"
            )
        if st.st_mode & 0o022 and not st.st_mode & stat.S_ISVTX:
            raise EngineTokenFileError(
                path, "ancestor-writable", f"another account can write {current}"
            )
        parent = os.path.dirname(current)
        if parent == current:
            return
        current = parent


def get_engine_token_optional() -> str | None:
    """Return the engine bearer, or ``None`` when none is configured.

    The bearer token the **thin client** presents to a remote engine when
    ``engine.base_url`` is configured (``HttpBackend``). It is deliberately
    *distinct* from ``PARTICLES_API_KEY`` (which is the key the *server* expects,
    read by the FastAPI auth dependency): the engine reads its copy, the client
    reads the token it sends. For a single operator they may hold the same value,
    but they are separate roles, so they are separate secrets.

    Two sources, never both: ``PARTICLES_ENGINE_TOKEN`` holds the
    token itself; ``PARTICLES_ENGINE_TOKEN_FILE`` names a file it is read from,
    checked on every read (absolute path, no symbolic link, one link, owned by
    this user, no group or other bit, a private parent directory, one line of at
    most 4 KiB). An empty variable counts as unset, which is how the installed
    hook commands make the file their only source. Both set, or a file that
    fails a check, raises :class:`EngineTokenFileError`: the getter never falls
    back to sending no token. A file's value is read once per process and never
    enters ``os.environ``.

    Optional: a loopback dev engine runs with the ``dev-key`` skip and needs no
    token, so ``HttpBackend`` omits the ``Authorization`` header when this is
    ``None``. A non-loopback engine will reject the unauthenticated request
    (401), which is the intended, legible failure.
    """
    token = os.environ.get("PARTICLES_ENGINE_TOKEN")
    path = (os.environ.get("PARTICLES_ENGINE_TOKEN_FILE") or "").strip()
    if token and token.strip():
        if path:
            raise EngineTokenFileError(
                path,
                "both-set",
                "PARTICLES_ENGINE_TOKEN is set as well; set one of the two, not both",
            )
        return token
    if not path:
        return None
    if path not in _token_file_values:
        _token_file_values[path] = _read_token_file(path)
    return _token_file_values[path]


def check_engine_token_file(path: str) -> str:
    """Read ``path`` through the token-file checks, without caching (for the installer)."""
    return _read_token_file(path)


def _forget_engine_token_files() -> None:
    """Drop the per-process token-file values (tests only)."""
    _token_file_values.clear()


def provider_key_names(provider_name: str) -> tuple[str, ...]:
    """The central environment names, also used for read-only job diagnostics."""
    if provider_name == "anthropic":
        return ("ANTHROPIC_API_KEY",)
    suffix = "".join(c if c.isalnum() else "_" for c in provider_name).upper()
    primary = f"PARTICLES_LLM_API_KEY_{suffix}"
    return (primary, "PARTICLES_LOCAL_LLM_API_KEY") if provider_name == "local" else (primary,)


def get_llm_api_key_optional(provider_name: str, *, endpoint: str | None = None) -> str | None:
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
    key = os.environ.get(provider_key_names(provider_name)[0]) or None
    if key is None and provider_name == "local":
        return get_local_llm_api_key_optional(endpoint=endpoint)
    return key or _scheduled_key(provider_name, endpoint)


def get_local_llm_api_key_optional(*, endpoint: str | None = None) -> str | None:
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
    return os.environ.get("PARTICLES_LOCAL_LLM_API_KEY") or _scheduled_key("local", endpoint)


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
    values.extend(_scheduled_values.get() or ())
    values.extend(_token_file_values.values())
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
ENGINE_KEY_SHAPE = re.compile(r"ptk_(?:read|agent|op|codex)_[0-9a-f]{12}_[A-Za-z0-9_-]{43,}")

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
