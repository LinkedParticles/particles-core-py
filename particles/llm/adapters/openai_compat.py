# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Generic ``CompletionProvider`` adapter — OpenAI-compatible endpoints.

Born as the ``LocalProvider``; generalized into the one
adapter behind every **named provider** in ``llm.providers`` — hosted
(api.openai.com, api.deepseek.com, gateways) or local (Ollama, llama.cpp's
server, vLLM, LM Studio), all of which expose the OpenAI
``POST {base_url}/chat/completions`` contract. Per-endpoint dialect quirks
(``max_tokens`` vs ``max_completion_tokens``, temperature support, strict
structured-output schemas) are declarative config on the entry, never runtime
sniffing. Transport is :mod:`httpx` (already a project dependency) over a
**dedicated** ``AsyncClient`` rather than :func:`particles.http.particles_client`:
the shared client routes through the SSRF-validating transport, which would
reject the loopback / private-network endpoints a local model lives on. The
endpoint is operator-configured (trusted), not user-supplied, so it is exempt
from that allow-listing by design — hosted or loopback alike.

The Anthropic SDK retries internally; raw ``httpx`` does not, so this adapter
implements a small bounded retry loop with exponential backoff on transient
failures (connection errors, timeouts, HTTP 429 / 5xx) and raises
:class:`particles.llm.registry.CompletionError` on exhaustion or an empty
response — the port's failure contract, which every call site already handles.
A 401 / 403 is surfaced with its HTTP status attached so the account-level circuit breaker (duck-typed on ``status_code``) can trip on a
mis-configured endpoint just as it does for a hosted provider. Budget
exhaustion — ``finish_reason: length``, an HTTP 200 whose text stops mid-token
— is named as truncation rather than handed to the call site as unexplained
malformed JSON; see :func:`_extract_text`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx
from opentelemetry import metrics, trace

from particles.llm.registry import CompletionError, EmptyCompletionError, VisionImage

log = logging.getLogger(__name__)

# A misbehaving gateway can echo the request's ``Authorization: Bearer <token>``
# header back in its response body; embedding that body verbatim in an exception
# would surface the bearer token to logs / stderr (security review F33). Redact
# any Authorization/Bearer header value before the body is put in an error.
_BEARER_RE = re.compile(r"(?i)(authorization\s*:\s*)?bearer\s+\S+")


def _scrub_response_body(text: str, *, limit: int = 500) -> str:
    """Redact bearer tokens from a gateway response body, then truncate.

    Keeps the body useful for diagnosing a 4xx/5xx (status detail, error JSON)
    while ensuring an echoed ``Authorization: Bearer …`` never reaches an
    exception string. Redaction runs before truncation so a token straddling the
    cut is still scrubbed.
    """
    return _BEARER_RE.sub("Bearer [REDACTED]", text)[:limit]


# Hand-rolled telemetry, mirroring the Anthropic adapter (Phase 2).
# The OTel **API** is used directly (no-op until a provider is installed), so
# this Client-layer adapter stays free of the Engine-side SDK.
_tracer = trace.get_tracer("particles.llm")
_meter = metrics.get_meter("particles.llm")
_llm_duration = _meter.create_histogram(
    "particles.llm.duration", unit="s", description="LLM completion call wall time"
)


class OpenAICompatCompletionError(CompletionError):
    """A :class:`CompletionError` carrying the HTTP status of a failed call.

    The circuit breaker duck-types account-level failures on an
    exception's ``status_code`` attribute (``_is_account_level`` in
    ``particles/operations/_llm.py``). Raising this — rather than a bare
    ``CompletionError`` — lets the endpoint's 401 / 403 trip the breaker
    exactly as the native provider's SDK exception does. ``status_code`` is
    ``None`` for non-HTTP failures (connection refused, timeout), which the
    breaker correctly treats as per-call/transient rather than account-level.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# HTTP statuses worth retrying: rate limit + transient server errors. A 4xx
# other than 429 is a deterministic client error (bad request, auth) — retrying
# it just wastes calls, so it fails fast.
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


# A reply that exhausted the completion-token budget comes back HTTP 200 with
# ``finish_reason == "length"`` and text that simply stops mid-token — so the
# call site's JSON parser reports "Unterminated string", naming the symptom and
# burying the cause. Reasoning models make this the common failure rather than
# the rare one: their thinking tokens are drawn from the same budget as the
# answer, so a prompt that fit comfortably under 8192 on a non-reasoning model
# can spend the whole budget before the answer starts. Observed live 2026-08
# across DeepSeek-V4 flash/pro and Kimi K3; 16384 cleared all three.
_TRUNCATION_HINT = (
    "for reasoning models the thinking spends from the same budget as the "
    "answer — raise extraction.max_tokens (or the call site's budget)"
)


def _budget_phrase(max_tokens: int | None) -> str:
    return f"max_tokens={max_tokens}" if max_tokens is not None else "its max_tokens budget"


# ---------------------------------------------------------------------------
# Structured output (folding)
# ---------------------------------------------------------------------------


def _wrap_array_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Wrap an array-root schema for object-root ``json_schema`` dialects.

    OpenAI's strict mode requires an object root; Ollama's accepts an array
    root but tolerates the wrap. Sending the wrapped shape everywhere keeps
    the adapter dialect-agnostic; :func:`_unwrap_items_reply` restores the
    array on the way out so the port's text contract and every call-site
    parser stay untouched.
    """
    return {
        "type": "object",
        "properties": {"items": schema},
        "required": ["items"],
        "additionalProperties": False,
    }


def _unwrap_items_reply(text: str) -> str:
    """Undo :func:`_wrap_array_schema` on the model's reply, transparently.

    A well-behaved enforced reply is ``{"items": [...]}`` — return the inner
    array serialized. An endpoint that honoured the array root anyway (or a
    reply that does not parse at all — the tolerant call-site parser's
    territory) passes through unchanged.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return json.dumps(data["items"])
    return text


def _response_format(schema: dict[str, Any]) -> dict[str, Any]:
    """The OpenAI-style schema-carrying ``response_format`` body member."""
    return {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": schema, "strict": True},
    }


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    """Make one property schema accept ``null`` (strict-dialect optionality)."""
    t = schema.get("type")
    if isinstance(t, str) and t != "null":
        return {**schema, "type": [t, "null"]}
    if isinstance(t, list) and "null" not in t:
        return {**schema, "type": [*t, "null"]}
    for comb in ("anyOf", "oneOf"):
        branches = schema.get(comb)
        if isinstance(branches, list) and {"type": "null"} not in branches:
            return {**schema, comb: [*branches, {"type": "null"}]}
    return schema


def _to_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Transform a JSON Schema to the OpenAI-strict dialect.

    Strict-mode validators (api.openai.com, vLLM / LM Studio strict modes)
    reject any object whose ``required`` is a proper subset of ``properties``.
    The transform lists every property key in ``required``, re-expresses the
    originally-optional keys as unions with ``null``, and stamps
    ``additionalProperties: false`` on every object, recursing through
    ``properties`` / ``items`` / combinators / ``$defs``. Pure and
    non-mutating — call sites keep their original schemas.
    """
    out = dict(schema)
    if out.get("type") == "object" and isinstance(out.get("properties"), dict):
        originally_required = set(out.get("required") or [])
        props: dict[str, Any] = {}
        for key, sub in out["properties"].items():
            strict_sub = _to_strict_schema(sub) if isinstance(sub, dict) else sub
            if key not in originally_required and isinstance(strict_sub, dict):
                strict_sub = _nullable(strict_sub)
            props[key] = strict_sub
        out["properties"] = props
        out["required"] = list(props)
        out["additionalProperties"] = False
    if isinstance(out.get("items"), dict):
        out["items"] = _to_strict_schema(out["items"])
    for comb in ("anyOf", "oneOf", "allOf"):
        if isinstance(out.get(comb), list):
            out[comb] = [_to_strict_schema(s) if isinstance(s, dict) else s for s in out[comb]]
    if isinstance(out.get("$defs"), dict):
        out["$defs"] = {
            k: _to_strict_schema(v) if isinstance(v, dict) else v for k, v in out["$defs"].items()
        }
    return out


def _is_unsupported_response_format(exc: OpenAICompatCompletionError) -> bool:
    """True when the endpoint rejected the ``response_format`` parameter itself.

    An HTTP 400 naming the parameter (or its ``json_schema`` payload) is the
    documented shape across OpenAI-compatible servers; anything else is a real
    request failure and must not be masked by a silent downgrade.
    """
    if exc.status_code != 400:
        return False
    message = str(exc).lower()
    return "response_format" in message or "json_schema" in message


@dataclass(frozen=True)
class OpenAICompatProvider:
    """Completion provider backed by an OpenAI-compatible chat endpoint.

    ``name`` is the ``llm.providers`` entry this instance reads its endpoint,
    resilience, and dialect policy from (at call time, so ``reset_config()``
    is honoured); ``model`` is the per-purpose model string (e.g.
    ``"gpt-5.6-luna"``, ``"llama3.1:8b"``).
    """

    name: str
    model: str

    @property
    def provider_model(self) -> str:
        """``"<name>:<model>"`` — the calibration / disclosure key.

        Keyed by the operator-chosen provider name, so renaming a provider
        orphans its calibration rows. A fresh pairing is
        uncalibrated until the benchmark harness is run for it, so
        its particles carry the ``EXTRACTOR_DIRECT`` disclosure until then
        .
        """
        return f"{self.name}:{self.model}"

    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        system: str | None = None,
        temperature: float | None = None,
        images: Sequence[VisionImage] | None = None,
        response_schema: dict[str, Any] | None = None,
        cache_prefix: str | None = None,
        **opts: object,
    ) -> str:
        # Vision via this adapter is deferred (
        # § Deferred); reject images loudly rather than silently dropping them,
        # so a caller routing a multimodal pass here fails legibly.
        if images:
            raise CompletionError("OpenAICompatProvider does not support image input")

        # prompt caching is an Anthropic-native feature; this adapter
        # ignores the cache *marker* but must preserve the *content* — fold the
        # cache prefix back into the system turn so the effective prompt is
        # identical (many OpenAI-compatible endpoints cache such prefixes
        # transparently anyway).
        if cache_prefix:
            system = cache_prefix + (system or "")

        from particles.config import get_config
        from particles.secrets import get_llm_api_key_optional

        try:
            cfg = get_config().llm.providers[self.name]
        except KeyError:
            # Config-load validation prevents this for registry-resolved
            # providers; a directly-constructed instance can still miss.
            raise CompletionError(f"no llm.providers entry named {self.name!r}") from None
        api_key = get_llm_api_key_optional(self.name)

        # Structured output: only schema-carrying enforcement is
        # used — OpenAI's schemaless ``json_object`` mode forces an object root
        # where extraction wants an *array*, so it is rejected outright rather
        # than silently sent.
        if response_schema is not None and response_schema.get("type") == "json_object":
            raise CompletionError(
                "response_schema must be a JSON Schema; the schemaless "
                "'json_object' response mode is rejected — it "
                "cannot express an array-root reply"
            )
        enforce = response_schema is not None and cfg.structured_output in ("auto", "strict")
        wrapped = enforce and response_schema is not None and response_schema.get("type") == "array"

        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            cfg.max_tokens_param: max_tokens,
        }
        if temperature is not None and cfg.send_temperature:
            body["temperature"] = temperature
        if enforce and response_schema is not None:
            schema = _wrap_array_schema(response_schema) if wrapped else response_schema
            if cfg.structured_output == "strict":
                schema = _to_strict_schema(schema)
            body["response_format"] = _response_format(schema)

        headers = {"Content-Type": "application/json"}
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"

        url = f"{cfg.base_url.rstrip('/')}/chat/completions"

        with _tracer.start_as_current_span("llm.complete") as span:
            span.set_attribute("llm.model", self.provider_model)
            span.set_attribute("llm.max_tokens", max_tokens)
            span.set_attribute("llm.structured_output", enforce)
            start = time.perf_counter()
            try:
                try:
                    text = await _post_with_retry(
                        url,
                        body,
                        headers,
                        max_retries=cfg.max_retries,
                        backoff=cfg.retry_backoff_seconds,
                        timeout=cfg.timeout_seconds,
                        provider_model=self.provider_model,
                        max_tokens=max_tokens,
                    )
                except OpenAICompatCompletionError as exc:
                    if not (enforce and _is_unsupported_response_format(exc)):
                        raise
                    # Graceful degradation: the endpoint's dialect
                    # does not support response_format — retry ONCE without it.
                    # The tolerant call-site parser is the backstop, as before
                    # enforcement existed.
                    log.warning(
                        "LLM endpoint %s rejected response_format; retrying "
                        "without structured output (downgraded to tolerant "
                        "parsing): %s",
                        self.name,
                        exc,
                    )
                    span.set_attribute("llm.structured_output_downgraded", True)
                    # A fresh dict, not a pop: the original request body may
                    # still be referenced by transport/test instrumentation.
                    body = {k: v for k, v in body.items() if k != "response_format"}
                    wrapped = False
                    text = await _post_with_retry(
                        url,
                        body,
                        headers,
                        max_retries=cfg.max_retries,
                        backoff=cfg.retry_backoff_seconds,
                        timeout=cfg.timeout_seconds,
                        provider_model=self.provider_model,
                        max_tokens=max_tokens,
                    )
            finally:
                _llm_duration.record(time.perf_counter() - start, {"model": self.provider_model})
        return _unwrap_items_reply(text) if wrapped else text


async def _post_with_retry(
    url: str,
    body: dict[str, object],
    headers: dict[str, str],
    *,
    max_retries: int,
    backoff: float,
    timeout: float,
    provider_model: str = "<unknown>",
    max_tokens: int | None = None,
) -> str:
    """POST the request, retrying transient failures with exponential backoff.

    Raises :class:`OpenAICompatCompletionError` after exhausting ``max_retries``,
    carrying the HTTP status when the last failure was an error response. A
    module-level function (not a method) so a test can drive the retry loop
    without constructing a provider. ``provider_model`` / ``max_tokens`` are
    carried for diagnostics only — they name the pairing and the budget in a
    truncation warning (see :func:`_extract_text`).
    """
    last_status: int | None = None
    last_detail = ""
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(max_retries + 1):
            try:
                resp = await client.post(url, json=body, headers=headers)
            except httpx.RequestError as exc:
                # Connection refused, DNS, timeout — transient, no status.
                last_status, last_detail = None, f"{type(exc).__name__}: {exc}"
                log.warning(
                    "LLM endpoint request error (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
            else:
                if resp.status_code == 200:
                    return _extract_text(
                        resp.json(), provider_model=provider_model, max_tokens=max_tokens
                    )
                last_status = resp.status_code
                last_detail = _scrub_response_body(resp.text)
                if resp.status_code not in _RETRYABLE_STATUSES:
                    # Deterministic client error (auth, bad request) — fail fast.
                    raise OpenAICompatCompletionError(
                        f"LLM endpoint returned HTTP {resp.status_code}: {last_detail}",
                        status_code=resp.status_code,
                    )
                log.warning(
                    "LLM endpoint HTTP %d (attempt %d/%d)",
                    resp.status_code,
                    attempt + 1,
                    max_retries + 1,
                )
            if attempt < max_retries:
                await asyncio.sleep(backoff * (2**attempt))

    raise OpenAICompatCompletionError(
        f"LLM endpoint call failed after {max_retries + 1} attempt(s): {last_detail}",
        status_code=last_status,
    )


def _extract_text(
    payload: object,
    *,
    provider_model: str = "<unknown>",
    max_tokens: int | None = None,
) -> str:
    """Pull ``choices[0].message.content`` from an OpenAI-compatible response.

    Also reads ``choices[0].finish_reason``: ``"length"`` means the endpoint
    stopped because the completion budget ran out, which the reply body itself
    cannot show. Truncated-but-present text is returned (the tolerant call-site
    parser is still the backstop) with a WARNING naming the budget; a truncated
    *empty* reply is a budget failure, not a model failure, and its
    :class:`CompletionError` says so.

    Raises :class:`EmptyCompletionError` — the deterministic
    ``CompletionError`` subclass — when the response carries no text, the same
    contract :class:`particles.llm.adapters.anthropic.AnthropicProvider`
    upholds for an empty Anthropic response. A call site that retries
    transient failures must not retry this one: the budget is the cause.
    """
    content: str | None = None
    finish_reason: object = None
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                finish_reason = first.get("finish_reason")
                message = first.get("message")
                if isinstance(message, dict):
                    raw = message.get("content")
                    if isinstance(raw, str) and raw.strip():
                        content = raw.strip()

    truncated = finish_reason == "length"
    if truncated:
        # The span opened by ``complete()`` is still current here, so the fact
        # reaches telemetry as well as the log (no-op without an OTel SDK).
        trace.get_current_span().set_attribute("llm.truncated", True)

    if content is not None:
        if truncated:
            log.warning(
                "LLM reply from %s was truncated at %s (finish_reason=length): "
                "the text stops mid-token and will not parse — %s.",
                provider_model,
                _budget_phrase(max_tokens),
                _TRUNCATION_HINT,
            )
        return content

    if truncated:
        raise EmptyCompletionError(
            f"LLM endpoint {provider_model} returned an empty reply truncated "
            f"at {_budget_phrase(max_tokens)} (finish_reason=length): the whole "
            f"budget was spent before any answer text — {_TRUNCATION_HINT}"
        )
    detail = f" (finish_reason={finish_reason})" if finish_reason is not None else ""
    raise EmptyCompletionError(f"LLM endpoint response carried no text content{detail}")
