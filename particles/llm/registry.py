# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The ``CompletionProvider`` port and per-purpose provider selection.

Every chat/completion call in the SDK goes through this port rather than
reaching for a provider SDK directly. The port mirrors the extractor /
exporter plugin pattern: a ``Protocol`` plus a small
registry that resolves the concrete adapter from config.

Completion has distinct *purposes* with different cost/quality trade-offs
(``LLMPurpose``); the operator can route each purpose to a different provider
and model — e.g. high-volume ``extraction`` on a cheap local model while
``synthesis`` stays on a hosted one. Selection lives in ``config.llm``
(``LLMConfig``); credentials/endpoints resolve through ``particles/secrets.py``.

``complete()`` is the one-line convenience every call site uses::

    text = await complete("extraction", prompt, max_tokens=8192)

``max_tokens`` is supplied per call (it is genuinely per-call: 8192 for an
extraction pass, 16 for the benchmark judge), so it is *not* part of the
config selection — only the (provider, model) pairing is.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

log = logging.getLogger(__name__)

# The six completion purposes. The string values match the field names on
# ``LLMConfig`` so ``LLMConfig.for_purpose`` is a plain ``getattr``.
# ``benchmark_answer`` is the memory-benchmark *answering* model —
# distinct from ``benchmark`` (the judge) so the two can be routed and pinned
# independently; the runner enforces one resolved answer model across the QA
# conditions.
LLMPurpose = Literal[
    "extraction",
    "semantic_lint",
    "query_response",
    "synthesis",
    "benchmark",
    "benchmark_answer",
    "abstraction",
]


class CompletionError(RuntimeError):
    """Raised by an adapter when a completion cannot be produced.

    A call site catches this (and any provider-SDK exception) and applies its
    own purpose-specific fallback — extraction records a quality note, the
    contradiction check returns ``False``, query concatenates particle
    contents, and so on. The port never decides the fallback; it only reports
    that no usable text came back.
    """


class EmptyCompletionError(CompletionError):
    """The provider replied, but the reply carried no usable text.

    The narrow, *deterministic* half of :class:`CompletionError`: the call
    reached the model and came back HTTP-200, yet there is no text to return —
    an extended-thinking model that spent its whole ``max_tokens`` budget
    before emitting an answer block, a refusal, or an endpoint whose
    ``finish_reason: length`` truncated the reply to nothing. It is a subclass,
    so every existing ``except CompletionError`` still catches it unchanged.

    Why it is worth naming apart from a transport failure: retrying an
    identical call at an identical budget reproduces it, so a call site that
    retries transient errors must *not* retry this one — the fix is a bigger
    budget, and the operator has to be told which of the two they are looking
    at. The memory benchmark's excluded-call disclosure splits its two counts
    on exactly this distinction.
    """


@dataclass(frozen=True)
class VisionImage:
    """An image to include in a multimodal completion call.

    ``media_type`` is an IANA image type (``"image/png"``, ``"image/jpeg"``);
    ``data`` is the raw image bytes — the adapter base64-encodes for transport.
    A provider whose model cannot accept image input raises
    :class:`CompletionError` when ``images`` is non-empty rather than silently
    dropping them, so a mis-configured (non-multimodal) provider fails loudly.
    """

    media_type: str
    data: bytes


@dataclass(frozen=True)
class CompletionRequest:
    """One independent unit of work in a :func:`complete_many` call.

    ``system`` is per-request rather than shared across the batch because the
    trusted-instruction turn often carries per-call state — most importantly
    the F3 data-fence nonce, which is regenerated per probe so a crafted claim
    in one request cannot close the fence in another. Everything genuinely
    uniform across the batch (``max_tokens``, ``response_schema``) stays a
    keyword argument on the call.

    ``cache_prefix`` is an optional leading slice of the system turn
    to mark as a cache boundary. The *effective* system the model sees is
    ``cache_prefix + system`` (content-preserving); the only difference is where
    the prompt-cache boundary falls. An adapter that supports prompt caching (the
    Anthropic adapter) sends it as a cached system block; every other adapter
    concatenates it and ignores the caching, so the return contract is unchanged.
    ``None`` (default) is exactly today's behaviour.
    """

    prompt: str
    system: str | None = None
    cache_prefix: str | None = None


@runtime_checkable
class CompletionProvider(Protocol):
    """A chat/completion model behind one port.

    Two adapter kinds ship: ``AnthropicProvider`` (the native SDK)
    and ``OpenAICompatProvider`` (any OpenAI-compatible endpoint — hosted or
    local, instantiated per named ``llm.providers`` entry), selected per
    purpose by a ``config.llm`` change with no call-site edits.
    """

    @property
    def provider_model(self) -> str:
        """``"<provider>:<model>"`` — the calibration / disclosure key."""
        ...

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
        """Return the model's text completion for ``prompt``.

        When ``images`` is supplied, they are sent alongside the
        prompt as a multimodal request; a provider whose model is not
        vision-capable raises :class:`CompletionError`. ``cache_prefix``
         marks a leading slice of the system turn as a prompt-cache
        boundary; the effective system is ``cache_prefix + system`` and an
        adapter MAY cache it (the Anthropic adapter does) or MAY ignore the
        marker and concatenate (every other adapter). ``response_schema``
         is the JSON Schema of the reply the
        caller will parse — advisory: an adapter MAY enforce it (the
        ``LocalProvider`` sends OpenAI-style ``response_format`` when
        ``structured_output`` on the named entry is ``"auto"``/``"strict"``) and MAY ignore it (the
        ``AnthropicProvider`` does in v1); the return contract stays plain
        text either way. Raises :class:`CompletionError` (or a provider-SDK
        exception) on failure or when the response carries no text — callers
        translate that into their own fallback.
        """
        ...


@runtime_checkable
class BatchCompletionProvider(Protocol):
    """A provider that can run many independent completions as one job.

    An **optional capability**, deliberately not folded into
    :class:`CompletionProvider`: batch submission is a property of a hosted
    provider's API surface, not of completion itself, and the ``LocalProvider`` has no equivalent. :func:`complete_many` duck-types on
    this protocol and falls back to sequential :meth:`CompletionProvider.complete`
    calls for any provider that does not implement it, so a new adapter opts
    in by adding the method and opts out by doing nothing.

    The return contract is ``list[str | None]``, **positionally aligned with
    the requests** and never short: a batch is a set of independent units of
    work, and one request that errored, was cancelled, or expired must not
    discard the answers to the others. ``None`` is that per-request failure,
    which every existing call site already handles as "probe unavailable".
    """

    async def complete_many(
        self,
        requests: Sequence[CompletionRequest],
        *,
        max_tokens: int,
        temperature: float | None = None,
        response_schema: dict[str, Any] | None = None,
        **opts: object,
    ) -> list[str | None]:
        """Run ``requests`` as one batch job, returning per-request results.

        Raises :class:`CompletionError` (or a provider-SDK exception) when the
        *job* could not be run at all — a submission the provider rejected, an
        expired credential. A job that ran but whose individual requests failed
        reports those as ``None`` entries, not as an exception.
        """
        ...


# ---------------------------------------------------------------------------
# Adapter-kind registry — the plugin-registry house style
#: a lazily-built keyed factory, one registration line per
# wire protocol. See particles/llm/AGENTS.md for the two-file procedure.
# ---------------------------------------------------------------------------

#: (provider_name, model) → a constructed provider. ``provider_name`` is the
#: ``llm.providers`` entry name the instance reads its config from (the native
#: ``anthropic`` adapter ignores it — its client/config live on the SDK seam).
AdapterFactory = Callable[[str, str], CompletionProvider]

_adapters: dict[str, AdapterFactory] | None = None


def _make_adapters() -> dict[str, AdapterFactory]:
    # defer: lazy-init — adapter modules are imported inside the factories so
    # building the kind map costs nothing until a provider is constructed, and
    # a broken adapter module cannot cascade-fail registry import (AGENTS.md
    # § Deferred imports, case 2; the extraction/exporter registries' pattern).
    def _anthropic(name: str, model: str) -> CompletionProvider:
        from particles.llm.adapters.anthropic import AnthropicProvider

        return AnthropicProvider(model=model)

    def _openai_compat(name: str, model: str) -> CompletionProvider:
        from particles.llm.adapters.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(name=name, model=model)

    return {
        "anthropic": _anthropic,
        "openai_compat": _openai_compat,
        # Register new adapter kinds here
    }


def _get_adapters() -> dict[str, AdapterFactory]:
    """Return the kind-keyed adapter factory map (cached after first call)."""
    global _adapters
    if _adapters is None:
        _adapters = _make_adapters()
    return _adapters


def adapter_kinds() -> frozenset[str]:
    """The registered adapter-kind names — ``LLMConfig`` validates against this."""
    return frozenset(_get_adapters())


def get_provider(purpose: LLMPurpose) -> CompletionProvider:
    """Resolve the provider configured for ``purpose``.

    ``provider: "anthropic"`` selects the native-SDK adapter directly; any
    other name is an ``llm.providers`` entry whose ``adapter`` field picks
    the factory from the kind registry. Reads config at call time (never
    captured at import) so ``reset_config()`` is honoured. A fresh adapter
    is constructed per call; adapters are cheap handles over cached clients,
    so this is not a hot-path cost.
    """
    from particles.config import get_config

    selection = get_config().llm.for_purpose(purpose)
    adapters = _get_adapters()
    if selection.provider == "anthropic":
        return adapters["anthropic"]("anthropic", selection.model)
    entry = get_config().llm.providers.get(selection.provider)
    if entry is None:
        # Config-load validation prevents this for file-loaded config; a
        # programmatically-mutated config can still miss.
        raise CompletionError(
            f"llm purpose {purpose!r} routes to unknown provider "
            f"{selection.provider!r} (not 'anthropic', not in llm.providers)"
        )
    factory = adapters.get(entry.adapter)
    if factory is None:
        raise CompletionError(
            f"llm.providers.{selection.provider} names unregistered adapter "
            f"kind {entry.adapter!r}; known kinds: {sorted(adapters)}"
        )
    return factory(selection.provider, selection.model)


async def complete(
    purpose: LLMPurpose,
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
    """Resolve the provider for ``purpose`` and run one completion.

    The one-liner every call site uses. Pass ``images`` for a multimodal call
    , ``response_schema`` when the reply will be parsed as JSON
    , and ``cache_prefix`` to mark a prompt-cache boundary on the
    system turn. See :class:`CompletionProvider` for the failure
    contract.

    Call :func:`complete_with_provider_model` instead when the caller needs to
    record *which* pairing served the request.
    """
    text, _ = await complete_with_provider_model(
        purpose,
        prompt,
        max_tokens=max_tokens,
        system=system,
        temperature=temperature,
        images=images,
        response_schema=response_schema,
        cache_prefix=cache_prefix,
        **opts,
    )
    return text


async def complete_with_provider_model(
    purpose: LLMPurpose,
    prompt: str,
    *,
    max_tokens: int,
    system: str | None = None,
    temperature: float | None = None,
    images: Sequence[VisionImage] | None = None,
    response_schema: dict[str, Any] | None = None,
    cache_prefix: str | None = None,
    **opts: object,
) -> tuple[str, str]:
    """Run one completion and report the ``"<provider>:<model>"`` that served it.

    Identical to :func:`complete` but returns ``(text, provider_model)``. The
    provider is resolved **once** and both values come off that one object, so
    the reported pairing is necessarily the one that ran — :func:`get_provider`
    reads live config on every call (a reload mid-pass changes the answer), so
    resolving a second time to ask "who served that?" could disagree with the
    call it claims to describe.

    This is the seam the extraction path stamps particles from.
    The pairing is what was *requested*: no adapter reads the served model back
    off the response, so a vendor alias resolving to a dated snapshot is
    invisible here.
    """
    provider = get_provider(purpose)
    text = await provider.complete(
        prompt,
        max_tokens=max_tokens,
        system=system,
        temperature=temperature,
        images=images,
        response_schema=response_schema,
        cache_prefix=cache_prefix,
        **opts,
    )
    return text, provider.provider_model


async def complete_many(
    purpose: LLMPurpose,
    requests: Sequence[CompletionRequest],
    *,
    max_tokens: int,
    temperature: float | None = None,
    response_schema: dict[str, Any] | None = None,
    latency_tolerant: bool = False,
) -> list[str | None]:
    """Run many independent completions, batching them when that is allowed.

    The counterpart to :func:`complete` for a call site that has a *set* of
    unrelated prompts rather than one — a contradiction probe per candidate
    pair, a behavioural-match judgement per batch of beliefs. Results are
    positionally aligned with ``requests``; ``None`` marks a request that
    failed, mirroring the per-call degradation every such call site already
    implements.

    ``latency_tolerant`` is the caller's assertion that nobody is waiting for
    this — the nightly consolidation cycle sets it, an interactive
    ``particles lint`` does not. Only then, and only when the configured
    provider implements :class:`BatchCompletionProvider` and the request count
    clears ``llm.batch.min_requests``, is the work submitted as one batch (half
    price, minutes-to-hours). Every other path — flag unset, batching disabled,
    a provider without the capability, a job the provider refused — runs the
    same sequential ``complete()`` calls as before, so batching is a cost
    optimisation that can never change what a call site can compute.

    Call :func:`complete_many_with_provider_model` instead when the caller
    needs to record *which* pairing served the set.
    """
    results, _ = await complete_many_with_provider_model(
        purpose,
        requests,
        max_tokens=max_tokens,
        temperature=temperature,
        response_schema=response_schema,
        latency_tolerant=latency_tolerant,
    )
    return results


async def complete_many_with_provider_model(
    purpose: LLMPurpose,
    requests: Sequence[CompletionRequest],
    *,
    max_tokens: int,
    temperature: float | None = None,
    response_schema: dict[str, Any] | None = None,
    latency_tolerant: bool = False,
) -> tuple[list[str | None], str]:
    """Run many independent completions and report the pairing that served them.

    Identical to :func:`complete_many` but returns
    ``(results, "<provider>:<model>")`` — the many-request twin of
    :func:`complete_with_provider_model`, and the seam the pooled
    extraction path stamps particles from. The provider is resolved **once**
    and both values come off that one object, for the same reason as the
    single-call variant: config can reload mid-pass, so re-resolving to ask
    "who served that?" could disagree with the calls it claims to describe.
    An empty request set returns ``([], "")`` without resolving a provider.
    """
    from particles.config import get_config

    if not requests:
        return [], ""

    provider = get_provider(purpose)
    batch_cfg = get_config().llm.batch
    if (
        latency_tolerant
        and batch_cfg.enabled
        and len(requests) >= batch_cfg.min_requests
        and isinstance(provider, BatchCompletionProvider)
    ):
        try:
            results = await provider.complete_many(
                requests,
                max_tokens=max_tokens,
                temperature=temperature,
                response_schema=response_schema,
            )
        except Exception as exc:
            # The *job* could not be run (submission rejected, SDK too old to
            # expose batches, a transport failure). Fall through to sequential
            # rather than surfacing a cost optimisation as a capability loss —
            # an account-level failure raises again on the first sequential
            # call, where the breaker sees it as usual.
            log.warning(
                "Batch completion unavailable for purpose %s (%s); "
                "falling back to %d sequential call(s) at full price.",
                purpose,
                exc,
                len(requests),
            )
        else:
            # A provider that under-returns would silently misalign every
            # result with its request, so pad rather than trust the length.
            if len(results) != len(requests):
                log.error(
                    "Batch provider returned %d result(s) for %d request(s); "
                    "treating the remainder as unavailable.",
                    len(results),
                    len(requests),
                )
                results = [*results[: len(requests)]] + [None] * max(
                    0, len(requests) - len(results)
                )
            return results, provider.provider_model

    out: list[str | None] = []
    for request in requests:
        try:
            out.append(
                await provider.complete(
                    request.prompt,
                    max_tokens=max_tokens,
                    system=request.system,
                    temperature=temperature,
                    response_schema=response_schema,
                    cache_prefix=request.cache_prefix,
                )
            )
        except Exception as exc:
            from particles.llm.errors import is_account_level_failure

            if is_account_level_failure(exc):
                # Every remaining call would fail the same way. Raise so the
                # caller's breaker trips once instead of the loop logging N
                # identical failures.
                raise
            log.info("Completion failed for one request of %d: %s", len(requests), exc)
            out.append(None)
    return out, provider.provider_model
