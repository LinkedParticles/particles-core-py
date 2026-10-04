# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Anthropic ``CompletionProvider`` adapter.

Ports today's behaviour verbatim: one ``client.messages.create`` call against
the shared Anthropic SDK client (``particles/llm/client.py``), returning the
first text block. A call whose ``max_tokens`` is above the SDK's non-streaming
ceiling is streamed instead (``client.messages.stream`` read to its final
message), which yields the same ``Message`` and so the same text, stop reason
and usage. The call is delegated to a worker thread via
:func:`asyncio.to_thread` — the SDK call is synchronous, and running it off
the event loop keeps ``KeyboardInterrupt`` responsive between calls (the
property the article-synthesis seam relied on before this consolidation) and
avoids blocking concurrent coroutines.

Since this adapter also implements the optional
:class:`~particles.llm.registry.BatchCompletionProvider` capability on top of
the Message Batches API (``client.messages.batches``), which prices all token
usage at 50% in exchange for a completion time measured in minutes-to-hours.
Same thread-offload discipline: submit, poll, and result-collection are each
synchronous SDK calls dispatched via :func:`asyncio.to_thread`, with the waiting
done in :func:`asyncio.sleep` so the event loop stays live throughout.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from anthropic import Omit, omit
from opentelemetry import metrics, trace

from particles.llm.batch_budget import current_batch_wait_budget
from particles.llm.registry import (
    CompletionError,
    CompletionRequest,
    EmptyCompletionError,
    RequestFailure,
    VisionImage,
    budget_hint,
    run_sequential,
)
from particles.llm.spend_budget import current_spend_budget
from particles.llm.usage import current_purpose, record_usage

if TYPE_CHECKING:
    from anthropic.types import MessageParam
    from anthropic.types.messages.batch_create_params import Request as BatchRequest

log = logging.getLogger(__name__)

# Hand-rolled telemetry (Phase 2). The OTel **API** is used directly
# (no-op until a provider is installed), so this Client-layer adapter stays free
# of the Engine-side SDK. The span wraps the ``await`` call site so the active
# context parents it across the ``asyncio.to_thread`` boundary.
_tracer = trace.get_tracer("particles.llm")
_meter = metrics.get_meter("particles.llm")
_llm_duration = _meter.create_histogram(
    "particles.llm.duration", unit="s", description="LLM completion call wall time"
)

#: Model ids that answer HTTP 400 when sent ``temperature``.
#:
#: Newer Claude models have deprecated the parameter — ``claude-sonnet-5``
#: returns ``400 `temperature` is deprecated for this model``. Learned at
#: runtime from the first such rejection rather than hardcoded, because a
#: compiled-in model list is wrong the day the next model ships and there is no
#: capability endpoint to ask. Process-local and never persisted: the cost of a
#: cold start is one wasted call per model, and the cost of a stale pin would be
#: silently dropping a parameter the operator asked for.
#:
#: Deliberately **not** a config knob. The dialect knobs
#: (`send_temperature` on the OpenAI-compatible adapter) are per-*provider
#: entry*, and this constraint is per-*model*: one `anthropic` provider serves
#: every Claude model an operator routes, and they do not agree.
_TEMPERATURE_UNSUPPORTED: set[str] = set()

#: The largest ``max_tokens`` this adapter sends as a plain (non-streaming)
#: request; a larger budget is streamed.
#:
#: The Anthropic Python SDK refuses a non-streaming ``messages.create`` whose
#: budget implies a reply longer than ten minutes, at its assumed rate of
#: 128,000 tokens an hour (``3600 * max_tokens / 128_000 > 600``). That puts the
#: ceiling at 21,333 tokens. The extraction budgets sit above it because a long
#: memory file needs more than 16,384 tokens of reply, and
#: claude-sonnet-5 takes up to 128K output tokens when streamed. Calls at or
#: under the ceiling keep the plain request, so short calls (probes, judges,
#: the §9.3 answer) are unchanged.
_NONSTREAMING_MAX_TOKENS = 21_333


def _is_unsupported_temperature(exc: Exception) -> bool:
    """True when Anthropic rejected the ``temperature`` parameter itself.

    An HTTP 400 naming the parameter is the shape the API uses; anything else
    is a real request failure and must not be masked by silently dropping a
    sampling setting the caller chose. Mirrors
    ``openai_compat._is_unsupported_response_format``.
    """
    if getattr(exc, "status_code", None) != 400:
        return False
    return "temperature" in str(exc).lower()


@dataclass(frozen=True)
class AnthropicProvider:
    """Completion provider backed by the Anthropic Messages API.

    Tests inject a mock SDK client via ``particles.llm.set_client`` — that seam
    reaches this adapter because ``complete`` resolves ``get_client()`` at call
    time rather than capturing it.
    """

    model: str

    @property
    def provider_model(self) -> str:
        return f"anthropic:{self.model}"

    def _temperature_arg(self, temperature: float | None) -> float | Omit:
        """The ``temperature`` to send for this model, or ``omit``.

        ``omit`` both when the caller supplied none and when this model is
        already known to reject the parameter.
        """
        if temperature is None or self.model in _TEMPERATURE_UNSUPPORTED:
            return omit
        return temperature

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
        # ``response_schema`` is deliberately ignored in v1:
        # Claude's schema adherence is what the tolerant call-site parsers were
        # tuned on; tool-use-based enforcement is a possible later upgrade
        # (captured at activation, out of scope here).
        del response_schema
        # A plain string content is the text-only path (unchanged); a list of
        # blocks carries the prompt plus one image block per VisionImage for a
        # multimodal request. Claude 3+/4 models are vision-capable;
        # a non-multimodal model surfaces the mismatch as an Anthropic API error.
        content: object
        if images:
            content = [{"type": "text", "text": prompt}, *(_image_block(img) for img in images)]
        else:
            content = prompt

        # The Anthropic Messages content is a union of richly-typed block
        # TypedDicts; build the multimodal list as plain dicts and cast once at
        # the call rather than threading the SDK's block-param types through.
        messages = [cast("MessageParam", {"role": "user", "content": content})]
        # render the system turn, caching ``cache_prefix`` as a
        # ``cache_control`` block when prompt caching is on.
        system_arg = _system_arg(system, cache_prefix)
        cache_usage: dict[str, int] = {}
        streamed = max_tokens > _NONSTREAMING_MAX_TOKENS

        def _call() -> str:
            from particles.llm.client import get_client

            client = get_client()

            def _create(temp: float | Omit) -> Any:
                # SDK 1.x dropped the typed sampling kwargs; models that still
                # honour ``temperature`` take it through ``extra_body``.
                extra_body = None if isinstance(temp, Omit) else {"temperature": temp}
                if streamed:
                    # The SDK's streaming helper accumulates the events into
                    # the same ``Message`` a plain request returns: content,
                    # ``stop_reason``, ``stop_details`` and the final ``usage``
                    # (cache fields included). Everything below reads that
                    # message, so both paths are handled identically.
                    with client.messages.stream(
                        model=self.model,
                        max_tokens=max_tokens,
                        messages=messages,
                        system=system_arg,
                        extra_body=extra_body,
                    ) as stream:
                        return stream.get_final_message()
                return client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=messages,
                    system=system_arg,
                    extra_body=extra_body,
                )

            temp_arg = self._temperature_arg(temperature)
            try:
                resp = _create(temp_arg)
            except Exception as exc:
                if isinstance(temp_arg, Omit) or not _is_unsupported_temperature(exc):
                    raise
                # this model has deprecated `temperature`. Retry ONCE
                # without it and remember, so the rest of the process skips the
                # wasted attempt — the equivalence judge issues one call per
                # contested pair, so paying a 400 on each would be the whole
                # cost of the run. Same graceful-degradation shape as the
                # openai_compat adapter's response_format downgrade.
                _TEMPERATURE_UNSUPPORTED.add(self.model)
                log.warning(
                    "Anthropic model %s rejected the temperature parameter; "
                    "retrying without it and omitting it for the rest of this "
                    "process. Sampling is now the model's default, so calls "
                    "that relied on temperature=0 are no longer deterministic: "
                    "%s",
                    self.model,
                    exc,
                )
                resp = _create(omit)
            # Billed whatever happens next (a refusal or an empty reply is
            # still paid for), so it is counted before either raises.
            _record_message_usage(self.provider_model, resp, batched=False)
            # Record prompt-cache usage for verification/observability without
            # widening the port's return; the widening is deferred.
            usage = getattr(resp, "usage", None)
            if usage is not None:
                cache_usage["read"] = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
                cache_usage["write"] = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
            stop_reason = getattr(resp, "stop_reason", None)
            if stop_reason == "refusal":
                # Claude 4+ safety classifiers decline with HTTP 200 and empty
                # (or partial) content — surface it as what it is, not as a
                # malformed response. ``stop_details`` is Opus 4.7+ and may be
                # absent or None even on a refusal; guard every hop.
                details = getattr(resp, "stop_details", None)
                category = getattr(details, "category", None) if details is not None else None
                raise CompletionError(
                    "Anthropic declined the request (stop_reason=refusal"
                    + (f", category={category}" if category else "")
                    + ") — the model's safety classifiers refused this content"
                )
            if stop_reason == "max_tokens":
                # Budget exhaustion returns HTTP 200 with text that stops
                # mid-token, so the call site's parser reports a malformed
                # reply instead of a short budget. Say which it is. (The
                # openai_compat adapter reads finish_reason for the same
                # reason; extended-thinking models spend the same budget on
                # their thinking as on the answer.)
                log.warning(
                    "Anthropic reply from %s was truncated at max_tokens=%d "
                    "(stop_reason=max_tokens): the text stops mid-token and may "
                    "not parse. To fix: %s.",
                    self.model,
                    max_tokens,
                    budget_hint(),
                )
            text_block = next((b for b in resp.content if hasattr(b, "text")), None)
            if text_block is None:
                # Deterministic at this budget (an extended-thinking model that
                # spent max_tokens before answering reproduces it on a retry),
                # so it is raised as the narrower EmptyCompletionError — a
                # CompletionError subclass, invisible to every call site that
                # does not care about the distinction.
                raise EmptyCompletionError(
                    f"Anthropic response carried no text block "
                    f"(stop_reason={stop_reason}, max_tokens={max_tokens}). "
                    f"To fix: {budget_hint()}"
                )
            return str(text_block.text).strip()

        with _tracer.start_as_current_span("llm.complete") as span:
            span.set_attribute("llm.model", self.model)
            span.set_attribute("llm.max_tokens", max_tokens)
            span.set_attribute("llm.streamed", streamed)
            span.set_attribute("llm.images", len(images) if images else 0)
            start = time.perf_counter()
            try:
                result = await asyncio.to_thread(_call)
                if cache_usage:
                    span.set_attribute("llm.cache.read_tokens", cache_usage.get("read", 0))
                    span.set_attribute("llm.cache.write_tokens", cache_usage.get("write", 0))
                    if cache_usage.get("read"):
                        log.debug(
                            "Anthropic prompt-cache read %d token(s) on %s",
                            cache_usage["read"],
                            self.model,
                        )
                return result
            finally:
                _llm_duration.record(time.perf_counter() - start, {"model": self.model})

    async def complete_many(
        self,
        requests: Sequence[CompletionRequest],
        *,
        max_tokens: int,
        temperature: float | None = None,
        response_schema: dict[str, Any] | None = None,
        failures_out: list[RequestFailure | None] | None = None,
        **opts: object,
    ) -> list[str | None]:
        """Run ``requests`` through the Message Batches API.

        One ``POST /v1/messages/batches`` submission keyed by ``custom_id``,
        polled to ``ended``, then read back and re-aligned to the input order —
        **results arrive in arbitrary order**, so the mapping is by
        ``custom_id`` and never by position. All token usage is billed at 50%,
        which is the entire reason this path exists.

        ``response_schema`` is ignored exactly as in :meth:`complete`; the
        tolerant call-site parsers are what the Anthropic path was tuned on.
        A batch that outlives ``llm.batch.max_wait_seconds`` is cancelled
        rather than stalling the caller. The requests it had finished are
        still read back (a bounded grace for the cancellation to end, then the
        usual collection), and only the rest come back ``None``.

        A batch is never streamed, and needs no streaming: the SDK's
        non-streaming ceiling (``_NONSTREAMING_MAX_TOKENS``) guards only
        ``messages.create``, whose reply is awaited on the open connection.
        ``batches.create`` returns as soon as the job is queued, so each request
        in it takes the full ``max_tokens`` up to the model's own output limit.

        ``failures_out`` is extended with why each ``None`` came back: a reply
        with no text block is :attr:`RequestFailure.EMPTY` (a budget spent on
        thinking, or a refusal), anything that never produced a reply is
        :attr:`RequestFailure.UNAVAILABLE`.
        """
        del response_schema, opts
        if not requests:
            return []

        from particles.config import get_config

        cfg = get_config().llm.batch
        results: list[str | None] = []
        failures: list[RequestFailure | None] = []
        spend = current_spend_budget()
        for start in range(0, len(requests), cfg.max_requests_per_batch):
            chunk = requests[start : start + cfg.max_requests_per_batch]
            if spend is not None and not spend.allows_chunk(
                purpose=current_purpose(),
                provider_model=self.provider_model,
                requests=chunk,
                max_tokens=max_tokens,
            ):
                # the run's dollar budget would be passed. The chunk
                # is not submitted; its requests read as unavailable, the way a
                # cancelled batch's lost requests do.
                log.info(
                    "Spend budget: %d request(s) not submitted (spent US$%.2f of US$%.2f).",
                    len(chunk),
                    spend.spent(),
                    spend.budget_usd,
                )
                results.extend([None] * len(chunk))
                failures.extend([RequestFailure.UNAVAILABLE] * len(chunk))
                continue
            chunk_results, chunk_failures = await self._run_batch(
                chunk,
                max_tokens=max_tokens,
                temperature=temperature,
                poll_interval=cfg.poll_interval_seconds,
                max_wait=cfg.max_wait_seconds,
                cancel_grace=cfg.cancel_grace_seconds,
                rerun_max_output_tokens=cfg.cancel_rerun_max_output_tokens,
            )
            results.extend(chunk_results)
            failures.extend(chunk_failures)
        if failures_out is not None:
            failures_out.extend(failures)
        return results

    async def _run_batch(
        self,
        requests: Sequence[CompletionRequest],
        *,
        max_tokens: int,
        temperature: float | None,
        poll_interval: float,
        max_wait: float,
        cancel_grace: float = 0.0,
        rerun_max_output_tokens: int = 0,
    ) -> tuple[list[str | None], list[RequestFailure | None]]:
        """Submit one batch, poll it to completion, and return aligned results and failures.

        Under a run's batch-wait budget this is also where the budget
        is kept, once per chunk: a chunk that finds less than the floor left is
        not submitted but served by sequential ``complete()`` calls; otherwise
        its wait is capped at the balance left and the time it waited is
        charged as it returns, so the next chunk never sees an unpaid balance.

        A batch that outlives its wait is cancelled, then given up to
        ``cancel_grace`` seconds to reach ``ended`` so the requests it finished
        are kept. The grace is charged with the rest of the wait.
        """
        budget = current_batch_wait_budget()
        if budget is not None and not budget.can_submit():
            log.info(
                "Batch-wait budget spent (%.0fs of %.0fs); %d request(s) run "
                "sequentially at full price instead of as a batch.",
                budget.spent_seconds,
                budget.budget_seconds,
                len(requests),
            )
            budget.record_sequential(len(requests))
            return await run_sequential(
                self,
                requests,
                max_tokens=max_tokens,
                temperature=temperature,
                response_schema=None,
            )
        capped_by_budget = False
        if budget is not None:
            max_wait, capped_by_budget = budget.wait_cap(max_wait)

        # ``custom_id`` is this chunk's index as a string; each chunk is its own
        # batch, so the ids only have to be unique within it.
        params: list[dict[str, Any]] = []
        for i, request in enumerate(requests):
            body: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": request.prompt}],
            }
            # the batch honours the same cache-prefix rendering as the
            # single-shot path — a repeated prefix caches across batch entries too.
            sys_arg = _system_arg(request.system, request.cache_prefix)
            if not isinstance(sys_arg, Omit):
                body["system"] = sys_arg
            # honour the same per-model memo `complete` populates. A
            # batch submission is all-or-nothing, so a rejected parameter fails
            # the whole job rather than one pair — there is no per-request retry
            # to fall back on, which is why this path only reads the memo and
            # never learns from a failure of its own.
            batch_temp = self._temperature_arg(temperature)
            if not isinstance(batch_temp, Omit):
                body["temperature"] = batch_temp
            params.append({"custom_id": str(i), "params": body})

        def _submit() -> str:
            from particles.llm.client import get_client

            # Same discipline as ``complete``: build the richly-typed request
            # TypedDicts as plain dicts and cast once at the call boundary.
            batch = get_client().messages.batches.create(
                requests=cast("list[BatchRequest]", params)
            )
            return str(batch.id)

        with _tracer.start_as_current_span("llm.complete_many") as span:
            span.set_attribute("llm.model", self.model)
            span.set_attribute("llm.batch.requests", len(requests))
            started = time.perf_counter()
            try:
                batch_id = await asyncio.to_thread(_submit)
                log.info(
                    "Submitted Anthropic message batch %s (%d request(s), model %s)",
                    batch_id,
                    len(requests),
                    self.model,
                )
                span.set_attribute("llm.batch.id", batch_id)
                waiting_since = time.monotonic()
                # The wait is reported while it runs, not only charged when it
                # returns, so the run owner can show it.
                wait_token = budget.begin_wait(max_wait) if budget is not None else None
                on_poll = budget.note_wait if budget is not None else None
                try:
                    ended = await self._await_batch(
                        batch_id,
                        poll_interval=poll_interval,
                        max_wait=max_wait,
                        by_budget=capped_by_budget,
                        on_poll=on_poll,
                    )
                    cancelled = not ended
                    if cancelled:
                        ended = await self._await_cancelled(
                            batch_id, poll_interval=poll_interval, grace=cancel_grace
                        )
                finally:
                    if budget is not None and wait_token is not None:
                        budget.end_wait(wait_token)
                if budget is not None:
                    budget.charge(
                        time.monotonic() - waiting_since,
                        requests=len(requests),
                        cut_short=cancelled and capped_by_budget,
                    )
                if not ended:
                    log.warning(
                        "Anthropic message batch %s did not end within %.0fs of being "
                        "cancelled (llm.batch.cancel_grace_seconds); its %d request(s) are "
                        "reported unavailable. Any it finished can be read back by id once "
                        "it ends.",
                        batch_id,
                        cancel_grace,
                        len(requests),
                    )
                    if budget is not None:
                        budget.record_cancellation(kept=0, rerun=0, lost=len(requests))
                    return [None] * len(requests), [RequestFailure.UNAVAILABLE] * len(requests)
                canceled_ids: set[str] = set()
                by_id = await asyncio.to_thread(
                    self._collect, batch_id, max_tokens, self.provider_model, canceled_ids
                )
            finally:
                _llm_duration.record(time.perf_counter() - started, {"model": self.model})

        replies = [by_id.get(str(i), _MISSING_REPLY) for i in range(len(requests))]
        texts = [r.text for r in replies]
        failures = [r.failure for r in replies]
        if cancelled:
            await self._settle_cancelled(
                batch_id,
                requests,
                texts,
                failures,
                canceled_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                rerun_max_output_tokens=rerun_max_output_tokens,
            )
        return texts, failures

    async def _settle_cancelled(
        self,
        batch_id: str,
        requests: Sequence[CompletionRequest],
        texts: list[str | None],
        failures: list[RequestFailure | None],
        canceled_ids: set[str],
        *,
        max_tokens: int,
        temperature: float | None,
        rerun_max_output_tokens: int,
    ) -> None:
        """Re-run a cancelled batch's small remainder, then disclose it (§5).

        Fills ``texts`` / ``failures`` in place. The remainder is only the
        requests whose result was ``canceled``: an ``errored`` or ``expired``
        one failed on its own and would fail again. It is re-run only inside a
        batch-wait budget scope, which is the marker for a run that already
        chooses batch-or-sequential per set, and only while
        ``len(remainder) * max_tokens`` stays within the bound, so short
        judgements qualify and extraction requests never do.
        """
        budget = current_batch_wait_budget()
        unavailable = sum(1 for f in failures if f is RequestFailure.UNAVAILABLE)
        kept = len(requests) - unavailable
        remainder = [i for i in range(len(requests)) if str(i) in canceled_ids]
        rerun = 0
        if (
            budget is not None
            and remainder
            and len(remainder) * max_tokens <= rerun_max_output_tokens
        ):
            re_texts, re_failures = await run_sequential(
                self,
                [requests[i] for i in remainder],
                max_tokens=max_tokens,
                temperature=temperature,
                response_schema=None,
            )
            for i, text, failure in zip(remainder, re_texts, re_failures, strict=True):
                texts[i] = text
                failures[i] = failure
                if failure is not RequestFailure.UNAVAILABLE:
                    rerun += 1
        lost = unavailable - rerun
        log.warning(
            "Anthropic message batch %s cancelled: %d of %d request(s) had finished and "
            "are kept; %d re-run sequentially; %d unavailable.",
            batch_id,
            kept,
            len(requests),
            rerun,
            lost,
        )
        if budget is not None:
            budget.record_cancellation(kept=kept, rerun=rerun, lost=lost)

    async def _await_batch(
        self,
        batch_id: str,
        *,
        poll_interval: float,
        max_wait: float,
        by_budget: bool = False,
        on_poll: Callable[[], None] | None = None,
    ) -> bool:
        """Poll ``batch_id`` until it ends. False ⇒ gave up (and cancelled it).

        ``on_poll`` runs after every status check that finds the batch still
        pending, so a watcher sees the wait advance.
        """
        deadline = time.monotonic() + max_wait
        while True:
            if await asyncio.to_thread(self._status, batch_id) == "ended":
                return True
            if on_poll is not None:
                on_poll()
            if time.monotonic() >= deadline:
                if by_budget:
                    log.warning(
                        "Anthropic message batch %s still processing after %.0fs, the "
                        "run's remaining batch-wait budget "
                        "(consolidation.batch_wait.budget_seconds); cancelling. The "
                        "requests it finished are kept, the rest reported unavailable.",
                        batch_id,
                        max_wait,
                    )
                    await asyncio.to_thread(self._cancel, batch_id)
                    return False
                log.warning(
                    "Anthropic message batch %s still processing after %.0fs "
                    "(llm.batch.max_wait_seconds); cancelling. The requests it "
                    "finished are kept, the rest reported unavailable.",
                    batch_id,
                    max_wait,
                )
                await asyncio.to_thread(self._cancel, batch_id)
                return False
            await asyncio.sleep(poll_interval)

    async def _await_cancelled(self, batch_id: str, *, poll_interval: float, grace: float) -> bool:
        """Poll a just-cancelled batch for up to ``grace`` seconds. True ⇒ it ended.

        A cancelled batch passes through ``canceling`` before ``ended``, and
        only an ended batch has results to read. ``grace`` 0
        skips the wait, which restores the pre-0290 discard.
        """
        if grace <= 0:
            return False
        deadline = time.monotonic() + grace
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(poll_interval, remaining))
            if await asyncio.to_thread(self._status, batch_id) == "ended":
                return True

    @staticmethod
    def _status(batch_id: str) -> str:
        """The batch's ``processing_status``."""
        from particles.llm.client import get_client

        batch = get_client().messages.batches.retrieve(batch_id)
        return str(getattr(batch, "processing_status", ""))

    @staticmethod
    def _cancel(batch_id: str) -> None:
        """Best-effort cancel of an over-running batch; never raises."""
        from particles.llm.client import get_client

        try:
            get_client().messages.batches.cancel(batch_id)
        except Exception as exc:  # noqa: BLE001 — cancellation is courtesy, not contract
            log.info("Could not cancel message batch %s: %s", batch_id, exc)

    @staticmethod
    def _collect(
        batch_id: str,
        max_tokens: int,
        provider_model: str,
        canceled_out: set[str] | None = None,
    ) -> dict[str, _BatchReply]:
        """Read a finished batch's results into ``{custom_id: _BatchReply}``.

        ``max_tokens`` is the batch's budget, carried for the truncation
        warning only — a request whose reply hit the ceiling is reported as
        the budget problem it is rather than as a malformed answer. The
        warning is one line per batch with the truncated count; each
        truncated request is an INFO line.
        ``provider_model`` keys each succeeded request's usage, which is
        recorded as batched (billed at the batch discount). ``canceled_out``,
        when given, collects the ids whose result was ``canceled``: the
        remainder a cancelled batch may re-run.
        """
        from particles.llm.client import get_client

        out: dict[str, _BatchReply] = {}
        truncated: list[str] = []
        for entry in get_client().messages.batches.results(batch_id):
            custom_id = str(entry.custom_id)
            result = entry.result
            if result.type != "succeeded":
                # errored / canceled / expired — one dead request, not a dead
                # batch. The call site degrades on the None as it would on any
                # other unavailable probe.
                log.info(
                    "Batch %s request %s did not succeed (%s)", batch_id, custom_id, result.type
                )
                if canceled_out is not None and result.type == "canceled":
                    canceled_out.add(custom_id)
                out[custom_id] = _BatchReply(None, RequestFailure.UNAVAILABLE)
                continue
            message = result.message
            _record_message_usage(provider_model, message, batched=True)
            stop_reason = getattr(message, "stop_reason", None)
            if stop_reason == "max_tokens":
                # Same masquerade as the single-shot path: a truncated batch
                # reply either parses as garbage downstream or comes back with
                # no text block at all. The per-request line is detail at INFO
                # (``--verbose``); the batch's one WARNING below names the
                # budget (aggregate by default, detail on demand).
                truncated.append(custom_id)
                log.info(
                    "Batch %s request %s was truncated at max_tokens=%d (stop_reason=max_tokens)",
                    batch_id,
                    custom_id,
                    max_tokens,
                )
            text_block = next((b for b in message.content if hasattr(b, "text")), None)
            if text_block is None:
                # No text block: a refusal, or an answer that spent its whole
                # max_tokens on thinking. Same treatment as an errored request.
                log.info(
                    "Batch %s request %s carried no text block (stop_reason=%s)",
                    batch_id,
                    custom_id,
                    stop_reason,
                )
                out[custom_id] = _BatchReply(None, RequestFailure.EMPTY)
                continue
            out[custom_id] = _BatchReply(str(text_block.text).strip(), None)
        if truncated:
            log.warning(
                "Batch %s: %d of %d request(s) were truncated at max_tokens=%d "
                "(stop_reason=max_tokens, llm purpose %s). To fix: %s. "
                "Per-request detail with --verbose.",
                batch_id,
                len(truncated),
                len(out),
                max_tokens,
                current_purpose() or "unscoped",
                budget_hint(),
            )
        return out


class _BatchReply(NamedTuple):
    """One batch entry read back: its text, or why there is none."""

    text: str | None
    failure: RequestFailure | None


_MISSING_REPLY = _BatchReply(None, RequestFailure.UNAVAILABLE)


def _usage_tokens(usage: object, name: str) -> int:
    """One token count off an SDK ``usage`` object; 0 when absent or not an int."""
    value = getattr(usage, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _record_message_usage(provider_model: str, message: object, *, batched: bool) -> None:
    """Report one Messages API response's ``usage`` to the open usage scopes."""
    usage = getattr(message, "usage", None)
    record_usage(
        provider_model,
        input_tokens=_usage_tokens(usage, "input_tokens"),
        output_tokens=_usage_tokens(usage, "output_tokens"),
        cache_creation_tokens=_usage_tokens(usage, "cache_creation_input_tokens"),
        cache_read_tokens=_usage_tokens(usage, "cache_read_input_tokens"),
        hit_max_tokens=getattr(message, "stop_reason", None) == "max_tokens",
        batched=batched,
    )


def _system_arg(system: str | None, cache_prefix: str | None) -> Any:
    """Build the Anthropic ``system`` argument, honouring a cache prefix.

    With a ``cache_prefix`` and ``llm.prompt_cache.enabled``, the system is two
    text blocks — the prefix carrying ``cache_control: {"type": "ephemeral"}``,
    then the variable remainder — so a prefix repeated across calls (within the
    cache TTL) bills as a ~10% cache read. Otherwise the prefix is folded into
    the plain system string (``cache_prefix + system``, content-preserving) and
    no ``cache_control`` is sent, so billing and behaviour are exactly pre-0252.
    Returns ``omit`` when there is nothing to send (today's ``system is None``
    path).
    """
    from particles.config import get_config

    if cache_prefix and get_config().llm.prompt_cache.enabled:
        blocks: list[dict[str, Any]] = [
            {"type": "text", "text": cache_prefix, "cache_control": {"type": "ephemeral"}}
        ]
        if system:
            blocks.append({"type": "text", "text": system})
        return blocks
    if cache_prefix:
        return cache_prefix + (system or "")
    return system if system is not None else omit


def _image_block(img: VisionImage) -> dict[str, object]:
    """Render a :class:`VisionImage` as an Anthropic base64 image content block."""
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": img.media_type,
            "data": base64.standard_b64encode(img.data).decode("ascii"),
        },
    }
