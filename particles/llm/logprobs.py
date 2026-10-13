# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Opt-in token log-probability capture, a side channel beside the port.

The ``CompletionProvider`` port returns plain text, and every call site relies
on that. A measurement that also needs the model's per-token log-probabilities
(whether they predict which extracted claims are correct) opens
:func:`capture_logprobs` around the calls it wants to observe. Inside the
scope, the OpenAI-compatible adapter asks the endpoint for ``logprobs`` and
``top_logprobs`` and records each reply's tokens here; the text it returns to
its caller is unchanged. With no scope open nothing is requested and nothing
is recorded, so the default request is byte-for-byte what it was before.

The scope is ``ContextVar``-held, the pattern :func:`particles.llm.usage.track_usage`
uses: it sees the calls made by the task that opened it and by tasks spawned
inside it, and nothing else. Only the OpenAI-compatible adapter can fill it.
The Anthropic API exposes no token log-probabilities, so a call routed there
records nothing, and a consumer must treat a missing record as "no signal",
never as a low one. The same holds for a call served from the replay cache:
the cache stores reply text only, so a replayed call never reaches
the adapter and records nothing. Do not open a capture inside a replay scope.

:func:`span_tokens` maps a character span of a recorded reply back to the
tokens that produced it, which is how a caller turns a parsed field (a claim's
text, its stated confidence) into per-token features.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TokenLogprob:
    """One generated token, its log-probability, and the endpoint's alternatives."""

    token: str
    logprob: float
    #: ``(token, logprob)`` for the ``top_logprobs`` most likely tokens at this
    #: position, most likely first. Empty when the endpoint sent none.
    top: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class CapturedCompletion:
    """One reply's tokens, in order, with the pairing that produced them."""

    provider_model: str
    tokens: tuple[TokenLogprob, ...]

    @property
    def text(self) -> str:
        """The reply as the concatenation of its tokens.

        This is the raw generated text, before any adapter post-processing
        (whitespace strip, the structured-output unwrap), so character offsets
        into it line up with the tokens.
        """
        return "".join(t.token for t in self.tokens)


@dataclass
class LogprobCapture:
    """The completions recorded inside one :func:`capture_logprobs` scope.

    Thread-safe, like the usage accumulator, so a worker thread spawned inside
    the scope can record into it.
    """

    top_logprobs: int
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _completions: list[CapturedCompletion] = field(default_factory=list, repr=False)

    def add(self, completion: CapturedCompletion) -> None:
        """Record one reply."""
        with self._lock:
            self._completions.append(completion)

    @property
    def completions(self) -> list[CapturedCompletion]:
        """A copy of the replies recorded so far, in arrival order."""
        with self._lock:
            return list(self._completions)


_capture: ContextVar[LogprobCapture | None] = ContextVar("llm_logprob_capture", default=None)


@contextlib.contextmanager
def capture_logprobs(top_logprobs: int = 5) -> Iterator[LogprobCapture]:
    """Request and record token log-probabilities for the calls made in the block.

    ``top_logprobs`` is how many alternatives per position to ask for; OpenAI
    caps it at 20, and some local servers cap it lower.
    """
    if not 0 <= top_logprobs <= 20:
        raise ValueError(f"top_logprobs must be in [0, 20], got {top_logprobs}")
    capture = LogprobCapture(top_logprobs=top_logprobs)
    token = _capture.set(capture)
    try:
        yield capture
    finally:
        _capture.reset(token)


def active_capture() -> LogprobCapture | None:
    """The innermost open capture scope, or ``None`` (the default)."""
    return _capture.get()


def parse_openai_logprobs(payload: object) -> tuple[TokenLogprob, ...]:
    """Read ``choices[0].logprobs.content`` off a ``chat/completions`` reply.

    Returns an empty tuple when the reply carries no log-probabilities (the
    endpoint ignored the request, or the dialect has none). Malformed entries
    are skipped rather than raised on: this is a measurement side channel and
    must never fail the completion it rides on.
    """
    if not isinstance(payload, dict):
        return ()
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ()
    logprobs = choices[0].get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(content, list):
        return ()
    out: list[TokenLogprob] = []
    for entry in content:
        if not isinstance(entry, dict):
            continue
        token, logprob = entry.get("token"), entry.get("logprob")
        if not isinstance(token, str) or not isinstance(logprob, int | float):
            continue
        top: list[tuple[str, float]] = []
        raw_top = entry.get("top_logprobs")
        if isinstance(raw_top, list):
            for alt in raw_top:
                if isinstance(alt, dict):
                    alt_token, alt_lp = alt.get("token"), alt.get("logprob")
                    if isinstance(alt_token, str) and isinstance(alt_lp, int | float):
                        top.append((alt_token, float(alt_lp)))
        out.append(TokenLogprob(token=token, logprob=float(logprob), top=tuple(top)))
    return tuple(out)


def record_logprobs(provider_model: str, payload: object) -> None:
    """Record a reply's log-probabilities into the open scope, if there is one."""
    capture = _capture.get()
    if capture is None:
        return
    tokens = parse_openai_logprobs(payload)
    if tokens:
        capture.add(CapturedCompletion(provider_model=provider_model, tokens=tokens))


def span_tokens(completion: CapturedCompletion, start: int, end: int) -> list[TokenLogprob]:
    """The tokens whose characters overlap ``[start, end)`` of ``completion.text``.

    A token that straddles a boundary is included, so the span is never
    under-covered. An empty or out-of-range span returns no tokens.
    """
    if end <= start:
        return []
    out: list[TokenLogprob] = []
    pos = 0
    for tok in completion.tokens:
        tok_end = pos + len(tok.token)
        if tok_end > start and pos < end:
            out.append(tok)
        if pos >= end:
            break
        pos = tok_end
    return out
