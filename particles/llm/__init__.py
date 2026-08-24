# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""LLM completion port + Anthropic adapter.

The public surface:

* :class:`CompletionProvider` — the port every completion call site depends on.
* :func:`complete` / :func:`get_provider` — resolve the adapter configured for
  an :data:`LLMPurpose` and run a completion.
* :func:`complete_many` / :class:`CompletionRequest` — the same for a *set* of
  independent prompts, submitted as one half-price batch when the caller is
  latency-tolerant and the adapter implements :class:`BatchCompletionProvider`
  , and run sequentially otherwise.
* :func:`get_client` / :func:`set_client` — the shared Anthropic SDK client
  and its test seam, re-exported from ``particles/llm/client.py`` so the
  long-standing ``from particles.llm import get_client`` /
  ``set_client(None)`` mocking pattern keeps working unchanged.

This module grew out of the former single-file ``particles/llm.py`` seam; the
client helpers moved to ``client.py`` and the port was layered on top.
"""

from __future__ import annotations

from particles.llm.adapters.anthropic import AnthropicProvider
from particles.llm.adapters.local import LocalProvider
from particles.llm.adapters.openai_compat import OpenAICompatProvider
from particles.llm.client import get_client, set_client
from particles.llm.errors import AccountLevelLLMError, is_account_level_failure
from particles.llm.fencing import (
    data_fence_instruction,
    fence,
    fenced_prompt,
    make_nonce,
)
from particles.llm.pool import CompletionPool
from particles.llm.registry import (
    BatchCompletionProvider,
    CompletionError,
    CompletionProvider,
    CompletionRequest,
    EmptyCompletionError,
    LLMPurpose,
    VisionImage,
    complete,
    complete_many,
    complete_many_with_provider_model,
    complete_with_provider_model,
    get_provider,
)

__all__ = [
    "AccountLevelLLMError",
    "AnthropicProvider",
    "BatchCompletionProvider",
    "CompletionError",
    "CompletionPool",
    "CompletionProvider",
    "CompletionRequest",
    "EmptyCompletionError",
    "LLMPurpose",
    "LocalProvider",
    "OpenAICompatProvider",
    "VisionImage",
    "complete",
    "complete_many",
    "complete_many_with_provider_model",
    "data_fence_instruction",
    "fence",
    "fenced_prompt",
    "get_client",
    "complete_with_provider_model",
    "get_provider",
    "is_account_level_failure",
    "make_nonce",
    "set_client",
]
