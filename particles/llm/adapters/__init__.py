# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Concrete :class:`~particles.llm.registry.CompletionProvider` adapters.

Two adapters ship: ``AnthropicProvider`` (hosted) and ``LocalProvider`` (an
OpenAI-compatible / Ollama endpoint) for cheap local extraction.
"""

from __future__ import annotations

from particles.llm.adapters.anthropic import AnthropicProvider
from particles.llm.adapters.local import LocalProvider

__all__ = ["AnthropicProvider", "LocalProvider"]
