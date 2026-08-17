"""Shared Anthropic SDK client (the ``anthropic`` adapter's transport).

The ``AnthropicProvider`` adapter (``particles/llm/adapters/anthropic.py``) is
the only production caller of ``get_client()``; every completion call site
reaches the model through the :class:`CompletionProvider` port, not
this client directly. ``ANTHROPIC_API_KEY`` is read at first use and the
client is cached for the process lifetime.

Tests override the client via ``set_client()``; passing ``None`` clears the
cache so the next ``get_client()`` rebuilds against the current environment.
This remains the canonical Anthropic mocking seam — patching it reaches every
purpose that resolves to the ``anthropic`` provider. See ``tests/AGENTS.md``.
"""

from __future__ import annotations

import anthropic

from particles.secrets import get_anthropic_api_key

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    """Return the process-level Anthropic client, creating it on first use.

    Raises ``ValueError`` if ``ANTHROPIC_API_KEY`` is unset and no client has
    been injected via ``set_client()``. Without this pre-flight check the
    Anthropic SDK fails much later with the cryptic message
    "Could not resolve authentication method…".
    """
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=get_anthropic_api_key())
    return _client


def set_client(client: anthropic.Anthropic | None) -> None:
    """Override or clear the cached client.

    Used by tests; passing ``None`` clears the cache so the next ``get_client()``
    rebuilds. See ``tests/AGENTS.md`` for the mocking pattern.
    """
    global _client
    _client = client
