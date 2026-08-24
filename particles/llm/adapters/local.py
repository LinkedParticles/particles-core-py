# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Back-compat shim: the adapter moved to ``openai_compat.py``.

``LocalProvider`` was the single-endpoint ancestor of the generic
:class:`particles.llm.adapters.openai_compat.OpenAICompatProvider`;
``LocalProvider(model)`` is now exactly ``OpenAICompatProvider(name="local",
model=model)`` — the compiled-in ``local`` entry in ``llm.providers``. New
code should import from ``particles.llm.adapters.openai_compat``.
"""

from __future__ import annotations

from particles.llm.adapters.openai_compat import (
    OpenAICompatCompletionError,
    OpenAICompatProvider,
    _extract_text,  # noqa: F401  — re-exported for existing test imports
    _post_with_retry,  # noqa: F401
    _scrub_response_body,  # noqa: F401
)

#: The pre-0227 name for the status-carrying failure; same class, so existing
#: ``except LocalCompletionError`` / breaker duck-typing is untouched.
LocalCompletionError = OpenAICompatCompletionError


class LocalProvider(OpenAICompatProvider):
    """The pre-0227 constructor signature: ``LocalProvider(model=...)``."""

    def __init__(self, model: str) -> None:
        super().__init__(name="local", model=model)
