# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for particles/llm/call_meter.py: every completion in a scope is charged."""

from __future__ import annotations

from typing import Any

import pytest

from particles.llm import complete, complete_many, override_providers
from particles.llm.call_meter import ModelCallBudgetSpent, current_meter, metered_calls
from particles.llm.registry import CompletionRequest


class _Counting:
    calls = 0

    @property
    def provider_model(self) -> str:
        return "test:counting"

    async def complete(self, prompt: str, **_kw: Any) -> str:
        type(self).calls += 1
        return "ok"


class _Allowance:
    """Admits ``n`` completions, then refuses."""

    def __init__(self, n: int) -> None:
        self.left = n
        self.charged = 0

    def charge(self) -> None:
        self.charged += 1
        if self.left <= 0:
            raise ModelCallBudgetSpent("spent")
        self.left -= 1


@pytest.fixture(autouse=True)
def _reset_counter() -> None:
    _Counting.calls = 0


@pytest.mark.asyncio
async def test_no_scope_charges_nothing() -> None:
    assert current_meter() is None
    with override_providers({"semantic_lint": _Counting()}):
        assert await complete("semantic_lint", "p", max_tokens=8) == "ok"
    assert _Counting.calls == 1


@pytest.mark.parametrize("allowance", [0, 1, 3])
@pytest.mark.asyncio
async def test_a_scope_sends_no_more_completions_than_its_meter_admits(allowance: int) -> None:
    meter = _Allowance(allowance)
    sent = 0
    with override_providers({"semantic_lint": _Counting()}), metered_calls(meter):
        for _ in range(5):
            try:
                await complete("semantic_lint", "p", max_tokens=8)
                sent += 1
            except ModelCallBudgetSpent:
                pass
    assert sent == _Counting.calls == allowance
    assert meter.charged == 5


@pytest.mark.asyncio
async def test_a_set_of_requests_is_charged_one_by_one() -> None:
    """A batch-capable path is served one charged call at a time; the refused ones are None."""
    meter = _Allowance(2)
    requests = [CompletionRequest(prompt=f"p{i}") for i in range(4)]
    with override_providers({"semantic_lint": _Counting()}), metered_calls(meter):
        results = await complete_many("semantic_lint", requests, max_tokens=8)
    assert results == ["ok", "ok", None, None]
    assert _Counting.calls == 2


def test_a_refusal_is_a_completion_error() -> None:
    """Call sites degrade on it as on any failed call."""
    from particles.llm import CompletionError

    assert issubclass(ModelCallBudgetSpent, CompletionError)
