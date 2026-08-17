"""Account-level LLM failure classification — the shared predicate.

An *account-level* failure fails **every** call until an operator fixes the
account: a bad or missing key (401), no permission (403), or an out-of-credits
billing error (a 400 whose message names a credit-balance / billing problem).
A *per-call* failure — one malformed prompt, a transient 429 / 5xx, a network
blip — does not.

The distinction has two consumers with different needs, which is why the
predicate lives here (the Client-layer ``llm`` package) rather than in either
of them:

* ``particles.operations._llm`` (Engine) trips the circuit breaker on
  it, so the semantic seam stops probing a dead API.
* ``particles.extraction`` (Client) raises :class:`AccountLevelLLMError` on it,
  so a bulk run stops instead of repeating the same failure per snapshot.

Before this module the predicate existed only in the first, so extraction —
which cannot import Engine code — had no way to tell "your key has no credit"
from "this one page confused the model", and ``extract --all-pending`` walked
every pending snapshot re-issuing a request that could not succeed.
"""

from __future__ import annotations


class AccountLevelLLMError(RuntimeError):
    """An LLM failure that will recur until the operator fixes the account.

    Raised by the extraction seam so a bulk caller aborts rather than repeating
    a doomed call per snapshot. Deliberately **not** caught as a transient
    extraction error: the snapshot's content is fine, so the pipeline's
    interrupt handler resets it ``IN_PROGRESS → PENDING`` and re-raises, leaving
    the work queued for a retry once the account is fixed.
    """

    def __init__(self, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.cause = cause


def is_account_level_failure(exc: BaseException) -> bool:
    """True if ``exc`` is an account-level LLM failure that will recur.

    Duck-typed on ``status_code`` plus the message, so it is not tied to one
    provider's SDK — a local endpoint's 401 / 403 trips it too. A per-call 400
    (one malformed prompt) and a transient 429 / 5xx deliberately do not.
    """
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return True
    msg = str(exc).lower()
    return status == 400 and ("credit balance" in msg or "billing" in msg)
