# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The episode reading: is a transcript claim true only of its session's moment.

Two writers ask the question, each under its own stamp so admission can tell
them apart:

* **The extractor.** On a session transcript the general extractor's prompt
  gains the ``prompt.episode`` component
  (:func:`particles.extraction.general.episode_prompt_component`), and each
  candidate carries the model's answer in its ``episode`` field. The stamp is
  the component's ``name@digest``.
* **The regeneration pass** (``particles episode``) reads stored claims,
  several of one source at a time, each beside its source passage
  (:func:`read_episodes`). It has its own prompt, and its stamp is
  ``regen.episode@<digest of that prompt>`` (:func:`regeneration_stamp`).

Both share :data:`EPISODE_DEFINITION`, so the two prompts ask one question,
and neither writes ``ABSTAIN``: a writer with no answer writes nothing.

Client layer: no store, corpus, db or ingest import.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from particles.core.episode import REGEN_WRITER, reading_verdict
from particles.core.rung_ladder import EpisodeVerdict
from particles.extraction.components import component_digest
from particles.llm.breaker import PROBE_MAX_TOKENS

log = logging.getLogger(__name__)

__all__ = [
    "EPISODE_DEFINITION",
    "EpisodeItem",
    "EpisodeReply",
    "parse_episode_reply",
    "read_episodes",
    "regen_instructions",
    "regeneration_stamp",
]

#: The question both writers ask, word for word.
EPISODE_DEFINITION = """\
  "session_bound" when the claim reports the state of something at this
  session's moment, or an act performed in this session, so that a reader of
  the store after the session would not ask for it as a standing fact: "PR #447
  is open", "the branch is 7 commits behind main", "the version was bumped from
  1.114.1 to 1.114.2", "was minted", "the lint reports 0 warnings on
  the rebased tree". "standing" when the claim holds beyond the session, or
  records history a reader would ask for: "backgrounded commits die at the GPG
  signing step", "the API rejects budget_tokens on this model", "the owner
  prefers one-command releases", "is titled 'symbol-aware docstring
  extraction'". When unsure, "standing"."""

_REGEN_FRAME = """You read claims extracted from the transcript of one working
session (a person and a coding agent). Each claim is given with the passage of
the transcript it came from. For each claim, decide only whether it is true
only of the session's moment; do not judge whether it is true. Answer
"""

_REGEN_OUTPUT = """

The fenced block holds the claims, each headed "Claim N:" with its passage
under it. Reply with one JSON array and nothing else, one object per claim in
order: [{"claim": 1, "episode": "session_bound"}, {"claim": 2, "episode":
"standing"}]. Use only "session_bound" or "standing"."""


def regen_instructions() -> str:
    """The regeneration reader's instructions, as its system turn carries them."""
    return _REGEN_FRAME + EPISODE_DEFINITION + _REGEN_OUTPUT


def regeneration_stamp() -> str:
    """The stamp a regenerated reading carries: ``regen.episode@<digest>``."""
    return f"{REGEN_WRITER}@{component_digest(regen_instructions())[:12]}"


@dataclass(frozen=True)
class EpisodeItem:
    """One claim to read: its text and its source passage (may be empty)."""

    content: str
    passage: str = ""


@dataclass(frozen=True)
class EpisodeReply:
    """One call's readings, in item order (``None`` where the reply gave none)."""

    verdicts: tuple[EpisodeVerdict | None, ...]
    provider_model: str
    stamp: str


def parse_episode_reply(raw: str, count: int) -> tuple[EpisodeVerdict | None, ...]:
    """The verdict the reply gives each of ``count`` claims; ``None`` where it gives none."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    out: list[EpisodeVerdict | None] = [None] * count
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return tuple(out)
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return tuple(out)
    if not isinstance(payload, list):
        return tuple(out)
    for position, obj in enumerate(payload):
        if not isinstance(obj, dict):
            continue
        index = obj.get("claim")
        i = index - 1 if isinstance(index, int) and not isinstance(index, bool) else position
        if 0 <= i < count and out[i] is None:
            out[i] = reading_verdict(obj.get("episode"))
    return tuple(out)


def _block(items: Sequence[EpisodeItem], passage_chars: int) -> str:
    parts: list[str] = []
    for n, item in enumerate(items, start=1):
        passage = item.passage.strip()
        if passage_chars > 0 and len(passage) > passage_chars:
            passage = passage[:passage_chars].rstrip() + " …"
        if passage_chars <= 0:
            passage = ""
        lines = [f"Claim {n}: {item.content}"]
        if passage:
            lines.append(f"Passage:\n{passage}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


async def read_episodes(items: Sequence[EpisodeItem], *, passage_chars: int) -> EpisodeReply | None:
    """Read several claims of one session in one completion.

    One call on the ``extraction`` purpose. The claims and passages came from
    an untrusted source, so they ride the user turn inside the data fence.
    A failed call returns ``None`` and leaves every claim as it was.
    """
    from particles.llm import complete_with_provider_model, fenced_prompt

    if not items:
        return None
    system, user = fenced_prompt(regen_instructions(), _block(items, passage_chars), label="claims")
    try:
        raw, provider_model = await complete_with_provider_model(
            "extraction", user, max_tokens=PROBE_MAX_TOKENS, system=system
        )
    except Exception as exc:
        log.warning("Episode reader call failed: %s", exc)
        return None
    return EpisodeReply(
        verdicts=parse_episode_reply(raw, len(items)),
        provider_model=provider_model,
        stamp=regeneration_stamp(),
    )
