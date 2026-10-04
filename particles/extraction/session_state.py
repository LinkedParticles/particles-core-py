# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""An agent session's own working state is never a durable belief.

A coding-agent transcript narrates where the session is standing: which
worktree it was handed, which directory the shell reset to, which branch is
checked out, which file it just edited. Every one of those is true for minutes.
The general extractor read them as ordinary falsifiable claims, so the store
accumulated beliefs like "The session now operates on a fresh worktree at
…/.claude/worktrees/frosty-cray-4f8f06." Two sessions that were each handed a
different worktree then collided as an INCONSISTENCY the operator had to
resolve by hand, and each minted a Subject named after a directory that no
longer exists.

The filter here is deterministic and runs after extraction, on conversational
source types only (``extraction.session_state_source_types``). It is a
post-extraction filter rather than a prompt rule so it holds on every path:
the single-pass, the chunked carry-forward, and the pooled batch path all
return through the same place, and it does not depend on the model complying.

It is deliberately narrow. A candidate is dropped only when it names a
concrete Claude Code worktree path, or when it opens by stating where the
session, the working directory, or the current branch is. A durable claim
*about* worktrees ("Git worktrees in this project are placed under
``.claude/worktrees/<name>``") uses a placeholder and is kept, and so is a
conditional ("When the session runs in a git worktree, …"). A kept candidate
also loses any subject that is a worktree path or a worktree name the source
itself spells as one, so a durable claim cannot mint a directory as a Subject.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from particles.extraction.general import CandidateParticle

#: A concrete Claude Code worktree path: ``.claude/worktrees/<name>`` where the
#: name is a real directory name, not a ``<name>`` / ``{name}`` / ``*``
#: placeholder. The per-session worktree is created for one task and removed
#: after it, so any claim pinned to its path is pinned to a dead location.
_WORKTREE_PATH = re.compile(r"\.claude/worktrees/(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)")

#: Sentence-initial statements of where the session is standing. Anchored at
#: the start so a conditional ("When the session runs in …") is untouched.
_SESSION_LOCUS = re.compile(
    r"^(?:the|this) (?:current )?session(?:'s)? (?:now )?"
    r"(?:operates|runs|is (?:now )?(?:running|operating|working|located|using))"
    r" (?:on|in|at|from|inside)\b",
    re.IGNORECASE,
)

#: "The session now …", anywhere: the adverb is the momentary marker.
_SESSION_NOW = re.compile(r"\b(?:the|this) session now\b", re.IGNORECASE)

#: Sentence-initial statements of the shell's working directory.
_WORKING_DIRECTORY = re.compile(
    r"^(?:the|this) (?:current )?(?:working directory|cwd)"
    r"(?: (?:for|of|used (?:in|during|by)) (?:this|the) session)?"
    r" (?:is|was)\b",
    re.IGNORECASE,
)

#: Statements of which branch is checked out right now.
_CURRENT_BRANCH = re.compile(
    r"^(?:the|this) current (?:git )?branch\b|\bthe current (?:git )?branch is\b",
    re.IGNORECASE,
)

_PATTERNS: tuple[re.Pattern[str], ...] = (
    _SESSION_LOCUS,
    _SESSION_NOW,
    _WORKING_DIRECTORY,
    _CURRENT_BRANCH,
)


@dataclass(frozen=True)
class SessionStateFilterResult:
    """What :func:`drop_session_state` kept, and how much it removed."""

    candidates: list[CandidateParticle]
    dropped: int
    subjects_stripped: int


def worktree_names(text: str) -> frozenset[str]:
    """Return every concrete worktree name ``text`` spells as a worktree path."""
    return frozenset(m.group("name").rstrip(".").lower() for m in _WORKTREE_PATH.finditer(text))


def is_session_state(content: str) -> bool:
    """True when ``content`` describes the session's momentary working state."""
    text = content.strip().lstrip("`*_ ")
    if _WORKTREE_PATH.search(text):
        return True
    return any(p.search(text) for p in _PATTERNS)


def _is_worktree_subject(name: str, names: frozenset[str]) -> bool:
    lowered = name.strip().strip("`'\"").rstrip("/").lower()
    if _WORKTREE_PATH.search(lowered):
        return True
    return lowered in names


def drop_session_state(
    candidates: list[CandidateParticle], source_text: str
) -> SessionStateFilterResult:
    """Drop session-state candidates and strip worktree-named subjects.

    Candidate order is preserved, and the stance indices are
    remapped onto the surviving list: a stance whose target was dropped
    degrades to a plain claim, the same default-safe direction the parser takes
    for an unusable target.
    """
    names = worktree_names(source_text)
    remap: dict[int, int] = {}
    kept: list[CandidateParticle] = []
    for i, candidate in enumerate(candidates):
        if is_session_state(candidate.content):
            continue
        remap[i] = len(kept)
        kept.append(candidate)

    stripped = 0
    for candidate in kept:
        if candidate.stance_target_index is not None:
            target = remap.get(candidate.stance_target_index)
            if target is None:
                candidate.stance_kind = None
                candidate.stance_magnitude = None
            candidate.stance_target_index = target
        if candidate.subjects:
            subjects = [s for s in candidate.subjects if not _is_worktree_subject(s, names)]
            stripped += len(candidate.subjects) - len(subjects)
            candidate.subjects = subjects
    return SessionStateFilterResult(
        candidates=kept, dropped=len(candidates) - len(kept), subjects_stripped=stripped
    )
