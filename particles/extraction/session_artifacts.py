# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""A session record names the artifact it is about, never a description of it.

A coding-agent transcript talks about "the pull request", "the commit", "the
branch": in the session each description has one referent, but the claim is
stored alone and read beside the records of hundreds of other sessions, where
"the pull request" is every pull request at once. Two true records of two
events then read as one claim with two values. In the owner's labels of the
contradiction census, 79 of the 98 compatible pairs the probe
flagged were "The pull request was created at <URL>" against the same sentence
with another URL.

Two deterministic repairs run after extraction, on the same conversational
source types as the session-state filter (``extraction.session_state_source_types``):

* **A pull-request URL names the pull request.** A claim that opens with a
  description of a pull request and carries exactly one pull-request URL is
  rewritten to open with the name the URL gives, ``acme/widgets#326``, and
  that name joins its subjects. The rewrite adds only what the claim's own URL
  states, so the claim says the same thing with its referent explicit.
* **A bare description is never a Subject.** "the pull request", "the
  commit", "the user", "the session" and the like name a different thing in
  every session, so a Subject minted from one pools the records of every
  session under one node. Such subjects are stripped.

Both are narrow on purpose: a claim that already names its pull request ("PR
#218 is …"), or carries two different ones, is left alone, and only a fixed
list of session-artifact nouns counts as a description. The prompt rule beside
``prompt.session_validity`` (``prompt.session_artifacts``) asks the model for
the same naming where no regex can settle it: a commit, a branch, a version.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from particles.core.schema import ClaimTerm, TermKind

if TYPE_CHECKING:
    from particles.extraction.general import CandidateParticle

#: A GitHub pull-request URL; the comment fragment, if any, is not part of the name.
_PR_URL = re.compile(
    r"https?://github\.com/(?P<owner>[A-Za-z0-9][A-Za-z0-9-]*)/(?P<repo>[A-Za-z0-9._-]+?)"
    r"/pull/(?P<number>\d+)\b"
)

#: A claim-initial description of a pull request: "The pull request", "A PR",
#: "The created pull request", "The PR URL". Not when a number follows
#: ("The PR #326 …"): that claim already names its pull request.
_PR_DESCRIPTION = re.compile(
    r"^(?:the|a|an)\s+(?:(?:new|newly created|created|opened|resulting)\s+)?"
    r"(?:pull request|PR)(?P<url>(?:'s)?\s+(?:URL|link))?"
    r"(?=\s)(?!\s+(?:#|no\.?\s*|number\s+)?\d)",
    re.IGNORECASE,
)

#: Session-artifact nouns whose bare definite description is never a Subject.
_ARTIFACT_NOUNS = (
    "pull request|pr|commit|branch|worktree|version|release|session|assistant|"
    "agent|user|owner|operator|repository|repo|project|codebase|adr|pdr|chip|"
    "brief|change|fix|patch|test suite|tests|file|issue|task|tag"
)
_DESCRIPTION_SUBJECT = re.compile(
    r"^(?:the|this|that)\s+(?:(?:current|new|created|opened|merged|latest|resulting|same)\s+)?"
    rf"(?:{_ARTIFACT_NOUNS})s?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SessionArtifactResult:
    """How many claims :func:`name_session_artifacts` rewrote and subjects it stripped."""

    named: int
    subjects_stripped: int


def is_description_subject(name: str) -> bool:
    """True when ``name`` is a bare definite description of a session artifact."""
    return bool(_DESCRIPTION_SUBJECT.match(name.strip().strip("`'\"").rstrip(".").strip()))


def pull_request_name(content: str) -> str | None:
    """The ``owner/repo#number`` of the one pull request ``content``'s URLs name, else ``None``."""
    names = {
        f"{m.group('owner')}/{m.group('repo')}#{m.group('number')}"
        for m in _PR_URL.finditer(content)
    }
    if len(names) != 1:
        return None
    return names.pop()


def name_pull_request(content: str) -> tuple[str, str] | None:
    """``(rewritten content, pull-request name)``, or ``None`` when the rule does not apply.

    Applies when ``content`` opens with a description of a pull request and its
    URLs name exactly one: "The pull request was created at
    https://github.com/acme/widgets/pull/326." becomes "Pull request
    acme/widgets#326 was created at https://github.com/acme/widgets/pull/326.".
    """
    match = _PR_DESCRIPTION.match(content)
    if match is None:
        return None
    name = pull_request_name(content)
    if name is None:
        return None
    head = f"The URL of pull request {name}" if match.group("url") else f"Pull request {name}"
    return head + content[match.end() :], name


def name_session_artifacts(candidates: list[CandidateParticle]) -> SessionArtifactResult:
    """Name each described pull request and strip description subjects, in place.

    A rewritten claim gains the pull request's name as a subject, and a
    structured-claim subject term that was the description is replaced with
    it, so the triple keeps pointing at one of the claim's subjects.
    """
    named = 0
    stripped = 0
    for candidate in candidates:
        rewrite = name_pull_request(candidate.content)
        subjects = [s for s in candidate.subjects if not is_description_subject(s)]
        stripped += len(candidate.subjects) - len(subjects)
        if rewrite is not None:
            candidate.content, name = rewrite
            if name.lower() not in {s.lower() for s in subjects}:
                subjects.append(name)
            triple = candidate.structured_claim
            if triple is not None and is_description_subject(triple.subject.value):
                candidate.structured_claim = triple.model_copy(
                    update={"subject": ClaimTerm(kind=TermKind.TOKEN, value=name)}
                )
            named += 1
        candidate.subjects = subjects
    return SessionArtifactResult(named=named, subjects_stripped=stripped)
