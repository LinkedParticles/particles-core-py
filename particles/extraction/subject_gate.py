# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Non-entity subject gate.

A pure, lexical classifier recognizing candidate *subject names* that are never
real-world entities — the project's own vocabulary, reference / doc-ID codes,
filenames, CLI command strings, and snake_case identifiers — so the Extract
pipeline can drop them before they are promoted to Subjects.

**Precision-first.** A real-world entity must never be suppressed, even at the
cost of letting some pollution through. Ambiguous shapes — bare CamelCase
(``PyTorch`` / ``OpenAI`` are CamelCase products) and lone lowercase / common
words (``index`` / ``reindex``) — are deliberately *out of scope* (
§ Deferred); the residual tail is handled by lint + ``subjects gc``.

**Client layer.** Pure functions over strings and the Client
``CandidateParticle`` dataclass: no store, no config read, no I/O. The Engine
pipeline reads ``get_config().subject_gate`` and passes the knobs in, so these
functions stay trivially unit-testable.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from particles.core.schema import AssertionModality, ParticleType, RelationType
from particles.core.status import Status, StatusReason

if TYPE_CHECKING:
    from particles.extraction.general import CandidateParticle

# Class A — self-vocabulary constants. The project's own ``StrEnum`` member
# values, derived from the enum classes so the set never drifts. These are the
# relation-kind / modality / status / type names a normative or self-referential
# document defines and then has minted *about* (e.g. ``CO_EVIDENTIAL``,
# ``FALSIFIABLE``, ``CONTRADICTS``). Matched case-sensitively against the exact
# all-caps value, so a real entity "Active" / "active" is never gated.
_SELF_VOCABULARY: frozenset[str] = frozenset(
    member.value
    for enum in (RelationType, AssertionModality, ParticleType, Status, StatusReason)
    for member in enum
)

# Class B — reference / identifier codes. A digit-bearing code of uppercase /
# digit segments joined by at least one separator (so ``3M`` / ``M3`` brand
# tokens, which have no separator, are spared). Catches record-id-shaped
# tokens (ADR / PDR / lint-rule / gate forms); spares ``PSUM`` / ``NASA``
# (no digit) and ``iPhone 15`` (lowercase run breaks the all-caps segment).
_REFERENCE_CODE_RE = re.compile(r"^[A-Z0-9]+(?:[ ._/\-][A-Z0-9]+)+$")
_REFERENCE_CODE_MAX_LEN = 32

# Class C — filenames. A token ending in a known code / text file extension.
# (A bare path separator is intentionally *not* a trigger: "TCP/IP" is a real
# entity.) Catches ``roadmap.md``, ``config.yaml``, ``pipeline.py``.
_FILE_EXT_RE = re.compile(
    r"\.(?:md|markdown|rst|txt|py|pyi|ipynb|json|jsonld|ya?ml|toml|ttl|cfg|ini"
    r"|sh|bash|zsh|js|ts|tsx|jsx|rs|go|c|h|cpp|java|rb|html?|csv|tsv|lock|cff)$",
    re.IGNORECASE,
)

# Class E — snake_case code identifiers: lowercase with at least one underscore
# (``subject_store``). Lone lowercase words (no underscore) are out of scope.
_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")

# Class D — a CLI subcommand / flag token after the binary name.
_CLI_TAIL_RE = re.compile(r"^(?:[a-z][a-z0-9-]*|--?[a-z0-9][a-z0-9-]*)$")


def _is_cli_command(name: str, cli_binaries: Sequence[str]) -> bool:
    """Class D — a multi-token string led by a configured CLI binary name.

    Anchored case-sensitively to ``cli_binaries`` (default ``["particles"]``) so
    the project name ``Particles`` (a legitimate subject) is never gated. Catches
    ``particles subjects merge`` / ``pin`` / ``split``.
    """
    tokens = name.split()
    if len(tokens) < 2 or tokens[0] not in cli_binaries:
        return False
    return all(_CLI_TAIL_RE.match(token) for token in tokens[1:])


def classify_non_entity(
    name: str,
    *,
    cli_binaries: Sequence[str] = ("particles",),
    allowlist: Sequence[str] = (),
) -> str | None:
    """Return the matched non-entity token-class name, or ``None`` if ``name``
    may be a real-world entity. Pure: no store, no I/O, no LLM.

    Classes (precision-first): ``self_vocabulary``, ``reference_code``,
    ``filename``, ``cli_command``, ``snake_case``. An ``allowlist`` entry always
    returns ``None`` (operator override).
    """
    candidate = name.strip()
    if not candidate or candidate in allowlist:
        return None
    if candidate in _SELF_VOCABULARY:
        return "self_vocabulary"
    if (
        len(candidate) <= _REFERENCE_CODE_MAX_LEN
        and any(char.isdigit() for char in candidate)
        and _REFERENCE_CODE_RE.match(candidate)
    ):
        return "reference_code"
    if _FILE_EXT_RE.search(candidate):
        return "filename"
    if _is_cli_command(candidate, cli_binaries):
        return "cli_command"
    if _SNAKE_CASE_RE.match(candidate):
        return "snake_case"
    return None


def gate_candidate_subjects(
    candidate: CandidateParticle,
    *,
    cli_binaries: Sequence[str] = ("particles",),
    allowlist: Sequence[str] = (),
) -> list[tuple[str, str]]:
    """Drop non-entity names from ``candidate`` **in place**; return the
    suppressed ``(name, class)`` pairs (for logging).

    Filters the three name-keyed fields together — ``subjects`` plus the
    ``subject_classes`` / ``external_refs`` maps — so the pipeline's positional
    ``zip(candidate.subjects, subject_ids, …)`` stays aligned. A candidate left
    with no subjects becomes a general (subjectless) claim, which is already a
    valid extractor output; the claim is never dropped.
    """
    suppressed: list[tuple[str, str]] = []
    kept: list[str] = []
    for name in candidate.subjects:
        token_class = classify_non_entity(name, cli_binaries=cli_binaries, allowlist=allowlist)
        if token_class is None:
            kept.append(name)
            continue
        suppressed.append((name, token_class))
        candidate.subject_classes.pop(name, None)
        candidate.external_refs.pop(name, None)
    if suppressed:
        candidate.subjects = kept
    return suppressed
