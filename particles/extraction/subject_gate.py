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
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING

from particles.core import claims as _claims
from particles.core import conflict_resolution as _conflict_resolution
from particles.core import contradiction_disclosure as _contradiction_disclosure
from particles.core import observer_scope as _observer_scope
from particles.core import reanchor as _reanchor
from particles.core import schema as _schema
from particles.core import status as _status
from particles.core.schema import AssertionModality, ParticleType, RelationType
from particles.core.scoring import confidence as _confidence
from particles.core.status import Status, StatusReason
from particles.extraction.components import component_digest
from particles.extraction.scope import SCOPE_DOCUMENT_META

if TYPE_CHECKING:
    from particles.extraction.general import CandidateParticle

#: ``properties`` key recording the names the gate classified, as
#: ``[{"name": ..., "class": ..., "disposition": ...}]`` in the order the
#: extractor emitted them. A *record*, never a subject: it says "the extractor
#: named these and the gate withheld or qualified them", so a later change to
#: what the gate does with a class is a backfill over stored particles rather
#: than a re-extraction. An entry with no ``disposition`` predates the
#: dispositions and was suppressed. The key carries the registered
#: ``extraction:`` namespace prefix (and §5).
GATED_SUBJECTS_KEY = "extraction:gated_subjects"

#: The two things the gate may do with a classified name.
SUPPRESS = "suppress"
QUALIFY = "qualify"

# Class A — self-vocabulary constants. The project's own ``StrEnum`` member
# values, derived from the enum classes so the set never drifts. These are the
# relation-kind / modality / status / type names a normative or self-referential
# document defines and then has minted *about* (e.g. ``CO_EVIDENTIAL``,
# ``FALSIFIABLE``, ``CONTRADICTS``). Matched case-sensitively against the exact
# all-caps value, so a real entity "Active" / "active" is never gated.
#
# Beyond those five, every compound (underscore-bearing) value of any ``StrEnum``
# in the Client ``core`` modules, plus the document-scope label. A compound
# constant such as ``DOCUMENT_META``, ``HUMAN_REVIEW`` or ``LOCAL_MARKDOWN`` is
# this implementation's identifier and never a real-world name; the one-word
# values of the other enums (``PDF``, ``CSV``, ``URI``, ``AGENT``) are left out,
# since those are also names of real things.
_CORE_ENUM_MODULES = (
    _schema,
    _status,
    _confidence,
    _claims,
    _conflict_resolution,
    _contradiction_disclosure,
    _observer_scope,
    _reanchor,
)


def _compound_enum_values() -> frozenset[str]:
    values: set[str] = set()
    for module in _CORE_ENUM_MODULES:
        for obj in vars(module).values():
            if (
                isinstance(obj, type)
                and issubclass(obj, StrEnum)
                and obj is not StrEnum
                and obj.__module__ == module.__name__
            ):
                values.update(m.value for m in obj if "_" in m.value and m.value.isupper())
    return frozenset(values)


_SELF_VOCABULARY: frozenset[str] = frozenset(
    {
        member.value
        for enum in (RelationType, AssertionModality, ParticleType, Status, StatusReason)
        for member in enum
    }
    | _compound_enum_values()
    | {SCOPE_DOCUMENT_META}
)

# Class B — reference / identifier codes. A digit-bearing code of uppercase /
# digit segments joined by at least one separator (so ``3M`` / ``M3`` brand
# tokens, which have no separator, are spared). Catches record-id-shaped
# tokens (ADR / PDR / lint-rule / gate forms); spares ``PSUM`` / ``NASA``
# (no digit) and ``iPhone 15`` (lowercase run breaks the all-caps segment).
_REFERENCE_CODE_RE = re.compile(r"^[A-Z0-9]+(?:[ ._/\-][A-Z0-9]+)+$")
_REFERENCE_CODE_MAX_LEN = 32

# Class B0 — version numbers and decimals: digit runs joined by separators
# (``3.11``, ``1.65.0``, ``0.90``). The reference-code shape below matches
# them too, and they are never a record, so they are split off
# first and are never qualified.
_VERSION_NUMBER_RE = re.compile(r"^\d+(?:[ ._/\-]\d+)+$")

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


#: The gate's name in an extraction's component record.
GATE_COMPONENT = "gate.subject"


def gate_digest(
    *,
    cli_binaries: Sequence[str],
    allowlist: Sequence[str],
    dispositions: Mapping[str, str] | None,
) -> str:
    """The component digest of the gate as configured.

    Covers what decides the gate's verdict on a name: the class tables and
    patterns above, and the three knobs that change a verdict or a disposition.
    ``exempt_source_types`` is left out, because it decides whether the gate
    runs on a source, which the component record already says by recording
    the gate or not.
    """
    return component_digest(
        "\n".join(sorted(_SELF_VOCABULARY)),
        _VERSION_NUMBER_RE.pattern,
        _REFERENCE_CODE_RE.pattern,
        str(_REFERENCE_CODE_MAX_LEN),
        _FILE_EXT_RE.pattern,
        _SNAKE_CASE_RE.pattern,
        _CLI_TAIL_RE.pattern,
        "cli_binaries=" + ",".join(cli_binaries),
        "allowlist=" + ",".join(sorted(allowlist)),
        "dispositions=" + ",".join(f"{k}:{v}" for k, v in sorted((dispositions or {}).items())),
    )


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

    Classes (precision-first): ``self_vocabulary``, ``version_number``,
    ``reference_code``, ``filename``, ``cli_command``, ``snake_case``. An
    ``allowlist`` entry always returns ``None`` (operator override).
    """
    candidate = name.strip()
    if not candidate or candidate in allowlist:
        return None
    if candidate in _SELF_VOCABULARY:
        return "self_vocabulary"
    if _VERSION_NUMBER_RE.match(candidate):
        return "version_number"
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


def disposition_for(
    token_class: str,
    dispositions: Mapping[str, str] | None,
    namespace_key: str | None,
) -> str:
    """What the gate does with a name of ``token_class``.

    ``qualify`` only when the class is configured to qualify **and** a
    namespace key scopes the name's identity; a class missing from the map, and
    any name with no key, is suppressed (the fail-closed rule).
    """
    if namespace_key and (dispositions or {}).get(token_class) == QUALIFY:
        return QUALIFY
    return SUPPRESS


def is_qualifiable(name: str, token_class: str) -> bool:
    """Whether a classified name has a shape worth an identity.

    A "filename" with whitespace in it is a command line ending in a path
    (``uv run python scripts/cut_changelog.py``), which the precision check
    found attached as a subject; it stays suppressed.
    """
    return not (token_class == "filename" and any(ch.isspace() for ch in name.strip()))


def gate_candidate_subjects(
    candidate: CandidateParticle,
    *,
    cli_binaries: Sequence[str] = ("particles",),
    allowlist: Sequence[str] = (),
    dispositions: Mapping[str, str] | None = None,
    namespace_key: str | None = None,
) -> list[tuple[str, str]]:
    """Suppress or qualify non-entity names on ``candidate`` **in place**; return
    the suppressed ``(name, class)`` pairs (for logging).

    A suppressed name is dropped from the three name-keyed fields together —
    ``subjects`` plus the ``subject_classes`` / ``external_refs`` maps — so the
    pipeline's positional ``zip(candidate.subjects, subject_ids, …)`` stays
    aligned. A candidate left with no subjects becomes a general (subjectless)
    claim; the claim is never dropped.

    A qualified name stays in ``subjects`` and is recorded in
    ``candidate.qualified_subjects`` with its class, which the pipeline hands
    to the resolver with ``namespace_key``. With the default
    ``dispositions=None`` every class is suppressed, which is the original
    gate's behaviour exactly.

    Every classified name is also recorded on the candidate's ``properties``
    under :data:`GATED_SUBJECTS_KEY` with its disposition, so the stored
    particle keeps what the extractor named whatever the gate did with it.
    """
    suppressed: list[tuple[str, str]] = []
    record: list[dict[str, str]] = []
    kept: list[str] = []
    for name in candidate.subjects:
        token_class = classify_non_entity(name, cli_binaries=cli_binaries, allowlist=allowlist)
        if token_class is None:
            kept.append(name)
            continue
        disposition = (
            disposition_for(token_class, dispositions, namespace_key)
            if is_qualifiable(name, token_class)
            else SUPPRESS
        )
        record.append({"name": name, "class": token_class, "disposition": disposition})
        if disposition == QUALIFY:
            kept.append(name)
            candidate.qualified_subjects[name] = token_class
            continue
        suppressed.append((name, token_class))
        candidate.subject_classes.pop(name, None)
        candidate.external_refs.pop(name, None)
    if record:
        candidate.subjects = kept
        # A fresh dict: an extractor may share one ``properties`` mapping
        # between candidates, and the record belongs to this candidate alone.
        properties = dict(candidate.properties or {})
        properties[GATED_SUBJECTS_KEY] = record
        candidate.properties = properties
    return suppressed
