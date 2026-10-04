# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""What one extraction exercised: named, content-hashed components.

An extractor version bump re-runs every snapshot stamped with the old version,
though most bumps change one prompt section or one rule. This module is the
record that lets a later bump ask which snapshots reached the changed part.

A **component** is one named piece of what decides an extraction's output: a
section of the prompt (a rule and its JSON-schema field), the chunking path a
source took, the vision channel, the subject gate. Each carries a **digest**,
a short content hash of the text that defines it, so a component whose text
changes gets a new digest under the same name and a component whose text does
not keeps it.

Three parts:

- :class:`PromptComponent` and :func:`assemble_prompt`. An extractor's prompt
  is assembled from a list of components and from nothing else, so a
  component's name can never drift from the text it names.
- :class:`ComponentTally`, opened by the pipeline around one snapshot's
  extraction (:func:`tally_components`) the way ``tally_replies`` is. The
  extractor adds each component it exercises (:func:`record_component`) and
  declares the table of everything it could have exercised
  (:func:`declare_components`). With no tally open both are no-ops.
- :class:`ComponentRecord`, the stored form, written on the snapshot with its
  COMPLETE status.

**Client layer.** Pure values and a ``ContextVar``: no store, no
config read, no I/O.
"""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

#: Version of the stored :class:`ComponentRecord`; additive keys do not bump it.
COMPONENT_RECORD_FORMAT = 1

#: Hex characters kept from the SHA-256 of a component's text. 64 bits is far
#: beyond the few dozen components a table holds; the digest names a text, it
#: is not a security boundary.
DIGEST_LENGTH = 16

#: The digest recorded when one extraction exercised the same component name
#: under two different texts (a config reload mid-pass, or a carried-forward
#: claim made under an older text). It equals no real digest, so the selection
#: rule always treats the component as changed.
MIXED_DIGEST = "mixed"


def component_digest(*parts: str) -> str:
    """The digest of a component defined by ``parts``, in order.

    Parts are separated by a unit separator before hashing, so ``("ab", "c")``
    and ``("a", "bc")`` never collide.
    """
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\x1f")
    return hasher.hexdigest()[:DIGEST_LENGTH]


@dataclass(frozen=True)
class PromptComponent:
    """One named section of an extraction prompt.

    ``rule`` is the text the component adds to the rules block and ``field``
    the text it adds to the JSON-schema block. A prompt is the concatenation
    of every component's ``rule`` followed by every component's ``field``
    (:func:`assemble_prompt`), so the output-schema frame is itself a
    component, placed last: its ``rule`` is the schema's head and its
    ``field`` the schema's tail.
    """

    name: str
    rule: str = ""
    field: str = ""

    @property
    def digest(self) -> str:
        """The content hash of this component's text (rule, then field)."""
        return component_digest(self.rule, self.field)


def assemble_prompt(components: Sequence[PromptComponent]) -> str:
    """The prompt the ``components`` make: every rule, then every field.

    The only way an extractor that records components builds its prompt, so
    every byte of the prompt belongs to exactly one component.
    """
    return "".join(c.rule for c in components) + "".join(c.field for c in components)


@dataclass(frozen=True)
class ComponentTable:
    """Every component one extractor could exercise under the current code and config.

    ``digests`` maps each name to its current digest. ``always`` names the
    components every extraction by this extractor exercises (the core rules,
    the output schema, each config-enabled rule); the rest are exercised only
    by some sources (the tool-turn rule, a chunking path, the vision channel).
    """

    digests: Mapping[str, str]
    always: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        unknown = self.always - set(self.digests)
        if unknown:
            raise ValueError(f"always names components not in the table: {sorted(unknown)}")

    def merged(self, other: ComponentTable) -> ComponentTable:
        """This table and ``other`` together (a pipeline-level component joins an extractor's)."""
        return ComponentTable(
            digests={**self.digests, **other.digests}, always=self.always | other.always
        )


class ComponentRecord(BaseModel):
    """The components one snapshot's extraction exercised, as stored.

    Written on the snapshot in the transaction that marks it COMPLETE, beside
    ``extracted_through``. A snapshot extracted before the record existed has
    none, and the selection rule reads that as "every component exercised".
    """

    format: int = COMPONENT_RECORD_FORMAT
    #: The extractor that ran (its registered id), when the pipeline knew it.
    extractor: str | None = None
    #: Component name to digest, for every component the extraction exercised.
    exercised: dict[str, str] = Field(default_factory=dict)
    #: Every component name the extractor could have exercised at stamp time,
    #: under the config of the day. A name absent here was unknown or disabled
    #: then, so a later table that has it cannot tell whether this snapshot
    #: would exercise it.
    available: list[str] = Field(default_factory=list)
    #: ``False`` when part of the snapshot's claims came from an extraction
    #: whose components are not known (claims carried forward from a snapshot
    #: with no record). The selection rule treats an incomplete record as no
    #: record.
    complete: bool = True

    def merged(self, other: ComponentRecord | None) -> ComponentRecord:
        """This record with ``other``'s folded in (a carried-forward claim's source).

        Exercised names union; a name exercised under two digests becomes
        :data:`MIXED_DIGEST`. ``None`` (a source with no record) makes the
        result incomplete.

        Available names **intersect**. ``available`` says "this extraction could
        have exercised the name and did not", and the merged record covers
        claims from both extractions, so that holds only for a name both knew.
        A union would credit the older extraction with a component that did
        not exist when it ran (a rule added by a version bump between the
        two), and the selection rule would then skip a snapshot whose older
        claims never saw it. A name exercised by either side stays known
        through ``exercised``, which the selection rule reads beside
        ``available``.
        """
        if other is None:
            return self.model_copy(update={"complete": False})
        exercised = dict(self.exercised)
        for name, digest in other.exercised.items():
            if exercised.get(name, digest) != digest:
                exercised[name] = MIXED_DIGEST
            else:
                exercised[name] = digest
        return ComponentRecord(
            extractor=self.extractor,
            exercised=exercised,
            available=sorted(set(self.available) & set(other.available)),
            complete=self.complete and other.complete,
        )


@dataclass
class ComponentTally:
    """The components exercised during one snapshot's extraction.

    Opened by :func:`tally_components` and filled by the extractor (and by the
    pipeline, for the subject gate) as the extraction runs.
    """

    exercised: dict[str, str] = field(default_factory=dict)
    available: set[str] = field(default_factory=set)

    def add(self, name: str, digest: str) -> None:
        """Record ``name`` as exercised under ``digest``."""
        previous = self.exercised.get(name)
        self.exercised[name] = digest if previous in (None, digest) else MIXED_DIGEST
        self.available.add(name)

    def declare(self, names: Iterable[str]) -> None:
        """Record ``names`` as components this extraction could have exercised."""
        self.available.update(names)

    def record(self, extractor: str | None = None) -> ComponentRecord:
        """The stored form of what this tally saw."""
        return ComponentRecord(
            extractor=extractor,
            exercised=dict(sorted(self.exercised.items())),
            available=sorted(self.available | set(self.exercised)),
        )


_TALLY: ContextVar[ComponentTally | None] = ContextVar("particles_component_tally", default=None)


@contextlib.contextmanager
def tally_components() -> Iterator[ComponentTally]:
    """Open a component tally for the extraction run inside the block.

    Every task spawned inside the block sees the same tally, so the pooled and
    per-page paths record into it without new plumbing (the ``tally_replies``
    pattern).
    """
    tally = ComponentTally()
    token = _TALLY.set(tally)
    try:
        yield tally
    finally:
        _TALLY.reset(token)


def record_component(name: str, digest: str) -> None:
    """Record one exercised component on the open tally; a no-op with none open."""
    tally = _TALLY.get()
    if tally is not None:
        tally.add(name, digest)


def record_prompt_components(components: Iterable[PromptComponent]) -> None:
    """Record every component of an assembled prompt as exercised."""
    for component in components:
        record_component(component.name, component.digest)


def declare_components(table: ComponentTable) -> None:
    """Declare the extractor's table on the open tally; a no-op with none open."""
    tally = _TALLY.get()
    if tally is not None:
        tally.declare(table.digests)
