# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Which spellings of one file fold into its path Subject.

A transcript can name one file five ways: its path from the repository root,
a partial path, its bare name, an absolute (often worktree) path, and its
dotted module name. The first four each get an ``artifact`` identity,
and the fifth is an unscoped bare-local Subject. This module decides,
over plain values, which of them are the root-relative path's Subject spelled
another way. The shell (``operations/artifact_fold.py``) gathers the values
and performs the merges.

Three rules, each refusing what it cannot settle:

- **absolute** (§3): an absolute path the Surface maps to a root-relative path
  folds into that path's Subject;
- **suffix** (§4): a bare name or partial path folds into the one file of the
  repository's file list that ends in it;
- **module** (§5): a dotted module name folds into the file it names.

The store's own recorded paths are never the oracle for §4 and §5: a store
sees part of a repository, and on the owner's store that oracle put up to 36
of 450 merges on the wrong file.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

#: The ``ExternalRef.namespace`` of every artifact identity.
ARTIFACT_REF_NAMESPACE = "artifact"

#: Last segments a dotted name may not end in and still be read as a module,
#: beside the gate's own file extensions: ``particles.db`` is the store file.
MODULE_EXTENSION_DENYLIST: frozenset[str] = frozenset({"db", "sqlite", "sqlite3", "log"})

#: The ``subject_class`` values a dotted module Subject may carry and still fold.
FOLDABLE_MODULE_CLASSES: frozenset[str | None] = frozenset({None, "artifact:symbol"})

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def artifact_ref_project(ref_id: str) -> str:
    """The namespace key an ``artifact`` ref id is scoped by.

    A ref id is ``<namespace key>/<normalized name>``. The key is opaque to
    the Engine, and a Claude Code key never holds ``/``, so the id
    splits at its first one.
    """
    return ref_id.partition("/")[0]


def artifact_ref_name(ref_id: str) -> str:
    """The normalized name an ``artifact`` ref id carries after its key."""
    return ref_id.partition("/")[2]


class FoldRule(StrEnum):
    """Which rule proposed or refused a fold."""

    ABSOLUTE = "absolute"
    SUFFIX = "suffix"
    MODULE = "module"


class RefusalReason(StrEnum):
    """Why a candidate was left as it is."""

    NO_FILE = "no file in the checkout"
    AMBIGUOUS = "more than one file in the checkout"
    NO_PATH_SUBJECT = "no Subject for the file's path yet"
    FILE_EXTENSION = "ends in a file extension"
    SCOPED_OR_CLASSED = "carries an external ref or another class"
    SEVERAL_NAMESPACES = "the file has a Subject in more than one project"


@dataclass(frozen=True)
class FileSubject:
    """An ``artifact:file`` Subject of the namespace being folded."""

    subject_id: str
    path: str


@dataclass(frozen=True)
class ModuleCandidate:
    """An unscoped Subject whose name may be a dotted module path."""

    subject_id: str
    name: str
    subject_class: str | None
    has_external_refs: bool


@dataclass(frozen=True)
class Fold:
    """One source Subject to merge into one target, and the rule that says so."""

    source_id: str
    source_name: str
    target_id: str
    target_path: str
    rule: FoldRule


@dataclass(frozen=True)
class Refusal:
    """A candidate the planner left alone, with its reason."""

    subject_id: str
    name: str
    rule: FoldRule
    reason: RefusalReason


@dataclass
class FoldPlan:
    """Everything one namespace's fold would do, and everything it declined."""

    folds: list[Fold] = field(default_factory=list)
    refusals: list[Refusal] = field(default_factory=list)


def is_absolute_path(path: str) -> bool:
    """A path the §3 rule applies to: absolute, or rooted at a home directory."""
    return path.startswith(("/", "~"))


def module_files(name: str) -> tuple[str, str] | None:
    """The two files a dotted module name can be, or ``None`` if it is not one.

    A module name is two or more dotted Python identifiers, so a single word
    (``query``, ``tests``) never is.
    """
    segments = name.split(".")
    if len(segments) < 2 or not all(_IDENTIFIER.match(s) for s in segments):
        return None
    base = "/".join(segments)
    return f"{base}.py", f"{base}/__init__.py"


def plan_artifact_fold(
    files: Sequence[FileSubject],
    *,
    relative_paths: Mapping[str, str],
    repo_files: Collection[str] | None,
    modules: Sequence[ModuleCandidate] = (),
    foreign_paths: Collection[str] = (),
    is_file_extension: Callable[[str], bool] = lambda word: False,
) -> FoldPlan:
    """Plan one namespace's fold (§5). Pure.

    Args:
        files: The namespace's ``artifact:file`` Subjects.
        relative_paths: Root-relative path per absolute-path Subject id, as the
            Surface hook answered; an id it could not map is absent.
        repo_files: The checkout's file list, or ``None`` when the operator
            named no checkout. ``None`` disables §4 and §5 (fail closed).
        modules: Unscoped Subjects whose names may be dotted module paths.
        foreign_paths: Paths that have an ``artifact:file`` Subject in some
            other namespace, for §5's single-namespace condition.
        is_file_extension: The gate's file-extension test, applied to a module
            name's last segment beside :data:`MODULE_EXTENSION_DENYLIST`.
    """
    plan = FoldPlan()
    by_path: dict[str, FileSubject] = {}
    for f in files:
        if not is_absolute_path(f.path):
            by_path.setdefault(f.path, f)

    # §3: an absolute path folds into its root-relative twin.
    for f in files:
        if not is_absolute_path(f.path) or f.subject_id not in relative_paths:
            continue
        target = by_path.get(relative_paths[f.subject_id])
        if target is None:
            plan.refusals.append(
                Refusal(f.subject_id, f.path, FoldRule.ABSOLUTE, RefusalReason.NO_PATH_SUBJECT)
            )
            continue
        plan.folds.append(
            Fold(f.subject_id, f.path, target.subject_id, target.path, FoldRule.ABSOLUTE)
        )

    if repo_files is None:
        return plan
    repo = set(repo_files)

    # §4: a bare name or partial path folds into the one file ending in it.
    for f in files:
        if is_absolute_path(f.path) or f.path in repo:
            continue
        hits = [r for r in repo if r.endswith("/" + f.path)]
        if not hits:
            reason = RefusalReason.NO_FILE
        elif len(hits) > 1:
            reason = RefusalReason.AMBIGUOUS
        elif hits[0] not in by_path:
            reason = RefusalReason.NO_PATH_SUBJECT
        else:
            target = by_path[hits[0]]
            plan.folds.append(
                Fold(f.subject_id, f.path, target.subject_id, target.path, FoldRule.SUFFIX)
            )
            continue
        plan.refusals.append(Refusal(f.subject_id, f.path, FoldRule.SUFFIX, reason))

    # §5: a dotted module name folds into the file it names. A name that names
    # no file of this checkout is not a module here and is not reported.
    foreign = set(foreign_paths)
    for m in modules:
        candidates = module_files(m.name)
        if candidates is None:
            continue
        present = [c for c in candidates if c in repo]
        if not present:
            continue
        last = m.name.rsplit(".", 1)[1]
        if len(present) > 1:
            reason = RefusalReason.AMBIGUOUS
        elif last.lower() in MODULE_EXTENSION_DENYLIST or is_file_extension(last):
            reason = RefusalReason.FILE_EXTENSION
        elif m.has_external_refs or m.subject_class not in FOLDABLE_MODULE_CLASSES:
            reason = RefusalReason.SCOPED_OR_CLASSED
        elif present[0] not in by_path:
            reason = RefusalReason.NO_PATH_SUBJECT
        elif present[0] in foreign:
            reason = RefusalReason.SEVERAL_NAMESPACES
        else:
            target = by_path[present[0]]
            plan.folds.append(
                Fold(m.subject_id, m.name, target.subject_id, target.path, FoldRule.MODULE)
            )
            continue
        plan.refusals.append(Refusal(m.subject_id, m.name, FoldRule.MODULE, reason))

    return plan
