# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Markdown Bridge renderer — renders particles and lint reports as Obsidian-compatible
callout blocks (§C.5).

Callout format:
  > [!type] Title
  > Body line 1
  > Body line 2

Also exposes :func:`atomic_write_text`, a shared utility every exporter
should use when writing into operator-visible directories.

Layering: this is a **Client-layer rendering utility** —
pure functions over Pydantic models plus filesystem-safe write/slug helpers,
with no store or graph. It lives in ``particles.render`` so both the Engine
*reasoning* layer (``operations`` — digest, inbox, lint) and the Engine
*output* layer (``exporters``) can depend on it **downward**. It used to live
under ``particles.exporters``, which made ``operations`` import ``exporters``
(an upward dependency / package cycle). ``particles.exporters.markdown`` is now
a back-compat shim re-exporting this module.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from particles.core.schema import (
    ContestedBadge,
    ContestednessReading,
    LintFinding,
    LintReport,
    ObserverScopeNote,
    Particle,
    RelationType,
    StancePosition,
    Subject,
)
from particles.core.status import Status
from particles.extraction.polarity import is_non_asserted

# ---------------------------------------------------------------------------
# Projected-region sentinels (hoisted here).
#
# The sentinel pair delimits a machine-owned region spliced into a file other
# writers own (README architecture, MEMORY.md memory-index). The format
# strings and parse regexes live in this Client-layer module — not in
# ``operations/projection`` — so both the Engine-side renderer
# (``operations.projection.project``) and the harvest-side strip
# (``api/cli/_claude_code.filter_memory_file_for_deposit``) import them
# DOWNWARD, never sideways (belt 1).
# ---------------------------------------------------------------------------

#: Format strings for the block-splice sentinels. The BEGIN
#: sentinel names the region *and* the manifest that drives it, so a reader of
#: the raw Markdown can find the source; the END sentinel names only the
#: region. HTML comments so the boundary is invisible in rendered Markdown.
PROJECTED_BEGIN_TMPL = "<!-- BEGIN PROJECTED: {region} (manifest: {manifest}) -->"
PROJECTED_END_TMPL = "<!-- END PROJECTED: {region} -->"

#: Matches an open sentinel for a *named* region, capturing the manifest path
#: so a re-splice can preserve it. Tolerant of surrounding whitespace inside
#: the comment so a hand-tidied file still parses.
PROJECTED_BEGIN_RE_TMPL = (
    r"<!--\s*BEGIN PROJECTED:\s*{region}\s*\(manifest:\s*(?P<manifest>[^)]*?)\s*\)\s*-->"
)
PROJECTED_END_RE_TMPL = r"<!--\s*END PROJECTED:\s*{region}\s*-->"

#: Matches *any* BEGIN sentinel, capturing the region name — the harvest-side
#: strip must find every projected region without knowing their names.
_ANY_PROJECTED_BEGIN_RE = re.compile(
    r"<!--\s*BEGIN PROJECTED:\s*(?P<region>[^\s(]+)\s*\(manifest:\s*(?P<manifest>[^)]*?)\s*\)\s*-->"
)


@dataclass(frozen=True)
class ProjectedRegion:
    """One ``BEGIN/END PROJECTED`` region located in a host file's text."""

    region: str
    """The region name from the BEGIN sentinel (e.g. ``memory-index``)."""
    manifest: str
    """The manifest attribution captured from the BEGIN sentinel."""
    body: str
    """The text strictly between the sentinel lines (own newlines trimmed)."""
    start: int
    """Offset of the BEGIN sentinel's first character in the host text."""
    end: int
    """Offset one past the END sentinel's last character in the host text."""


def find_projected_regions(text: str) -> list[ProjectedRegion]:
    """Locate every well-formed ``BEGIN/END PROJECTED`` region in ``text``.

    A BEGIN sentinel whose region has no matching END *after* it is skipped —
    a structurally damaged pair is never treated as a region (the caller's
    splice path raises ``SpliceError`` for that; the harvest-side strip leaves
    the damaged text untouched rather than guess at its extent).
    """
    regions: list[ProjectedRegion] = []
    for begin in _ANY_PROJECTED_BEGIN_RE.finditer(text):
        name = begin.group("region")
        end_re = re.compile(PROJECTED_END_RE_TMPL.format(region=re.escape(name)))
        end = end_re.search(text, begin.end())
        if end is None:
            continue
        regions.append(
            ProjectedRegion(
                region=name,
                manifest=begin.group("manifest"),
                body=text[begin.end() : end.start()].strip("\n"),
                start=begin.start(),
                end=end.end(),
            )
        )
    return regions


def strip_projected_regions_for_deposit(text: str, snapshot_bodies: Mapping[str, str]) -> str:
    """Remove PRISTINE projected regions from a memory file before deposit (§2/§6).

    The corpus must never contain the store's own rendered output (belt 1 of
    the round-trip contract), so a region whose body still matches its render
    snapshot (``snapshot_bodies[region]``, newline-normalised) is removed
    wholesale — sentinels and body. A **dirtied** region (body ≠ snapshot, or
    no snapshot known) is, by definition, human/agent signal: its *body* is
    kept — deposited as ordinarily-authored input for the §6.6 ladder — and
    only the sentinel comment lines are dropped. Structurally damaged pairs
    (BEGIN without END) are left untouched.
    """
    regions = find_projected_regions(text)
    if not regions:
        return text
    out: list[str] = []
    cursor = 0
    for region in regions:
        out.append(text[cursor : region.start])
        snapshot = snapshot_bodies.get(region.region)
        pristine = snapshot is not None and region.body.strip("\n") == snapshot.strip("\n")
        if not pristine and region.body:
            out.append(region.body + "\n")
        cursor = region.end
    out.append(text[cursor:])
    stripped = "".join(out)
    # Collapse the blank-line runs a region removal leaves behind.
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped


def insert_projected_region_at_top(text: str, region: str, manifest: str) -> str:
    """Insert an empty sentinel pair at the top of ``text``, preserving all content below.

    The init-time seeding: the region sits at the top of the file
    so the ranked view always lands inside the harness's load window. Existing
    content (byte-for-byte) follows after one blank line. The caller is
    responsible for not calling this when the region already exists.
    """
    pair = (
        PROJECTED_BEGIN_TMPL.format(region=region, manifest=manifest)
        + "\n"
        + PROJECTED_END_TMPL.format(region=region)
        + "\n"
    )
    remainder = text.lstrip("\n")
    return pair + ("\n" + remainder if remainder else "")


def exclude_non_asserted(
    particles: list[Particle], options: Mapping[str, object]
) -> list[Particle]:
    """Drop non-asserted particles (cap. 1) from a one-way export set.

    Returns ``particles`` unchanged when ``options['include_non_asserted']`` is
    truthy; otherwise filters out ``DECLINED`` / ``HYPOTHETICAL`` particles — a
    document's rejected / superseded / deferred / counterfactual prose, which is
    off the default factual surface. Shared by the projection exporters
    (Obsidian / Logseq / wiki / Anki / JSONL) so the ``--include-non-asserted``
    opt-in behaves identically across them. The round-trippable *interchange*
    export does NOT call this — it must preserve every particle (polarity rides
    on ``properties`` and survives the round-trip).
    """
    if bool(options.get("include_non_asserted", False)):
        return particles
    return [p for p in particles if not is_non_asserted(p.properties)]


# ---------------------------------------------------------------------------
# Filesystem-safe subject slug — shared across every exporter that writes
# one file per Subject so that the GitHub-author / Reddit-user nesting
# is consistent across Obsidian / Wiki / future Logseq exporters.
# ---------------------------------------------------------------------------

_INVALID_FS_CHARS = re.compile(r'[\\/:*?"<>|]+')
_WHITESPACE_RUN = re.compile(r"\s+")

# Windows reserved device names — illegal as a filename stem regardless of
# extension (``CON``, ``CON.md``, … all fail). Subject canonical names are
# LLM-extracted from untrusted documents, so a hostile or accidental "NUL"
# could otherwise become an unwritable export file on a Windows vault.
_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def sanitize_filename(name: str) -> str:
    """Filesystem-safe filename (no extension), preserving dashes.

    Invalid filesystem chars are replaced with a dash; runs of whitespace
    are collapsed to a single space. Single dashes (e.g. date ranges
    like ``1948-1950``) are preserved so that ``[[wiki-links]]`` resolve
    to the actual filename.

    Subject canonical names are LLM-extracted from untrusted documents, so
    this also hardens three robustness edges (no directory escape is
    possible — path separators are already neutralised above — these are
    defence-in-depth): a ``..`` path token is replaced with a dash, leading
    dots are stripped (no hidden ``.foo`` / bare ``.`` / ``..`` files), and
    a Windows reserved device-name stem (``CON``, ``NUL``, ``COM1`` …) is
    prefixed with a dash so the export file stays writable on every OS.
    """
    cleaned = _INVALID_FS_CHARS.sub("-", name)
    cleaned = _WHITESPACE_RUN.sub(" ", cleaned).strip()
    cleaned = cleaned.replace("..", "-").lstrip(".")
    if cleaned.split(".", 1)[0].upper() in _RESERVED_NAMES:
        cleaned = f"-{cleaned}"
    return cleaned or "unnamed"


def _sanitize_shard_tail(remainder: str) -> str:
    """Sanitise the user-controlled tail of a sharded subject slug, per segment.

    The ``reddit.com/`` / ``github.com/`` shards prepend a fixed prefix to an
    LLM-extracted remainder (``u/spez`` → ``reddit.com/u/spez``). That
    remainder is **untrusted** — a Subject ``canonical_name`` is extracted from
    deposited documents and validated only for ``min_length=1`` — so each
    ``/``-separated segment is routed through :func:`sanitize_filename`
    (collapsing ``..`` to a dash and dashing out any embedded separators)
    before being re-joined. Legitimate names keep their nesting
    (``u/spez`` → ``u/spez``, ``login`` → ``login``); a poisoned
    ``u/../../../evil`` is flattened to ``u/-/-/-/evil`` and can no longer
    escape the export directory. Empty segments (leading / doubled slashes)
    are dropped; an all-empty remainder degrades to ``unnamed``.
    """
    segments = [sanitize_filename(seg) for seg in remainder.split("/") if seg]
    return "/".join(segments) or "unnamed"


def subject_slug(canonical_name: str) -> str:
    """Map a canonical subject name to its vault path (no extension).

    Reddit users (``u/…``) and subreddits (``r/…``) nest under
    ``reddit.com/``. GitHub authors (``github:login``) nest under
    ``github.com/``. All other names go through :func:`sanitize_filename`
    for filesystem safety.

    The ``reddit.com/`` / ``github.com/`` prefix is a fixed shard; the
    user-controlled remainder is sanitised per path-segment by
    :func:`_sanitize_shard_tail` so a poisoned ``canonical_name`` such as
    ``github:../../../../etc/cron.d/x`` cannot ``..``-escape the export
    directory (those segments previously bypassed :func:`sanitize_filename`
    entirely). The disambiguation call site re-slugs ``"{name} ({qualifier})"``
    through this same function, so it inherits the same protection.

    This is the **single source of truth** for subject → path mapping
    across exporters. Every exporter that writes one file per Subject
    must use this helper so an operator who exports the same store via
    two exporters gets identical paths (no ``github.com/foo.md`` in one
    vault and ``github-foo.md`` in another).
    """
    if canonical_name.startswith(("u/", "r/")):
        return f"reddit.com/{_sanitize_shard_tail(canonical_name)}"
    if canonical_name.startswith("github:"):
        return f"github.com/{_sanitize_shard_tail(canonical_name[len('github:') :])}"
    return sanitize_filename(canonical_name)


# ---------------------------------------------------------------------------
# Subject-name disambiguation — shared across every Markdown
# exporter that writes one file per Subject. Two distinct Subjects can
# legitimately share a canonical_name (e.g. "Prometheus" the software vs
# the Greek Titan, two distinct Wikidata QIDs). Without disambiguation
# they collide on one filename and the last writer silently overwrites
# the rest. The convention: when ≥2 Subjects share a base slug, each gets
# a parenthetical qualifier appended to its *display name*, Wikipedia
# style ("Prometheus (software)"). Exporters feed the disambiguated
# display name through their own slug / link machinery, so the same
# convention works for filename-keyed links (Obsidian, Wiki) and
# name-keyed links (Logseq). A `(disambiguation)` note is emitted per
# collision group.
# ---------------------------------------------------------------------------


def _class_qualifier(subject: Subject) -> str:
    """Tier 1: the subject's ontology class, namespace prefix stripped."""
    if not subject.subject_class:
        return ""
    return subject.subject_class.split(":", 1)[-1].strip()


def _desc_qualifier(subject: Subject) -> str:
    """Tier 2: a short readable qualifier distilled from the description.

    Wikidata glosses are category-shaped, so the most distinctive token is
    usually the trailing noun of the pre-comma head:
    "event monitoring and alerting *software*" → ``software``,
    "genus of the order *Lepidoptera*" → ``Lepidoptera``,
    "*Titan*, culture hero, …" → ``Titan``. Prefer that single trailing
    token when it is a clean word (alphabetic, length >= 3); otherwise
    fall back to the leading phrase (pre-comma, capped to 6 words / 48
    chars) so noisy glosses like "American politician (born 1947)" still
    yield a deterministic, filename-safe qualifier.

    The cascade's distinctness guard in :func:`_choose_qualifiers` still
    applies: if two members of a collision group distil to the same token
    (e.g. both "… software"), this tier is rejected and the next tier
    (external-id) is tried.
    """
    if not subject.description:
        return ""
    phrase = _WHITESPACE_RUN.sub(" ", subject.description.split(",", 1)[0]).strip()
    if not phrase:
        return ""
    words = phrase.split(" ")
    last = words[-1]
    if last.isalpha() and len(last) >= 3:
        return last
    capped = " ".join(words[:6])
    if len(capped) > 48:
        capped = capped[:48].rstrip()
    return capped


def _extid_qualifier(subject: Subject) -> str:
    """Tier 3: the highest-confidence external reference (e.g. "wikidata Q83160")."""
    if not subject.external_ids:
        return ""
    ref = sorted(
        subject.external_ids,
        key=lambda r: (-(r.confidence or 0.0), r.namespace, r.id),
    )[0]
    return f"{ref.namespace} {ref.id}"


def _id_qualifier(subject: Subject) -> str:
    """Tier 4: a short subject-id prefix — always unique, the terminal fallback."""
    return f"id {subject.id[:8]}"


_QUALIFIER_TIERS = (_class_qualifier, _desc_qualifier, _extid_qualifier, _id_qualifier)


def _choose_qualifiers(members: list[Subject]) -> dict[str, str]:
    """Pick a qualifier per subject from the first cascade tier that makes
    the whole colliding group distinct."""
    for tier in _QUALIFIER_TIERS:
        quals = {s.id: tier(s) for s in members}
        vals = list(quals.values())
        if not all(vals) or len(set(vals)) != len(vals):
            continue
        # The chosen qualifiers must also yield distinct slugs (sanitize
        # can collapse otherwise-distinct strings).
        slugs = {subject_slug(f"{s.canonical_name} ({quals[s.id]})").lower() for s in members}
        if len(slugs) == len(members):
            return quals
    # _id_qualifier is guaranteed unique + slug-safe, so this is unreachable
    # in practice; kept as an explicit terminal for total correctness.
    return {s.id: _id_qualifier(s) for s in members}


@dataclass(frozen=True)
class DisambiguationGroup:
    """One set of Subjects whose canonical names collided on a base slug."""

    base_name: str
    """Display name for the bare collision (the shared canonical_name)."""
    base_slug: str
    """The colliding base slug, e.g. ``Prometheus``."""
    member_ids: tuple[str, ...]
    """Subject ids in the group, sorted for deterministic output."""


@dataclass(frozen=True)
class SubjectNaming:
    """Per-export disambiguation map.

    Built once from the full Subject set via :func:`build_subject_naming`.
    ``display_name(subject)`` returns the canonical name for unique
    subjects and the qualified name (``"Prometheus (software)"``) for
    members of a collision group. Exporters route that display name
    through their own slug/link helpers.
    """

    display_by_id: Mapping[str, str]
    qualifier_by_id: Mapping[str, str]
    groups: tuple[DisambiguationGroup, ...] = field(default_factory=tuple)

    def display_name(self, subject: Subject) -> str:
        """Disambiguated display name for ``subject`` (canonical if unique)."""
        return self.display_by_id.get(subject.id, subject.canonical_name)

    @property
    def has_collisions(self) -> bool:
        return bool(self.groups)


def disambiguation_name(base_name: str) -> str:
    """Wikipedia-style name for a collision group's disambiguation note."""
    return f"{base_name} (disambiguation)"


def build_subject_naming(subjects: Iterable[Subject]) -> SubjectNaming:
    """Compute the disambiguated display name for every Subject.

    Subjects whose ``subject_slug(canonical_name)`` is unique keep their
    bare canonical name. When ≥2 share a base slug, each gets a
    parenthetical qualifier chosen by the cascade in
    :func:`_choose_qualifiers`.
    """
    subjects = list(subjects)
    groups: dict[str, list[Subject]] = defaultdict(list)
    for s in subjects:
        groups[subject_slug(s.canonical_name).lower()].append(s)

    display_by_id: dict[str, str] = {}
    qualifier_by_id: dict[str, str] = {}
    collision_groups: list[DisambiguationGroup] = []

    for base_slug, members in groups.items():
        if len(members) < 2:
            s = members[0]
            display_by_id[s.id] = s.canonical_name
            qualifier_by_id[s.id] = ""
            continue
        quals = _choose_qualifiers(members)
        for s in members:
            qualifier_by_id[s.id] = quals[s.id]
            display_by_id[s.id] = f"{s.canonical_name} ({quals[s.id]})"
        # The bare name for the group: when names are identical (the
        # common case) any member's canonical_name works; pick the
        # lexicographically-first for determinism when they differ
        # (distinct names that merely sanitized to the same slug).
        base_name = sorted(s.canonical_name for s in members)[0]
        collision_groups.append(
            DisambiguationGroup(
                base_name=base_name,
                base_slug=base_slug,
                member_ids=tuple(sorted(s.id for s in members)),
            )
        )

    return SubjectNaming(
        display_by_id=display_by_id,
        qualifier_by_id=qualifier_by_id,
        groups=tuple(sorted(collision_groups, key=lambda g: g.base_slug)),
    )


# per-NARRATIVE note naming. A NARRATIVE has a one-sentence *label*
# (its ``content``), not a canonical name, so this is a simpler sibling of
# ``build_subject_naming``: slug the label (truncated to a sane length), and on
# collision append the narrative's short id. The id suffix is the only qualifier
# tier and always disambiguates.
_NARRATIVE_SLUG_MAXLEN = 80


def build_narrative_naming(narratives: Iterable[Particle]) -> dict[str, str]:
    """Map each NARRATIVE id to a unique leaf slug for its vault note.

    Returns ``{narrative_id: slug}`` — the leaf filename without extension or the
    ``Narratives/`` directory prefix (the caller adds those). Colliding label
    slugs are disambiguated by appending the narrative's eight-char id.
    """
    narratives = list(narratives)
    base: dict[str, str] = {}
    for n in narratives:
        slug = subject_slug((n.content or "narrative").strip())[:_NARRATIVE_SLUG_MAXLEN]
        base[n.id] = slug.rstrip("-. ") or "narrative"
    counts: dict[str, int] = defaultdict(int)
    for slug in base.values():
        counts[slug] += 1
    result: dict[str, str] = {}
    for n in narratives:
        slug = base[n.id]
        result[n.id] = slug if counts[slug] == 1 else f"{slug}-{n.id[:8]}"
    return result


def narrative_as_subject(narrative: Particle) -> Subject:
    """Adapt a NARRATIVE to the ``Subject`` shape the render engine expects.

    The label becomes the title; the narrative id becomes the synthesis-cache
    key — a distinct id space from real Subjects. Shared by
    every prose exporter that emits per-narrative notes (Obsidian, Logseq, Wiki).
    """
    return Subject(
        id=narrative.id,
        canonical_name=narrative.content or "Narrative",
        asserted_by=narrative.asserted_by,
    )


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write ``content`` to ``path`` atomically (write-then-rename).

    The default ``Path.write_text`` opens with ``O_WRONLY|O_CREAT|O_TRUNC``
    — i.e. it *first truncates the file to zero bytes* and then writes
    the new content. iCloud / OneDrive / Dropbox sync daemons observe
    the truncate as one filesystem event and the subsequent write(s) as
    others, and resolve the conflict by leaving the original file in
    place and creating a sibling ``"subject 2.md"``. This breaks vault
    hygiene on every export.

    The fix is the standard POSIX atomic-rename idiom: write into a
    hidden tempfile sibling of the target, then ``os.replace`` it onto
    the target. ``os.replace`` is a single rename(2) syscall, atomic at
    the inode level, and cloud sync daemons see it as one event
    (replacement of the file). Same disk = same filesystem = the rename
    cannot cross a filesystem boundary and become a copy.

    Hidden-prefix + PID disambiguates concurrent writes within the same
    process tree and prevents the tempfile from being indexed by sync
    daemons before the rename. The target directory is created if
    missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(content, encoding=encoding)
        os.replace(tmp, path)
    except Exception:
        # Best-effort cleanup of the tempfile if anything between
        # write_text and replace blew up.
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise


def is_within_directory(directory: Path, path: Path) -> bool:
    """Whether ``path`` resolves to a location inside ``directory`` (containment guard).

    Defence-in-depth for the one-file-per-subject exporters: their write
    targets derive from :func:`subject_slug`, which slugs an LLM-extracted
    ``canonical_name`` (untrusted — sourced from deposited documents).
    :func:`subject_slug` already neutralises ``..`` / separators in the slug;
    this is an **independent** second gate every exporter applies right before
    writing, so a traversal that somehow survived slugging — or a symlinked
    output directory — still cannot land a write outside the export root. Both
    paths are fully :meth:`Path.resolve`\\d first so ``..`` segments and
    symlinks are collapsed before the comparison.
    """
    try:
        return path.resolve().is_relative_to(directory.resolve())
    except (OSError, ValueError):
        # A path that cannot even be resolved (e.g. a symlink loop) is, by
        # definition, not safely contained — fail closed.
        return False


#: Ownership ledger a one-file-per-subject exporter keeps at its export root
#:: the root-relative paths of every ``.md`` file its last run wrote.
EXPORT_MANIFEST_NAME = ".particles-export.json"

#: Append-only journal of files a run has written that its prior manifest did
#: not list. It is unioned into ownership on the next run, so an export killed
#: before it could save its manifest still owns what it wrote.
EXPORT_JOURNAL_NAME = ".particles-export.pending"


class ExportTargetNotOwnedError(ValueError):
    """An export target holds Markdown files the exporter did not write.

    Raised before anything is written, the first time an exporter is pointed
    at a populated directory, unless the caller passed ``force``. A
    ``ValueError`` so every surface reports it as a usage error.
    """


def prune_obsolete_markdown(
    directory: Path, written: set[Path], *, owned: set[Path], recursive: bool
) -> int:
    """Remove the ``.md`` files a previous export run wrote that this run did
    not, returning the count removed.

    Shared by every exporter that writes one-file-per-subject
    (Obsidian, Logseq, future flat-markdown exporters) so the
    "subject suppressed / renamed in DB but file lingers on disk"
    case is handled identically across them. Replaces the pre-0.42.4
    pattern of wiping the entire output directory before writing.

    Only paths in ``owned`` are candidates: the files the exporter provably
    wrote, from its manifest (see :class:`MarkdownExportLedger`). Before
    every ``.md`` under ``directory`` was a candidate, so pointing
    an export at a real vault deleted the user's own notes. A file not in
    ``owned`` is never touched, whatever its name.

    Paths are compared by :meth:`Path.resolve`. Each candidate must also
    resolve *inside* ``directory`` (via :func:`is_within_directory`), be a
    ``.md`` file and, with ``recursive=False``, sit directly in
    ``directory``, so a tampered manifest or a symlinked shard whose target
    lies outside the export root is skipped rather than followed.

    A subdirectory the prune emptied is removed (so old ``reddit.com/`` /
    ``github.com/`` shards disappear with their last child); a directory that
    was already empty, or that holds anything else, is left alone.
    """
    root = directory.resolve()
    resolved_written = {p.resolve() for p in written}
    files_pruned = 0
    emptied: set[Path] = set()
    for candidate in sorted({p.resolve() for p in owned} - resolved_written):
        if candidate.suffix != ".md" or not candidate.is_file():
            continue
        if not is_within_directory(root, candidate):
            continue
        if not recursive and candidate.parent != root:
            continue
        candidate.unlink()
        files_pruned += 1
        emptied.add(candidate.parent)
    # Children before parents, and never the export root itself.
    for d in sorted(emptied, key=lambda p: len(p.parts), reverse=True):
        while d != root and d.is_relative_to(root):
            if not d.is_dir() or any(d.iterdir()):
                break
            d.rmdir()
            d = d.parent
    return files_pruned


class MarkdownExportLedger:
    """Which ``.md`` files under an export target the exporter owns.

    A one-file-per-subject export prunes the notes of subjects that left its
    target set, and is pointed at directories users also write in (an
    Obsidian vault, a Logseq graph's ``pages/``). The ledger is what keeps
    the two apart. Ownership is the manifest at ``root`` naming every file
    the last run wrote, unioned with the journal of an interrupted run. An
    export written before the manifest existed is adopted through
    ``legacy_marker``, a predicate over a file's text, when no manifest is
    present.

    The lifecycle is :meth:`claim`, then :meth:`write` for every file, then
    :meth:`finish`, which prunes and saves the manifest. :meth:`claim`
    refuses a target that holds files it does not own and has never been
    exported to, unless ``force``. :meth:`write` never overwrites a file it
    does not own unless ``force``; it skips it and reports ``False``.
    """

    def __init__(
        self,
        root: Path,
        scope: Path,
        *,
        exporter: str,
        recursive: bool,
        force: bool,
        owned: set[Path],
    ) -> None:
        self.root = root
        self.scope = scope
        self.exporter = exporter
        self.recursive = recursive
        self.force = force
        self.owned = owned
        self.written: set[Path] = set()
        self.skipped: list[Path] = []

    @classmethod
    def claim(
        cls,
        root: Path,
        *,
        exporter: str,
        scope: Path | None = None,
        recursive: bool,
        force: bool = False,
        legacy_marker: Callable[[str], bool] | None = None,
    ) -> MarkdownExportLedger:
        """Load ownership for ``exporter`` at ``root`` and vet the target.

        ``scope`` is the directory whose ``.md`` files the exporter writes and
        prunes (default ``root``). Raises :class:`ExportTargetNotOwnedError`
        when the target has no manifest from this exporter, holds ``.md``
        files it does not own, and ``force`` is false. Writes nothing.
        """
        scope = scope if scope is not None else root
        manifest = _read_export_manifest(root, exporter)
        journal = _read_export_journal(root)
        owned = {p for p in (manifest or set()) | journal if _ownable(scope, p, recursive)}
        present = _markdown_files(scope, recursive)
        if manifest is None and legacy_marker is not None:
            for path in present - owned:
                try:
                    text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                if legacy_marker(text):
                    owned.add(path)
        foreign = sorted(present - owned)
        if foreign and manifest is None and not journal and not force:
            shown = ", ".join(str(p.relative_to(scope.resolve())) for p in foreign[:5])
            more = f" and {len(foreign) - 5} more" if len(foreign) > 5 else ""
            raise ExportTargetNotOwnedError(
                f"{scope} holds {len(foreign)} Markdown file(s) this {exporter} export "
                f"did not write ({shown}{more}). Export into an empty directory or a "
                f"folder of its own, or pass --force to export alongside them. The "
                f"export never deletes a file it did not write; with --force it "
                f"overwrites one whose path a note it writes takes."
            )
        return cls(root, scope, exporter=exporter, recursive=recursive, force=force, owned=owned)

    def write(self, path: Path, content: str) -> bool:
        """Write ``path`` atomically unless it is a file this exporter does not own.

        Returns ``False``, and writes nothing, for an existing file that is
        neither owned nor written earlier this run, unless ``force``.
        """
        resolved = path.resolve()
        if (
            not self.force
            and resolved.exists()
            and resolved not in self.owned
            and resolved not in self.written
        ):
            self.skipped.append(resolved)
            return False
        atomic_write_text(path, content)
        if resolved not in self.owned and resolved not in self.written:
            _append_export_journal(self.root, resolved)
        self.written.add(resolved)
        return True

    def finish(self) -> int:
        """Prune owned files this run did not write, save the manifest, clear
        the journal; return the number pruned."""
        pruned = prune_obsolete_markdown(
            self.scope, self.written, owned=self.owned, recursive=self.recursive
        )
        root = self.root.resolve()
        files = sorted(p.relative_to(root).as_posix() for p in self.written)
        manifest = {"exporter": self.exporter, "version": 1, "files": files}
        atomic_write_text(self.root / EXPORT_MANIFEST_NAME, json.dumps(manifest, indent=1) + "\n")
        with contextlib.suppress(FileNotFoundError):
            (self.root / EXPORT_JOURNAL_NAME).unlink()
        return pruned


def _markdown_files(scope: Path, recursive: bool) -> set[Path]:
    if not scope.is_dir():
        return set()
    iterator = scope.rglob("*.md") if recursive else scope.glob("*.md")
    return {p.resolve() for p in iterator if p.is_file() and is_within_directory(scope, p)}


def _ownable(scope: Path, path: Path, recursive: bool) -> bool:
    """A manifest entry is honoured only for a ``.md`` path inside ``scope``."""
    if path.suffix != ".md" or not is_within_directory(scope, path):
        return False
    return recursive or path.parent == scope.resolve()


def _read_export_manifest(root: Path, exporter: str) -> set[Path] | None:
    """The resolved paths ``exporter``'s manifest at ``root`` lists, or ``None``
    when there is none (or it belongs to another exporter, or is unreadable)."""
    try:
        data = json.loads((root / EXPORT_MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("exporter") != exporter:
        return None
    files = data.get("files")
    if not isinstance(files, list):
        return None
    return {(root / f).resolve() for f in files if isinstance(f, str)}


def _read_export_journal(root: Path) -> set[Path]:
    try:
        lines = (root / EXPORT_JOURNAL_NAME).read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()
    return {(root / line).resolve() for line in lines if line.strip()}


def _append_export_journal(root: Path, path: Path) -> None:
    rel = path.relative_to(root.resolve()).as_posix()
    with (root / EXPORT_JOURNAL_NAME).open("a", encoding="utf-8") as fh:
        fh.write(rel + "\n")


_SEVERITY_CALLOUT: dict[str, str] = {
    "ERROR": "danger",
    "WARNING": "warning",
    "INFO": "info",
}

_STATUS_CALLOUT: dict[str, str] = {
    Status.ACTIVE: "success",
    Status.INCONSISTENCY: "danger",
    Status.PROVENANCE_STALE: "warning",
    Status.SUPERSEDED: "note",
    Status.RETRACTED: "failure",
}


def render_particle(particle: Particle, effective_confidence: float | None = None) -> str:
    """Render a single particle as a Markdown callout block."""
    callout_type = _STATUS_CALLOUT.get(particle.status, "note")
    conf_display = (
        f"{effective_confidence:.2f}"
        if effective_confidence is not None
        else f"{particle.confidence.value:.2f}"
    )
    title = f"Particle [{particle.status}] — confidence: {conf_display}"

    lines = [
        f"> [!{callout_type}] {title}",
        f"> **Content:** {particle.content}",
        f"> **Uncertainty:** {particle.uncertainty_nature}",
        f"> **Asserted by:** {particle.asserted_by}",
        f"> **Asserted at:** {particle.asserted_at.isoformat()}",
        f"> **ID:** `{particle.id}`",
    ]

    if particle.provenance:
        pref = particle.provenance[0]
        lines.append(f"> **Source:** {pref.corpus_entry_id} / {pref.snapshot_id or '—'}")

    if particle.status_reason:
        lines.append(f"> **Status reason:** {particle.status_reason}")

    return "\n".join(lines)


def render_stance_callout(positions: list[StancePosition]) -> str:
    """Render a claim's query-time agreement distribution as a callout.

    Holders are grouped Endorses / Disputes, each attributed and cited by the
    stance particle. Returns ``""`` for an empty distribution. The callout
    carries the M6 unverified-grouping caveat so the holder set is never read as
    a count of verified agents, and is labelled as *not* factual confidence (the
    §4 MUST: agreement is surfaced beside, never folded into, the claim's own
    confidence).
    """
    if not positions:
        return ""

    def _row(p: StancePosition) -> str:
        mag = f", magnitude {p.magnitude:.2f}" if p.magnitude is not None else ""
        short = p.stance_particle_id.split("-")[0]
        return f"> - {p.holder} (conf {p.effective_confidence:.2f}{mag}) `{short}`"

    lines = ["> [!agreement] Positions on this claim (query-time; not factual confidence)"]
    endorses = [p for p in positions if p.kind == RelationType.ENDORSES]
    disputes = [p for p in positions if p.kind == RelationType.DISPUTES]
    if endorses:
        lines.append("> **Endorses:**")
        lines.extend(_row(p) for p in endorses)
    if disputes:
        lines.append("> **Disputes:**")
        lines.extend(_row(p) for p in disputes)
    lines.append("> _Holders are unverified raw keys — a count of keys, not verified agents._")
    return "\n".join(lines)


def render_contested_callout(
    reading: ContestednessReading | None = None,
    *,
    badge: ContestedBadge | None = None,
    positions: list[StancePosition] | None = None,
) -> str:
    """Render one composed ``[!contested]`` callout for a claim.

    ``badge`` is the composed basis-carrying badge; the callout names the fired
    bases and adds one attribution block per basis from whichever drill-down
    payloads the caller supplied — the divergence ``reading`` (per-policy
    renderings, sorted most- to least-confident so the extremes are nameable),
    the stance ``positions`` (the disputing holders, plus the M6 caveat the
    badge carries), and the inconsistency id the badge itself holds. Returns
    ``""`` when no badge fired — absence of a badge is never rendered as an
    explicit "uncontested" (§3).

    Called with only a ``reading`` (the earlier signature), it derives a
    divergence-only badge gated by ``config.contestedness.callout_threshold``
    so existing callers keep today's behavior.

    The callout is **disclosure, not discount**: it
    never moves the claim's own confidence, which the particle block shows.
    """
    if badge is None:
        # Legacy divergence-only path: threshold-gate the reading.
        if reading is None:
            return ""
        from particles.config import get_config

        if reading.spread < get_config().contestedness.callout_threshold:
            return ""
        badge = ContestedBadge(bases=["divergence"])

    lines = [f"> [!contested] Contested ({', '.join(badge.bases)}; not factual confidence)"]
    if "stance" in badge.bases:
        disputes = [p for p in positions or [] if p.kind == RelationType.DISPUTES]
        if disputes:
            lines.append("> **Stance — disputed by:**")
            for p in disputes:
                mag = f", magnitude {p.magnitude:.2f}" if p.magnitude is not None else ""
                short = p.stance_particle_id.split("-")[0]
                lines.append(f"> - {p.holder} (conf {p.effective_confidence:.2f}{mag}) `{short}`")
        else:
            lines.append("> **Stance:** at least one holder disputes this claim.")
        if badge.caveat:
            lines.append(f"> _{badge.caveat}_")
    if "divergence" in badge.bases:
        if reading is not None:
            ordered = sorted(reading.renderings, key=lambda r: (-r.effective_confidence, r.policy))
            lines.append(f"> **Divergence across trust policies (spread {reading.spread:.2f}):**")
            lines.extend(f"> - **{r.policy}:** {r.effective_confidence:.2f}" for r in ordered)
        else:
            lines.append(
                "> **Divergence:** effective confidence spreads across the viewer's "
                "policy set (local policy + adopted lenses)."
            )
    if "inconsistency" in badge.bases and badge.inconsistency_id:
        lines.append(
            f"> **Inconsistency:** open INCONSISTENCY `{badge.inconsistency_id}` "
            "references this claim."
        )
    lines.append("> _The claim's own confidence is unchanged — disclosure, never a discount._")
    return "\n".join(lines)


def render_particles(
    particles: list[Particle],
    effective_confidences: list[float] | None = None,
    agreement_distributions: list[list[StancePosition]] | None = None,
    contestedness: list[ContestednessReading] | None = None,
    contested: list[ContestedBadge | None] | None = None,
) -> str:
    """Render a list of particles as a Markdown document.

    When ``contested`` is supplied (parallel to ``particles``),
    each badged particle gets **one** composed ``[!contested]`` callout naming
    its fired bases, with per-basis attribution drawn from whichever payloads
    are also supplied; the ``[!agreement]`` callout then renders only for the
    endorse-only (uncontested) distributions. Without badges the earlier
    behavior holds: ``agreement_distributions`` appends each
    stance callout, and ``contestedness`` appends a
    divergence-only callout when the spread crosses the configured threshold.
    """
    if not particles:
        return "_No particles to display._\n"
    parts: list[str] = [f"# Particles ({len(particles)})\n"]
    for i, p in enumerate(particles):
        ec = (
            effective_confidences[i]
            if effective_confidences and i < len(effective_confidences)
            else None
        )
        parts.append(render_particle(p, ec))
        badge = contested[i] if contested and i < len(contested) else None
        positions = (
            agreement_distributions[i]
            if agreement_distributions and i < len(agreement_distributions)
            else []
        )
        reading = contestedness[i] if contestedness and i < len(contestedness) else None
        if contested is not None:
            # one composed callout per badged claim; [!agreement]
            # remains for the endorse-only (uncontested) distribution.
            if badge is not None:
                parts.append(render_contested_callout(reading, badge=badge, positions=positions))
            elif positions and not any(sp.kind == RelationType.DISPUTES for sp in positions):
                parts.append(render_stance_callout(positions))
        else:
            if positions:
                parts.append(render_stance_callout(positions))
            callout = render_contested_callout(reading)
            if callout:
                parts.append(callout)
        parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sources trailer + memory bullets (fork #2, §4–§5).
# Shared, store-free formatting: the projection renderer emits these; the
# session-start freshness check parses them back.
# ---------------------------------------------------------------------------

#: Matches one sources trailer, capturing the comma-separated id list.
SOURCES_TRAILER_RE = re.compile(r"<!--\s*sources:\s*(?P<ids>[^>]*?)\s*-->")


def particle_short_id(particle_id: str) -> str:
    """Eight-char particle-id prefix — the ``p-<shortid>`` display body."""
    return particle_id[:8]


def format_sources_trailer(short_ids: Iterable[str]) -> str:
    """The per-section provenance trailer (fork #2).

    A single HTML-comment line listing the cited / selected particle short-ids
    (sorted, de-duplicated) — the machine-diffable selection fingerprint the
    freshness check compares. Returns ``""`` for an empty input.
    """
    unique = sorted(set(short_ids))
    if not unique:
        return ""
    return "<!-- sources: " + ", ".join(f"p-{sid}" for sid in unique) + " -->\n"


def parse_sources_trailers(text: str) -> set[str] | None:
    """Extract the union of all sources-trailer ids in ``text`` (``p-`` stripped).

    Returns ``None`` when no trailer is present — the caller's parse-failure
    signal (fall back to the full digest). An empty trailer
    (``<!-- sources: -->``) parses as the empty set, which is distinct.
    """
    trailers = SOURCES_TRAILER_RE.findall(text)
    if not trailers:
        return None
    ids: set[str] = set()
    for raw in trailers:
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if token.startswith("p-"):
                token = token[2:]
            ids.add(token)
    return ids


def format_memory_bullet(
    content: str,
    short_id: str,
    contested_by: str | None = None,
    contested_bases: Iterable[str] = (),
) -> str:
    """One ranked memory-index bullet — digest-style, deterministic.

    The line carries the belief content and its short-id drill-down handle
    (resolved via the MCP tools); a contested belief is flagged
    rather than omitted — an agent must *know* a belief is disputed.
    ``contested_bases`` names the composed badge's fired bases
    — "⚠ contested (stance, divergence) — …" — keeping the "(vs. p-xxx)"
    drill-down when the inconsistency basis fired (``contested_by``). With no
    bases supplied, a bare ``contested_by`` renders the pre-badge
    inconsistency-only flag unchanged. No timestamp, no volatile field:
    byte-stable for a given input (the §2 idempotent-render belt).
    """
    flat = " ".join(content.split())
    bases = list(contested_bases)
    versus = f" (vs. p-{contested_by})" if contested_by else ""
    if bases:
        return f"- ⚠ contested ({', '.join(bases)}) — {flat}{versus} `p-{short_id}`"
    if contested_by:
        return f"- ⚠ contested — {flat}{versus} `p-{short_id}`"
    return f"- {flat} `p-{short_id}`"


#: One memory-index bullet as :func:`format_memory_bullet` writes it: the
#: contested bases when the badge named them, and the closing short-id handle.
_MEMORY_BULLET_RE = re.compile(
    r"^- (?:⚠ contested(?: \((?P<bases>[^)]*)\))? — )?.*`p-(?P<id>[0-9A-Za-z-]+)`\s*$"
)


def parse_memory_bullet(line: str) -> tuple[str, list[str]] | None:
    """Read back a :func:`format_memory_bullet` line: ``(short_id, contested_bases)``.

    Returns ``None`` for a line that is not a memory bullet. A line flagged
    contested without named bases (the pre-badge inconsistency-only form)
    yields an empty basis list, as does an uncontested one.
    """
    match = _MEMORY_BULLET_RE.match(line)
    if match is None:
        return None
    raw = match.group("bases")
    bases = [b.strip() for b in raw.split(",") if b.strip()] if raw else []
    return match.group("id"), bases


@dataclass(frozen=True)
class DigestEntry:
    """One belief's pre-gathered data for the session-start digest.

    The assembly side (``particles/mcp/resources``) fills these — content,
    query-time effective confidence, subject canonical names, ``asserted_at``,
    and the referencing INCONSISTENCY id when the belief is contested—
    and hands a pre-ordered list to :func:`render_digest`. This keeps the
    formatter pure (no session, no trust math, no clock).
    """

    content: str
    effective_confidence: float
    subjects: tuple[str, ...]
    asserted_at: datetime
    contested: str | None = None
    #: the composed badge's fired basis labels (empty when no badge
    #: fired or the badge is disabled). ``contested`` above keeps its §7
    #: meaning — the open INCONSISTENCY id, now exactly the ``inconsistency``
    #: basis's drill-down.
    contested_bases: tuple[str, ...] = ()


def observer_scope_line(note: ObserverScopeNote) -> str:
    """The one-line disclosure a project-observer read carries. Never silent."""
    if not note.engaged:
        return (
            f"_Project observer `{note.project}` not applied: this store has not been "
            "rescoped yet (`particles memory rescope`), so this is the whole store._"
        )
    line = (
        f"_Observer: project `{note.project}` — {note.in_scope} of {note.total} ACTIVE "
        "belief(s) in scope (global, or observed in this project)"
    )
    if note.unattributed:
        line += f"; {note.unattributed} unattributed"
    if note.lapsed:
        line += f"; {note.lapsed} lapsed (no source states them any more)"
    return line + "._"


def render_digest(
    store: str,
    entries: list[DigestEntry],
    total_active: int,
    observer: ObserverScopeNote | None = None,
    spend_line: str | None = None,
) -> str:
    """Render a store's ACTIVE beliefs as a terse session-start digest.

    The text of :func:`render_digest_located`, which documents the format.
    """
    return render_digest_located(store, entries, total_active, observer, spend_line)[0]


def render_digest_located(
    store: str,
    entries: list[DigestEntry],
    total_active: int,
    observer: ObserverScopeNote | None = None,
    spend_line: str | None = None,
) -> tuple[str, list[int]]:
    """Render the digest, and where each entry's line starts in it.

    Returns ``(markdown, offsets)``: ``offsets[i]`` is the character offset in
    ``markdown`` at which ``entries[i]``'s line begins. A caller that cuts the
    text to a prefix can tell which entries survived the cut from these alone,
    whatever the entries' content holds.

    The ``MEMORY.md`` analog: one line per belief, **already ordered by the
    caller** (effective confidence, descending). An index, not a dump —
    provenance and ids are a ``particle_show`` away, so the artifact stays within
    a client's context budget. A contested belief is flagged with the composed
    badge's fired bases and, when the inconsistency basis fired, the
    open INCONSISTENCY id (the MUST this digest inherits). When
    fewer entries are shown than ``total_active`` the footer discloses the
    truncation (no silent cap).

    ``observer`` is set when the digest was read through a project
    observer: ``total_active`` is then the in-scope population the entries were
    ranked from, and one line says which observer, how many beliefs are in
    scope, and how many the store holds. With no observer the output is
    byte-identical to before.

    ``spend_line`` is the store's recorded LLM spend, rendered by
    the caller; it closes the digest as one italic line. ``None`` (no run has
    recorded usage yet) leaves the output byte-identical to before.

    Pure: no session and no clock — ``asserted_at`` renders as its date, and
    recency is already folded into the effective-confidence ordering, so the
    output is deterministic for a given input.
    """
    header = f"# Memory digest — {store}"
    scope_lines = [observer_scope_line(observer), ""] if observer is not None else []
    if not entries:
        where = (
            f"in view for project `{observer.project}` in `{store}`"
            if observer is not None and observer.engaged
            else f"in `{store}`"
        )
        empty = [header, "", *scope_lines, f"_No ACTIVE beliefs {where}._"]
        if spend_line is not None:
            empty += ["", f"_{spend_line}_"]
        return "\n".join(empty) + "\n", []

    shown = len(entries)
    entry_rows: list[int] = []
    lines = [
        header,
        "",
        *scope_lines,
        f"_{shown} of {total_active} ACTIVE belief(s), ranked by effective confidence._",
        "",
    ]
    for e in entries:
        subjects = f" _({', '.join(e.subjects)})_" if e.subjects else ""
        # the composed badge names its fired bases; the INCONSISTENCY
        # id stays as the inconsistency basis's drill-down. A bases-less entry
        # renders the pre-badge inconsistency-only flag unchanged.
        if e.contested_bases:
            contested = f" · contested ({', '.join(e.contested_bases)})"
            if e.contested:
                contested += f" by `{e.contested}`"
        elif e.contested:
            contested = f" · contested by `{e.contested}`"
        else:
            contested = ""
        entry_rows.append(len(lines))
        lines.append(
            f"- **{e.effective_confidence:.2f}** {e.content}{subjects} · "
            f"{e.asserted_at.date().isoformat()}{contested}"
        )
    if shown < total_active:
        lines += [
            "",
            f"_Showing the top {shown} of {total_active} by effective confidence; "
            f"{total_active - shown} not shown._",
        ]
    if spend_line is not None:
        lines += ["", f"_{spend_line}_"]
    # Each line is joined by one "\n", so a line starts after the lengths of
    # every line before it plus their separators.
    starts: list[int] = []
    pos = 0
    for line in lines:
        starts.append(pos)
        pos += len(line) + 1
    return "\n".join(lines) + "\n", [starts[row] for row in entry_rows]


def render_lint_finding(finding: LintFinding) -> str:
    """Render a single lint finding as a Markdown callout block."""
    callout_type = _SEVERITY_CALLOUT.get(finding.severity, "note")
    ref = finding.particle_id or finding.corpus_entry_id or "—"
    title = f"Lint [{finding.finding_type}] — {finding.severity}"

    lines = [
        f"> [!{callout_type}] {title}",
        f"> **Detail:** {finding.detail}",
        f"> **Reference:** `{ref}`",
    ]
    if finding.recommended_action:
        lines.append(f"> **Action:** {finding.recommended_action}")

    return "\n".join(lines)


def render_lint_report(report: LintReport) -> str:
    """Render a full LintReport as a Markdown document."""
    lines = [
        f"# Lint Report — {report.run_at.isoformat()}",
        "",
        "## Summary",
        "",
    ]
    if report.summary:
        for finding_type, count in sorted(report.summary.items()):
            lines.append(f"- **{finding_type}**: {count}")
    else:
        lines.append("_No findings._")

    lines += ["", "## Findings", ""]

    if not report.findings:
        lines.append("_No findings._")
    else:
        for finding in report.findings:
            lines.append(render_lint_finding(finding))
            lines.append("")

    return "\n".join(lines)
