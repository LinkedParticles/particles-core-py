"""Fixture corpus loader for the extractor conformance validator.

A fixture is a directory under ``tests/conformance/fixtures/`` containing:

  manifest.yaml      # fixture_id, source_type, expected_acceptors, notes
  content.bin        # raw bytes the extractor would receive
  snapshot.json      # serialised Snapshot (sha256, etag, content_published_at, …)

The fixture corpus is version-pinned by ``compute_corpus_hash`` — a SHA-256
over every fixture's ``(fixture_id, content, snapshot)`` triple. Reports
record this hash so two reports for the same extractor are only comparable
when their corpora match.

``MANIFEST.yaml`` at the corpus root is a human-readable inventory and is
**not** read by discovery: :func:`iter_fixtures` walks the directory and
loads every subdirectory carrying a ``manifest.yaml``. A fixture missing
from ``MANIFEST.yaml`` therefore still runs and still contributes to the
corpus hash.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml

from particles.core.schema import Snapshot


@dataclass(frozen=True)
class Fixture:
    """One loaded fixture ready to feed to an extractor.

    ``content`` is the raw byte payload the importer would have written to
    the corpus blob store. ``snapshot`` is the corresponding ``Snapshot`` the
    extractor's ``extract()`` is called with.
    """

    fixture_id: str
    source_type: str
    expected_acceptors: list[str]
    content: bytes
    snapshot: Snapshot
    notes: str = ""


def _load_one(fixture_dir: Path) -> Fixture:
    """Load a single fixture directory into a :class:`Fixture`."""
    manifest_path = fixture_dir / "manifest.yaml"
    content_path = fixture_dir / "content.bin"
    snapshot_path = fixture_dir / "snapshot.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Fixture {fixture_dir.name}: missing manifest.yaml")
    if not content_path.exists():
        raise FileNotFoundError(f"Fixture {fixture_dir.name}: missing content.bin")
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Fixture {fixture_dir.name}: missing snapshot.json")

    manifest = yaml.safe_load(manifest_path.read_text())
    content = content_path.read_bytes()
    snapshot = Snapshot.model_validate_json(snapshot_path.read_text())

    return Fixture(
        fixture_id=manifest["fixture_id"],
        source_type=manifest["source_type"],
        expected_acceptors=list(manifest.get("expected_acceptors", [])),
        content=content,
        snapshot=snapshot,
        notes=manifest.get("notes", ""),
    )


def iter_fixtures(fixture_dir: Path) -> Iterator[Fixture]:
    """Yield every fixture in ``fixture_dir``, sorted by ``fixture_id``.

    Hidden directories and files (``.`` prefix), the corpus ``MANIFEST.yaml``,
    and any ``__pycache__`` are skipped. Subdirectories without a
    ``manifest.yaml`` are also skipped silently so partial / WIP fixtures do
    not break a run.
    """
    if not fixture_dir.exists():
        return
    for child in sorted(fixture_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name.startswith("__"):
            continue
        if not (child / "manifest.yaml").exists():
            continue
        yield _load_one(child)


_MANIFEST_HEADER = """\
# Conformance fixture corpus manifest.
#
# This file is informational — the *authoritative* corpus hash is computed
# by ``particles.conformance.fixtures.compute_corpus_hash`` and surfaced
# inside every ConformanceReport's ``fixture_corpus_hash`` field. Reports
# are only comparable when their hashes match, so corpus growth is a
# deliberate, reviewed event.
#
# Adding a fixture: create a new sibling directory with manifest.yaml +
# content.bin + snapshot.json, then update this list (the
# ``extractor generate-fixture`` verb does both). Removing or
# renaming a fixture invalidates every prior report."""


def write_fixture(
    output_dir: Path,
    fixture_id: str,
    *,
    source_type: str,
    content: bytes,
    snapshot: Snapshot,
    expected_acceptors: list[str] | None = None,
    notes: str = "",
    force: bool = False,
) -> Path:
    """Write a fixture skeleton (``manifest.yaml`` + ``content.bin`` +
    ``snapshot.json``) under ``output_dir/fixture_id`` and register it in the
    corpus ``MANIFEST.yaml``.

    Pure / store-free: the caller supplies the raw ``content`` bytes and the
    ``Snapshot`` model (read from the store at the CLI layer). ``expected_acceptors``
    defaults to an empty list — the operator fills it after verifying which
    extractors the fixture should exercise (decision 4). Raises
    ``FileExistsError`` if the directory already exists and ``force`` is False.

    Returns the fixture directory path.
    """
    fixture_dir = output_dir / fixture_id
    if fixture_dir.exists() and not force:
        raise FileExistsError(f"Fixture directory already exists: {fixture_dir} (use --force)")
    fixture_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "fixture_id": fixture_id,
        "source_type": source_type,
        "expected_acceptors": expected_acceptors or [],
        "notes": notes,
    }
    (fixture_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (fixture_dir / "content.bin").write_bytes(content)
    (fixture_dir / "snapshot.json").write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")

    _register_in_manifest(
        output_dir / "MANIFEST.yaml", fixture_id, source_type, expected_acceptors or []
    )
    return fixture_dir


def _register_in_manifest(
    manifest_path: Path,
    fixture_id: str,
    source_type: str,
    expected_acceptors: list[str],
) -> None:
    """Add (or replace) the fixture's entry in the corpus ``MANIFEST.yaml``.

    Preserves the file's header comment block (the only comments it carries),
    re-emitting the ``fixtures:`` list sorted by id so re-running the generator
    is idempotent.
    """
    text = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else ""
    data = yaml.safe_load(text) if text.strip() else None
    fixtures: list[dict[str, object]] = list((data or {}).get("fixtures") or [])
    fixtures = [f for f in fixtures if f.get("id") != fixture_id]
    fixtures.append(
        {"id": fixture_id, "source_type": source_type, "expected_acceptors": expected_acceptors}
    )
    fixtures.sort(key=lambda f: str(f.get("id")))
    header = text.split("\nfixtures:", 1)[0].rstrip() if "fixtures:" in text else _MANIFEST_HEADER
    body = yaml.safe_dump({"fixtures": fixtures}, sort_keys=False, allow_unicode=True)
    manifest_path.write_text(f"{header}\n\n{body}", encoding="utf-8")


def compute_corpus_hash(fixture_dir: Path) -> str:
    """SHA-256 of the fixture corpus contents (manifests + content + snapshots).

    Stable across machines: the hash is computed over (fixture_id, content,
    snapshot) tuples sorted by fixture_id, so the hash only depends on what
    the validator actually consumes.
    """
    hasher = hashlib.sha256()
    for fixture in iter_fixtures(fixture_dir):
        hasher.update(fixture.fixture_id.encode())
        hasher.update(b"\x00")
        hasher.update(fixture.content)
        hasher.update(b"\x00")
        hasher.update(json.dumps(fixture.snapshot.model_dump(mode="json"), sort_keys=True).encode())
        hasher.update(b"\x00")
    return hasher.hexdigest()


DEFAULT_FIXTURE_DIR = Path("tests/conformance/fixtures")
