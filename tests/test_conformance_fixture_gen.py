"""Tests for the fixture writer (`conformance.fixtures.write_fixture`).

The store-reading half (resolving an entry → snapshot → blob) lives in the
`extractor generate-fixture` CLI verb and is covered in tests/test_cli.py; this
pins the pure writer + MANIFEST registration.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from particles.conformance.fixtures import iter_fixtures, write_fixture
from particles.core.schema import ExtractionStatus, Snapshot, WarcRecordType


def _snapshot() -> Snapshot:
    return Snapshot(
        snapshot_id="snap-fixture-001",
        content_hash="a" * 64,
        warc_record_type=WarcRecordType.RESPONSE,
        extraction_status=ExtractionStatus.COMPLETE,
    )


class TestWriteFixture:
    def test_round_trips_through_iter_fixtures(self, tmp_path: Path) -> None:
        write_fixture(
            tmp_path,
            "my-fixture-001",
            source_type="WEB_PAGE",
            content=b"<html>hello</html>",
            snapshot=_snapshot(),
            notes="exercises the html path",
        )
        loaded = list(iter_fixtures(tmp_path))
        assert len(loaded) == 1
        fx = loaded[0]
        assert fx.fixture_id == "my-fixture-001"
        assert fx.source_type == "WEB_PAGE"
        assert fx.content == b"<html>hello</html>"
        assert fx.expected_acceptors == []  # left empty for the operator to fill
        assert fx.notes == "exercises the html path"
        assert fx.snapshot.content_hash == "a" * 64

    def test_registers_in_manifest(self, tmp_path: Path) -> None:
        write_fixture(
            tmp_path,
            "coin-001",
            source_type="NUMISTA_API_COIN",
            content=b"{}",
            snapshot=_snapshot(),
        )
        manifest = yaml.safe_load((tmp_path / "MANIFEST.yaml").read_text())
        assert manifest["fixtures"] == [
            {"id": "coin-001", "source_type": "NUMISTA_API_COIN", "expected_acceptors": []}
        ]

    def test_manifest_preserves_header_and_sorts(self, tmp_path: Path) -> None:
        (tmp_path / "MANIFEST.yaml").write_text(
            "# Hand-written header comment\n\nfixtures:\n  - id: zeta-001\n"
            "    source_type: PDF\n    expected_acceptors: []\n",
            encoding="utf-8",
        )
        write_fixture(
            tmp_path, "alpha-001", source_type="WEB_PAGE", content=b"x", snapshot=_snapshot()
        )
        text = (tmp_path / "MANIFEST.yaml").read_text()
        assert "# Hand-written header comment" in text  # header preserved
        manifest = yaml.safe_load(text)
        # alpha sorts before zeta; both present.
        assert [f["id"] for f in manifest["fixtures"]] == ["alpha-001", "zeta-001"]

    def test_refuses_existing_dir_without_force(self, tmp_path: Path) -> None:
        write_fixture(tmp_path, "dup-001", source_type="PDF", content=b"a", snapshot=_snapshot())
        import pytest

        with pytest.raises(FileExistsError):
            write_fixture(
                tmp_path, "dup-001", source_type="PDF", content=b"b", snapshot=_snapshot()
            )

    def test_force_overwrites(self, tmp_path: Path) -> None:
        write_fixture(tmp_path, "dup-002", source_type="PDF", content=b"old", snapshot=_snapshot())
        write_fixture(
            tmp_path,
            "dup-002",
            source_type="PDF",
            content=b"new",
            snapshot=_snapshot(),
            force=True,
        )
        assert (tmp_path / "dup-002" / "content.bin").read_bytes() == b"new"
        # MANIFEST keeps a single entry (idempotent re-registration).
        manifest = yaml.safe_load((tmp_path / "MANIFEST.yaml").read_text())
        assert [f["id"] for f in manifest["fixtures"]] == ["dup-002"]
