# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for vision / multimodal PDF extraction.

Covers the deterministic seams: the page-modality decision, the rasterizer
wiring (mocked — pypdfium2 + Pillow live behind the [vision] extra, absent in
the unit venv), the actionable ImportError when the extra is missing, the
vision source-modality marker, and the per-page routing in the paged-PDF loop
(image-bearing → multimodal call; text → text-only; page cap; disabled = no
vision). The model is never called and no PDF is rendered for real.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from particles.config import ExtractionVisionConfig, ParticlesConfig
from particles.core.schema import Snapshot, UncertaintyNature
from particles.extraction.general import (
    VISION_SOURCE_MODALITY,
    VISION_SOURCE_MODALITY_KEY,
    CandidateParticle,
    GeneralExtractor,
    _call_llm,
    _open_pdfium_for_vision,
    _page_is_visual,
    _render_page_png,
    sniff_image_media_type,
)
from particles.llm import VisionImage

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _candidate(content: str = "a claim") -> CandidateParticle:
    return CandidateParticle(
        content=content,
        confidence_value=0.8,
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
    )


class TestPageIsVisual:
    def test_always_trigger_is_always_visual(self) -> None:
        page = MagicMock()
        page.images = []
        assert _page_is_visual(
            page, "lots of text " * 100, trigger="always", low_text_threshold=200
        )

    def test_low_text_page_is_visual(self) -> None:
        page = MagicMock()
        page.images = []
        assert _page_is_visual(page, "tiny", trigger="image_bearing", low_text_threshold=200)

    def test_embedded_images_make_page_visual(self) -> None:
        page = MagicMock()
        page.images = ["img0"]
        # Text well above the threshold, but the embedded image still wins.
        assert _page_is_visual(page, "x" * 500, trigger="image_bearing", low_text_threshold=200)

    def test_text_page_without_images_is_not_visual(self) -> None:
        page = MagicMock()
        page.images = []
        assert not _page_is_visual(page, "x" * 500, trigger="image_bearing", low_text_threshold=200)

    def test_images_access_error_falls_back_to_text_signal(self) -> None:
        page = MagicMock()
        type(page).images = property(lambda self: (_ for _ in ()).throw(RuntimeError("bad")))
        # Long text + image probe raises → not visual (no crash).
        assert not _page_is_visual(page, "x" * 500, trigger="image_bearing", low_text_threshold=200)


class TestRenderPagePng:
    def test_renders_via_pdfium_and_pillow(self) -> None:
        image = MagicMock()
        image.save = MagicMock(
            side_effect=lambda buf, format: buf.write(b"\x89PNG-bytes")  # noqa: A006
        )
        bitmap = MagicMock()
        bitmap.to_pil = MagicMock(return_value=image)
        page = MagicMock()
        page.render = MagicMock(return_value=bitmap)
        doc = MagicMock()
        doc.__getitem__ = MagicMock(return_value=page)

        out = _render_page_png(doc, 2, dpi=150)
        assert out == b"\x89PNG-bytes"
        doc.__getitem__.assert_called_once_with(2)
        # dpi/72 scale handed to the renderer.
        assert page.render.call_args.kwargs["scale"] == pytest.approx(150 / 72)


class TestOpenPdfiumForVision:
    def test_missing_extra_raises_actionable_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Setting the module to None makes ``import pypdfium2`` raise ImportError.
        monkeypatch.setitem(sys.modules, "pypdfium2", None)
        with pytest.raises(ImportError, match=r"particles\[vision\]"):
            _open_pdfium_for_vision(b"%PDF-1.4 fake")


class TestCallLlmVisionMarker:
    @pytest.mark.asyncio
    async def test_vision_candidates_get_source_modality_marker(self) -> None:
        json_array = '[{"content": "a claim from a figure", "subjects": [], '
        json_array += '"confidence_value": 0.9, "uncertainty_nature": "EPISTEMIC"}]'
        with patch(
            "particles.llm.complete_with_provider_model",
            AsyncMock(return_value=(json_array, "anthropic:test-model")),
        ):
            candidates, _notes, transient = await _call_llm(
                "page text", images=[VisionImage(media_type="image/png", data=b"PNG")]
            )
        assert transient is False
        assert candidates[0].properties is not None
        assert candidates[0].properties[VISION_SOURCE_MODALITY_KEY] == VISION_SOURCE_MODALITY

    @pytest.mark.asyncio
    async def test_text_only_candidates_unmarked(self) -> None:
        json_array = '[{"content": "a plain claim", "subjects": [], '
        json_array += '"confidence_value": 0.9, "uncertainty_nature": "EPISTEMIC"}]'
        with patch(
            "particles.llm.complete_with_provider_model",
            AsyncMock(return_value=(json_array, "anthropic:test-model")),
        ):
            candidates, _notes, _transient = await _call_llm("page text")
        # No images → no marker (and properties may stay None).
        assert not (candidates[0].properties or {}).get(VISION_SOURCE_MODALITY_KEY)


def _vision_config(**overrides: Any) -> ParticlesConfig:
    return ParticlesConfig(extraction_vision=ExtractionVisionConfig(enabled=True, **overrides))


def _fake_page(text: str, images: list[str]) -> MagicMock:
    page = MagicMock()
    page.extract_text = MagicMock(return_value=text)
    page.images = images
    return page


class TestExtractPdfPagedVisionRouting:
    @pytest.mark.asyncio
    async def test_visual_page_calls_multimodal_text_page_does_not(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        extractor = GeneralExtractor()
        # page 1: embedded image → visual; page 2: long text, no image → text-only.
        visual = _fake_page("Figure caption.", ["img0"])
        textual = _fake_page("x" * 500, [])
        fake_reader = MagicMock()
        fake_reader.pages = [visual, textual]

        monkeypatch.setattr("particles.extraction.general.get_config", _vision_config)
        call = AsyncMock(return_value=([_candidate()], [], False))
        with (
            patch("pypdf.PdfReader", return_value=fake_reader),
            patch("particles.extraction.general._open_pdfium_for_vision", return_value=MagicMock()),
            patch("particles.extraction.general._render_page_png", return_value=b"PNG"),
            patch("particles.extraction.general._call_llm", call),
        ):
            await extractor._extract_pdf_paged(b"%PDF-1.4 fake")

        assert call.call_count == 2
        # Page 1 carried a rendered image; page 2 did not.
        assert call.call_args_list[0].kwargs["images"] == [
            VisionImage(media_type="image/png", data=b"PNG")
        ]
        assert call.call_args_list[1].kwargs["images"] is None

    @pytest.mark.asyncio
    async def test_vision_disabled_never_renders(self, monkeypatch: pytest.MonkeyPatch) -> None:
        extractor = GeneralExtractor()
        fake_reader = MagicMock()
        fake_reader.pages = [_fake_page("tiny", ["img0"])]  # would be visual if enabled

        # Default config → extraction_vision.enabled is False.
        monkeypatch.setattr("particles.extraction.general.get_config", ParticlesConfig)
        call = AsyncMock(return_value=([_candidate()], [], False))
        opener = MagicMock()
        with (
            patch("pypdf.PdfReader", return_value=fake_reader),
            patch("particles.extraction.general._open_pdfium_for_vision", opener),
            patch("particles.extraction.general._call_llm", call),
        ):
            await extractor._extract_pdf_paged(b"%PDF-1.4 fake")

        opener.assert_not_called()
        assert call.call_args_list[0].kwargs["images"] is None

    @pytest.mark.asyncio
    async def test_vision_page_cap_falls_back_to_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        extractor = GeneralExtractor()
        fake_reader = MagicMock()
        fake_reader.pages = [_fake_page("Fig A", ["img0"]), _fake_page("Fig B", ["img1"])]

        monkeypatch.setattr(
            "particles.extraction.general.get_config", lambda: _vision_config(max_pages=1)
        )
        call = AsyncMock(return_value=([_candidate()], [], False))
        with (
            patch("pypdf.PdfReader", return_value=fake_reader),
            patch("particles.extraction.general._open_pdfium_for_vision", return_value=MagicMock()),
            patch("particles.extraction.general._render_page_png", return_value=b"PNG"),
            patch("particles.extraction.general._call_llm", call),
        ):
            result = await extractor._extract_pdf_paged(b"%PDF-1.4 fake")

        # First visual page used the budget; the second fell back to text-only.
        assert call.call_args_list[0].kwargs["images"] is not None
        assert call.call_args_list[1].kwargs["images"] is None
        assert any("VISION_PAGE_CAP" in n for n in result.quality_notes)


# ---------------------------------------------------------------------------
# Standalone image deposits
# ---------------------------------------------------------------------------


class TestSniffImageMediaType:
    def test_png(self) -> None:
        assert sniff_image_media_type(_PNG) == "image/png"

    def test_jpeg(self) -> None:
        assert sniff_image_media_type(b"\xff\xd8\xff\xe0" + b"\x00" * 16) == "image/jpeg"

    def test_gif(self) -> None:
        assert sniff_image_media_type(b"GIF89a" + b"\x00" * 16) == "image/gif"

    def test_webp(self) -> None:
        assert sniff_image_media_type(b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 8) == "image/webp"

    def test_non_image_is_none(self) -> None:
        assert sniff_image_media_type(b"hello world, just text") is None
        assert sniff_image_media_type(b"%PDF-1.4 ...") is None


class TestExtractStandaloneImage:
    @pytest.mark.asyncio
    async def test_extract_routes_image_to_one_multimodal_call(self) -> None:
        call = AsyncMock(return_value=([_candidate()], [], False))
        with patch("particles.extraction.general._call_llm", call):
            result = await GeneralExtractor().extract(MagicMock(), _PNG)
        assert call.call_count == 1
        # The raw image bytes are sent as a VisionImage — no rasterization.
        assert call.call_args.kwargs["images"] == [VisionImage(media_type="image/png", data=_PNG)]
        assert len(result.candidates) == 1

    @pytest.mark.asyncio
    async def test_extracted_image_candidates_marked_vision(self) -> None:
        json_array = '[{"content": "a claim from the image", "subjects": [], '
        json_array += '"confidence_value": 0.9, "uncertainty_nature": "EPISTEMIC"}]'
        # Drive the real _call_llm (which stamps the marker) via a mocked complete.
        with patch(
            "particles.llm.complete_with_provider_model",
            AsyncMock(return_value=(json_array, "anthropic:test-model")),
        ):
            result = await GeneralExtractor().extract(Snapshot(content_hash="a" * 64), _PNG)
        assert result.candidates
        props = result.candidates[0].properties or {}
        assert props.get(VISION_SOURCE_MODALITY_KEY) == VISION_SOURCE_MODALITY

    @pytest.mark.asyncio
    async def test_oversized_image_skipped_with_note(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.config import ExtractionConfig, ParticlesConfig

        cfg = ParticlesConfig(extraction=ExtractionConfig(max_image_bytes=10))
        monkeypatch.setattr("particles.extraction.general.get_config", lambda: cfg)
        call = AsyncMock(return_value=([_candidate()], [], False))
        with patch("particles.extraction.general._call_llm", call):
            result = await GeneralExtractor().extract(
                Snapshot(content_hash="a" * 64), _PNG
            )  # 40 bytes > 10
        call.assert_not_called()
        assert any("IMAGE_BYTES_CAP" in n for n in result.quality_notes)
