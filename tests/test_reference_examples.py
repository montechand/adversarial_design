"""
tests/test_reference_examples.py

Validates the human-made creative reference loader, the prompt-interleaving
helper, and the reverse-prompter encoder.

We do NOT call Playwright here (no chromium in CI). HTML-only references
exercise the discovery + sidecar-caption + manifest branches; PNG references
exercise the downscale path. Encoding tests use synthetic 1x1 PNGs so they
work offline.
"""

from __future__ import annotations

import base64
import json
import struct
import zlib
from pathlib import Path
from typing import Iterable

import pytest

from pipeline.state import ReferenceExample
from pipeline.utils.reference_examples import (
    MAX_REFERENCE_PX,
    discover_reference_examples,
    normalize_reference_examples,
    to_serializable,
)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_solid_png(path: Path, width: int = 8, height: int = 8, rgb=(255, 0, 0)) -> Path:
    """Write a tiny PNG with the given solid color. Avoids the Pillow dependency
    for the basic discovery tests so they run even on a slim CI image."""
    r, g, b = rgb

    def _chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    raw = b""
    row = bytes([r, g, b]) * width
    for _ in range(height):
        raw += b"\x00" + row
    idat = zlib.compress(raw, 9)
    path.write_bytes(sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b""))
    return path


def _make_big_png_with_pillow(path: Path, w: int, h: int) -> Path:
    pytest.importorskip("PIL")
    from PIL import Image
    img = Image.new("RGB", (w, h), color=(64, 128, 200))
    img.save(path, format="PNG")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# discover_reference_examples
# ─────────────────────────────────────────────────────────────────────────────

class TestDiscover:
    def test_missing_directory_returns_empty(self, tmp_path):
        assert discover_reference_examples(tmp_path / "does_not_exist") == []

    def test_none_returns_empty(self):
        assert discover_reference_examples(None) == []

    def test_basic_png_discovery(self, tmp_path):
        _make_solid_png(tmp_path / "ref_alpha.png")
        _make_solid_png(tmp_path / "ref_beta.png")
        refs = discover_reference_examples(tmp_path)
        assert len(refs) == 2
        names = [r.name for r in refs]
        assert names == sorted(names), "results should be alphabetically ordered"
        assert all(Path(r.image_path).exists() for r in refs)
        # default caption = prettified stem
        assert refs[0].description == "Ref Alpha"

    def test_sidecar_md_caption(self, tmp_path):
        _make_solid_png(tmp_path / "pfizer_xeljanz.png")
        (tmp_path / "pfizer_xeljanz.md").write_text(
            "Pfizer XELJANZ HCP email — asymmetric grid, oversized type.\n",
            encoding="utf-8",
        )
        refs = discover_reference_examples(tmp_path)
        assert len(refs) == 1
        assert "asymmetric grid" in refs[0].description

    def test_sidecar_txt_caption(self, tmp_path):
        _make_solid_png(tmp_path / "ad1.png")
        (tmp_path / "ad1.txt").write_text("Caption from txt.", encoding="utf-8")
        refs = discover_reference_examples(tmp_path)
        assert refs[0].description == "Caption from txt."

    def test_sidecar_files_are_not_treated_as_references(self, tmp_path):
        """`.md` / `.txt` files must NOT show up as their own references."""
        _make_solid_png(tmp_path / "img.png")
        (tmp_path / "img.md").write_text("a caption", encoding="utf-8")
        (tmp_path / "loose_notes.md").write_text("not a caption for anything", encoding="utf-8")
        refs = discover_reference_examples(tmp_path)
        assert {r.name for r in refs} == {"img"}

    def test_manifest_overrides_discovery(self, tmp_path):
        _make_solid_png(tmp_path / "a.png")
        _make_solid_png(tmp_path / "b.png")
        _make_solid_png(tmp_path / "c.png")   # NOT in manifest → excluded
        manifest = {
            "items": [
                {"name": "Bee", "image": "b.png", "description": "b first"},
                {"name": "Ay",  "image": "a.png", "description": "a second"},
            ]
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        refs = discover_reference_examples(tmp_path)
        assert [r.name for r in refs] == ["Bee", "Ay"]
        assert refs[0].description == "b first"

    def test_manifest_skips_missing_files(self, tmp_path):
        _make_solid_png(tmp_path / "exists.png")
        manifest = {
            "items": [
                {"name": "Real", "image": "exists.png"},
                {"name": "Phantom", "image": "missing.png"},
            ]
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        refs = discover_reference_examples(tmp_path)
        assert [r.name for r in refs] == ["Real"]

    def test_downscaling_triggers_above_cap(self, tmp_path):
        pytest.importorskip("PIL")
        from pipeline.utils.reference_examples import _downscale_if_needed
        cache = tmp_path / ".cache"
        cache.mkdir()
        big = _make_big_png_with_pillow(
            tmp_path / "huge.png", w=MAX_REFERENCE_PX * 2, h=400
        )
        out = _downscale_if_needed(big, cache_dir=cache, name="huge")
        assert out != big
        from PIL import Image
        with Image.open(out) as img:
            assert max(img.size) == MAX_REFERENCE_PX

    def test_small_images_pass_through_unchanged(self, tmp_path):
        from pipeline.utils.reference_examples import _downscale_if_needed
        small = _make_solid_png(tmp_path / "small.png", width=10, height=10)
        out = _downscale_if_needed(small, cache_dir=tmp_path / ".cache", name="small")
        assert out == small

    def test_html_discovery_returns_html_path_not_png(self, tmp_path):
        """Discovery defers Playwright rendering — image_path stays the .html."""
        html = tmp_path / "ref.html"
        html.write_text("<html><body><p>ref</p></body></html>", encoding="utf-8")
        refs = discover_reference_examples(tmp_path)
        assert len(refs) == 1
        assert refs[0].image_path.endswith(".html")
        assert refs[0].source_path == str(html)


# ─────────────────────────────────────────────────────────────────────────────
# normalize_reference_examples
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalize:
    def test_dataclass_passthrough(self, tmp_path):
        ref = ReferenceExample(name="x", image_path=str(tmp_path / "x.png"), description="d")
        out = normalize_reference_examples([ref])
        assert out == [ref]

    def test_dict_conversion(self, tmp_path):
        out = normalize_reference_examples(
            [{"name": "x", "image_path": "/p/x.png", "description": "d", "source_path": "/p/x.html"}]
        )
        assert out == [
            ReferenceExample(
                name="x", image_path="/p/x.png", description="d", source_path="/p/x.html"
            )
        ]

    def test_mixed_passthrough(self, tmp_path):
        ref = ReferenceExample(name="a", image_path="/a.png", description="")
        out = normalize_reference_examples(
            [ref, {"name": "b", "image_path": "/b.png"}]
        )
        assert [r.name for r in out] == ["a", "b"]

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            normalize_reference_examples(["not a ref"])

    def test_none_and_empty_returns_empty(self):
        assert normalize_reference_examples(None) == []
        assert normalize_reference_examples([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# to_serializable
# ─────────────────────────────────────────────────────────────────────────────

def test_to_serializable_round_trip():
    refs = [
        ReferenceExample(name="a", image_path="/p/a.png", description="ad"),
        ReferenceExample(name="b", image_path="/p/b.png", description="bd"),
    ]
    rows = to_serializable(refs)
    assert isinstance(rows, list)
    assert all(isinstance(r, dict) for r in rows)
    assert rows[0]["name"] == "a"
    assert rows[1]["description"] == "bd"
    again = normalize_reference_examples(rows)
    assert again == refs


# ─────────────────────────────────────────────────────────────────────────────
# build_user_prompt — interleaving of reference examples
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildUserPromptInterleave:
    def _slice(self) -> tuple[str, str]:
        return ("image/png", base64.b64encode(b"slice").decode("ascii"))

    def _ref(self, name: str, desc: str) -> tuple[str, str, str, str]:
        return (name, desc, "image/png", base64.b64encode(b"ref:" + name.encode()).decode("ascii"))

    def test_no_refs_yields_no_reference_block(self):
        from pipeline.prompts.reverse_prompter import build_user_prompt
        content = build_user_prompt(
            source="baseline",
            example_id="baseline",
            blueprint_summary="(empty)",
            html_source_compressed="<html></html>",
            screenshot_slices_b64=[self._slice()],
        )
        joined = "\n".join(b.get("text", "") for b in content if b["type"] == "text")
        assert "REFERENCE CREATIVE EXAMPLES" not in joined

    def test_refs_interleave_with_captions(self):
        from pipeline.prompts.reverse_prompter import build_user_prompt
        refs = [
            self._ref("pfizer_x", "Pfizer X — asymmetric grid"),
            self._ref("lilly_y",  "Lilly Y — editorial"),
        ]
        content = build_user_prompt(
            source="baseline",
            example_id="baseline",
            blueprint_summary="(empty)",
            html_source_compressed="<html></html>",
            screenshot_slices_b64=[self._slice()],
            reference_examples_b64=refs,
        )
        # Count text-then-image pairs for the references: 1 header + 2 (caption, image) pairs
        # Find the reference header.
        header_idx = next(
            i for i, b in enumerate(content)
            if b["type"] == "text" and "REFERENCE CREATIVE EXAMPLES" in b.get("text", "")
        )
        # Right after the header we expect: text(caption1), image(1), text(caption2), image(2)
        assert content[header_idx + 1]["type"] == "text"
        assert "pfizer_x" in content[header_idx + 1]["text"]
        assert content[header_idx + 2]["type"] == "image"
        assert content[header_idx + 3]["type"] == "text"
        assert "lilly_y" in content[header_idx + 3]["text"]
        assert content[header_idx + 4]["type"] == "image"

    def test_refs_appear_AFTER_baseline_slices(self):
        from pipeline.prompts.reverse_prompter import build_user_prompt
        refs = [self._ref("only_ref", "the only ref")]
        slices = [self._slice(), self._slice()]
        content = build_user_prompt(
            source="baseline",
            example_id="baseline",
            blueprint_summary="(empty)",
            html_source_compressed="<html></html>",
            screenshot_slices_b64=slices,
            reference_examples_b64=refs,
        )
        # Find indices of: last baseline image, the references header
        baseline_slice_indices = [
            i for i, b in enumerate(content)
            if b["type"] == "text" and "[baseline slice" in b.get("text", "")
        ]
        ref_header_idx = next(
            i for i, b in enumerate(content)
            if b["type"] == "text" and "REFERENCE CREATIVE EXAMPLES" in b.get("text", "")
        )
        assert baseline_slice_indices[-1] < ref_header_idx, \
            "reference gallery must come AFTER all baseline slices"

    def test_html_assets_interleave_with_token_labels(self):
        from pipeline.prompts.reverse_prompter import build_user_prompt
        html_assets = [
            ("[images_base64_0]", "image/png", base64.b64encode(b"asset0").decode("ascii")),
            ("[images_base64_1]", "image/png", base64.b64encode(b"asset1").decode("ascii")),
        ]
        content = build_user_prompt(
            source="baseline",
            example_id="baseline",
            blueprint_summary="(empty)",
            html_source_compressed="<img src='[images_base64_0]'><img src='[images_base64_1]'>",
            screenshot_slices_b64=[self._slice()],
            html_images_b64=html_assets,
        )
        html_header_idx = next(
            i for i, b in enumerate(content)
            if b["type"] == "text" and "HTML PLACEHOLDER IMAGE ASSETS" in b.get("text", "")
        )
        assert content[html_header_idx + 1]["type"] == "text"
        assert "[images_base64_0]" in content[html_header_idx + 1]["text"]
        assert content[html_header_idx + 2]["type"] == "image"
        assert content[html_header_idx + 3]["type"] == "text"
        assert "[images_base64_1]" in content[html_header_idx + 3]["text"]
        assert content[html_header_idx + 4]["type"] == "image"

    def test_html_assets_appear_before_reference_examples(self):
        from pipeline.prompts.reverse_prompter import build_user_prompt
        refs = [self._ref("r1", "reference")]
        html_assets = [
            ("[images_base64_0]", "image/png", base64.b64encode(b"asset0").decode("ascii")),
        ]
        content = build_user_prompt(
            source="baseline",
            example_id="baseline",
            blueprint_summary="(empty)",
            html_source_compressed="<img src='[images_base64_0]'>",
            screenshot_slices_b64=[self._slice()],
            html_images_b64=html_assets,
            reference_examples_b64=refs,
        )
        html_header_idx = next(
            i for i, b in enumerate(content)
            if b["type"] == "text" and "HTML PLACEHOLDER IMAGE ASSETS" in b.get("text", "")
        )
        ref_header_idx = next(
            i for i, b in enumerate(content)
            if b["type"] == "text" and "REFERENCE CREATIVE EXAMPLES" in b.get("text", "")
        )
        assert html_header_idx < ref_header_idx

    def test_trailing_instruction_changes_when_refs_present(self):
        from pipeline.prompts.reverse_prompter import build_user_prompt
        with_refs = build_user_prompt(
            source="baseline",
            example_id="baseline",
            blueprint_summary="(empty)",
            html_source_compressed="<html></html>",
            screenshot_slices_b64=[self._slice()],
            reference_examples_b64=[self._ref("r1", "first")],
        )
        without_refs = build_user_prompt(
            source="baseline",
            example_id="baseline",
            blueprint_summary="(empty)",
            html_source_compressed="<html></html>",
            screenshot_slices_b64=[self._slice()],
        )
        last_with = with_refs[-1]["text"]
        last_without = without_refs[-1]["text"]
        assert "reference_patterns_to_emulate" in last_with.lower()
        assert "reference" not in last_without.lower()
