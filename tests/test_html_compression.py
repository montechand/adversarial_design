"""
tests/test_html_compression.py

Smoke tests for the base64 ⇄ placeholder bridge that the pipeline relies on
for cheap LLM text calls.

Run:
    cd adversarial_design
    PYTHONPATH=. pytest tests/test_html_compression.py -v
"""

from __future__ import annotations
import json
from pathlib import Path

import pytest

from pipeline.utils.html_compression import (
    compress_html_base64,
    rehydrate_html,
    find_placeholders,
    placeholder_tokens,
    load_images_b64_json,
    PLACEHOLDER_REGEX,
    TRANSPARENT_PNG_DATA_URI,
)
from pipeline.state import BlueprintSection


# ─────────────────────────────────────────────────────────────────────────────
# compression / rehydration round-trip
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_HTML = """\
<html><body>
  <img src="data:image/png;base64,AAAAAA==" alt="a">
  <img src='data:image/jpeg;base64,BBBBBB=='>
  <div style="background:url('data:image/png;base64,CCCCCC==')"></div>
  <img src="data:image/png;base64,AAAAAA==" alt="dup">
</body></html>
"""


def test_compress_replaces_each_unique_uri_with_token():
    compressed, images = compress_html_base64(SAMPLE_HTML)
    assert "data:image/png;base64,AAAAAA==" not in compressed
    assert "data:image/jpeg;base64,BBBBBB==" not in compressed
    assert "data:image/png;base64,CCCCCC==" not in compressed
    # 3 unique URIs (the duplicate "AAAAAA==" gets the same token)
    assert len(images) == 3


def test_compress_deduplicates_identical_uris():
    compressed, images = compress_html_base64(SAMPLE_HTML)
    tokens = find_placeholders(compressed)
    # 4 image references in source, but only 3 unique → 4 token occurrences but
    # token #0 should appear twice
    assert len(tokens) == 4
    assert tokens.count("0") == 2  # find_placeholders returns the captured digits


def test_round_trip_is_lossless():
    compressed, images = compress_html_base64(SAMPLE_HTML)
    restored = rehydrate_html(compressed, images)
    assert restored == SAMPLE_HTML


def test_rehydrate_unknown_token_uses_transparent_png_fallback():
    compressed = "<img src='[images_base64_42]'>"
    restored = rehydrate_html(compressed, {})
    assert TRANSPARENT_PNG_DATA_URI in restored
    assert "[images_base64_42]" not in restored


def test_placeholder_regex_only_matches_our_format():
    text = "ok [images_base64_0] [images_base64_99] bad [images_base64] also [Images_Base64_1]"
    matches = PLACEHOLDER_REGEX.findall(text)
    assert matches == ["0", "99"]


def test_rehydrate_leaves_tokens_in_html_comments_alone():
    """LLMs sometimes name tokens inside `<!-- ASSET CHECK: [images_base64_0]
    must be the gMG hero -->` style commentary. Inlining a megabyte of base64
    into a comment is never what we want — rehydration must scope itself to
    real image-loading contexts (img/source src/srcset, CSS url())."""
    images = {
        "[images_base64_0]": "data:image/png;base64,REAL0",
        "[images_base64_1]": "data:image/png;base64,REAL1",
    }
    html = (
        "<!-- ASSET CHECK: [images_base64_0] must be the gMG hero asset -->"
        '<img src="[images_base64_0]" alt="hero">'
        "<p>This token in prose stays literal: [images_base64_1]</p>"
    )
    restored = rehydrate_html(html, images)
    # The comment text and prose text must be left literally as written.
    assert "ASSET CHECK: [images_base64_0]" in restored
    assert "literal: [images_base64_1]" in restored
    # The img src must still be rehydrated.
    assert 'src="data:image/png;base64,REAL0"' in restored


def test_rehydrate_handles_srcset_source_and_css_url():
    images = {
        "[images_base64_0]": "data:image/png;base64,SRC0",
        "[images_base64_1]": "data:image/png;base64,SET1",
        "[images_base64_2]": "data:image/png;base64,CSS2",
    }
    html = (
        '<img src="[images_base64_0]" srcset="[images_base64_1] 2x">'
        "<source src='[images_base64_0]' />"
        "<div style=\"background:url('[images_base64_2]')\"></div>"
    )
    restored = rehydrate_html(html, images)
    assert 'src="data:image/png;base64,SRC0"' in restored
    assert 'srcset="data:image/png;base64,SET1 2x"' in restored
    assert "src='data:image/png;base64,SRC0'" in restored
    assert "url('data:image/png;base64,CSS2')" in restored


def test_placeholder_tokens_returns_in_numeric_order():
    images = {
        "[images_base64_10]": "data:image/png;base64,zzz",
        "[images_base64_2]": "data:image/png;base64,bbb",
        "[images_base64_0]": "data:image/png;base64,aaa",
    }
    assert placeholder_tokens(images) == [
        "[images_base64_0]",
        "[images_base64_2]",
        "[images_base64_10]",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Example assets
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_example_blueprint_parses_to_section_specs():
    raw = json.loads((EXAMPLES / "blueprint.json").read_text())
    sections_raw = raw["content_blueprint"]
    sections = [BlueprintSection.from_dict(s) for s in sections_raw]
    assert len(sections) >= 5, f"blueprint has only {len(sections)} sections"
    # Sanity: every section has a non-empty id and an `order` that's an int
    for s in sections:
        assert s.section_id, "blueprint section missing section_id"
        assert isinstance(s.order, int), f"{s.section_id} order is not int"
    # The `order` field should be unique and renderable as a sorted sequence
    orders = sorted(s.order for s in sections)
    assert orders == list(range(orders[0], orders[0] + len(sections))) or \
           len(set(orders)) == len(orders), "section orders are not unique"


def test_example_baseline_html_loads():
    """The bundled baseline HTML may be either pre-compressed (with
    `[images_base64_N]` tokens) OR raw (with `data:image` URIs). The pipeline
    accepts both — ingestion compresses raw HTML on the fly."""
    html = (EXAMPLES / "baseline_email.html").read_text()
    assert html.startswith("<!DOCTYPE html") or html.startswith("<html")
    # Should look like a real email (>= 1KB, contains a <body>)
    assert len(html) >= 1024
    assert "<body" in html.lower()


def test_example_images_b64_map_loads_or_is_absent():
    """If the example ships an image map, every URI in it must be a data URI
    and the file must parse as JSON. If it's absent, that's also fine — the
    HTML doesn't necessarily reference placeholders."""
    image_map_path = EXAMPLES / "baseline_images_b64.json"
    if not image_map_path.exists():
        pytest.skip("no example image map shipped")
    images = load_images_b64_json(image_map_path)
    assert isinstance(images, dict)
    for token, uri in images.items():
        assert token.startswith("[images_base64_") and token.endswith("]"), \
            f"unexpected token format: {token!r}"
        assert uri.startswith("data:image/"), \
            f"{token!r} is not a data URI: {uri[:60]}"


def test_example_round_trip_or_compress_first():
    """If the example HTML uses placeholders, rehydration must restore data
    URIs. If it uses raw data: URIs, compression-then-rehydration must be
    lossless. We accept either shape."""
    html = (EXAMPLES / "baseline_email.html").read_text()
    image_map_path = EXAMPLES / "baseline_images_b64.json"

    if PLACEHOLDER_REGEX.search(html):
        # Pre-compressed flavor
        assert image_map_path.exists(), "html uses placeholders but no map shipped"
        images = load_images_b64_json(image_map_path)
        restored = rehydrate_html(html, images)
        assert not PLACEHOLDER_REGEX.search(restored)
    else:
        # Raw flavor — compress then rehydrate
        from pipeline.utils.html_compression import compress_html_base64
        compressed, extracted = compress_html_base64(html)
        if not extracted:
            pytest.skip("example HTML has no embedded base64 images to round-trip")
        restored = rehydrate_html(compressed, extracted)
        assert restored == html, "compress→rehydrate is not lossless"
