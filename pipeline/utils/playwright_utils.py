"""
utils/playwright_utils.py

Headless Chromium screenshot helpers via Playwright **plus** post-render
image slicing via Pillow.

Two flavors of rendering:

  * `render_html_to_png(html_path, output_path)` — the original helper.
    Renders a self-contained HTML *file* on disk.
  * `render_compressed_html_to_png(html, images_b64, output_path)` — accepts a
    compressed HTML string (`[images_base64_i]` tokens) plus the side-car map,
    rehydrates internally, writes the rehydrated HTML to a temp file, and
    renders that. This is what the screenshotter node uses for candidates the
    code-generator returns in compressed form.

Why slicing
-----------
A 600px-wide pharma email is often 4000–10000px tall. VLM providers downsample
large images aggressively which loses small typography and erodes judge
reliability. We slice the full-page screenshot into ~1600px-tall chunks and
hand the VLM the slices in order; the judge prompts say "these images form
one scrollable email read top → bottom".
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional


def render_html_to_png(
    html_path: str,
    output_path: str,
    viewport_width: int = 600,
    viewport_height: int = 900,
    full_page: bool = True,
) -> str:
    """Render a local HTML file to PNG. Returns output_path."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": viewport_width, "height": viewport_height})
        page.goto(f"file://{os.path.abspath(html_path)}")
        page.wait_for_load_state("networkidle")
        page.screenshot(path=output_path, full_page=full_page)
        browser.close()

    return output_path


def render_compressed_html_to_png(
    compressed_html: str,
    images_b64: dict[str, str],
    output_path: str,
    viewport_width: int = 600,
    viewport_height: int = 900,
    full_page: bool = True,
) -> str:
    """
    Rehydrate a compressed HTML string, write it to a temp file, render it.

    Returns the absolute path of the PNG that was produced.
    """
    from pipeline.utils.html_compression import rehydrate_html

    rendered = rehydrate_html(compressed_html, images_b64 or {})
    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(rendered)
        tmp_path = fh.name
    try:
        return render_html_to_png(
            html_path=tmp_path,
            output_path=output_path,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            full_page=full_page,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def screenshot_html_files(
    html_paths: list[str],
    output_dir: str | None = None,
) -> list[str]:
    """Batch screenshot a list of HTML files. Caches per-stem PNGs."""
    png_paths = []
    for html_path in html_paths:
        stem = Path(html_path).stem
        if output_dir:
            png_path = os.path.join(output_dir, f"{stem}.png")
        else:
            png_path = str(Path(html_path).with_suffix(".png"))

        if not os.path.exists(png_path):
            render_html_to_png(html_path, png_path)

        png_paths.append(png_path)

    return png_paths


# ─────────────────────────────────────────────────────────────────────────────
# Image slicing (post-render)
# ─────────────────────────────────────────────────────────────────────────────

def slice_image_vertically(
    png_path: str,
    output_dir: Optional[str] = None,
    slice_height: int = 1600,
    output_stem: Optional[str] = None,
) -> list[str]:
    """
    Slice a tall PNG into horizontal strips of height `slice_height` each.

    Args:
        png_path:     source PNG (full-page screenshot)
        output_dir:   directory for slice files; defaults to png's directory
        slice_height: pixel height per slice; the last slice may be shorter
        output_stem:  filename stem for slices; defaults to source stem +
                      "_slice"

    Returns:
        Ordered list of slice PNG paths (top→bottom).

    Behavior:
        - If the image is shorter than `slice_height`, returns `[png_path]`
          (no slicing performed).
        - Slices are written to `<output_dir>/<stem>_<idx>.png`, 0-indexed.
    """
    from PIL import Image

    src = Path(png_path)
    if output_dir is None:
        output_dir = str(src.parent)
    os.makedirs(output_dir, exist_ok=True)
    stem = output_stem or src.stem

    with Image.open(png_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if h <= slice_height:
            return [png_path]

        slices: list[str] = []
        idx = 0
        for top in range(0, h, slice_height):
            bottom = min(top + slice_height, h)
            crop = img.crop((0, top, w, bottom))
            out = os.path.join(output_dir, f"{stem}_{idx}.png")
            crop.save(out, format="PNG")
            slices.append(out)
            idx += 1
        return slices


def render_and_slice_compressed_html(
    compressed_html: str,
    images_b64: dict[str, str],
    output_dir: str,
    base_stem: str,
    viewport_width: int = 600,
    slice_height: int = 1600,
) -> tuple[str, list[str]]:
    """
    Convenience: rehydrate → render full-page PNG → slice → return both paths.

    Returns:
        (full_page_png_path, [slice_paths...])
    """
    os.makedirs(output_dir, exist_ok=True)
    full_page = os.path.join(output_dir, f"{base_stem}.png")
    render_compressed_html_to_png(
        compressed_html=compressed_html,
        images_b64=images_b64,
        output_path=full_page,
        viewport_width=viewport_width,
    )
    slices = slice_image_vertically(
        png_path=full_page,
        output_dir=output_dir,
        slice_height=slice_height,
        output_stem=f"{base_stem}_slice",
    )
    return full_page, slices
