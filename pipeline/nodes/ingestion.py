"""
pipeline/nodes/ingestion.py

Stage 0 — ensure the pipeline has everything it needs before stage 1 fires.

For the **baseline-vs-candidates** mode (default):
  - Validates that `blueprint_sections` and `baseline_html_compressed` are
    present.
  - If `baseline_screenshot_slices` is empty, rehydrates the baseline HTML
    and re-renders it through Playwright, slicing the resulting PNG.
  - If `baseline_images_b64` is missing, attempts to extract it from the
    HTML if (somehow) raw `data:` URIs are still embedded.
  - Materializes `reference_examples` — for each ReferenceExample whose
    image_path points at an HTML file (or is missing), renders the HTML
    via Playwright and downscales large images. After this node the
    list contains only ready-to-encode PNG paths.
  - Pre-warms the legacy-reference screenshots in parallel via
    `asyncio.to_thread` so the slow Playwright sub-process doesn't block
    the async loop.

For the **legacy multi-reference mode**:
  - Ensures every `example_html_paths` entry has a corresponding screenshot.
"""

from __future__ import annotations

import os
import asyncio
import logging

from pipeline.state import PipelineState, ReferenceExample
from pipeline.utils.playwright_utils import (
    screenshot_html_files,
    render_and_slice_compressed_html,
    render_html_to_png,
)
from pipeline.utils.html_compression import (
    compress_html_base64,
    estimate_compression_savings,
)
from pipeline.utils.image_descriptors import build_descriptors
from pipeline.utils.reference_examples import (
    normalize_reference_examples,
    _downscale_if_needed,
    _is_stale,
    MAX_REFERENCE_PX,
)
from pipeline.utils.storage import run_dir

logger = logging.getLogger(__name__)


async def ingest_examples(state: PipelineState) -> PipelineState:
    out: dict = dict(state)
    layout_width = int(state.get("layout_width", 600) or 600)
    slice_height = int(state.get("slice_height", 1600) or 1600)

    # ── Baseline validation + slice generation ───────────────────────────────
    baseline_html = state.get("baseline_html_compressed", "") or ""
    baseline_images = dict(state.get("baseline_images_b64", {}) or {})
    baseline_slices = list(state.get("baseline_screenshot_slices", []) or [])

    if baseline_html:
        # Defensive: if the caller passed a still-rehydrated HTML (raw data
        # URIs embedded), compress it now and merge into the map.
        if "data:image" in baseline_html and "base64," in baseline_html:
            logger.info("[ingest] baseline HTML still contains raw data URIs — compressing")
            original_size = len(baseline_html)
            compressed, extracted = compress_html_base64(baseline_html)
            baseline_html = compressed
            for token, uri in extracted.items():
                baseline_images.setdefault(token, uri)
            stats = estimate_compression_savings(state["baseline_html_compressed"])
            logger.info(
                "[ingest] baseline compression: %d → %d chars (%d images extracted)",
                stats.get("original_chars", original_size),
                stats.get("compressed_chars", len(baseline_html)),
                stats.get("image_count", len(extracted)),
            )

        if not baseline_slices:
            run_id = state.get("run_id", "default_run")
            out_dir = os.path.join(run_dir(run_id), "baseline")
            os.makedirs(out_dir, exist_ok=True)
            logger.info("[ingest] no baseline_screenshot_slices supplied — rendering + slicing")
            try:
                _full, baseline_slices = await asyncio.to_thread(
                    render_and_slice_compressed_html,
                    compressed_html=baseline_html,
                    images_b64=baseline_images,
                    output_dir=out_dir,
                    base_stem="baseline",
                    viewport_width=layout_width,
                    slice_height=slice_height,
                )
                logger.info("[ingest] baseline produced %d slice(s)", len(baseline_slices))
            except Exception as e:
                logger.warning("[ingest] baseline rendering failed: %s", e)
                baseline_slices = []

        # Mine image descriptors (intrinsic size + HTML usage) so the
        # reverse prompter can reason about how each image was originally
        # used. Pre-existing descriptors (loaded from JSON) take precedence
        # over freshly-mined fields so callers can override anything we'd
        # otherwise infer.
        prior_descriptors = dict(state.get("baseline_image_descriptors", {}) or {})
        descriptors = build_descriptors(
            images_b64=baseline_images,
            compressed_html=baseline_html,
            existing=prior_descriptors,
        )
        referenced = sum(
            1 for d in descriptors.values()
            if (d.get("usage") or {}).get("is_referenced_in_html")
        )
        logger.info(
            "[ingest] mined image descriptors: %d token(s); %d referenced in HTML",
            len(descriptors), referenced,
        )

        out["baseline_html_compressed"] = baseline_html
        out["baseline_images_b64"] = baseline_images
        out["baseline_image_descriptors"] = descriptors
        out["baseline_screenshot_slices"] = baseline_slices

    elif not state.get("example_html_paths"):
        raise ValueError(
            "Pipeline requires either `baseline_html_compressed` (preferred) or "
            "`example_html_paths` (legacy) in initial state."
        )

    # ── Reference creative examples (human-made ads) ────────────────────────
    raw_refs = normalize_reference_examples(state.get("reference_examples", []) or [])
    if raw_refs:
        run_id = state.get("run_id", "default_run")
        ref_cache_dir = os.path.join(run_dir(run_id), "reference_examples")
        os.makedirs(ref_cache_dir, exist_ok=True)
        out["reference_examples"] = await _materialize_references(
            raw_refs,
            cache_dir=ref_cache_dir,
            viewport_width=layout_width,
        )

    # ── Legacy reference examples ───────────────────────────────────────────
    html_paths = state.get("example_html_paths", []) or []
    if html_paths:
        screenshot_paths = await asyncio.to_thread(screenshot_html_files, html_paths)
        out["example_screenshot_paths"] = screenshot_paths

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Reference example materialization (HTML → PNG, downscale large images)
# ─────────────────────────────────────────────────────────────────────────────

async def _materialize_references(
    refs: list[ReferenceExample],
    *,
    cache_dir: str,
    viewport_width: int,
) -> list[ReferenceExample]:
    """Walk the list; for each ref whose image_path is an HTML file (or
    missing) render it through Playwright. Then downscale any oversize
    images. Runs Playwright in a thread so the async loop isn't blocked."""
    if not refs:
        return []

    sem = asyncio.Semaphore(2)  # don't fire >2 chromium instances in parallel

    async def materialize_one(ref: ReferenceExample) -> ReferenceExample | None:
        try:
            path = ref.image_path
            ext = os.path.splitext(path)[1].lower()
            if ext in (".html", ".htm"):
                png_path = os.path.join(cache_dir, f"{ref.name}.png")
                if (not os.path.exists(png_path)) or _is_stale(
                    __import__("pathlib").Path(png_path),
                    __import__("pathlib").Path(path),
                ):
                    logger.info("[ingest] rendering reference %s → %s", path, png_path)
                    async with sem:
                        await asyncio.to_thread(
                            render_html_to_png,
                            html_path=path,
                            output_path=png_path,
                            viewport_width=viewport_width,
                            full_page=True,
                        )
                path = png_path
            elif not os.path.exists(path):
                logger.warning("[ingest] reference image missing on disk: %s", path)
                return None

            # Downscale to keep VLM payload bounded.
            from pathlib import Path
            downscaled = await asyncio.to_thread(
                _downscale_if_needed,
                Path(path),
                cache_dir=Path(cache_dir),
                name=ref.name,
            )
            return ReferenceExample(
                name=ref.name,
                image_path=str(downscaled),
                description=ref.description,
                source_path=ref.source_path or ref.image_path,
            )
        except Exception as e:
            logger.warning("[ingest] reference %s failed to materialize: %s", ref.name, e)
            return None

    results = await asyncio.gather(*(materialize_one(r) for r in refs))
    materialized = [r for r in results if r is not None]
    logger.info(
        "[ingest] reference creative examples: %d input → %d ready (cap=%dpx)",
        len(refs), len(materialized), MAX_REFERENCE_PX,
    )
    return materialized
