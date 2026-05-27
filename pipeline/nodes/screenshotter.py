"""
nodes/screenshotter.py

Stage 4 — render each candidate's compressed HTML to a full-page PNG and
slice it into VLM-friendly chunks.

Async: Playwright is synchronous, so each candidate's render runs in a
thread via `asyncio.to_thread`. We gather them with a bounded `Semaphore`
so we don't spawn a Chromium process per candidate at the same time.

Each candidate's HTML may contain `[images_base64_N]` tokens. We rehydrate
them against the baseline image map (the only image source we have for
unknown tokens), then render. Unknown tokens fall back to a 1×1 transparent
PNG so the browser never 404s.

The candidate is updated in-place with:
  - `screenshot_path`        full-page PNG
  - `screenshot_slice_paths` ordered list of slice PNGs
"""

from __future__ import annotations

import os
import shutil
import asyncio
import logging
from typing import Any

import yaml

from pipeline.state import PipelineState, Candidate
from pipeline.utils.html_compression import rehydrate_html
from pipeline.utils.playwright_utils import render_and_slice_compressed_html
from pipeline.utils.storage import run_dir

logger = logging.getLogger(__name__)

_PIPELINE_CONFIG = os.path.join(os.path.dirname(__file__), "../../config/pipeline_config.yaml")
# Default low — Playwright + Chromium is heavy. 2 parallel renders is usually
# the sweet spot for a laptop; bump in pipeline_config.yaml on bigger hosts.
_DEFAULT_SCREENSHOT_CONCURRENCY = 2


async def screenshot_candidates(state: PipelineState) -> PipelineState:
    run_root = run_dir(state.get("run_id", "default_run"))
    out_dir = os.path.join(run_root, "candidates")
    os.makedirs(out_dir, exist_ok=True)
    iteration = int(state.get("iteration", 0) or 0)
    history_dir = os.path.join(run_root, "candidate_history", f"iteration_{iteration:03d}")
    os.makedirs(history_dir, exist_ok=True)

    images_b64 = state.get("baseline_images_b64", {}) or {}
    layout_width = int(state.get("layout_width", 600) or 600)
    slice_height = int(state.get("slice_height", 1600) or 1600)

    candidates = state.get("candidates", []) or []
    if not candidates:
        return {**state, "candidates": []}

    sem = asyncio.Semaphore(_load_concurrency())
    logger.info(
        "[screenshot] rendering %d candidate(s) (concurrency=%d, out=%s)",
        len(candidates), sem._value, out_dir,  # type: ignore[attr-defined]
    )

    tasks = [
        _render_one_candidate(
            candidate=c,
            out_dir=out_dir,
            history_dir=history_dir,
            images_b64=images_b64,
            layout_width=layout_width,
            slice_height=slice_height,
            sem=sem,
        )
        for c in candidates
    ]
    updated: list[Candidate] = await asyncio.gather(*tasks)
    return {**state, "candidates": updated}


async def _render_one_candidate(
    *,
    candidate: Candidate,
    out_dir: str,
    history_dir: str,
    images_b64: dict[str, str],
    layout_width: int,
    slice_height: int,
    sem: asyncio.Semaphore,
) -> Candidate:
    """Render one candidate's HTML in a worker thread, under the screenshot
    semaphore. Returns the updated Candidate (preserving every other field).

    Persisted artifacts (both `candidates/` and `candidate_history/<iter>/`)
    are written with `[images_base64_N]` tokens REHYDRATED to their original
    data URIs so the on-disk HTML/MJML opens directly in a browser without
    a sidecar map. The in-memory candidate keeps the compressed form because
    every downstream node (compliance check, judges) still expects tokens.
    """
    stem = f"candidate_{candidate.candidate_id}"
    rehydrated_html = rehydrate_html(candidate.html, images_b64)
    html_path = os.path.join(out_dir, f"{stem}.html")
    history_html_path = os.path.join(history_dir, f"{stem}.html")
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(rehydrated_html)
        with open(history_html_path, "w", encoding="utf-8") as f:
            f.write(rehydrated_html)
    except OSError as e:
        logger.warning("[screenshot] could not write %s: %s", html_path, e)

    # Save the candidate's MJML alongside the rehydrated HTML so the history
    # captures both the source-of-truth (MJML) and its compiled output. The
    # baseline skips MJML generation, so `mjml_document` is None there.
    if candidate.mjml_document:
        rehydrated_mjml = rehydrate_html(candidate.mjml_document, images_b64)
        mjml_path = os.path.join(out_dir, f"{stem}.mjml")
        history_mjml_path = os.path.join(history_dir, f"{stem}.mjml")
        try:
            with open(mjml_path, "w", encoding="utf-8") as f:
                f.write(rehydrated_mjml)
            with open(history_mjml_path, "w", encoding="utf-8") as f:
                f.write(rehydrated_mjml)
        except OSError as e:
            logger.warning("[screenshot] could not write %s: %s", mjml_path, e)

    async with sem:
        try:
            full_path, slices = await asyncio.to_thread(
                render_and_slice_compressed_html,
                compressed_html=candidate.html,
                images_b64=images_b64,
                output_dir=out_dir,
                base_stem=stem,
                viewport_width=layout_width,
                slice_height=slice_height,
            )
        except Exception as e:
            logger.warning("[screenshot] candidate %d render failed: %s",
                           candidate.candidate_id, e)
            full_path = ""
            slices = []

    _archive_candidate_images(
        history_dir=history_dir,
        full_path=full_path,
        slices=slices,
        candidate_id=candidate.candidate_id,
    )

    return Candidate(**{
        **candidate.__dict__,
        "screenshot_path": full_path or None,
        "screenshot_slice_paths": slices,
    })


def _archive_candidate_images(
    *,
    history_dir: str,
    full_path: str,
    slices: list[str],
    candidate_id: int,
) -> None:
    """Copy rendered candidate PNG artifacts into per-iteration history."""
    for src in [full_path, *slices]:
        if not src:
            continue
        try:
            if not os.path.exists(src):
                logger.warning(
                    "[screenshot] candidate %d artifact missing for history copy: %s",
                    candidate_id, src,
                )
                continue
            shutil.copy2(src, os.path.join(history_dir, os.path.basename(src)))
        except OSError as e:
            logger.warning(
                "[screenshot] candidate %d failed to archive %s to history: %s",
                candidate_id, src, e,
            )


def _load_concurrency() -> int:
    try:
        with open(_PIPELINE_CONFIG) as f:
            cfg = yaml.safe_load(f) or {}
        conc = (cfg.get("concurrency", {}) if isinstance(cfg, dict) else {}) or {}
        return max(1, int(conc.get("screenshotter", _DEFAULT_SCREENSHOT_CONCURRENCY)))
    except Exception:
        return _DEFAULT_SCREENSHOT_CONCURRENCY
