"""
utils/reference_examples.py

Discover + materialize **reference creative examples** (human-made ads) for
the reverse prompter.

A reference directory may contain a flat mix of:

  * ``*.html`` — auto-rendered to PNG via Playwright (full-page, downscaled).
  * ``*.png`` / ``*.jpg`` / ``*.jpeg`` — used as-is (downscaled if huge).
  * Sidecar caption files: ``<stem>.md`` or ``<stem>.txt`` — the file body
    becomes the description shown in the VLM prompt right before the image.
    Falls back to the prettified stem if no sidecar exists.
  * Optional ``manifest.json`` — top-level ``{ "items": [ {name, image|html,
    description}, ... ] }``. When present, it overrides directory discovery
    and lets you order/curate references precisely.

Why downscale
-------------
A human-made reference is **context**, not the subject of analysis. The VLM
needs to "see" the design language at a glance — not pixel-peep. We cap the
longest side at ``MAX_REFERENCE_PX`` (default 1280px) before base64-encoding
so the prompt payload stays small enough to leave room for the baseline +
its slices. Anthropic recommends ~5MB per image; we land well under that.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from pipeline.state import ReferenceExample

logger = logging.getLogger(__name__)

# Anything larger than this on either side gets thumb-nailed before encoding.
# 1280px is plenty for the VLM to grok a layout while keeping the payload small.
MAX_REFERENCE_PX = 1280

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_HTML_SUFFIXES = {".html", ".htm"}
_CAPTION_SUFFIXES = (".md", ".txt")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def discover_reference_examples(
    directory: str | Path | None,
    *,
    render_cache_dir: str | Path | None = None,  # kept for API compat; unused here
    viewport_width: int = 600,                    # kept for API compat; unused here
) -> list[ReferenceExample]:
    """Walk ``directory`` and return discovered references (not yet rendered).

    This is a **discovery-only** step: it finds files, reads sidecar captions,
    and returns ``ReferenceExample`` stubs whose ``image_path`` points at the
    on-disk source (``.html`` or image). HTML rendering and downscaling happen
    later in the ingestion node via ``asyncio.to_thread`` — Playwright's sync
    API cannot run inside an asyncio event loop.

    If a ``manifest.json`` is present in ``directory``, only the items listed
    there (in order) are returned.

    Returns an empty list if ``directory`` is None, missing, or empty —
    the pipeline always treats references as optional.
    """
    del render_cache_dir, viewport_width  # materialization deferred to ingestion

    if directory is None:
        return []
    src_dir = Path(directory)
    if not src_dir.exists() or not src_dir.is_dir():
        return []

    manifest_path = src_dir / "manifest.json"
    if manifest_path.exists():
        items = _load_manifest(manifest_path, src_dir)
    else:
        items = _discover_directory(src_dir)

    discovered: list[ReferenceExample] = []
    for item in items:
        try:
            ref = _reference_from_item(item)
        except Exception as e:
            logger.warning(
                "[references] failed to load %s (%s) — skipping: %s",
                item.get("name", "?"), item.get("source_path", "?"), e,
            )
            continue
        if ref is not None:
            discovered.append(ref)
    return discovered


def normalize_reference_examples(items: Iterable) -> list[ReferenceExample]:
    """Accept a heterogeneous list (dicts or ReferenceExample) and normalize.

    Used by the LangGraph runner to accept either typed dataclasses or
    serialized state dicts.
    """
    out: list[ReferenceExample] = []
    for it in items or []:
        if isinstance(it, ReferenceExample):
            out.append(it)
        elif isinstance(it, dict):
            out.append(
                ReferenceExample(
                    name=str(it.get("name", "") or ""),
                    image_path=str(it.get("image_path", "") or ""),
                    description=str(it.get("description", "") or ""),
                    source_path=str(it.get("source_path", "") or ""),
                )
            )
        else:
            raise TypeError(
                f"reference_examples entries must be ReferenceExample or dict, got {type(it).__name__}"
            )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────

def _load_manifest(manifest_path: Path, src_dir: Path) -> list[dict]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    items_raw = raw.get("items") if isinstance(raw, dict) else raw
    if not isinstance(items_raw, list):
        logger.warning("[references] manifest at %s has no 'items' list", manifest_path)
        return []

    items: list[dict] = []
    for entry in items_raw:
        if not isinstance(entry, dict):
            continue
        rel = entry.get("image") or entry.get("html") or entry.get("path")
        if not rel:
            continue
        src = (src_dir / rel).resolve()
        if not src.exists():
            logger.warning("[references] manifest entry missing on disk: %s", src)
            continue
        items.append(
            {
                "name": str(entry.get("name") or src.stem),
                "source_path": str(src),
                "kind": _classify(src),
                "description": str(entry.get("description") or "").strip(),
            }
        )
    return items


def _discover_directory(src_dir: Path) -> list[dict]:
    """Auto-discovery: every .html/.png/.jpg/.jpeg in the directory becomes one
    reference. Sidecar .md/.txt files become captions. Order: alphabetical."""
    candidates: list[dict] = []
    for path in sorted(src_dir.iterdir()):
        if path.is_dir():
            continue
        kind = _classify(path)
        if kind == "skip":
            continue
        description = _read_sidecar_caption(path) or _prettify_stem(path.stem)
        candidates.append(
            {
                "name": path.stem,
                "source_path": str(path),
                "kind": kind,
                "description": description,
            }
        )
    return candidates


def _classify(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in _IMAGE_SUFFIXES:
        return "image"
    if suf in _HTML_SUFFIXES:
        return "html"
    return "skip"


def _read_sidecar_caption(path: Path) -> str | None:
    for ext in _CAPTION_SUFFIXES:
        sidecar = path.with_suffix(ext)
        if sidecar.exists() and sidecar.is_file():
            text = sidecar.read_text(encoding="utf-8").strip()
            if text:
                return text
    return None


def _prettify_stem(stem: str) -> str:
    return stem.replace("_", " ").replace("-", " ").strip().title()


# ─────────────────────────────────────────────────────────────────────────────
# Discovery → ReferenceExample stub (render/downscale deferred to ingestion)
# ─────────────────────────────────────────────────────────────────────────────

def _reference_from_item(item: dict) -> ReferenceExample | None:
    """Build a ReferenceExample stub from a discovered item.

    ``image_path`` is the source file on disk (HTML or image). The ingestion
    node renders HTML → PNG and downscales oversize images in a worker thread.
    """
    name = item["name"]
    source_path = item["source_path"]
    kind = item["kind"]
    description = item.get("description", "") or _prettify_stem(name)

    if kind in ("html", "image"):
        return ReferenceExample(
            name=name,
            image_path=source_path,
            description=description,
            source_path=source_path,
        )

    return None


def _is_stale(rendered: Path, src: Path) -> bool:
    try:
        return rendered.stat().st_mtime < src.stat().st_mtime
    except OSError:
        return True


def _downscale_if_needed(image_path: Path, *, cache_dir: Path, name: str) -> Path:
    """If the image exceeds MAX_REFERENCE_PX on either side, write a downscaled
    copy under cache_dir and return its path. Otherwise return image_path."""
    try:
        from PIL import Image
    except Exception:
        logger.warning("[references] Pillow unavailable — sending %s at full resolution", image_path)
        return image_path

    try:
        with Image.open(image_path) as img:
            w, h = img.size
            longest = max(w, h)
            if longest <= MAX_REFERENCE_PX:
                return image_path
            scale = MAX_REFERENCE_PX / float(longest)
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            out = cache_dir / f"{name}__thumb.png"
            resized = img.convert("RGB").resize(new_size, Image.LANCZOS)
            resized.save(out, format="PNG", optimize=True)
            logger.info(
                "[references] downscaled %s from %dx%d → %dx%d (cap=%dpx)",
                image_path.name, w, h, new_size[0], new_size[1], MAX_REFERENCE_PX,
            )
            return out
    except Exception as e:
        logger.warning("[references] downscale failed for %s (%s) — using original", image_path, e)
        return image_path


# ─────────────────────────────────────────────────────────────────────────────
# Serialization (state.py round-trip)
# ─────────────────────────────────────────────────────────────────────────────

def to_serializable(refs: Iterable[ReferenceExample]) -> list[dict]:
    """Pickle/JSON-friendly representation, useful for run artifact dumps."""
    return [asdict(r) for r in refs or []]
