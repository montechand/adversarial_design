"""
nodes/reverse_prompter.py

Stage 1 — extract structured design intent from the BASELINE design, using
optional human-made creative reference ads as comparison anchors.

Inputs (from PipelineState):
  - blueprint_sections          → human-readable blueprint summary
  - baseline_html_compressed    → HTML with `[images_base64_N]` placeholders
  - baseline_screenshot_slices  → ordered PNG slices (top→bottom)
  - reference_examples          → list[ReferenceExample] — human-made ads,
                                  each rendered to a single PNG with a
                                  description caption. Interleaved with their
                                  captions into the SAME VLM call as the
                                  baseline so the analyzer can use them as a
                                  comparison anchor when listing the
                                  baseline's weaknesses.

The legacy ``example_html_paths`` slot is still honored for backward
compatibility: those entries still get their own descriptor (one VLM call
each) so existing flows do not break. New code should populate
``reference_examples`` instead.
"""

from __future__ import annotations

import os
import re
import json
import asyncio
import logging
from dataclasses import asdict
from typing import Any, Iterable

import yaml

from pipeline.state import (
    PipelineState,
    DesignDescriptor,
    BlueprintSection,
    ReferenceExample,
)
from pipeline.utils.image_utils import encode_image_b64
from pipeline.utils.llm_clients import get_async_router
from pipeline.utils.html_compression import (
    compress_html_base64,
    placeholder_tokens,
    split_data_uri,
)
from pipeline.utils.image_descriptors import (
    build_descriptors,
    format_descriptor_for_prompt,
)
from pipeline.utils.storage import run_dir
from pipeline.prompts.reverse_prompter import (
    SYSTEM_PROMPT,
    build_user_prompt,
    build_blueprint_summary,
)
from pipeline.prompts.creativizer import (
    summarize_descriptor,
    summarize_reference_descriptor,
)

logger = logging.getLogger(__name__)

# Provider/model is resolved at runtime from `config/pipeline_config.yaml >
# models.reverse_prompter`. The default below is used only if the config
# row is missing or unparseable.
_DEFAULT_MODEL = "claude-opus-4-6"
_MAX_TOKENS = 8192  # fallback if config is missing

_PIPELINE_CONFIG = os.path.join(os.path.dirname(__file__), "../../config/pipeline_config.yaml")
_DEFAULT_REVERSE_CONCURRENCY = 4
_DEFAULT_MAX_REFERENCES = 6   # cap to keep the VLM payload reasonable


async def reverse_prompt(state: PipelineState) -> PipelineState:
    router = get_async_router("reverse_prompter", default=_DEFAULT_MODEL)

    blueprint_sections = state.get("blueprint_sections", []) or []
    blueprint_summary = build_blueprint_summary(
        [_section_to_dict(s) for s in blueprint_sections]
    )

    # Encode the human-made reference creative ads once; they're reused
    # verbatim across every VLM call this node makes.
    max_refs = _load_max_references()
    raw_refs: list[ReferenceExample] = list(state.get("reference_examples", []) or [])
    if len(raw_refs) > max_refs:
        logger.info(
            "[reverse_prompt] capping reference_examples %d → %d (config: max_references)",
            len(raw_refs), max_refs,
        )
        raw_refs = raw_refs[:max_refs]
    reference_examples_b64 = _encode_reference_examples(raw_refs)
    if reference_examples_b64:
        logger.info(
            "[reverse_prompt] using %d reference creative example(s) as comparison anchors",
            len(reference_examples_b64),
        )

    # Collect inputs to describe (baseline + each legacy reference HTML)
    jobs: list[dict[str, Any]] = []

    baseline_html = state.get("baseline_html_compressed", "") or ""
    baseline_slices = state.get("baseline_screenshot_slices", []) or []
    baseline_images_map = dict(state.get("baseline_images_b64", {}) or {})
    baseline_html_images = _encode_html_placeholder_images(baseline_images_map)
    # Descriptors flow through state; fall back to mining at runtime if
    # nothing was attached (defensive — ingestion should have populated this).
    baseline_descriptors = dict(state.get("baseline_image_descriptors", {}) or {})
    if baseline_images_map and not baseline_descriptors:
        baseline_descriptors = build_descriptors(
            images_b64=baseline_images_map,
            compressed_html=baseline_html,
        )
    if baseline_html and baseline_slices:
        # Attach previous-winner feedback on iterations > 0, if the toggle is on.
        iteration = int(state.get("iteration", 0) or 0)
        winner_feedback: str | None = None
        if iteration > 0 and _load_winner_feedback_enabled():
            winner_feedback = state.get("winner_feedback") or None
            if winner_feedback:
                logger.info(
                    "[reverse_prompt] attaching previous-winner judge feedback "
                    "to baseline prompt (iteration=%d, len=%d)",
                    iteration, len(winner_feedback),
                )
        jobs.append({
            "source": "baseline",
            "example_id": "baseline",
            "html_source_compressed": baseline_html,
            "screenshot_slices_b64": _encode_slices(baseline_slices),
            "html_images_b64": baseline_html_images,
            "html_image_descriptors": baseline_descriptors,
            # Only the baseline call receives the interleaved references —
            # legacy per-reference descriptors should describe themselves,
            # not be cross-contaminated with other refs.
            "reference_examples_b64": reference_examples_b64,
            "winner_feedback": winner_feedback,
        })

    ref_html_paths = state.get("example_html_paths", []) or []
    ref_screenshot_paths = state.get("example_screenshot_paths", []) or []
    for i, (html_path, png_path) in enumerate(zip(ref_html_paths, ref_screenshot_paths)):
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                html_source = f.read()
            compressed, _imgs = compress_html_base64(html_source)
            jobs.append({
                "source": "reference",
                "example_id": f"ref_{i}",
                "html_source_compressed": compressed,
                "screenshot_slices_b64": _encode_slices([png_path]),
                "html_images_b64": [],
                "html_image_descriptors": {},
                "reference_examples_b64": [],   # legacy refs do not get cross-refs
            })
        except Exception as e:
            logger.warning("[reverse_prompt] reference %s failed to load: %s", html_path, e)

    # Human reference gallery — each example gets its own descriptor so the
    # creativizer receives full design vocabulary, not just baseline comparisons.
    for ref in raw_refs:
        html_compressed = "<!-- reference: screenshot-only; no HTML available -->"
        source_path = (ref.source_path or ref.image_path or "").strip()
        if source_path.lower().endswith((".html", ".htm")):
            try:
                with open(source_path, "r", encoding="utf-8") as f:
                    html_compressed, _imgs = compress_html_base64(f.read())
            except Exception as e:
                logger.warning(
                    "[reverse_prompt] reference %s HTML load failed (%s): %s",
                    ref.name, source_path, e,
                )
        slices = _encode_slices([ref.image_path])
        if not slices:
            logger.warning(
                "[reverse_prompt] skipping reference %s — no encodable screenshot",
                ref.name,
            )
            continue
        jobs.append({
            "source": "reference",
            "example_id": ref.name,
            "html_source_compressed": html_compressed,
            "screenshot_slices_b64": slices,
            "html_images_b64": [],
            "html_image_descriptors": {},
            "reference_examples_b64": [],
        })

    if not jobs:
        return {**state, "design_descriptors": [], "baseline_descriptor": None}

    sem = asyncio.Semaphore(_load_concurrency())
    logger.info(
        "[reverse_prompt] fanning out %d VLM call(s) in parallel "
        "(provider=%s model=%s, concurrency=%d, refs_inlined=%d)",
        len(jobs), router.provider, router.model,
        sem._value, len(reference_examples_b64),  # type: ignore[attr-defined]
    )
    coros = [
        _call_vlm(
            client=router,
            source=job["source"],
            example_id=job["example_id"],
            blueprint_summary=blueprint_summary,
            html_source_compressed=job["html_source_compressed"],
            screenshot_slices_b64=job["screenshot_slices_b64"],
            html_images_b64=job["html_images_b64"],
            html_image_descriptors=job.get("html_image_descriptors", {}),
            reference_examples_b64=job["reference_examples_b64"],
            winner_feedback=job.get("winner_feedback"),
            sem=sem,
        )
        for job in jobs
    ]
    descriptors: list[DesignDescriptor] = await asyncio.gather(*coros)

    baseline_descriptor = next((d for d in descriptors if d.source == "baseline"), None)
    _persist_descriptors(
        run_id=str(state.get("run_id", "default_run")),
        iteration=int(state.get("iteration", 0) or 0),
        descriptors=descriptors,
        baseline_descriptor=baseline_descriptor,
    )
    out: dict = dict(state)
    out["design_descriptors"] = descriptors
    if baseline_descriptor is not None:
        out["baseline_descriptor"] = baseline_descriptor
    return out


def _persist_descriptors(
    *,
    run_id: str,
    iteration: int,
    descriptors: list[DesignDescriptor],
    baseline_descriptor: DesignDescriptor | None,
) -> None:
    """Write reverse-prompter output to disk for audit.

    Layout:
        storage/runs/<run_id>/descriptors/iteration_<NNN>.json
            Full structured dump — every DesignDescriptor field including
            raw_description, plus the prompt-ready summary the creativizer
            actually receives via summarize_*_descriptor().
        storage/runs/<run_id>/descriptors/iteration_<NNN>.creativizer_view.txt
            Just the strings the creativizer sees per descriptor, for easy
            human inspection.
    """
    try:
        out_dir = os.path.join(run_dir(run_id), "descriptors")
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, f"iteration_{iteration:03d}.json")
        view_path = os.path.join(out_dir, f"iteration_{iteration:03d}.creativizer_view.txt")

        entries: list[dict[str, Any]] = []
        for d in descriptors:
            d_dict = asdict(d)
            if d.source == "baseline":
                summary = summarize_descriptor(d_dict)
            else:
                summary = summarize_reference_descriptor(d_dict)
            entries.append({
                **d_dict,
                "creativizer_prompt_summary": summary,
            })

        payload = {
            "run_id": run_id,
            "iteration": iteration,
            "baseline_example_id": baseline_descriptor.example_id if baseline_descriptor else None,
            "descriptors": entries,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        view_lines: list[str] = [
            f"# Creativizer view — what the LLM actually receives",
            f"# run_id={run_id}  iteration={iteration}",
            "",
        ]
        for entry in entries:
            view_lines.append(f"--- {entry.get('source','?').upper()}: {entry.get('example_id','?')} ---")
            view_lines.append(entry["creativizer_prompt_summary"])
            view_lines.append("")
        with open(view_path, "w", encoding="utf-8") as f:
            f.write("\n".join(view_lines))

        logger.info(
            "[reverse_prompt] persisted %d descriptor(s) → %s",
            len(entries), json_path,
        )
    except OSError as e:
        logger.warning(
            "[reverse_prompt] failed to persist descriptors for run=%s iteration=%d: %s",
            run_id, iteration, e,
        )


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _call_vlm(
    *,
    client,
    source: str,
    example_id: str,
    blueprint_summary: str,
    html_source_compressed: str,
    screenshot_slices_b64: list[tuple[str, str]],
    html_images_b64: list[tuple[str, str, str]],
    html_image_descriptors: dict[str, dict],
    reference_examples_b64: list[tuple[str, str, str, str]],
    sem: asyncio.Semaphore,
    winner_feedback: str | None = None,
) -> DesignDescriptor:
    async with sem:
        try:
            response = await client.messages.create(
                max_tokens=_load_max_tokens(),
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": build_user_prompt(
                            source=source,
                            example_id=example_id,
                            blueprint_summary=blueprint_summary,
                            html_source_compressed=html_source_compressed,
                            screenshot_slices_b64=screenshot_slices_b64,
                            html_images_b64=html_images_b64,
                            html_image_descriptors=html_image_descriptors,
                            reference_examples_b64=reference_examples_b64,
                            winner_feedback=winner_feedback,
                        ),
                    }
                ],
            )
            raw = response.content[0].text
        except Exception as e:
            logger.warning("[reverse_prompt] %s VLM call failed: %s", example_id, e)
            return _empty_descriptor(example_id=example_id, source=source, raw=str(e))
    return _parse_descriptor(raw, example_id=example_id, source=source)


def _parse_descriptor(raw: str, *, example_id: str, source: str) -> DesignDescriptor:
    try:
        json_str = raw.split("```json")[-1].split("```")[0].strip()
        data = json.loads(json_str)
        return DesignDescriptor(
            example_id=example_id,
            layout_type=str(data.get("layout_type", "") or ""),
            color_mood=str(data.get("color_mood", "") or ""),
            typography_personality=str(data.get("typography_personality", "") or ""),
            information_hierarchy=str(data.get("information_hierarchy", "") or ""),
            emotional_tone=str(data.get("emotional_tone", "") or ""),
            visual_motifs=list(data.get("visual_motifs", []) or []),
            creative_risks_taken=str(data.get("creative_risks_taken", "") or ""),
            weaknesses=list(data.get("weaknesses", []) or []),
            reference_patterns_to_emulate=list(data.get("reference_patterns_to_emulate", []) or []),
            reference_standards_summary=str(data.get("reference_standards_summary", "") or ""),
            raw_description=raw,
            source=source,
        )
    except Exception:
        return _empty_descriptor(example_id=example_id, source=source, raw=raw)


def _empty_descriptor(*, example_id: str, source: str, raw: str) -> DesignDescriptor:
    return DesignDescriptor(
        example_id=example_id,
        layout_type="",
        color_mood="",
        typography_personality="",
        information_hierarchy="",
        emotional_tone="",
        visual_motifs=[],
        creative_risks_taken="",
        weaknesses=[],
        raw_description=raw,
        source=source,
    )


def _encode_slices(paths: Iterable[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for p in paths:
        try:
            out.append(("image/png", encode_image_b64(p)))
        except Exception as e:
            logger.warning("[reverse_prompt] failed to encode slice %s: %s", p, e)
    return out


def _encode_reference_examples(
    refs: Iterable[ReferenceExample],
) -> list[tuple[str, str, str, str]]:
    """Encode references as [(name, description, media_type, base64), ...].

    Media type is inferred from the file extension (PNG / JPEG / WebP).
    Failures are logged and skipped so one bad reference doesn't kill the
    whole call.
    """
    out: list[tuple[str, str, str, str]] = []
    for ref in refs:
        try:
            b64 = encode_image_b64(ref.image_path)
        except Exception as e:
            logger.warning(
                "[reverse_prompt] failed to encode reference %s (%s): %s",
                ref.name, ref.image_path, e,
            )
            continue
        out.append((ref.name, ref.description, _guess_media_type(ref.image_path), b64))
    return out


def _encode_html_placeholder_images(
    images_b64: dict[str, str],
) -> list[tuple[str, str, str]]:
    """Encode baseline placeholder map as [(token, media_type, base64), ...].

    The order follows placeholder token index so prompt labels line up with
    `[images_base64_N]` mentions in the HTML.
    """
    out: list[tuple[str, str, str]] = []
    for token in placeholder_tokens(images_b64):
        data_uri = images_b64.get(token, "")
        if not data_uri:
            continue
        try:
            media_type, payload = split_data_uri(data_uri)
        except Exception as e:
            logger.warning(
                "[reverse_prompt] failed to decode baseline placeholder %s: %s",
                token, e,
            )
            continue
        out.append((token, media_type, re.sub(r"\s+", "", payload)))
    return out


def _guess_media_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif":  "image/gif",
    }.get(ext, "image/png")


def _section_to_dict(s) -> dict:
    """Accept either a BlueprintSection dataclass or a raw dict."""
    if isinstance(s, BlueprintSection):
        return asdict(s)
    if isinstance(s, dict):
        return s
    raise TypeError(f"Unsupported blueprint section type: {type(s).__name__}")


def _load_concurrency() -> int:
    try:
        with open(_PIPELINE_CONFIG) as f:
            cfg = yaml.safe_load(f) or {}
        conc = (cfg.get("concurrency", {}) if isinstance(cfg, dict) else {}) or {}
        return max(1, int(conc.get("reverse_prompter", _DEFAULT_REVERSE_CONCURRENCY)))
    except Exception:
        return _DEFAULT_REVERSE_CONCURRENCY


def _load_max_references() -> int:
    try:
        with open(_PIPELINE_CONFIG) as f:
            cfg = yaml.safe_load(f) or {}
        pipe = (cfg.get("pipeline", {}) if isinstance(cfg, dict) else {}) or {}
        return max(0, int(pipe.get("max_references", _DEFAULT_MAX_REFERENCES)))
    except Exception:
        return _DEFAULT_MAX_REFERENCES


def _load_max_tokens() -> int:
    try:
        with open(_PIPELINE_CONFIG) as f:
            cfg = yaml.safe_load(f) or {}
        mt = (cfg.get("max_tokens", {}) if isinstance(cfg, dict) else {}) or {}
        return max(1, int(mt.get("reverse_prompter", _MAX_TOKENS)))
    except Exception:
        return _MAX_TOKENS


def _load_winner_feedback_enabled() -> bool:
    try:
        with open(_PIPELINE_CONFIG) as f:
            cfg = yaml.safe_load(f) or {}
        fb = (cfg.get("feedback", {}) if isinstance(cfg, dict) else {}) or {}
        return bool(fb.get("winner_feedback_to_reverse_prompter", True))
    except Exception:
        return True
