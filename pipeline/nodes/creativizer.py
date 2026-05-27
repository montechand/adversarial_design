"""
nodes/creativizer.py

Stage 2 — generate k structurally distinct HTML design specs in parallel.

The creativizer runs ONE LLM call per candidate to produce a long-form design
spec. With k=3..5 candidates, that's k independent round-trips — so we fan
them out with `asyncio.gather` (plus a bounded `Semaphore` to play nice with
the Anthropic rate limits).

Each spec is anchored to:
  - the same blueprint (so candidates are comparable)
  - the same base_prompt seed
  - a DIFFERENT combination of forced design axes (to enforce diversity)
  - explicit baseline weaknesses as attack surface
  - the same Brand Rules block (supreme over forced axes)

Compliance feedback from a previous failed attempt is forwarded back into
the prompt for the corresponding candidate index (retry semantics unchanged).
"""

from __future__ import annotations

import os
import random
import asyncio
import logging
from dataclasses import asdict
from typing import Any

import yaml

from pipeline.state import PipelineState, DesignDescriptor, BlueprintSection
from pipeline.utils.llm_clients import get_async_router
from pipeline.utils.html_compression import placeholder_tokens
from pipeline.utils.image_descriptors import build_descriptors
from pipeline.utils.storage import run_dir
from pipeline.prompts.creativizer import (
    SYSTEM_PROMPT,
    build_user_prompt,
    summarize_descriptor,
    summarize_reference_descriptor,
)
from pipeline.prompts.reverse_prompter import build_blueprint_summary
import json

logger = logging.getLogger(__name__)

AXES_CONFIG = os.path.join(os.path.dirname(__file__), "../../config/design_axes.yaml")
PIPELINE_CONFIG = os.path.join(os.path.dirname(__file__), "../../config/pipeline_config.yaml")

# Provider/model is resolved at runtime from `config/pipeline_config.yaml >
# models.creativizer`. The default below is used only if the config row is
# missing or unparseable.
_DEFAULT_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 16000  # fallback if config is missing

_DEFAULT_CREATIVIZER_CONCURRENCY = 4


async def creativize(state: PipelineState) -> PipelineState:
    router = get_async_router("creativizer", default=_DEFAULT_MODEL)
    design_axes = _load_axes()
    k = _get_k()

    axis_combinations = _sample_axis_combinations(design_axes, k=k)

    blueprint_sections = state.get("blueprint_sections", []) or []
    blueprint_summary = build_blueprint_summary(
        [_section_to_dict(s) for s in blueprint_sections]
    )

    descriptors: list[DesignDescriptor] = state.get("design_descriptors", []) or []
    baseline = state.get("baseline_descriptor")
    if baseline is None:
        for d in descriptors:
            if d.source == "baseline":
                baseline = d
                break
    baseline_summary = (
        summarize_descriptor(asdict(baseline)) if baseline else "(no baseline available)"
    )
    reference_summaries = [
        summarize_reference_descriptor(asdict(d)) for d in descriptors if d.source != "baseline"
    ]
    reference_patterns: list[str] = []
    reference_standards = ""
    if baseline is not None:
        reference_patterns = list(baseline.reference_patterns_to_emulate or [])
        reference_standards = baseline.reference_standards_summary or ""

    base_prompt = state.get("base_prompt", "") or ""
    brand_rules = state.get("brand_rules") or None
    images_b64 = state.get("baseline_images_b64", {}) or {}
    available_image_tokens = placeholder_tokens(images_b64)

    # Pull image descriptors from state (populated by ingestion) so the
    # creativizer can reason about each token's intrinsic dimensions /
    # aspect ratio / original role rather than just its bare name.
    # Defensive fallback: rebuild from bytes + compressed HTML if
    # ingestion didn't populate them (e.g. partial state from tests).
    image_descriptors: dict[str, dict] = dict(
        state.get("baseline_image_descriptors", {}) or {}
    )
    if images_b64 and not image_descriptors:
        try:
            image_descriptors = build_descriptors(
                images_b64=images_b64,
                compressed_html=state.get("baseline_html_compressed", "") or "",
            )
        except Exception as e:
            logger.warning(
                "[creativize] failed to mine image descriptors at runtime: %s — "
                "creativizer will see token names only", e,
            )
            image_descriptors = {}

    sem = asyncio.Semaphore(_load_concurrency())
    descriptor_coverage = sum(
        1 for t in available_image_tokens
        if (image_descriptors.get(t) or {}).get("intrinsic")
    )
    logger.info(
        "[creativize] generating %d candidate prompt(s) in parallel "
        "(provider=%s model=%s, concurrency=%d, available_image_tokens=%d, "
        "descriptors_with_intrinsic=%d)",
        k, router.provider, router.model,
        sem._value, len(available_image_tokens), descriptor_coverage,  # type: ignore[attr-defined]
    )

    tasks = [
        _one_creativizer_call(
            client=router,
            blueprint_summary=blueprint_summary,
            base_prompt=base_prompt,
            baseline_summary=baseline_summary,
            reference_summaries=reference_summaries,
            reference_patterns=reference_patterns,
            reference_standards=reference_standards,
            forced_axes=axes,
            candidate_index=i,
            compliance_feedback=_get_compliance_feedback(state, i),
            brand_rules=brand_rules,
            available_image_tokens=available_image_tokens,
            image_descriptors=image_descriptors,
            sem=sem,
        )
        for i, axes in enumerate(axis_combinations)
    ]
    prompt_texts: list[str] = await asyncio.gather(*tasks)

    candidate_prompts: list[str] = list(prompt_texts)
    candidate_metadata: list[dict] = [
        {"axes": axes, "candidate_index": i}
        for i, axes in enumerate(axis_combinations)
    ]

    _persist_creativizer_prompts(
        run_id=str(state.get("run_id", "default_run")),
        iteration=int(state.get("iteration", 0) or 0),
        candidate_prompts=candidate_prompts,
        candidate_metadata=candidate_metadata,
        available_image_tokens=available_image_tokens,
        image_descriptors=image_descriptors,
        compliance_feedback=[
            _get_compliance_feedback(state, i) for i in range(len(candidate_prompts))
        ],
    )

    return {
        **state,
        "candidate_prompts": candidate_prompts,
        "candidate_prompt_metadata": candidate_metadata,
    }


def _persist_creativizer_prompts(
    *,
    run_id: str,
    iteration: int,
    candidate_prompts: list[str],
    candidate_metadata: list[dict],
    available_image_tokens: list[str],
    image_descriptors: dict[str, dict],
    compliance_feedback: list[str | None],
) -> None:
    """Write the creativizer's k design specs (the exact strings fed into
    code_generator) to disk for audit.

    Layout:
        storage/runs/<run_id>/creativizer/iteration_<NNN>/
            candidate_<i>.prompt.txt     full design spec
            candidate_<i>.metadata.json  axes, compliance feedback, token whitelist
            manifest.json                index + sizes + axes summary
                                         + image descriptors snapshot

    The descriptor snapshot captures exactly what the creativizer saw for
    each token (intrinsic dims / role / alt) — useful for auditing whether
    the design specs respected each asset's native aspect ratio.
    """
    try:
        out_dir = os.path.join(
            run_dir(run_id), "creativizer", f"iteration_{iteration:03d}"
        )
        os.makedirs(out_dir, exist_ok=True)

        descriptor_snapshot = {
            tok: {
                "intrinsic": dict((image_descriptors.get(tok) or {}).get("intrinsic") or {}),
                "usage":     dict((image_descriptors.get(tok) or {}).get("usage") or {}),
            }
            for tok in available_image_tokens
        }

        manifest_entries: list[dict] = []
        for i, (prompt, meta) in enumerate(zip(candidate_prompts, candidate_metadata)):
            stem = f"candidate_{i}"
            prompt_path = os.path.join(out_dir, f"{stem}.prompt.txt")
            meta_path = os.path.join(out_dir, f"{stem}.metadata.json")

            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt or "")

            meta_payload = {
                "run_id": run_id,
                "iteration": iteration,
                "candidate_index": i,
                "axes": meta.get("axes", {}),
                "available_image_tokens": list(available_image_tokens),
                "image_descriptors": descriptor_snapshot,
                "compliance_feedback": compliance_feedback[i] if i < len(compliance_feedback) else None,
                "prompt_char_count": len(prompt or ""),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_payload, f, indent=2, ensure_ascii=False)

            manifest_entries.append({
                "candidate_index": i,
                "prompt_file": f"{stem}.prompt.txt",
                "metadata_file": f"{stem}.metadata.json",
                "prompt_char_count": len(prompt or ""),
                "axes": meta.get("axes", {}),
            })

        manifest = {
            "run_id": run_id,
            "iteration": iteration,
            "available_image_tokens": list(available_image_tokens),
            "image_descriptors": descriptor_snapshot,
            "candidates": manifest_entries,
        }
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        logger.info(
            "[creativize] persisted %d creativizer prompt(s) → %s",
            len(candidate_prompts), out_dir,
        )
    except OSError as e:
        logger.warning(
            "[creativize] failed to persist creativizer prompts for run=%s iteration=%d: %s",
            run_id, iteration, e,
        )


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _one_creativizer_call(
    *,
    client: Any,
    blueprint_summary: str,
    base_prompt: str,
    baseline_summary: str,
    reference_summaries: list[str],
    reference_patterns: list[str],
    reference_standards: str,
    forced_axes: dict[str, str],
    candidate_index: int,
    compliance_feedback: str | None,
    brand_rules: dict | None,
    available_image_tokens: list[str],
    image_descriptors: dict[str, dict],
    sem: asyncio.Semaphore,
) -> str:
    user_prompt = build_user_prompt(
        blueprint_summary=blueprint_summary,
        base_prompt=base_prompt,
        baseline_descriptor_summary=baseline_summary,
        reference_descriptor_summaries=reference_summaries,
        reference_patterns_to_emulate=reference_patterns,
        reference_standards_summary=reference_standards,
        forced_axes=forced_axes,
        candidate_index=candidate_index,
        compliance_feedback=compliance_feedback,
        brand_rules=brand_rules,
        available_image_tokens=available_image_tokens,
        image_descriptors=image_descriptors,
    )
    async with sem:
        try:
            response = await client.messages.create(
                max_tokens=_load_max_tokens(),
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.warning(
                "[creativize] candidate %d failed (%s) — using fallback prompt",
                candidate_index, e,
            )
            return _fallback_prompt(base_prompt, forced_axes)


def _load_axes() -> dict[str, list[str]]:
    try:
        with open(AXES_CONFIG) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {
            "layout": ["hero-left content-right", "full-width stacked", "grid-mosaic", "editorial magazine"],
            "color_mood": ["clinical cool blues", "warm earthy trust", "bold high-contrast", "soft muted pastels"],
            "typography": ["modern sans geometric", "authoritative serif", "mixed editorial", "minimal monospace"],
        }


def _get_k() -> int:
    try:
        with open(PIPELINE_CONFIG) as f:
            cfg = yaml.safe_load(f) or {}
        pipeline_cfg = cfg.get("pipeline", {}) if isinstance(cfg, dict) else {}
        return int(pipeline_cfg.get("k", cfg.get("k", 3)) or 3)
    except Exception:
        return 3


def _load_concurrency() -> int:
    try:
        with open(PIPELINE_CONFIG) as f:
            cfg = yaml.safe_load(f) or {}
        conc = (cfg.get("concurrency", {}) if isinstance(cfg, dict) else {}) or {}
        return max(1, int(conc.get("creativizer", _DEFAULT_CREATIVIZER_CONCURRENCY)))
    except Exception:
        return _DEFAULT_CREATIVIZER_CONCURRENCY


def _load_max_tokens() -> int:
    try:
        with open(PIPELINE_CONFIG) as f:
            cfg = yaml.safe_load(f) or {}
        mt = (cfg.get("max_tokens", {}) if isinstance(cfg, dict) else {}) or {}
        return max(1, int(mt.get("creativizer", _MAX_TOKENS)))
    except Exception:
        return _MAX_TOKENS


def _sample_axis_combinations(axes: dict[str, list[str]], k: int) -> list[dict[str, str]]:
    combinations: list[dict[str, str]] = []
    seen: set[tuple] = set()
    attempts = 0
    while len(combinations) < k and attempts < 100:
        combo = {axis: random.choice(values) for axis, values in axes.items() if values}
        key = tuple(sorted(combo.items()))
        if key not in seen:
            seen.add(key)
            combinations.append(combo)
        attempts += 1
    return combinations


def _get_compliance_feedback(state: PipelineState, candidate_index: int) -> str | None:
    reasons = state.get("compliance_reasons", []) or []
    if candidate_index < len(reasons) and reasons[candidate_index]:
        return reasons[candidate_index]
    return None


def _section_to_dict(s) -> dict:
    if isinstance(s, BlueprintSection):
        return asdict(s)
    if isinstance(s, dict):
        return s
    raise TypeError(f"Unsupported blueprint section type: {type(s).__name__}")


def _fallback_prompt(base_prompt: str, axes: dict[str, str]) -> str:
    axes_block = "\n".join(f"- {k}: {v}" for k, v in axes.items())
    return (
        f"FALLBACK SPEC (LLM unreachable). Use the base prompt and the forced axes:\n\n"
        f"BASE:\n{base_prompt or '(none)'}\n\nFORCED AXES:\n{axes_block}\n"
    )
