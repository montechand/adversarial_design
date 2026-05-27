"""
nodes/code_generator.py

Stage 3 — candidate generation.

Two MJML flavors live side-by-side here; the active one is selected at
runtime by `pipeline_config.yaml > pipeline.candidate_mode`:

  ┌──────────────────────────────────────────────────────────────────────┐
  │  full_mjml      (default)                                            │
  │     ONE LLM call per candidate produces fragments for EVERY section  │
  │     in a single response, so the LLM can reason about color          │
  │     hierarchy, typographic rhythm, spacing cadence, and cross-       │
  │     section narrative arc as one coherent design.                    │
  │     Prompt: pipeline/prompts/mjml_full_designer.py                   │
  │                                                                      │
  │  section_wise   (kept for comparison / future fallback)              │
  │     N parallel LLM calls per candidate (one per section), gathered   │
  │     with asyncio.gather + Semaphore. Lower latency per candidate     │
  │     but design choices don't carry across sections → results often   │
  │     read as N stitched-together panels.                              │
  │     Prompt: pipeline/prompts/mjml_section_designer.py                │
  └──────────────────────────────────────────────────────────────────────┘

Both flavors share steps 2–4:

  2. COMPILE — `compile_mjml_document` assembles fragments by `order`,
     wraps them with `<!-- Section:Start sid -->` markers, prefixes the
     `<mj-head>` with the brand body font.

  3. MJML → HTML — `mjml_utils.mjml_to_html` runs `mjml-python` with an
     LLM-driven retry/fix loop on syntax failures (mirror of
     `_fix_mjml_with_error_feedback`).

  4. MECHANICAL VALIDATORS — `validate_all_sections` warns (no retry) on
     missing `data-section-id`, `<sup>` tags without `data-claim-id`, and
     sentences from `copy_outline` that don't appear verbatim. Production
     semantics: prose-drift is prompt-enforced only.

Across-candidate parallelism (both modes): the k candidates run concurrently
under a `candidate_sem` Semaphore. Inside ONE candidate:

  - full_mjml      → a single per-candidate LLM call (no per-section sem
                     needed; the call is itself one round-trip).
  - section_wise   → N parallel per-section LLM calls under section_sem.

The baseline is inserted as `candidate_id=0` with `is_baseline=True` and
skips MJML generation entirely.
"""

from __future__ import annotations

import os
import re
import json
import asyncio
import logging
from dataclasses import asdict
from typing import Any

import yaml

from pipeline.state import PipelineState, Candidate, BlueprintSection
from pipeline.utils.llm_clients import get_async_llm_client
from pipeline.utils.html_compression import placeholder_tokens
from pipeline.utils.storage import run_dir
from pipeline.utils.style_guide import extract_body_font
from pipeline.utils.mjml_utils import (
    compile_mjml_document,
    mjml_to_html,
    validate_all_sections,
)
from pipeline.prompts.mjml_section_designer import (
    SYSTEM_PROMPT as MJML_SECTION_SYSTEM_PROMPT,
    build_user_prompt as build_mjml_section_user_prompt,
)
from pipeline.prompts.mjml_full_designer import (
    SYSTEM_PROMPT as MJML_FULL_SYSTEM_PROMPT,
    build_user_prompt as build_mjml_full_user_prompt,
)
from pipeline.prompts.reverse_prompter import build_blueprint_summary

logger = logging.getLogger(__name__)

# Models
_SECTION_MODEL = "claude-sonnet-4-6"
_SECTION_MAX_TOKENS = 10240  # fallback if config is missing
_FULL_MODEL = "claude-sonnet-4-6"
_FULL_MAX_TOKENS = 16384     # fallback if config is missing — all sections in one response (~Nx2k)
_MJML_FIX_MODEL = "claude-sonnet-4-6"

# Concurrency caps (tunable via config/pipeline_config.yaml → concurrency.*)
_DEFAULT_SECTION_CONCURRENCY = 6      # max concurrent per-section LLM calls
_DEFAULT_CANDIDATE_CONCURRENCY = 3    # max concurrent candidate flows

# Candidate mode (tunable via config/pipeline_config.yaml → pipeline.candidate_mode)
#   "full_mjml"    → one LLM call per candidate covers every section (default)
#   "section_wise" → N parallel per-section LLM calls per candidate
_DEFAULT_CANDIDATE_MODE = "full_mjml"
_VALID_CANDIDATE_MODES = ("full_mjml", "section_wise")

_PIPELINE_CONFIG = os.path.join(os.path.dirname(__file__), "../../config/pipeline_config.yaml")


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph entrypoint (async)
# ─────────────────────────────────────────────────────────────────────────────

async def generate_candidates(state: PipelineState) -> PipelineState:
    """LangGraph node — dispatcher that picks the candidate mode (full_mjml
    vs section_wise) and runs the k candidates in parallel.

    Regardless of mode the per-candidate flow always ends in:
      compile → mjml→html → mechanical validate.
    """
    client = get_async_llm_client()
    blueprint_sections = state.get("blueprint_sections", []) or []
    blueprint_dicts = [_section_to_dict(s) for s in blueprint_sections]
    blueprint_summary = build_blueprint_summary(blueprint_dicts)

    layout_width = int(state.get("layout_width", 600) or 600)
    baseline_tokens = placeholder_tokens(state.get("baseline_images_b64", {}) or {})
    brand_rules = state.get("brand_rules") or None
    brand_body_font = extract_body_font(brand_rules)

    candidates: list[Candidate] = []

    # ── BASELINE — already designed upstream, insert as candidate #0 ─────────
    baseline_html = state.get("baseline_html_compressed", "") or ""
    if baseline_html:
        candidates.append(Candidate(
            candidate_id=0,
            prompt_used="(baseline — produced by backend-server pipeline)",
            prompt_metadata={"is_baseline": True},
            html=baseline_html,
            is_baseline=True,
            mjml_render_ok=None,  # baseline skipped MJML pipeline
        ))

    offset = len(candidates)
    prompts = state.get("candidate_prompts", []) or []
    metadata = state.get("candidate_prompt_metadata", []) or []
    if not prompts:
        return {**state, "candidates": candidates}

    # Concurrency caps + mode from config
    section_concurrency, candidate_concurrency = _load_concurrency()
    candidate_mode = _load_candidate_mode()
    candidate_sem = asyncio.Semaphore(candidate_concurrency)
    # Only needed by section_wise; cheap to construct unconditionally.
    section_sem = asyncio.Semaphore(section_concurrency)

    logger.info(
        "[code_gen] mode=%s  candidates=%d  sections=%d  "
        "candidate_concurrency=%d  section_concurrency=%d  brand_body_font=%r",
        candidate_mode, len(prompts), len(blueprint_dicts),
        candidate_concurrency, section_concurrency, brand_body_font,
    )

    run_id = str(state.get("run_id", "default_run"))
    iteration = int(state.get("iteration", 0) or 0)

    if candidate_mode == "full_mjml":
        per_candidate = _generate_one_candidate_full_mjml
        extra_kwargs = {}
    else:  # section_wise
        per_candidate = _generate_one_candidate_section_wise  # type: ignore[assignment]
        extra_kwargs = {"section_sem": section_sem}

    coros = [
        per_candidate(
            client=client,
            candidate_id=offset + i,
            design_prompt=prompt,
            metadata=meta,
            blueprint_dicts=blueprint_dicts,
            blueprint_summary=blueprint_summary,
            layout_width=layout_width,
            baseline_tokens=baseline_tokens,
            brand_rules=brand_rules,
            brand_body_font=brand_body_font,
            candidate_sem=candidate_sem,
            run_id=run_id,
            iteration=iteration,
            **extra_kwargs,
        )
        for i, (prompt, meta) in enumerate(zip(prompts, metadata))
    ]
    generated = await asyncio.gather(*coros, return_exceptions=False)
    candidates.extend(generated)

    return {**state, "candidates": candidates}


# ─────────────────────────────────────────────────────────────────────────────
# Per-candidate flow — FULL MJML (default)
#   ONE LLM call per candidate produces fragments for every section in one
#   response, so the LLM can reason about color hierarchy, typography rhythm,
#   spacing cadence, and narrative arc across the whole email at once.
# ─────────────────────────────────────────────────────────────────────────────

async def _generate_one_candidate_full_mjml(
    *,
    client: Any,
    candidate_id: int,
    design_prompt: str,
    metadata: dict,
    blueprint_dicts: list[dict],
    blueprint_summary: str,
    layout_width: int,
    baseline_tokens: list[str],
    brand_rules: dict | None,
    brand_body_font: str,
    candidate_sem: asyncio.Semaphore,
    run_id: str = "default_run",
    iteration: int = 0,
) -> Candidate:
    """Whole-email path: one LLM call → all fragments → compile → render → validate."""
    async with candidate_sem:
        user_prompt = build_mjml_full_user_prompt(
            blueprint_sections=blueprint_dicts,
            design_prompt=design_prompt,
            layout_width=layout_width,
            available_image_tokens=baseline_tokens or None,
            brand_rules=brand_rules,
        )
        raw: str = ""
        llm_exception: str | None = None
        parse_trace: dict[str, Any] = {
            "status": "llm_exception",
            "missing_section_ids": [str(s.get("section_id", "")) for s in blueprint_dicts],
            "sections_total": len(blueprint_dicts),
            "sections_filled_by_llm": 0,
        }
        try:
            _cfg_full_max, _ = _load_max_tokens()
            response = await client.messages.create(
                model=_FULL_MODEL,
                max_tokens=_cfg_full_max,
                system=MJML_FULL_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text
            section_artifacts, design_reasoning, parse_trace = _parse_full_response(
                raw, blueprint_dicts=blueprint_dicts,
            )
        except Exception as e:  # noqa: BLE001 — fall back gracefully
            llm_exception = f"{type(e).__name__}: {e}"
            logger.warning(
                "[code_gen] candidate %d full-mjml LLM call failed: %s — using fallback",
                candidate_id, e,
            )
            section_artifacts = {
                str(s.get("section_id", "")): _fallback_section_artifact(s)
                for s in blueprint_dicts
            }
            design_reasoning = ""

        if design_reasoning:
            metadata = {**metadata, "design_reasoning": design_reasoning}

        codegen_trace: dict[str, Any] = {
            "mode": "full_mjml",
            "model": _FULL_MODEL,
            "user_prompt_char_count": len(user_prompt),
            "raw_response_char_count": len(raw),
            "llm_exception": llm_exception,
            "parse_status": parse_trace["status"],
            "missing_section_ids": parse_trace["missing_section_ids"],
            "sections_total": parse_trace["sections_total"],
            "sections_filled_by_llm": parse_trace["sections_filled_by_llm"],
            "sections_used_fallback": len(parse_trace["missing_section_ids"]),
            "raw_response": raw,
            "user_prompt": user_prompt,
        }

        return await _finalize_candidate(
            client=client,
            candidate_id=candidate_id,
            design_prompt=design_prompt,
            metadata=metadata,
            blueprint_dicts=blueprint_dicts,
            blueprint_summary=blueprint_summary,
            layout_width=layout_width,
            brand_body_font=brand_body_font,
            section_artifacts=section_artifacts,
            codegen_trace=codegen_trace,
            run_id=run_id,
            iteration=iteration,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Per-candidate flow — SECTION_WISE (kept for comparison / fallback)
#   N parallel LLM calls per candidate, one per section, gathered with
#   asyncio.gather + Semaphore. Lower latency per candidate but designs
#   tend to feel disjunctive across sections because each call sees only
#   its own section.
# ─────────────────────────────────────────────────────────────────────────────

async def _generate_one_candidate_section_wise(
    *,
    client: Any,
    candidate_id: int,
    design_prompt: str,
    metadata: dict,
    blueprint_dicts: list[dict],
    blueprint_summary: str,
    layout_width: int,
    baseline_tokens: list[str],
    brand_rules: dict | None,
    brand_body_font: str,
    section_sem: asyncio.Semaphore,
    candidate_sem: asyncio.Semaphore,
    run_id: str = "default_run",
    iteration: int = 0,
) -> Candidate:
    """Section-wise path: per-section fan-out → compile → render → validate."""
    async with candidate_sem:
        section_tasks = [
            _design_one_section(
                client=client,
                section=section,
                design_prompt=design_prompt,
                layout_width=layout_width,
                baseline_tokens=baseline_tokens,
                brand_rules=brand_rules,
                other_sections_overview=blueprint_dicts,
                section_sem=section_sem,
            )
            for section in blueprint_dicts
        ]
        section_results: list[tuple[str, dict]] = await asyncio.gather(*section_tasks)
        section_artifacts: dict[str, dict] = {sid: art for sid, art in section_results}

        fallback_ids = [
            sid for sid, art in section_artifacts.items()
            if (art.get("design_reasoning") or "").startswith("Fallback fragment")
        ]
        codegen_trace: dict[str, Any] = {
            "mode": "section_wise",
            "model": _SECTION_MODEL,
            "sections_total": len(blueprint_dicts),
            "sections_filled_by_llm": len(blueprint_dicts) - len(fallback_ids),
            "sections_used_fallback": len(fallback_ids),
            "missing_section_ids": fallback_ids,
            "parse_status": "ok" if not fallback_ids else "partial_fallback",
            "llm_exception": None,
            "raw_response": None,
            "user_prompt": None,
        }

        return await _finalize_candidate(
            client=client,
            candidate_id=candidate_id,
            design_prompt=design_prompt,
            metadata=metadata,
            blueprint_dicts=blueprint_dicts,
            blueprint_summary=blueprint_summary,
            layout_width=layout_width,
            brand_body_font=brand_body_font,
            section_artifacts=section_artifacts,
            codegen_trace=codegen_trace,
            run_id=run_id,
            iteration=iteration,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Shared tail: compile fragments → mjml→html (with LLM fix retries) → validate
# ─────────────────────────────────────────────────────────────────────────────

async def _finalize_candidate(
    *,
    client: Any,
    candidate_id: int,
    design_prompt: str,
    metadata: dict,
    blueprint_dicts: list[dict],
    blueprint_summary: str,
    layout_width: int,
    brand_body_font: str,
    section_artifacts: dict[str, dict],
    codegen_trace: dict[str, Any] | None = None,
    run_id: str = "default_run",
    iteration: int = 0,
) -> Candidate:
    """Steps 2-4 common to both candidate modes. Takes the (per-sid)
    artifact dict and produces a fully-formed Candidate."""
    fragments_by_sid = {
        sid: art.get("mjml_fragment", "") for sid, art in section_artifacts.items()
    }

    mjml_document = compile_mjml_document(
        fragments_by_sid=fragments_by_sid,
        blueprint_sections=blueprint_dicts,
        brand_body_font=brand_body_font,
        layout_width=layout_width,
    )

    render = await mjml_to_html(
        mjml_document,
        async_llm_client=client,
        fix_model=_MJML_FIX_MODEL,
        max_retries=3,
    )

    if render.success:
        html = render.html
        logger.info(
            "[code_gen] candidate %d: MJML→HTML OK in %d attempt(s) (html=%d chars)",
            candidate_id, render.attempts, len(html),
        )
    else:
        logger.warning(
            "[code_gen] candidate %d: MJML→HTML failed after %d attempts: %s",
            candidate_id, render.attempts, render.error,
        )
        html = _placeholder_html(
            blueprint_summary=blueprint_summary,
            layout_width=layout_width,
            reason=render.error or "MJML conversion failed",
            mjml_dump=render.fixed_mjml or mjml_document,
        )

    _validations, warnings = validate_all_sections(
        fragments_by_sid=fragments_by_sid,
        blueprint_sections=blueprint_dicts,
        body_width=layout_width,
    )
    if warnings:
        logger.warning(
            "[code_gen] candidate %d: %d validator warning(s)",
            candidate_id, len(warnings),
        )
        for w in warnings:
            logger.warning("[code_gen]   %s", w)

    candidate = Candidate(
        candidate_id=candidate_id,
        prompt_used=design_prompt,
        prompt_metadata=metadata,
        html=html,
        is_baseline=False,
        mjml_document=render.fixed_mjml if render.success else mjml_document,
        section_artifacts=section_artifacts,
        validation_warnings=warnings,
        mjml_render_ok=render.success,
        mjml_render_error=render.error if not render.success else None,
    )

    _persist_codegen_artifacts(
        run_id=run_id,
        iteration=iteration,
        candidate=candidate,
        codegen_trace=codegen_trace or {},
        render_attempts=getattr(render, "attempts", None),
    )

    return candidate


def _persist_codegen_artifacts(
    *,
    run_id: str,
    iteration: int,
    candidate: Candidate,
    codegen_trace: dict[str, Any],
    render_attempts: int | None,
) -> None:
    """Write per-candidate code-generator audit artifacts.

    Layout:
        storage/runs/<run_id>/code_generator/iteration_<NNN>/
            candidate_<N>.raw_response.txt   raw LLM design text (or error message)
            candidate_<N>.user_prompt.txt    the full user prompt fed to the LLM
            candidate_<N>.metadata.json      status, fallback section ids, render result,
                                              validation warnings, attempt counts
    """
    try:
        out_dir = os.path.join(
            run_dir(run_id), "code_generator", f"iteration_{iteration:03d}"
        )
        os.makedirs(out_dir, exist_ok=True)
        stem = f"candidate_{candidate.candidate_id}"

        raw_response = codegen_trace.get("raw_response")
        if raw_response is None and codegen_trace.get("llm_exception"):
            raw_response = f"[LLM exception]\n{codegen_trace['llm_exception']}"
        if raw_response is not None:
            with open(os.path.join(out_dir, f"{stem}.raw_response.txt"), "w", encoding="utf-8") as f:
                f.write(raw_response)

        user_prompt = codegen_trace.get("user_prompt")
        if user_prompt:
            with open(os.path.join(out_dir, f"{stem}.user_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(user_prompt)

        per_section = {
            sid: {
                "is_fallback": (art.get("design_reasoning") or "").startswith("Fallback fragment"),
                "design_reasoning": art.get("design_reasoning", ""),
                "claims_used": list(art.get("claims_used", []) or []),
                "mjml_fragment_chars": len(art.get("mjml_fragment", "") or ""),
            }
            for sid, art in (candidate.section_artifacts or {}).items()
        }

        metadata_payload = {
            "run_id": run_id,
            "iteration": iteration,
            "candidate_id": candidate.candidate_id,
            "mode": codegen_trace.get("mode"),
            "model": codegen_trace.get("model"),
            "parse_status": codegen_trace.get("parse_status"),
            "llm_exception": codegen_trace.get("llm_exception"),
            "sections_total": codegen_trace.get("sections_total"),
            "sections_filled_by_llm": codegen_trace.get("sections_filled_by_llm"),
            "sections_used_fallback": codegen_trace.get("sections_used_fallback"),
            "missing_section_ids": codegen_trace.get("missing_section_ids", []),
            "raw_response_char_count": codegen_trace.get("raw_response_char_count"),
            "user_prompt_char_count": codegen_trace.get("user_prompt_char_count"),
            "mjml_render_ok": candidate.mjml_render_ok,
            "mjml_render_error": candidate.mjml_render_error,
            "mjml_render_attempts": render_attempts,
            "validation_warnings": list(candidate.validation_warnings or []),
            "per_section": per_section,
        }
        with open(os.path.join(out_dir, f"{stem}.metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2, ensure_ascii=False)

        logger.info(
            "[code_gen] persisted candidate %d artifacts → %s (status=%s, fallbacks=%s/%s)",
            candidate.candidate_id, out_dir,
            metadata_payload["parse_status"],
            metadata_payload["sections_used_fallback"],
            metadata_payload["sections_total"],
        )
    except OSError as e:
        logger.warning(
            "[code_gen] failed to persist codegen artifacts for run=%s iteration=%d candidate=%d: %s",
            run_id, iteration, candidate.candidate_id, e,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Per-section LLM call (one async LLM round-trip per blueprint section)
# ─────────────────────────────────────────────────────────────────────────────

async def _design_one_section(
    *,
    client: Any,
    section: dict,
    design_prompt: str,
    layout_width: int,
    baseline_tokens: list[str],
    brand_rules: dict | None,
    other_sections_overview: list[dict],
    section_sem: asyncio.Semaphore,
) -> tuple[str, dict]:
    """Make one LLM call for one section, return (section_id, artifact_dict).

    The artifact dict mirrors the production `mjml_section_artifacts[sid]`:
      { "mjml_fragment": str, "claims_used": list[str], "design_reasoning": str }
    """
    sid = str(section.get("section_id", "") or "")
    user_prompt = build_mjml_section_user_prompt(
        section=section,
        design_prompt=design_prompt,
        layout_width=layout_width,
        available_image_tokens=baseline_tokens or None,
        brand_rules=brand_rules,
        other_sections_overview=other_sections_overview,
    )

    async with section_sem:
        try:
            _, _cfg_section_max = _load_max_tokens()
            response = await client.messages.create(
                model=_SECTION_MODEL,
                max_tokens=_cfg_section_max,
                system=MJML_SECTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text
        except Exception as e:  # noqa: BLE001 — fall back gracefully
            logger.warning("[code_gen] section %s LLM call failed: %s — using fallback", sid, e)
            return sid, _fallback_section_artifact(section)

    artifact = _parse_section_artifact(raw, sid=sid)
    if not artifact.get("mjml_fragment"):
        logger.warning("[code_gen] section %s produced no fragment — using fallback", sid)
        return sid, _fallback_section_artifact(section)
    return sid, artifact


# ─────────────────────────────────────────────────────────────────────────────
# Parsing & fallback helpers
# ─────────────────────────────────────────────────────────────────────────────

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]+\}")
# Matches one complete <mj-section ...>...</mj-section> block (non-greedy).
_MJ_SECTION_BLOCK_RE = re.compile(
    r"<mj-section\b[^>]*>[\s\S]*?</mj-section>",
    re.IGNORECASE,
)


def _trim_to_assigned_section(frag: str, sid: str) -> tuple[str, bool]:
    """If `frag` contains more than one <mj-section> root, keep only the one
    whose data-section-id matches `sid`. Falls back to the first block when
    none match. Returns (trimmed_frag, was_trimmed).
    """
    blocks = _MJ_SECTION_BLOCK_RE.findall(frag)
    if len(blocks) <= 1:
        return frag, False
    # Prefer the block that carries the correct data-section-id.
    for block in blocks:
        if f'data-section-id="{sid}"' in block or f"data-section-id='{sid}'" in block:
            return block, True
    # No exact match — keep the first block (best-effort).
    return blocks[0], True


def _parse_full_response(
    raw: str,
    *,
    blueprint_dicts: list[dict],
) -> tuple[dict[str, dict], str, dict[str, Any]]:
    """Parse the whole-email designer's JSON response.

    Returns `(section_artifacts_by_sid, top_level_design_reasoning, parse_trace)`
    where `parse_trace` exposes:
        - status: "ok" | "non_json" | "missing_sections"
        - missing_section_ids: list[str]
        - sections_total / sections_filled_by_llm
    """
    data = _tolerant_json_load(raw)
    if not isinstance(data, dict):
        logger.warning("[code_gen] full-mjml response not a JSON object — using fallbacks")
        all_ids = [str(s.get("section_id", "")) for s in blueprint_dicts]
        return (
            {sid: _fallback_section_artifact(s) for sid, s in zip(all_ids, blueprint_dicts)},
            "",
            {
                "status": "non_json",
                "missing_section_ids": all_ids,
                "sections_total": len(blueprint_dicts),
                "sections_filled_by_llm": 0,
            },
        )

    sections_out = data.get("sections")
    design_reasoning = str(data.get("design_reasoning", "") or "")

    by_sid: dict[str, dict] = {}
    if isinstance(sections_out, list):
        for entry in sections_out:
            if not isinstance(entry, dict):
                continue
            sid = str(entry.get("section_id", "") or "")
            if not sid:
                continue
            by_sid[sid] = _normalize_section_artifact(entry, sid=sid)

    artifacts: dict[str, dict] = {}
    missing_from_response: list[str] = []
    for s in blueprint_dicts:
        sid = str(s.get("section_id", "") or "")
        if sid in by_sid and by_sid[sid].get("mjml_fragment"):
            artifacts[sid] = by_sid[sid]
        else:
            missing_from_response.append(sid)
            artifacts[sid] = _fallback_section_artifact(s)

    if missing_from_response:
        logger.warning(
            "[code_gen] full-mjml response missing %d section(s): %s — using fallback fragments",
            len(missing_from_response), missing_from_response,
        )

    parse_trace = {
        "status": "ok" if not missing_from_response else "missing_sections",
        "missing_section_ids": missing_from_response,
        "sections_total": len(blueprint_dicts),
        "sections_filled_by_llm": len(blueprint_dicts) - len(missing_from_response),
    }
    return artifacts, design_reasoning, parse_trace


def _normalize_section_artifact(entry: dict, *, sid: str) -> dict:
    """Same shape and self-healing as `_parse_section_artifact`, but applied
    to one entry from the full-mjml response's `sections[]`."""
    frag = str(entry.get("mjml_fragment", "") or "").strip()

    frag, was_trimmed = _trim_to_assigned_section(frag, sid)
    if was_trimmed:
        logger.warning(
            "[code_gen] section %s (full-mjml): fragment contained multiple "
            "<mj-section> blocks — trimmed to assigned section only",
            sid,
        )

    if frag and "data-section-id" not in frag.lower() and sid:
        frag = re.sub(
            r"<mj-section\b(?![^>]*\bdata-section-id\b)",
            f'<mj-section data-section-id="{sid}"',
            frag,
            count=1,
            flags=re.IGNORECASE,
        )
    return {
        "mjml_fragment": frag,
        "claims_used": list(entry.get("claims_used", []) or []),
        "design_reasoning": str(entry.get("design_reasoning", "") or ""),
    }


def _tolerant_json_load(raw: str) -> Any:
    """Best-effort JSON parse: strips fences, falls back to the largest
    {...} block. Returns None on total failure."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.split("\n", 1)[-1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(raw or "")
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def _parse_section_artifact(raw: str, *, sid: str) -> dict:
    """Tolerant JSON parser for the section-designer response.

    Accepts: bare JSON, fenced ```json ... ```, fenced ``` ... ```.
    Falls back to the largest `{...}` block we can find.
    """
    text = raw.strip()
    # Strip optional code fence
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.split("\n", 1)[-1] if "\n" in text else text
        # Trailing fence
        if text.endswith("```"):
            text = text[:-3]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(raw)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}

    frag = str(data.get("mjml_fragment", "") or "").strip()

    # Strip any extra <mj-section> roots the model produced beyond the
    # assigned section — enforce the one-section output contract.
    frag, was_trimmed = _trim_to_assigned_section(frag, sid)
    if was_trimmed:
        logger.warning(
            "[code_gen] section %s: model returned multiple <mj-section> "
            "blocks — trimmed to assigned section only",
            sid,
        )

    if frag and "data-section-id" not in frag.lower() and sid:
        # Inject the required attribute on the root tag (best-effort).
        frag = re.sub(
            r"<mj-section\b(?![^>]*\bdata-section-id\b)",
            f'<mj-section data-section-id="{sid}"',
            frag,
            count=1,
            flags=re.IGNORECASE,
        )

    return {
        "mjml_fragment": frag,
        "claims_used": list(data.get("claims_used", []) or []),
        "design_reasoning": str(data.get("design_reasoning", "") or ""),
    }


def _fallback_section_artifact(section: dict) -> dict:
    """Last-resort fragment used when the LLM call fails or returns garbage.

    Mirrors the spirit of the backend's `_generate_fallback_mjml_section`:
    keep the verbatim copy and headline so the compiled email still renders
    SOMETHING, even if it's unstyled. Renders as a plain text block with a
    simple headline.
    """
    sid = str(section.get("section_id", "") or "fallback")
    headline = (section.get("headline") or "").strip()
    copy = (section.get("copy_outline") or "").strip()
    headline_block = (
        f'<mj-text font-family="Arial, sans-serif" font-size="20px" '
        f'font-weight="700" color="#111111" padding="16px 24px 8px 24px">'
        f'{_escape_xml(headline)}</mj-text>'
        if headline else ""
    )
    copy_block = (
        f'<mj-text font-family="Arial, sans-serif" font-size="14px" '
        f'color="#333333" line-height="1.5" padding="0 24px 16px 24px">'
        f'{_escape_xml(copy)}</mj-text>'
        if copy else ""
    )
    fragment = (
        f'<mj-section data-section-id="{sid}" background-color="#ffffff">'
        f'<mj-column>{headline_block}{copy_block}</mj-column>'
        f'</mj-section>'
    )
    return {
        "mjml_fragment": fragment,
        "claims_used": [],
        "design_reasoning": "Fallback fragment (LLM call failed).",
    }


def _placeholder_html(*, blueprint_summary: str, layout_width: int,
                      reason: str, mjml_dump: str) -> str:
    """Visible placeholder HTML used when MJML→HTML conversion fails entirely.

    Keeps the failure visible so downstream judges can score this candidate
    LOWER than a real candidate, rather than silently rejecting it.
    """
    safe_summary = _escape_xml(blueprint_summary)
    safe_reason = _escape_xml(reason or "unknown error")
    safe_mjml = _escape_xml((mjml_dump or "")[:2000])
    return (
        f'<!DOCTYPE html><html><body style="max-width:{layout_width}px;'
        f'margin:0 auto;font-family:sans-serif;color:#900;background:#fff5f5">'
        f'<div style="padding:24px;border:2px dashed #c00;border-radius:6px">'
        f'<h2 style="margin-top:0">MJML render failed</h2>'
        f'<p><strong>Reason:</strong> {safe_reason}</p>'
        f'<details><summary>blueprint</summary><pre style="font-size:11px">'
        f'{safe_summary}</pre></details>'
        f'<details><summary>partial MJML</summary><pre style="font-size:11px">'
        f'{safe_mjml}</pre></details>'
        f'</div></body></html>'
    )


def _escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
    )


def _section_to_dict(s) -> dict:
    if isinstance(s, BlueprintSection):
        return asdict(s)
    if isinstance(s, dict):
        return s
    raise TypeError(f"Unsupported blueprint section type: {type(s).__name__}")


# ─────────────────────────────────────────────────────────────────────────────
# Concurrency config
# ─────────────────────────────────────────────────────────────────────────────

def _load_concurrency() -> tuple[int, int]:
    """Read concurrency caps from `config/pipeline_config.yaml`.

    Expected (all optional):
        concurrency:
          per_section: 6
          per_candidate: 3
    """
    try:
        with open(_PIPELINE_CONFIG) as f:
            cfg = yaml.safe_load(f) or {}
        conc = (cfg.get("concurrency", {}) if isinstance(cfg, dict) else {}) or {}
        per_section = int(conc.get("per_section", _DEFAULT_SECTION_CONCURRENCY))
        per_candidate = int(conc.get("per_candidate", _DEFAULT_CANDIDATE_CONCURRENCY))
        return max(1, per_section), max(1, per_candidate)
    except Exception:
        return _DEFAULT_SECTION_CONCURRENCY, _DEFAULT_CANDIDATE_CONCURRENCY


def _load_max_tokens() -> tuple[int, int]:
    """Read per-mode max_tokens from `config/pipeline_config.yaml`.

    Returns (full_max_tokens, section_max_tokens).
    """
    try:
        with open(_PIPELINE_CONFIG) as f:
            cfg = yaml.safe_load(f) or {}
        mt = (cfg.get("max_tokens", {}) if isinstance(cfg, dict) else {}) or {}
        full = max(1, int(mt.get("code_generator_full", _FULL_MAX_TOKENS)))
        section = max(1, int(mt.get("code_generator_section", _SECTION_MAX_TOKENS)))
        return full, section
    except Exception:
        return _FULL_MAX_TOKENS, _SECTION_MAX_TOKENS


def _load_candidate_mode() -> str:
    """Read the candidate generation mode from `config/pipeline_config.yaml`.

    Expected:
        pipeline:
          candidate_mode: "full_mjml"   # or "section_wise"

    Falls back to the default ("full_mjml") on any error or unknown value.
    """
    try:
        with open(_PIPELINE_CONFIG) as f:
            cfg = yaml.safe_load(f) or {}
        pipeline_cfg = (cfg.get("pipeline", {}) if isinstance(cfg, dict) else {}) or {}
        mode = str(pipeline_cfg.get("candidate_mode", _DEFAULT_CANDIDATE_MODE) or
                   _DEFAULT_CANDIDATE_MODE).lower().strip()
        if mode not in _VALID_CANDIDATE_MODES:
            logger.warning(
                "[code_gen] unknown candidate_mode=%r — falling back to %r",
                mode, _DEFAULT_CANDIDATE_MODE,
            )
            return _DEFAULT_CANDIDATE_MODE
        return mode
    except Exception:
        return _DEFAULT_CANDIDATE_MODE
