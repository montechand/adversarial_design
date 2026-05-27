"""
graph.py — LangGraph StateGraph definition.

Wires all nodes together into the full pipeline. Most nodes are async — the
candidate generation flow fans out per-candidate × per-section LLM calls
with `asyncio.gather`, and every other node that loops over candidates does
the same. We therefore drive the graph with `ainvoke` rather than `invoke`.

Two entry points:
  - `build_graph()`         compiled StateGraph for programmatic use.
  - `run_from_baseline()`   async convenience runner used by the CLI.
  - `run_from_baseline_sync()` thin sync wrapper that calls asyncio.run for
                              backwards compatibility with the previous CLI.

There is no longer a free-form `blueprint: str` runner — the structured
contract is mandatory because that's what the backend-server will hand us.
"""

from __future__ import annotations

import asyncio
import datetime
import os
from typing import Any

import yaml
from langgraph.graph import StateGraph, END

from pipeline.state import PipelineState, BlueprintSection, ReferenceExample
from pipeline.nodes.ingestion import ingest_examples
from pipeline.utils.reference_examples import normalize_reference_examples
from pipeline.nodes.reverse_prompter import reverse_prompt
from pipeline.nodes.creativizer import creativize
from pipeline.nodes.code_generator import generate_candidates
from pipeline.nodes.screenshotter import screenshot_candidates
from pipeline.nodes.compliance_check import check_compliance
from pipeline.nodes.quality_judge import judge_quality
from pipeline.nodes.anti_ai_judge import judge_anti_ai
from pipeline.nodes.aggregator import aggregate_and_promote


def build_graph():
    g = StateGraph(PipelineState)

    g.add_node("ingest_examples",       ingest_examples)
    g.add_node("reverse_prompt",        reverse_prompt)
    g.add_node("creativize",            creativize)
    g.add_node("generate_candidates",   generate_candidates)
    g.add_node("screenshot_candidates", screenshot_candidates)
    g.add_node("check_compliance",      check_compliance)
    g.add_node("judge_quality",         judge_quality)
    g.add_node("judge_anti_ai",         judge_anti_ai)
    g.add_node("aggregate_and_promote", aggregate_and_promote)

    g.set_entry_point("ingest_examples")
    g.add_edge("ingest_examples",       "reverse_prompt")
    g.add_edge("reverse_prompt",        "creativize")
    g.add_edge("creativize",            "generate_candidates")
    g.add_edge("generate_candidates",   "screenshot_candidates")
    g.add_edge("screenshot_candidates", "check_compliance")

    # Compliance gate: if at least one non-baseline candidate passes, judge.
    # Otherwise retry creativizer with the compliance feedback we collected.
    g.add_conditional_edges(
        "check_compliance",
        _any_non_baseline_compliant,
        {
            True:  "judge_quality",
            False: "creativize",
        },
    )

    g.add_edge("judge_quality",         "judge_anti_ai")
    g.add_edge("judge_anti_ai",         "aggregate_and_promote")
    g.add_edge("aggregate_and_promote", END)

    return g.compile()


def _any_non_baseline_compliant(state: PipelineState) -> bool:
    return any(
        c.compliance_passed for c in (state.get("candidates", []) or []) if not c.is_baseline
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience runners — used by scripts/run_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

async def run_from_baseline(
    *,
    blueprint_sections: list[BlueprintSection] | list[dict[str, Any]],
    base_prompt: str,
    baseline_html_compressed: str,
    baseline_images_b64: dict[str, str],
    baseline_image_descriptors: dict[str, dict] | None = None,
    baseline_screenshot_slices: list[str] | None = None,
    example_html_paths: list[str] | None = None,
    reference_examples: list[ReferenceExample] | list[dict] | None = None,
    iterations: int = 1,
    run_id: str | None = None,
    layout_width: int = 600,
    slice_height: int = 1600,
    brand_rules: dict | None = None,
) -> PipelineState:
    """
    Run the pipeline `iterations` times. Between iterations the winner's
    prompt is promoted as the new `base_prompt` (so each iteration tries to
    beat the last winner, not the original baseline).

    Returns the final PipelineState (winner accessible via state['winner']).

    Async: drives `graph.ainvoke` so every node's `asyncio.gather` calls
    actually run in parallel. Use `run_from_baseline_sync` if you need the
    legacy sync entrypoint.
    """
    norm_sections: list[BlueprintSection] = []
    for s in blueprint_sections:
        if isinstance(s, BlueprintSection):
            norm_sections.append(s)
        elif isinstance(s, dict):
            norm_sections.append(BlueprintSection.from_dict(s))
        else:
            raise TypeError(
                f"blueprint_sections entries must be BlueprintSection or dict, got {type(s).__name__}"
            )
    norm_sections.sort(key=lambda x: x.order)

    norm_refs = normalize_reference_examples(reference_examples or [])

    graph = build_graph()
    rid = run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    state: PipelineState = {
        "blueprint_sections": norm_sections,
        "base_prompt": base_prompt or "",
        "baseline_html_compressed": baseline_html_compressed,
        "baseline_images_b64": baseline_images_b64 or {},
        "baseline_image_descriptors": baseline_image_descriptors or {},
        "baseline_screenshot_slices": baseline_screenshot_slices or [],
        "example_html_paths": example_html_paths or [],
        "example_screenshot_paths": [],
        "reference_examples": norm_refs,
        "design_descriptors": [],
        "baseline_descriptor": None,
        "candidate_prompts": [],
        "candidate_prompt_metadata": [],
        "candidates": [],
        "compliance_results": [],
        "compliance_reasons": [],
        "quality_scores": [],
        "anti_ai_ranks": [],
        "winner": None,
        "winner_feedback": None,
        "iteration": 0,
        "run_id": rid,
        "layout_width": layout_width,
        "slice_height": slice_height,
        "brand_rules": brand_rules or None,
    }

    feedback_enabled = _winner_feedback_enabled()

    for i in range(max(iterations, 1)):
        state["iteration"] = i
        state = await graph.ainvoke(state)
        winner = state.get("winner") if isinstance(state, dict) else None
        if winner is not None and not winner.is_baseline:
            state["base_prompt"] = winner.prompt_used
            if feedback_enabled:
                rationale = getattr(winner, "quality_rationale", None)
                state["winner_feedback"] = rationale or None

    return state


def run_from_baseline_sync(**kwargs) -> PipelineState:
    """Sync wrapper around `run_from_baseline` for callers that aren't async
    yet. Internally calls `asyncio.run`."""
    return asyncio.run(run_from_baseline(**kwargs))


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

_PIPELINE_CONFIG = os.path.join(os.path.dirname(__file__), "../config/pipeline_config.yaml")


def _winner_feedback_enabled() -> bool:
    """Read feedback.winner_feedback_to_reverse_prompter from pipeline_config.yaml."""
    try:
        with open(_PIPELINE_CONFIG) as f:
            cfg = yaml.safe_load(f) or {}
        fb = (cfg.get("feedback", {}) if isinstance(cfg, dict) else {}) or {}
        return bool(fb.get("winner_feedback_to_reverse_prompter", True))
    except Exception:
        return True
