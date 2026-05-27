"""
nodes/anti_ai_judge.py

Stage 5c: VLM judge that ranks candidates from LEAST to MOST AI-like,
relative to a set of "human-designed" reference screenshots.

Reference pool, in priority order:
  1. `example_screenshot_paths` if provided (legacy multi-reference mode)
  2. `baseline_screenshot_slices` (the baseline produced by the upstream
     backend is treated as the "comparable human-designed" example here —
     candidates are pushed to be more distinctive than it).

Skips the baseline candidate itself when scoring.
"""

from __future__ import annotations
import json
import logging
import os
from pipeline.state import PipelineState, Candidate
from pipeline.utils.llm_clients import get_vlm_client
from pipeline.utils.image_utils import encode_image_b64
from pipeline.prompts.anti_ai_judge import SYSTEM_PROMPT, build_user_prompt

import yaml

logger = logging.getLogger(__name__)

_VLM_MODEL = "claude-opus-4-6"
_MAX_TOKENS = 8192  # fallback if config is missing

_PIPELINE_CONFIG = os.path.join(os.path.dirname(__file__), "../../config/pipeline_config.yaml")


def judge_anti_ai(state: PipelineState) -> PipelineState:
    candidates = state.get("candidates", []) or []
    worst_rank = len(candidates) + 1

    # Compliant non-baseline candidates that we rank
    rankable = [c for c in candidates if c.compliance_passed and not c.is_baseline]
    if not rankable:
        all_ranks = [worst_rank for _ in candidates]
        return {**state, "anti_ai_ranks": all_ranks}

    # Reference pool — prefer external references; fall back to baseline slices
    ref_paths = list(state.get("example_screenshot_paths", []) or [])
    if not ref_paths:
        ref_paths = list(state.get("baseline_screenshot_slices", []) or [])
    reference_b64 = _encode(ref_paths)

    candidate_b64: list[list[tuple[str, str]]] = []
    for c in rankable:
        paths = c.screenshot_slice_paths or ([c.screenshot_path] if c.screenshot_path else [])
        candidate_b64.append(_encode(paths))

    try:
        client = get_vlm_client()
        response = client.messages.create(
            model=_VLM_MODEL,
            max_tokens=_load_max_tokens(),
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": build_user_prompt(
                        reference_slices_b64=reference_b64,
                        candidate_slices_b64=candidate_b64,
                    ),
                }
            ],
        )
        ranks_local = _parse_ranks(response.content[0].text, n=len(rankable))
    except Exception as e:
        logger.warning("[anti_ai_judge] VLM call failed: %s", e)
        ranks_local = list(range(1, len(rankable) + 1))

    # `ranks_local[i]` = rank assigned to rankable[i] (1=best)
    # Map back to the full candidates list
    rankable_id_to_rank = {c.candidate_id: r for c, r in zip(rankable, ranks_local)}
    all_ranks: list[int] = []
    updated: list[Candidate] = []
    for c in candidates:
        if c.is_baseline:
            # Baseline is the anchor — give it the median rank so it neither
            # wins nor loses purely on this dimension.
            rank = max(1, (len(rankable) + 1) // 2)
        elif not c.compliance_passed:
            rank = worst_rank
        else:
            rank = rankable_id_to_rank.get(c.candidate_id, worst_rank)
        all_ranks.append(rank)
        updated.append(Candidate(**{**c.__dict__, "anti_ai_rank": rank}))

    return {**state, "candidates": updated, "anti_ai_ranks": all_ranks}


def _parse_ranks(raw: str, n: int) -> list[int]:
    """Parse `{"ranking": [...]}` then convert the ordering to per-candidate ranks."""
    try:
        json_str = raw.split("```json")[-1].split("```")[0].strip()
        data = json.loads(json_str)
        ordering = data.get("ranking") if isinstance(data, dict) else None
        if isinstance(ordering, list):
            # ordering[k] = candidate_local_index at rank k+1
            per_candidate = [n] * n
            for rank_pos, local_idx in enumerate(ordering):
                if isinstance(local_idx, int) and 0 <= local_idx < n:
                    per_candidate[local_idx] = rank_pos + 1
            return per_candidate
    except Exception:
        pass
    return list(range(1, n + 1))


def _load_max_tokens() -> int:
    try:
        with open(_PIPELINE_CONFIG) as f:
            cfg = yaml.safe_load(f) or {}
        mt = (cfg.get("max_tokens", {}) if isinstance(cfg, dict) else {}) or {}
        return max(1, int(mt.get("anti_ai_judge", _MAX_TOKENS)))
    except Exception:
        return _MAX_TOKENS


def _encode(paths: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for p in paths:
        if not p:
            continue
        try:
            out.append(("image/png", encode_image_b64(p)))
        except Exception as e:
            logger.warning("[anti_ai_judge] could not encode %s: %s", p, e)
    return out
