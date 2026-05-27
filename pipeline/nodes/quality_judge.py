"""
nodes/quality_judge.py

Stage 5b: VLM judge that scores each compliant candidate on creativity,
HCP relevance, layout, and how much it beats the baseline.

Pairwise / batch scoring (all compliant candidates in one VLM call) — more
stable than independent scoring when n ≤ 5.

For each candidate we send:
  - compressed HTML source (text)
  - ordered screenshot slices (images)

The judge also receives the baseline DesignDescriptor as an anchor for the
`beats_baseline` dimension.
"""

from __future__ import annotations
import json
import logging
import os
from dataclasses import asdict
from typing import Any

import yaml

from pipeline.state import PipelineState, Candidate, BlueprintSection
from pipeline.utils.llm_clients import get_vlm_client
from pipeline.utils.image_utils import encode_image_b64
from pipeline.prompts.quality_judge import SYSTEM_PROMPT, build_user_prompt
from pipeline.prompts.reverse_prompter import build_blueprint_summary
from pipeline.prompts.creativizer import summarize_descriptor

logger = logging.getLogger(__name__)

_VLM_MODEL = "claude-opus-4-6"
_MAX_TOKENS = 4096  # fallback if config is missing

_PIPELINE_CONFIG = os.path.join(os.path.dirname(__file__), "../../config/pipeline_config.yaml")


def judge_quality(state: PipelineState) -> PipelineState:
    client = get_vlm_client()
    compliant = [c for c in (state.get("candidates", []) or []) if c.compliance_passed]
    if not compliant:
        return {**state, "quality_scores": []}

    blueprint_summary = build_blueprint_summary([
        _section_to_dict(s) for s in (state.get("blueprint_sections", []) or [])
    ])
    baseline = state.get("baseline_descriptor")
    baseline_summary = summarize_descriptor(asdict(baseline)) if baseline else "(no baseline available)"

    cand_payloads: list[dict[str, Any]] = []
    for c in compliant:
        slices_b64 = _encode_slices(c.screenshot_slice_paths or ([c.screenshot_path] if c.screenshot_path else []))
        cand_payloads.append({
            "html_compressed": c.html or "",
            "slices_b64": slices_b64,
        })

    try:
        response = client.messages.create(
            model=_VLM_MODEL,
            max_tokens=_load_max_tokens(),
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": build_user_prompt(
                        blueprint_summary=blueprint_summary,
                        baseline_descriptor_summary=baseline_summary,
                        candidates=cand_payloads,
                    ),
                }
            ],
        )
        scores_data = _parse_scores(response.content[0].text, n=len(compliant))
    except Exception as e:
        logger.warning("[quality_judge] VLM call failed: %s", e)
        scores_data = [{"overall": 5.0, "rationale": f"judge_error: {e}"} for _ in compliant]

    # Map scores back into the full candidate list
    all_scores: list[float] = []
    updated: list[Candidate] = []
    score_iter = iter(scores_data)
    for cand in state.get("candidates", []) or []:
        if cand.compliance_passed:
            entry = next(score_iter, {})
            score = float(entry.get("overall", 0.0) or 0.0)
            updated.append(Candidate(**{
                **cand.__dict__,
                "quality_score": score,
                "quality_rationale": str(entry.get("rationale", "") or ""),
            }))
            all_scores.append(score)
        else:
            updated.append(cand)
            all_scores.append(0.0)

    return {**state, "candidates": updated, "quality_scores": all_scores}


def _parse_scores(raw: str, n: int) -> list[dict]:
    try:
        json_str = raw.split("```json")[-1].split("```")[0].strip()
        data = json.loads(json_str)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except Exception:
        pass
    return [{"overall": 5.0, "rationale": "parse_error"} for _ in range(n)]


def _encode_slices(paths: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for p in paths:
        if not p:
            continue
        try:
            out.append(("image/png", encode_image_b64(p)))
        except Exception as e:
            logger.warning("[quality_judge] could not encode %s: %s", p, e)
    return out


def _section_to_dict(s) -> dict:
    if isinstance(s, BlueprintSection):
        return asdict(s)
    if isinstance(s, dict):
        return s
    raise TypeError(f"Unsupported blueprint section type: {type(s).__name__}")


def _load_max_tokens() -> int:
    try:
        with open(_PIPELINE_CONFIG) as f:
            cfg = yaml.safe_load(f) or {}
        mt = (cfg.get("max_tokens", {}) if isinstance(cfg, dict) else {}) or {}
        return max(1, int(mt.get("quality_judge", _MAX_TOKENS)))
    except Exception:
        return _MAX_TOKENS
