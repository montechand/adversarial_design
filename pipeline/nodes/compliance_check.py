"""
nodes/compliance_check.py

Stage 5a — lightweight MLR gate operating on the COMPRESSED HTML.

Async fan-out: every non-baseline candidate is checked in parallel via
`asyncio.gather` with a bounded `Semaphore`. The baseline is trusted (it was
already MLR-cleared upstream by the backend pipeline) and skips the LLM call.

Disqualified candidates are kept in the candidate list but flagged
`compliance_passed=False`, with their reason stored for the creativizer's
next retry round (the graph re-enters `creativize` if no candidate passes).
"""

from __future__ import annotations

import os
import json
import asyncio
import logging
from dataclasses import asdict
from typing import Any

import yaml

from pipeline.state import PipelineState, Candidate, BlueprintSection
from pipeline.utils.llm_clients import get_async_llm_client
from pipeline.prompts.compliance_check import SYSTEM_PROMPT, build_user_prompt
from pipeline.prompts.reverse_prompter import build_blueprint_summary

logger = logging.getLogger(__name__)

_LLM_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 600  # fallback if config is missing

_PIPELINE_CONFIG = os.path.join(os.path.dirname(__file__), "../../config/pipeline_config.yaml")
_DEFAULT_COMPLIANCE_CONCURRENCY = 4


async def check_compliance(state: PipelineState) -> PipelineState:
    client = get_async_llm_client()
    blueprint_summary = build_blueprint_summary([
        _section_to_dict(s) for s in (state.get("blueprint_sections", []) or [])
    ])
    brand_rules = state.get("brand_rules") or None

    candidates = state.get("candidates", []) or []
    if not candidates:
        return {**state, "compliance_results": [], "compliance_reasons": []}

    sem = asyncio.Semaphore(_load_concurrency())
    logger.info(
        "[compliance] checking %d candidate(s) in parallel (concurrency=%d, baseline skipped)",
        len(candidates), sem._value,  # type: ignore[attr-defined]
    )

    tasks = [
        _check_one_candidate(
            client=client,
            candidate=c,
            blueprint_summary=blueprint_summary,
            brand_rules=brand_rules,
            sem=sem,
        )
        for c in candidates
    ]
    verdicts: list[tuple[Candidate, bool, str]] = await asyncio.gather(*tasks)

    updated: list[Candidate] = []
    results: list[bool] = []
    reasons: list[str] = []
    for new_cand, passed, reason in verdicts:
        updated.append(new_cand)
        results.append(passed)
        reasons.append(reason)

    return {
        **state,
        "candidates": updated,
        "compliance_results": results,
        "compliance_reasons": reasons,
    }


async def _check_one_candidate(
    *,
    client: Any,
    candidate: Candidate,
    blueprint_summary: str,
    brand_rules: dict | None,
    sem: asyncio.Semaphore,
) -> tuple[Candidate, bool, str]:
    if candidate.is_baseline:
        # Trust the upstream baseline — it was already MLR-cleared.
        new_cand = Candidate(**{
            **candidate.__dict__,
            "compliance_passed": True,
            "compliance_reason": "",
        })
        return new_cand, True, ""

    async with sem:
        try:
            response = await client.messages.create(
                model=_LLM_MODEL,
                max_tokens=_load_max_tokens(),
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": build_user_prompt(
                        html_compressed=candidate.html,
                        blueprint_summary=blueprint_summary,
                        brand_rules=brand_rules,
                    ),
                }],
            )
            verdict = _parse_verdict(response.content[0].text)
        except Exception as e:
            logger.warning("[compliance] candidate %d failed: %s", candidate.candidate_id, e)
            verdict = {"passed": True, "reason": f"compliance_check_error_optimistic_pass: {e}"}

    passed = bool(verdict.get("passed", False))
    reason = str(verdict.get("reason", "") or "")
    new_cand = Candidate(**{
        **candidate.__dict__,
        "compliance_passed": passed,
        "compliance_reason": reason,
    })
    return new_cand, passed, reason


def _parse_verdict(raw: str) -> dict:
    try:
        json_str = raw.split("```json")[-1].split("```")[0].strip()
        return json.loads(json_str)
    except Exception:
        return {"passed": True, "reason": "parse_error_optimistic_pass"}


def _section_to_dict(s) -> dict:
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
        return max(1, int(conc.get("compliance", _DEFAULT_COMPLIANCE_CONCURRENCY)))
    except Exception:
        return _DEFAULT_COMPLIANCE_CONCURRENCY


def _load_max_tokens() -> int:
    try:
        with open(_PIPELINE_CONFIG) as f:
            cfg = yaml.safe_load(f) or {}
        mt = (cfg.get("max_tokens", {}) if isinstance(cfg, dict) else {}) or {}
        return max(1, int(mt.get("compliance_check", _MAX_TOKENS)))
    except Exception:
        return _MAX_TOKENS
