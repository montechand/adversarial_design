"""
nodes/aggregator.py

Stage 6: combine quality_score + anti_ai_rank → aggregate_score, select
winner, persist artifacts, log the winning prompt.

The winner can be the baseline (i.e. none of the generated candidates beat
what the backend produced). When that happens we still emit a winner so
downstream consumers always get a deterministic best output.

Scoring formula (configurable via pipeline_config.yaml):
    aggregate = (quality_score / 10) * quality_weight
              + (1 - normalized_anti_ai_rank) * anti_ai_weight

where normalized_anti_ai_rank = (rank - 1) / max(n - 1, 1)
so rank=1 (least AI-like) → 0.0 → contributes max to score.
"""

from __future__ import annotations
import os
import json
import logging
from dataclasses import asdict

import yaml

from pipeline.state import PipelineState, Candidate
from pipeline.utils.html_compression import rehydrate_html
from pipeline.utils.storage import run_dir

logger = logging.getLogger(__name__)

PIPELINE_CONFIG = os.path.join(os.path.dirname(__file__), "../../config/pipeline_config.yaml")

_DEFAULT_QUALITY_WEIGHT = 0.55
_DEFAULT_ANTI_AI_WEIGHT = 0.45


def aggregate_and_promote(state: PipelineState) -> PipelineState:
    quality_w, anti_ai_w = _load_weights()
    candidates: list[Candidate] = state.get("candidates", []) or []
    n = len(candidates)

    scored: list[tuple[float, Candidate]] = []
    for c in candidates:
        if not c.compliance_passed:
            continue
        q = (c.quality_score or 0.0) / 10.0
        r = c.anti_ai_rank or n
        anti_ai_norm = 1.0 - ((r - 1) / max(n - 1, 1))
        agg = (q * quality_w) + (anti_ai_norm * anti_ai_w)
        scored.append((agg, c))

    if not scored:
        logger.warning("[aggregate] no compliant candidates — returning state unchanged")
        return state

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, winner = scored[0]
    updated_winner = Candidate(**{**winner.__dict__, "aggregate_score": best_score})

    _save_run_artifacts(state, scored, weights=(quality_w, anti_ai_w))
    _log_prompt(
        run_id=state.get("run_id", "default_run"),
        iteration=state.get("iteration", 0),
        winner=updated_winner,
        score=best_score,
    )

    return {**state, "winner": updated_winner}


# ─────────────────────────────────────────────────────────────────────────────

def _load_weights() -> tuple[float, float]:
    try:
        with open(PIPELINE_CONFIG) as f:
            cfg = yaml.safe_load(f) or {}
        scoring = cfg.get("scoring", {}) if isinstance(cfg, dict) else {}
        return (
            float(scoring.get("quality_weight", _DEFAULT_QUALITY_WEIGHT) or _DEFAULT_QUALITY_WEIGHT),
            float(scoring.get("anti_ai_weight", _DEFAULT_ANTI_AI_WEIGHT) or _DEFAULT_ANTI_AI_WEIGHT),
        )
    except Exception:
        return _DEFAULT_QUALITY_WEIGHT, _DEFAULT_ANTI_AI_WEIGHT


def _save_run_artifacts(
    state: PipelineState,
    scored: list[tuple[float, Candidate]],
    weights: tuple[float, float],
) -> None:
    out_dir = run_dir(state.get("run_id", "default_run"))
    os.makedirs(out_dir, exist_ok=True)

    payload = [
        {
            "candidate_id": c.candidate_id,
            "is_baseline": c.is_baseline,
            "aggregate_score": score,
            "quality_score": c.quality_score,
            "quality_rationale": c.quality_rationale,
            "anti_ai_rank": c.anti_ai_rank,
            "compliance_passed": c.compliance_passed,
            "compliance_reason": c.compliance_reason,
            "prompt_metadata": c.prompt_metadata,
        }
        for score, c in scored
    ]
    with open(os.path.join(out_dir, "scores.json"), "w") as f:
        json.dump({"weights": {"quality": weights[0], "anti_ai": weights[1]}, "candidates": payload}, f, indent=2)

    best_score, winner = scored[0]
    winner = Candidate(**{**winner.__dict__, "aggregate_score": best_score})
    images_b64 = state.get("baseline_images_b64", {}) or {}
    # Persist on-disk artifacts with [images_base64_N] tokens rehydrated to
    # their original data URIs so the winner files open directly in any
    # browser without a sidecar map.
    with open(os.path.join(out_dir, "winner.html"), "w", encoding="utf-8") as f:
        f.write(rehydrate_html(winner.html, images_b64))
    if winner.mjml_document:
        with open(os.path.join(out_dir, "winner.mjml"), "w", encoding="utf-8") as f:
            f.write(rehydrate_html(winner.mjml_document, images_b64))
    with open(os.path.join(out_dir, "winner.prompt.txt"), "w", encoding="utf-8") as f:
        f.write(winner.prompt_used)
    with open(os.path.join(out_dir, "winner.meta.json"), "w", encoding="utf-8") as f:
        meta = dict(asdict(winner))
        # Avoid dumping the full HTML / MJML twice
        meta.pop("html", None)
        meta.pop("mjml_document", None)
        json.dump(meta, f, indent=2, default=str)


def _log_prompt(*, run_id: str, iteration: int, winner: Candidate, score: float) -> None:
    log_path = os.path.join("storage", "prompt_history.jsonl")
    os.makedirs("storage", exist_ok=True)
    entry = {
        "run_id": run_id,
        "iteration": iteration,
        "score": score,
        "is_baseline_winner": winner.is_baseline,
        "metadata": winner.prompt_metadata,
        "prompt": winner.prompt_used,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
