"""utils/storage.py — path helpers for run artifact storage."""
import os


def run_dir(run_id: str) -> str:
    path = os.path.join("storage", "runs", run_id)
    os.makedirs(path, exist_ok=True)
    return path


def save_scores_log(run_id: str, scores: list[dict]) -> None:
    import json
    path = os.path.join(run_dir(run_id), "scores.json")
    with open(path, "w") as f:
        json.dump(scores, f, indent=2)
