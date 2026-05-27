#!/usr/bin/env python3
"""
scripts/run_pipeline.py — CLI entrypoint for the adversarial-design pipeline.

This is the standalone runner. The eventual integration with the
backend-server email generator will call `pipeline.graph.run_from_baseline`
directly with the same three core inputs:

  1. blueprint sections           (mirrors backend `content_blueprint`)
  2. baseline HTML (compressed)   (with `[images_base64_N]` placeholders)
  3. baseline images map          (placeholder → base64 data URI)

Usage — explicit inputs:

    python scripts/run_pipeline.py \\
        --blueprint-json examples/blueprint.json \\
        --baseline-html  examples/baseline_email.html \\
        --baseline-images examples/baseline_images_b64.json \\
        --baseline-screenshots examples/baseline_screenshots \\
        --iterations 1

Usage — bundled example (does everything above):

    python scripts/run_pipeline.py --from-example examples
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import datetime
import logging
from pathlib import Path

# Allow running this script directly (without `pip install -e .`)
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# Load .env from adversarial_design/ before any pipeline imports touch os.environ
from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env", override=False)

from pipeline.graph import run_from_baseline
from pipeline.state import BlueprintSection
from pipeline.utils.html_compression import (
    compress_html_base64,
    load_images_b64_with_descriptors,
)
from pipeline.utils.reference_examples import discover_reference_examples
from pipeline.utils.style_guide import load_style_guide


async def amain() -> int:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")

    p = argparse.ArgumentParser(
        description="Adversarial pharma email design pipeline (baseline-vs-candidates).",
    )
    p.add_argument("--from-example", default=None,
                   help="Convenience: load blueprint/baseline/images/screenshots from "
                        "the given example directory (e.g. examples/).")
    p.add_argument("--blueprint-json", default=None,
                   help="Path to JSON list of section dicts (matches backend SectionSpec). "
                        "Either {\"content_blueprint\": [...]} or a bare [...] array.")
    p.add_argument("--baseline-html", default=None,
                   help="Path to the baseline HTML. May contain raw data:image base64 "
                        "URIs OR `[images_base64_N]` placeholders — both are accepted.")
    p.add_argument("--baseline-images", default=None,
                   help="Path to a JSON map of {placeholder_token: data_uri or raw_b64}. "
                        "Required only if the HTML already uses placeholder tokens.")
    p.add_argument("--baseline-screenshots", default=None,
                   help="Directory of pre-sliced PNGs of the baseline rendering. "
                        "If omitted, the pipeline will render + slice from the HTML.")
    p.add_argument("--examples-dir", default=None,
                   help="Optional directory of human-designed reference HTMLs for the "
                        "anti-AI judge to use as a `human baseline` (legacy mode).")
    p.add_argument("--reference-examples", default=None,
                   help="Directory of HUMAN-MADE CREATIVE REFERENCE ADS. Each .html "
                        "is auto-rendered (Playwright); each .png/.jpg is used "
                        "as-is. Optional same-stem .md/.txt files are read as "
                        "captions. These are interleaved with their captions into "
                        "the reverse prompter's BASELINE VLM call as comparison "
                        "anchors. Defaults to `<from-example>/references/` when "
                        "--from-example is used, or `examples/references/` otherwise.")
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--layout-width", type=int, default=600)
    p.add_argument("--slice-height", type=int, default=1600)
    p.add_argument("--base-prompt", default=None,
                   help="Override seed base prompt (default: storage/base_prompt.txt or built-in).")
    p.add_argument("--style-guide", default=None,
                   help="Path to a style_guide_ruleset.json. When supplied, the four rule lists "
                        "(content_pattern_rules, color_scheme_rules, design_pattern_rules, "
                        "other_rules) are injected as SUPREME brand constraints into the "
                        "creativizer, code_generator, and compliance_check prompts. "
                        "Defaults to examples/style_guide_ruleset.json if it exists.")
    args = p.parse_args()

    # ── Resolve inputs ─────────────────────────────────────────────────────
    if args.from_example:
        ex = Path(args.from_example)
        blueprint_path = ex / "blueprint.json"
        baseline_html_path = ex / "baseline_email.html"
        baseline_images_path = ex / "baseline_images_b64.json"
        baseline_screenshots_dir = ex / "baseline_screenshots"
        default_references_dir: Path | None = ex / "references"
    else:
        if not (args.blueprint_json and args.baseline_html):
            p.error("--blueprint-json and --baseline-html are required (or use --from-example)")
        blueprint_path = Path(args.blueprint_json)
        baseline_html_path = Path(args.baseline_html)
        baseline_images_path = Path(args.baseline_images) if args.baseline_images else None
        baseline_screenshots_dir = Path(args.baseline_screenshots) if args.baseline_screenshots else None
        default_references_dir = Path("examples/references")

    blueprint_sections = _load_blueprint(blueprint_path)
    baseline_html_raw = baseline_html_path.read_text(encoding="utf-8")

    # Compress the HTML if needed and produce the image map (+ descriptors).
    baseline_image_descriptors: dict[str, dict] = {}
    if "[images_base64_" in baseline_html_raw:
        baseline_html_compressed = baseline_html_raw
        if baseline_images_path and baseline_images_path.exists():
            baseline_images, baseline_image_descriptors = (
                load_images_b64_with_descriptors(baseline_images_path)
            )
        else:
            baseline_images = {}
            print(
                f"[warn] HTML uses [images_base64_N] tokens but no --baseline-images JSON "
                f"was supplied; unknown tokens will render as a 1x1 transparent PNG.",
                file=sys.stderr,
            )
    else:
        compressed, extracted = compress_html_base64(baseline_html_raw)
        baseline_html_compressed = compressed
        baseline_images = dict(extracted)
        # Merge caller-supplied sidecar entries only for tokens that were NOT
        # extracted from this HTML. This avoids stale sidecars silently
        # overriding fresh extraction and breaking image-token alignment.
        if baseline_images_path and baseline_images_path.exists():
            sidecar_images, sidecar_descriptors = load_images_b64_with_descriptors(
                baseline_images_path
            )
            conflicting = [
                token for token in sidecar_images
                if token in baseline_images and sidecar_images[token] != baseline_images[token]
            ]
            if conflicting:
                print(
                    f"[warn] baseline sidecar {baseline_images_path} has {len(conflicting)} token "
                    "value mismatch(es) vs extracted HTML map; keeping extracted HTML values "
                    f"for those token(s): {', '.join(conflicting[:5])}"
                    + (" ..." if len(conflicting) > 5 else "")
                )
            for token, value in sidecar_images.items():
                baseline_images.setdefault(token, value)
            # Adopt sidecar descriptors verbatim — they're authoritative on
            # how the baseline used each image. Ingestion will fill in any
            # tokens missing from the sidecar by mining from the HTML.
            baseline_image_descriptors = dict(sidecar_descriptors)

    # Slices
    baseline_screenshots: list[str] = []
    if baseline_screenshots_dir and baseline_screenshots_dir.exists() and baseline_screenshots_dir.is_dir():
        baseline_screenshots = sorted(str(p) for p in baseline_screenshots_dir.glob("*.png"))

    # Reference examples (optional, legacy)
    example_html_paths: list[str] = []
    if args.examples_dir and Path(args.examples_dir).exists():
        example_html_paths = sorted(str(p) for p in Path(args.examples_dir).glob("*.html"))

    # Human-made creative reference ads — interleaved into the reverse
    # prompter's baseline VLM call.
    if args.reference_examples:
        reference_dir = Path(args.reference_examples)
    elif default_references_dir and default_references_dir.exists() and default_references_dir.is_dir():
        reference_dir = default_references_dir
    else:
        reference_dir = None
    reference_examples = discover_reference_examples(reference_dir)
    if reference_dir and not reference_examples:
        print(f"[warn] reference_examples dir {reference_dir} found but no usable assets discovered")

    # Base prompt
    base_prompt_file = Path("storage/base_prompt.txt")
    if args.base_prompt:
        base_prompt = args.base_prompt
    elif base_prompt_file.exists():
        base_prompt = base_prompt_file.read_text().strip()
    else:
        base_prompt = _default_base_prompt()
        base_prompt_file.parent.mkdir(parents=True, exist_ok=True)
        base_prompt_file.write_text(base_prompt)

    # Style guide — auto-discover from examples/ if not explicitly supplied
    style_guide_path = None
    if args.style_guide:
        style_guide_path = Path(args.style_guide)
    elif args.from_example:
        ex_dir = Path(args.from_example)
        for name in ("style_guide_ruleset_ultomiris.json", "style_guide_ruleset.json"):
            candidate = ex_dir / name
            if candidate.exists():
                style_guide_path = candidate
                break
    else:
        for name in ("style_guide_ruleset_ultomiris.json", "style_guide_ruleset.json"):
            candidate = Path("examples") / name
            if candidate.exists():
                style_guide_path = candidate
                break

    brand_rules = load_style_guide(style_guide_path)
    active_categories = sum(1 for k in ("content_pattern_rules", "color_scheme_rules",
                                        "design_pattern_rules", "other_rules")
                            if brand_rules.get(k))
    if style_guide_path and active_categories:
        print(f"[run] style_guide={style_guide_path.name}  "
              f"active_categories={active_categories}/4")
    elif style_guide_path:
        print(f"[warn] style_guide={style_guide_path.name} found but all rule lists empty "
              f"(populate the JSON before running)")

    # ── Run ────────────────────────────────────────────────────────────────
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[run] run_id={run_id} iterations={args.iterations} sections={len(blueprint_sections)} "
          f"baseline_chars={len(baseline_html_compressed)} images={len(baseline_images)} "
          f"slices={len(baseline_screenshots)} legacy_examples={len(example_html_paths)} "
          f"reference_examples={len(reference_examples)}"
          + (f" (from {reference_dir})" if reference_dir else ""))

    final_state = await run_from_baseline(
        blueprint_sections=blueprint_sections,
        base_prompt=base_prompt,
        baseline_html_compressed=baseline_html_compressed,
        baseline_images_b64=baseline_images,
        baseline_image_descriptors=baseline_image_descriptors,
        baseline_screenshot_slices=baseline_screenshots,
        example_html_paths=example_html_paths,
        reference_examples=reference_examples,
        iterations=args.iterations,
        run_id=run_id,
        layout_width=args.layout_width,
        slice_height=args.slice_height,
        brand_rules=brand_rules or None,
    )

    winner = final_state.get("winner") if isinstance(final_state, dict) else None
    if not winner:
        print("[run] no winner produced — compliance gate rejected every candidate.")
        return 2

    out_dir = Path("storage/runs") / run_id
    print(f"\n[result] winner: candidate_{winner.candidate_id}  is_baseline={winner.is_baseline}")
    print(f"[result] aggregate_score: {winner.aggregate_score:.3f}")
    if winner.quality_score is not None:
        print(f"[result] quality_score: {winner.quality_score:.2f}/10")
    if winner.anti_ai_rank is not None:
        print(f"[result] anti_ai_rank: {winner.anti_ai_rank}")
    print(f"[result] artifacts:  {out_dir}")
    print(f"[result] best HTML:  {out_dir / 'winner.html'}")
    print(f"[result] best prompt:{out_dir / 'winner.prompt.txt'}")

    if not winner.is_baseline:
        base_prompt_file.write_text(winner.prompt_used)
        print("[result] storage/base_prompt.txt updated with winning prompt.")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_blueprint(path: Path) -> list[BlueprintSection]:
    raw = json.loads(path.read_text())
    sections_raw: list[dict] = []
    if isinstance(raw, list):
        sections_raw = raw
    elif isinstance(raw, dict):
        # Accept the backend's "content_blueprint": [...] shape too
        if "content_blueprint" in raw and isinstance(raw["content_blueprint"], list):
            sections_raw = raw["content_blueprint"]
        elif "sections" in raw and isinstance(raw["sections"], list):
            sections_raw = raw["sections"]
    if not sections_raw:
        raise ValueError(f"{path}: could not find a section list ([...] or {{content_blueprint: [...]}})")
    return [BlueprintSection.from_dict(s) for s in sections_raw]


def _default_base_prompt() -> str:
    return (
        "Design a pharmaceutical HCP marketing HTML email at 600px width.\n"
        "Use a single centered column. Render every blueprint section in order.\n"
        "Visual hierarchy: section headline → key statistic → supporting copy → citation.\n"
        "Inline CSS only. Use [images_base64_N] placeholders for any image.\n"
        "Include a clearly labeled ISI section at 10px with full placeholder text.\n"
        "Avoid generic stock aesthetics — make deliberate, brand-credible design choices.\n"
    )


def main() -> int:
    """Sync entrypoint — runs the async pipeline under `asyncio.run`."""
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
