"""
PipelineState — the single TypedDict that flows through every LangGraph node.

The pipeline runs in **baseline-vs-candidates** mode:

  Input contract (what the backend-server will eventually hand us, and what the
  CLI `--from-example` flag stubs locally):

    1. `blueprint_sections` — the structured email blueprint (a list of
       `BlueprintSection`, mirroring backend `SectionSpec`).
    2. `baseline_html_compressed` — the HTML the backend-server produced, with
       every `data:image/...;base64,...` URI replaced by a `[images_base64_i]`
       placeholder so it is cheap to feed to LLM *text* calls.
    3. `baseline_images_b64` — `{placeholder_token: raw_base64}` side-car map,
       used to re-hydrate the HTML for screenshotting and VLM judging.
    4. `baseline_screenshot_slices` — pre-sliced PNG paths of the rendered
       baseline. If empty on entry, ingestion regenerates them.

  Optional inputs:
    - `reference_examples` — list of `ReferenceExample` (human-made creative
      ads). Each is interleaved with its description in the reverse
      prompter's BASELINE VLM call so the analyzer can compare the
      baseline against strong human designs when listing weaknesses.
    - `example_html_paths` / `example_screenshot_paths` — extra human-designed
      reference emails for the reverse prompter to mine vocabulary from
      (legacy free-form mode, still supported).
    - `base_prompt` — seed design spec (updated each iteration).

  Output:
    - `winner` — best Candidate (its `.prompt_used` is the "best prompt",
      its `.html` is the "best design").
"""

from __future__ import annotations

from typing import TypedDict, Optional, Any
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Blueprint structure (mirrors backend SectionSpec — keep field names aligned)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BlueprintSection:
    """One section of the email blueprint.

    Field names mirror the backend `SectionSpec` TypedDict
    (see Backend-Server/.../message_generator_from_scratch_v0_1.py line 693).
    Only a subset is required here; downstream nodes read defensively.
    """
    section_id: str
    order: int
    type: str = "body"                 # intro | body | cta | closing | hero
    intent: str = ""
    headline: str = ""
    copy_outline: str = ""
    clinical_fact_covered: str = ""
    constraints: dict[str, Any] = field(default_factory=dict)
    general_allowed: bool = True
    claim_required: bool = False
    # Extra structural hints we do not interpret but pass through verbatim.
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BlueprintSection":
        """Tolerant constructor — accepts the backend's superset schema."""
        known = {
            "section_id", "order", "type", "intent", "headline", "copy_outline",
            "clinical_fact_covered", "constraints", "general_allowed",
            "claim_required",
        }
        extras = {k: v for k, v in raw.items() if k not in known}
        return cls(
            section_id=str(raw.get("section_id", "")),
            order=int(raw.get("order", 0) or 0),
            type=str(raw.get("type", "body") or "body"),
            intent=str(raw.get("intent", "") or ""),
            headline=str(raw.get("headline", "") or ""),
            copy_outline=str(raw.get("copy_outline", "") or ""),
            clinical_fact_covered=str(raw.get("clinical_fact_covered", "") or ""),
            constraints=dict(raw.get("constraints", {}) or {}),
            general_allowed=bool(raw.get("general_allowed", True)),
            claim_required=bool(raw.get("claim_required", False)),
            extra=extras,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Reverse-prompter & candidate dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DesignDescriptor:
    """Structured output from the reverse prompter for one example/baseline."""
    example_id: str
    layout_type: str
    color_mood: str
    typography_personality: str
    information_hierarchy: str
    emotional_tone: str
    visual_motifs: list[str]
    creative_risks_taken: str = ""
    weaknesses: list[str] = field(default_factory=list)   # attack surface — baseline only
    reference_patterns_to_emulate: list[str] = field(default_factory=list)
    # concrete layout/color/type patterns mined from human reference examples
    reference_standards_summary: str = ""
    # one-paragraph synthesis of what makes the human references strong
    raw_description: str = ""                              # freeform VLM output (full)
    source: str = "reference"                              # "baseline" | "reference"


@dataclass
class ReferenceExample:
    """One human-made creative ad used as a visual reference example.

    Drop into `examples/references/` as a `.html` (auto-rendered) or
    `.png` / `.jpg` / `.jpeg` (used as-is). Optionally pair with a
    same-stem `.md` or `.txt` for a richer description; otherwise the
    pretty-printed file stem is used as the caption.

    These are passed verbatim into the reverse prompter's BASELINE VLM
    call as INTERLEAVED [text-caption, image] content blocks. They give
    the VLM a creative vocabulary to compare the baseline against when
    listing weaknesses and creative risks.
    """
    name: str                 # short stable identifier, e.g. "pfizer_xeljanz"
    image_path: str           # absolute path to the PNG/JPG to send to the VLM
    description: str = ""     # free-text caption shown in the prompt right before the image
    source_path: str = ""     # original on-disk path (html or image) — for logging only


@dataclass
class Candidate:
    """One generated HTML candidate and its evaluation artifacts."""
    candidate_id: int
    prompt_used: str
    prompt_metadata: dict
    html: str
    screenshot_path: Optional[str] = None
    screenshot_slice_paths: list[str] = field(default_factory=list)
    compliance_passed: Optional[bool] = None
    compliance_reason: str = ""
    quality_score: Optional[float] = None      # 0-10
    quality_rationale: Optional[str] = None
    anti_ai_rank: Optional[int] = None         # 1 = least AI-like
    aggregate_score: Optional[float] = None
    # If True, this Candidate is the baseline that came in as input
    # (not generated by code_generator). Useful for "must beat baseline" logic.
    is_baseline: bool = False

    # ── MJML artifacts (new candidates produced via the section-wise MJML
    #    flow). The baseline has these empty since it skips MJML generation.
    mjml_document: Optional[str] = None
    # {section_id: {mjml_fragment, claims_used, design_reasoning}}
    section_artifacts: dict[str, dict] = field(default_factory=dict)
    # Flat list of mechanical-validator warning strings (data-section-id
    # missing, <sup> without data-claim-id, sentences from copy_outline not
    # rendered verbatim, etc.). Logged at WARNING level by the node.
    validation_warnings: list[str] = field(default_factory=list)
    # Did MJML→HTML compilation succeed? None = the candidate skipped MJML
    # (e.g. baseline), True/False for generated candidates.
    mjml_render_ok: Optional[bool] = None
    mjml_render_error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline state
# ─────────────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict, total=False):
    # ── Required inputs ─────────────────────────────────────────────────────
    blueprint_sections: list[BlueprintSection]   # structured blueprint
    base_prompt: str                              # seed prompt (updated each iteration)
    run_id: str
    iteration: int

    # ── Brand style guide (injected as SUPREME rules into LLM prompts) ──────
    # Loaded from a style_guide_ruleset.json; four lists keyed by:
    #   content_pattern_rules, color_scheme_rules, design_pattern_rules, other_rules
    # None / missing means no brand rules are active.
    brand_rules: Optional[dict]

    # ── Baseline (the design produced by the backend-server pipeline) ──────
    baseline_html_compressed: str                 # HTML with [images_base64_i] tokens
    baseline_images_b64: dict[str, str]           # {placeholder_token: raw_b64}
    # {placeholder_token: {"intrinsic": {...}, "usage": {...}}} — mined in
    # ingestion from the baseline HTML + image bytes. The reverse prompter
    # passes a compact rendering of this to the VLM so it understands how
    # each image was *originally* used (rendered size, aspect ratio, alt
    # text, role hint) and can judge any candidate's image usage against
    # that anchor.
    baseline_image_descriptors: dict[str, dict]
    baseline_screenshot_slices: list[str]         # PNG paths (top→bottom slices)
    baseline_descriptor: Optional[DesignDescriptor]

    # ── Optional extra reference examples (legacy inspiration mode) ────────
    example_html_paths: list[str]
    example_screenshot_paths: list[str]

    # ── Reference creative examples (human-made ads, used as visual anchors
    #    by the reverse prompter when analyzing the baseline) ──────────────
    reference_examples: list[ReferenceExample]

    # ── Stage 1: reverse prompter ───────────────────────────────────────────
    design_descriptors: list[DesignDescriptor]

    # ── Stage 2: creativizer ────────────────────────────────────────────────
    candidate_prompts: list[str]
    candidate_prompt_metadata: list[dict]

    # ── Stage 3-4: generation + screenshots ─────────────────────────────────
    candidates: list[Candidate]

    # ── Stage 5: judging ────────────────────────────────────────────────────
    compliance_results: list[bool]
    compliance_reasons: list[str]
    quality_scores: list[float]
    anti_ai_ranks: list[int]

    # ── Stage 6: aggregation ────────────────────────────────────────────────
    winner: Optional[Candidate]

    # ── Cross-iteration feedback ─────────────────────────────────────────────
    # Set by graph.py after iteration 0 when feedback.winner_feedback_to_reverse_prompter
    # is enabled. Contains the previous winner's quality_rationale string so the
    # reverse prompter can ground its weakness analysis in what worked last round.
    winner_feedback: Optional[str]

    # ── Misc / non-public ───────────────────────────────────────────────────
    layout_width: int                             # email render width, default 600
    slice_height: int                             # screenshot slice height, default 1600
