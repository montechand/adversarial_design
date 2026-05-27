"""
tests/test_creativizer.py

Exercise the pure-Python pieces of the creativizer Stage 2 wiring:

  - System prompt requires the new image-utilization protocol (rule 10/11
    plus the descriptor-aware preamble).
  - `build_user_prompt` renders descriptors as a compact bulleted table
    inside the AVAILABLE IMAGE TOKENS block, with intrinsic + role + alt.
  - Empty descriptors fall back to "no descriptor mined" but still keep
    the utilization-plan requirement.
  - Empty whitelist still emits the `## Image utilization plan` requirement.
  - `_persist_creativizer_prompts` writes the descriptor snapshot into both
    the per-candidate metadata and the manifest so audits can confirm
    exactly what each candidate saw.
  - Top-level `creativize()` plumbs descriptors through to the prompt
    builder and falls back to runtime mining when state lacks them.

LLM round-trips are not exercised — the Anthropic client is monkey-patched.

Run:
    cd adversarial_design
    PYTHONPATH=. pytest tests/test_creativizer.py -v
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest  # noqa: F401  (kept for the module-level pytest fixtures)

from pipeline.prompts.creativizer import (
    SYSTEM_PROMPT,
    build_user_prompt,
)
from pipeline.nodes import creativizer as creativizer_node
from pipeline.utils.image_descriptors import format_descriptor_for_prompt


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — the new asset-utilization protocol must be present
# ─────────────────────────────────────────────────────────────────────────────

def test_system_prompt_documents_image_descriptor_input():
    """The 'IMAGE ASSET DESCRIPTORS' bullet must explain that intrinsic
    dims + aspect ratio + role + alt come into the call as ground truth."""
    assert "IMAGE ASSET DESCRIPTORS" in SYSTEM_PROMPT
    assert "intrinsic dimensions" in SYSTEM_PROMPT
    assert "aspect ratio" in SYSTEM_PROMPT
    assert "role" in SYSTEM_PROMPT


def test_system_prompt_requires_utilization_reasoning_step():
    """Rule 10 — the model must reason about every available token and
    classify it USE / SKIP, NOT silently drop it."""
    assert "IMAGE ASSET UTILIZATION" in SYSTEM_PROMPT
    assert "SKIP" in SYSTEM_PROMPT
    assert "Forced silence on an available token is NOT acceptable" in SYSTEM_PROMPT


def test_system_prompt_requires_image_utilization_plan_section():
    """Rule 11 — output spec must contain the structured utilization plan."""
    assert "Image utilization plan" in SYSTEM_PROMPT
    assert "REQUIRED OUTPUT SECTION" in SYSTEM_PROMPT


# ─────────────────────────────────────────────────────────────────────────────
# build_user_prompt — descriptor rendering
# ─────────────────────────────────────────────────────────────────────────────

def _minimal_kwargs() -> dict[str, Any]:
    """Tightest valid kwargs so individual tests stay focused."""
    return dict(
        blueprint_summary="(blueprint)",
        base_prompt="(base)",
        baseline_descriptor_summary="(baseline)",
        reference_descriptor_summaries=[],
        reference_patterns_to_emulate=[],
        reference_standards_summary="",
        forced_axes={"layout": "hero-left content-right"},
        candidate_index=0,
    )


def test_user_prompt_renders_descriptors_as_compact_table():
    """When descriptors are supplied, every token line must include
    intrinsic dimensions, aspect ratio, role hint, and alt text — not
    just the bare token name."""
    descriptors = {
        "[images_base64_0]": {
            "intrinsic": {
                "width_px": 1360, "height_px": 220,
                "aspect_ratio": "68:11", "aspect_ratio_decimal": 6.18,
                "orientation": "landscape", "mime": "image/png",
                "size_bytes": 178_000,
            },
            "usage": {
                "tag": "img", "via": "src",
                "parent_selector": "a.banner",
                "css_role_hint": "banner",
                "alt": "ULTOMIRIS branding banner",
                "is_referenced_in_html": True,
            },
        },
        "[images_base64_1]": {
            "intrinsic": {
                "width_px": 80, "height_px": 80,
                "aspect_ratio": "1:1", "aspect_ratio_decimal": 1.0,
                "orientation": "square", "mime": "image/png",
                "size_bytes": 5_000,
            },
            "usage": {
                "tag": "img", "via": "src",
                "parent_selector": "td.efficacy-row",
                "css_role_hint": "icon",
                "alt": "efficacy icon",
                "is_referenced_in_html": True,
            },
        },
    }
    out = build_user_prompt(
        **_minimal_kwargs(),
        available_image_tokens=["[images_base64_0]", "[images_base64_1]"],
        image_descriptors=descriptors,
    )

    # Both tokens render with intrinsic + role + alt
    assert "1360x220" in out
    assert "role=banner" in out
    assert 'alt: "ULTOMIRIS branding banner"' in out
    assert "80x80" in out
    assert "role=icon" in out
    assert 'alt: "efficacy icon"' in out

    # Intro line still tells the LLM the whitelist is exhaustive
    assert "EXACT, EXHAUSTIVE WHITELIST" in out

    # Tokens are bulleted (one per line) so the LLM doesn't lose them
    # in a comma-separated blob.
    assert "  • token [images_base64_0]" in out
    assert "  • token [images_base64_1]" in out


def test_user_prompt_includes_inline_utilization_reasoning_steps():
    """The user prompt must spell out the 1/2/3 reasoning checklist
    alongside the whitelist — not just refer to the system prompt."""
    out = build_user_prompt(
        **_minimal_kwargs(),
        available_image_tokens=["[images_base64_0]"],
        image_descriptors={
            "[images_base64_0]": {
                "intrinsic": {
                    "width_px": 600, "height_px": 600,
                    "aspect_ratio": "1:1", "orientation": "square",
                    "mime": "image/png", "size_bytes": 12_000,
                },
                "usage": {
                    "tag": "img", "via": "src",
                    "parent_selector": "div.hero",
                    "css_role_hint": "hero",
                    "alt": "",
                    "is_referenced_in_html": True,
                },
            },
        },
    )
    assert "Image asset utilization (REQUIRED reasoning" in out
    assert "intrinsic aspect ratio" in out
    assert "USE: section, slot, role, rationale" in out
    assert "SKIP: reason" in out
    assert "Silent omissions are treated as bugs." in out


def test_user_prompt_falls_back_when_descriptors_missing_for_some_tokens():
    """Tokens with no mined descriptor should still appear as a bullet,
    flagged so the LLM knows the gap is metadata-only (not the token
    itself missing)."""
    out = build_user_prompt(
        **_minimal_kwargs(),
        available_image_tokens=[
            "[images_base64_0]", "[images_base64_1]", "[images_base64_2]",
        ],
        image_descriptors={
            "[images_base64_0]": {
                "intrinsic": {
                    "width_px": 600, "height_px": 200,
                    "aspect_ratio": "3:1", "orientation": "landscape",
                    "mime": "image/png", "size_bytes": 20_000,
                },
                "usage": {
                    "tag": "img", "via": "src",
                    "parent_selector": "", "css_role_hint": "",
                    "alt": "", "is_referenced_in_html": False,
                },
            },
        },
    )
    assert "  • token [images_base64_0]" in out
    assert "600x200" in out
    assert "  • token [images_base64_1] — (no descriptor mined)" in out
    assert "  • token [images_base64_2] — (no descriptor mined)" in out


def test_user_prompt_handles_empty_whitelist_with_utilization_plan_requirement():
    """Even when there are zero baseline assets, the spec must still
    contain a `## Image utilization plan` section so reviewers can tell
    the case apart from a silently-dropped one."""
    out = build_user_prompt(
        **_minimal_kwargs(),
        available_image_tokens=[],
        image_descriptors={},
    )
    assert "AVAILABLE IMAGE TOKENS: (none" in out
    assert "Image utilization plan" in out
    assert "no baseline image assets" in out


def test_user_prompt_handles_descriptors_passed_without_token_whitelist():
    """Defensive: descriptors with no whitelist should still be treated
    as the empty-whitelist branch (we never invent tokens from descriptors)."""
    out = build_user_prompt(
        **_minimal_kwargs(),
        available_image_tokens=None,
        image_descriptors={
            "[images_base64_0]": {"intrinsic": {"width_px": 1, "height_px": 1}},
        },
    )
    assert "AVAILABLE IMAGE TOKENS: (none" in out
    assert "[images_base64_0]" not in out


def test_user_prompt_descriptor_lines_match_format_descriptor_helper():
    """Each rendered descriptor line must be exactly what
    `format_descriptor_for_prompt` produces — single source of truth so
    the reverse_prompter and creativizer give the LLM identical wording."""
    desc = {
        "intrinsic": {
            "width_px": 684, "height_px": 300,
            "aspect_ratio": "57:25", "orientation": "landscape",
            "mime": "image/png", "size_bytes": 91_000,
        },
        "usage": {
            "tag": "img", "via": "src",
            "parent_selector": "div.hero.gradient",
            "css_role_hint": "hero",
            "alt": "Strive for Zero",
            "is_referenced_in_html": True,
        },
    }
    expected_line = "  • " + format_descriptor_for_prompt("[images_base64_0]", desc)
    out = build_user_prompt(
        **_minimal_kwargs(),
        available_image_tokens=["[images_base64_0]"],
        image_descriptors={"[images_base64_0]": desc},
    )
    assert expected_line in out


# ─────────────────────────────────────────────────────────────────────────────
# _persist_creativizer_prompts — descriptors snapshot is preserved on disk
# ─────────────────────────────────────────────────────────────────────────────

def test_persist_creativizer_prompts_writes_descriptor_snapshot(tmp_path, monkeypatch):
    """Per-candidate metadata.json AND the iteration manifest must contain
    the exact descriptor snapshot the LLM saw — keyed by token."""
    monkeypatch.setattr(
        creativizer_node,
        "run_dir",
        lambda run_id: str(tmp_path),
    )

    descriptors = {
        "[images_base64_0]": {
            "intrinsic": {"width_px": 1360, "height_px": 220, "aspect_ratio": "68:11"},
            "usage": {"css_role_hint": "banner", "alt": "x", "is_referenced_in_html": True},
        },
        "[images_base64_1]": {
            "intrinsic": {"width_px": 80, "height_px": 80},
            "usage": {"css_role_hint": "icon"},
        },
    }
    creativizer_node._persist_creativizer_prompts(
        run_id="test_run",
        iteration=2,
        candidate_prompts=["spec A", "spec B"],
        candidate_metadata=[
            {"axes": {"layout": "stacked"}, "candidate_index": 0},
            {"axes": {"layout": "grid"},    "candidate_index": 1},
        ],
        available_image_tokens=["[images_base64_0]", "[images_base64_1]"],
        image_descriptors=descriptors,
        compliance_feedback=[None, "previous attempt missed ISI"],
    )

    iteration_dir = Path(tmp_path) / "creativizer" / "iteration_002"
    assert iteration_dir.is_dir()

    # Per-candidate metadata
    cand0_meta = json.loads((iteration_dir / "candidate_0.metadata.json").read_text())
    assert cand0_meta["available_image_tokens"] == [
        "[images_base64_0]", "[images_base64_1]",
    ]
    snapshot = cand0_meta["image_descriptors"]
    assert snapshot["[images_base64_0]"]["intrinsic"]["width_px"] == 1360
    assert snapshot["[images_base64_0]"]["usage"]["css_role_hint"] == "banner"
    assert snapshot["[images_base64_1]"]["intrinsic"]["height_px"] == 80

    # Manifest
    manifest = json.loads((iteration_dir / "manifest.json").read_text())
    assert manifest["image_descriptors"] == snapshot

    # Per-candidate metadata is identical across candidates within an iteration
    cand1_meta = json.loads((iteration_dir / "candidate_1.metadata.json").read_text())
    assert cand1_meta["image_descriptors"] == snapshot


def test_persist_creativizer_prompts_handles_empty_descriptors(tmp_path, monkeypatch):
    """No descriptors = empty snapshot, but the keys still exist so audit
    code doesn't have to special-case the missing field."""
    monkeypatch.setattr(
        creativizer_node,
        "run_dir",
        lambda run_id: str(tmp_path),
    )
    creativizer_node._persist_creativizer_prompts(
        run_id="test_run",
        iteration=0,
        candidate_prompts=["spec"],
        candidate_metadata=[{"axes": {}, "candidate_index": 0}],
        available_image_tokens=["[images_base64_0]"],
        image_descriptors={},
        compliance_feedback=[None],
    )
    iteration_dir = Path(tmp_path) / "creativizer" / "iteration_000"
    cand_meta = json.loads((iteration_dir / "candidate_0.metadata.json").read_text())
    assert cand_meta["image_descriptors"] == {
        "[images_base64_0]": {"intrinsic": {}, "usage": {}},
    }


# ─────────────────────────────────────────────────────────────────────────────
# creativize() top-level flow — descriptors are forwarded to the prompt
# builder and runtime-mined when missing
# ─────────────────────────────────────────────────────────────────────────────

class _StubMessage:
    def __init__(self, text: str):
        self.content = [type("Block", (), {"text": text})()]


class _StubMessages:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _StubMessage("STUB SPEC OUTPUT")


class _StubRouter:
    provider = "stub"
    model = "stub-model"

    def __init__(self):
        self.messages = _StubMessages()


def _patch_creativize_runtime(monkeypatch, tmp_path, router, k=1, axes=None):
    """Stub out every external dependency (LLM router, config readers,
    storage path) so `creativize()` runs in-process with no I/O beyond
    `tmp_path`."""
    monkeypatch.setattr(creativizer_node, "get_async_router", lambda *_a, **_k: router)
    monkeypatch.setattr(creativizer_node, "_get_k", lambda: k)
    monkeypatch.setattr(creativizer_node, "_load_concurrency", lambda: 1)
    monkeypatch.setattr(creativizer_node, "_load_max_tokens", lambda: 1024)
    monkeypatch.setattr(creativizer_node, "run_dir", lambda run_id: str(tmp_path))
    monkeypatch.setattr(
        creativizer_node,
        "_load_axes",
        lambda: axes or {"layout": ["stacked"], "color_mood": ["clinical"]},
    )


def test_creativize_forwards_state_descriptors_to_prompt(monkeypatch, tmp_path):
    """When `baseline_image_descriptors` is in state, the rendered user
    prompt MUST contain the descriptor's role + intrinsic dims so the
    LLM sees them. We assert by inspecting the kwargs the stub LLM
    received.

    Coroutine is driven via `asyncio.run` to keep the suite free of any
    pytest-asyncio plugin requirement (matches the rest of this repo's
    sync-only test layout)."""
    router = _StubRouter()
    _patch_creativize_runtime(monkeypatch, tmp_path, router, k=1)

    state = {
        "blueprint_sections": [],
        "base_prompt": "old spec",
        "baseline_images_b64": {
            "[images_base64_0]": "data:image/png;base64,AAAA",
        },
        "baseline_image_descriptors": {
            "[images_base64_0]": {
                "intrinsic": {
                    "width_px": 1360, "height_px": 220,
                    "aspect_ratio": "68:11", "orientation": "landscape",
                    "mime": "image/png", "size_bytes": 178_000,
                },
                "usage": {
                    "tag": "img", "via": "src",
                    "parent_selector": "a.banner",
                    "css_role_hint": "banner",
                    "alt": "ULTOMIRIS banner",
                    "is_referenced_in_html": True,
                },
            },
        },
        "design_descriptors": [],
        "baseline_descriptor": None,
        "iteration": 0,
        "run_id": "test_run",
    }
    asyncio.run(creativizer_node.creativize(state))  # type: ignore[arg-type]

    assert len(router.messages.calls) == 1
    user_msg = router.messages.calls[0]["messages"][0]["content"]
    # User prompt is a single string in this codepath (not multimodal)
    assert isinstance(user_msg, str)
    assert "1360x220" in user_msg
    assert "role=banner" in user_msg
    assert 'alt: "ULTOMIRIS banner"' in user_msg
    assert "Image utilization plan" in user_msg


def test_creativize_runtime_mines_descriptors_when_state_missing(monkeypatch, tmp_path):
    """Defensive path: if ingestion didn't populate descriptors but the
    image bytes are present, creativize() must rebuild them via
    `build_descriptors` so the LLM still sees intrinsic dims."""
    router = _StubRouter()
    _patch_creativize_runtime(monkeypatch, tmp_path, router, k=1, axes={"layout": ["stacked"]})

    fake_descriptors = {
        "[images_base64_0]": {
            "intrinsic": {
                "width_px": 600, "height_px": 200,
                "aspect_ratio": "3:1", "orientation": "landscape",
                "mime": "image/png", "size_bytes": 9_000,
            },
            "usage": {
                "tag": "img", "via": "src",
                "parent_selector": "div.hero",
                "css_role_hint": "hero",
                "alt": "stub hero",
                "is_referenced_in_html": True,
            },
        },
    }
    captured: list[dict[str, Any]] = []

    def _stub_build(*, images_b64, compressed_html, existing=None):
        captured.append({"images_b64": dict(images_b64), "compressed_html": compressed_html})
        return fake_descriptors

    monkeypatch.setattr(creativizer_node, "build_descriptors", _stub_build)

    state = {
        "blueprint_sections": [],
        "base_prompt": "",
        "baseline_images_b64": {
            "[images_base64_0]": "data:image/png;base64,AAAA",
        },
        # Descriptors deliberately MISSING from state.
        "baseline_image_descriptors": {},
        "baseline_html_compressed": "<html><img src='[images_base64_0]'></html>",
        "design_descriptors": [],
        "baseline_descriptor": None,
        "iteration": 0,
        "run_id": "test_run",
    }
    asyncio.run(creativizer_node.creativize(state))  # type: ignore[arg-type]

    assert len(captured) == 1, "build_descriptors must run exactly once"
    assert "[images_base64_0]" in captured[0]["images_b64"]
    user_msg = router.messages.calls[0]["messages"][0]["content"]
    assert "600x200" in user_msg
    assert "role=hero" in user_msg


def test_creativize_persists_descriptors_alongside_specs(monkeypatch, tmp_path):
    """End-to-end: after creativize() runs, the iteration manifest must
    contain the descriptor snapshot for every available token."""
    router = _StubRouter()
    _patch_creativize_runtime(monkeypatch, tmp_path, router, k=2, axes={"layout": ["a", "b"]})

    state = {
        "blueprint_sections": [],
        "base_prompt": "",
        "baseline_images_b64": {
            "[images_base64_0]": "data:image/png;base64,AAAA",
        },
        "baseline_image_descriptors": {
            "[images_base64_0]": {
                "intrinsic": {"width_px": 100, "height_px": 100, "aspect_ratio": "1:1"},
                "usage": {"css_role_hint": "icon", "is_referenced_in_html": True},
            },
        },
        "design_descriptors": [],
        "baseline_descriptor": None,
        "iteration": 5,
        "run_id": "persist_test",
    }
    asyncio.run(creativizer_node.creativize(state))  # type: ignore[arg-type]

    iteration_dir = Path(tmp_path) / "creativizer" / "iteration_005"
    assert iteration_dir.is_dir()
    manifest = json.loads((iteration_dir / "manifest.json").read_text())
    assert manifest["image_descriptors"]["[images_base64_0]"]["intrinsic"]["width_px"] == 100
    assert manifest["image_descriptors"]["[images_base64_0]"]["usage"]["css_role_hint"] == "icon"
    # Both candidates (k=2) must have written metadata
    assert (iteration_dir / "candidate_0.metadata.json").is_file()
    assert (iteration_dir / "candidate_1.metadata.json").is_file()
