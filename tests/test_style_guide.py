"""
tests/test_style_guide.py — exercises the style-guide loader + helpers.

Validates that:
  - the loader handles the production-shape (`design_bible.website`)
  - the loader handles the legacy / flat shape
  - missing / invalid paths return EMPTY_RULES without raising
  - the "TEMPLATE — " documentation filter actually filters
  - format_brand_rules_block produces something or empty-string predictably
  - the body-font extractor picks a sensible font from rule text
  - the bundled examples/style_guide_ruleset.json round-trips through the loader
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.utils.style_guide import (
    EMPTY_RULES,
    extract_body_font,
    format_brand_rules_block,
    has_any_rules,
    load_raw_style_guide,
    load_style_guide,
    normalize_rules,
    rules_block_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_FILE = REPO_ROOT / "examples" / "style_guide_ruleset.json"


# ─────────────────────────────────────────────────────────────────────────────
# load_style_guide / normalize_rules
# ─────────────────────────────────────────────────────────────────────────────

def test_load_style_guide_handles_none_and_missing():
    assert load_style_guide(None) == EMPTY_RULES
    assert load_style_guide("/tmp/__definitely_not_a_file_12345__.json") == EMPTY_RULES


def test_load_raw_style_guide_returns_full_payload(tmp_path):
    f = tmp_path / "ruleset.json"
    raw = {"design_bible": {"website": {"color_scheme_rules": ["use violet #4B2D8F"]}}}
    f.write_text(json.dumps(raw))
    assert load_raw_style_guide(f) == raw


def test_normalize_production_shape():
    payload = {
        "design_bible": {
            "website": {
                "content_pattern_rules": ["headlines lead with the claim"],
                "color_scheme_rules": ["primary #4B2D8F"],
                "design_pattern_rules": ["max-width 600px"],
                "other_rules": ["always include REMS warning"],
            }
        }
    }
    rules = normalize_rules(payload)
    assert rules["content_pattern_rules"] == ["headlines lead with the claim"]
    assert rules["color_scheme_rules"] == ["primary #4B2D8F"]
    assert rules["design_pattern_rules"] == ["max-width 600px"]
    assert rules["other_rules"] == ["always include REMS warning"]


def test_normalize_flat_shape():
    payload = {
        "color_scheme_rules": ["accent navy #002F6C"],
    }
    rules = normalize_rules(payload)
    assert rules["color_scheme_rules"] == ["accent navy #002F6C"]
    # missing categories default to empty
    assert rules["content_pattern_rules"] == []
    assert rules["design_pattern_rules"] == []
    assert rules["other_rules"] == []


def test_normalize_filters_template_placeholders():
    payload = {
        "design_bible": {
            "website": {
                "color_scheme_rules": [
                    "TEMPLATE — replace with real rules extracted from the brand PDF.",
                    "Real rule: navy #002F6C for headlines.",
                ]
            }
        }
    }
    rules = normalize_rules(payload)
    assert rules["color_scheme_rules"] == ["Real rule: navy #002F6C for headlines."]


def test_normalize_handles_garbage_input():
    assert normalize_rules({}) == EMPTY_RULES
    assert normalize_rules({"design_bible": "not a dict"}) == EMPTY_RULES
    assert normalize_rules({"design_bible": {"website": "not a dict"}}) == EMPTY_RULES
    assert normalize_rules(None) == EMPTY_RULES  # type: ignore[arg-type]


def test_has_any_rules():
    assert not has_any_rules(None)
    assert not has_any_rules({})
    assert not has_any_rules(EMPTY_RULES)
    assert has_any_rules({"color_scheme_rules": ["x"], "content_pattern_rules": [],
                          "design_pattern_rules": [], "other_rules": []})


# ─────────────────────────────────────────────────────────────────────────────
# format_brand_rules_block
# ─────────────────────────────────────────────────────────────────────────────

def test_format_brand_rules_block_empty():
    assert format_brand_rules_block(None) == ""
    assert format_brand_rules_block(EMPTY_RULES) == ""


def test_format_brand_rules_block_includes_supremacy_header_and_json():
    rules = {
        "content_pattern_rules": [],
        "color_scheme_rules": ["primary navy #002F6C"],
        "design_pattern_rules": [],
        "other_rules": [],
    }
    block = format_brand_rules_block(rules)
    assert "Brand Rules (SUPREME" in block
    assert "primary navy #002F6C" in block
    assert "```json" in block


def test_rules_block_summary_truncates_per_category():
    rules = {
        "content_pattern_rules": [f"rule {i}" for i in range(10)],
        "color_scheme_rules": [],
        "design_pattern_rules": [],
        "other_rules": [],
    }
    summary = rules_block_summary(rules, max_per_list=3)
    assert "rule 0" in summary
    assert "rule 2" in summary
    assert "rule 3" not in summary
    assert "+7 more rules" in summary


# ─────────────────────────────────────────────────────────────────────────────
# extract_body_font
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_body_font_explicit_wins():
    rules = {"content_pattern_rules": ["body copy is Roboto 14px"],
             "color_scheme_rules": [], "design_pattern_rules": [], "other_rules": []}
    assert extract_body_font(rules, explicit="Inter") == "Inter, Helvetica, Arial, sans-serif"


def test_extract_body_font_finds_body_hint():
    rules = {
        "content_pattern_rules": ["Body copy is Lato 14px, 1.5 line-height; no narrative tone."],
        "color_scheme_rules": [], "design_pattern_rules": [], "other_rules": [],
    }
    font = extract_body_font(rules)
    assert "Lato" in font
    assert "sans-serif" in font  # fallback chain appended


def test_extract_body_font_falls_back_to_any_known_font():
    rules = {
        "content_pattern_rules": ["uses Oswald for the brand wordmark"],
        "color_scheme_rules": [], "design_pattern_rules": [], "other_rules": [],
    }
    font = extract_body_font(rules)
    assert "Oswald" in font


def test_extract_body_font_default_when_no_match():
    rules = EMPTY_RULES
    assert extract_body_font(rules) == "Arial, Helvetica, sans-serif"
    assert extract_body_font(None) == "Arial, Helvetica, sans-serif"


def test_extract_body_font_preserves_existing_fallback():
    rules = EMPTY_RULES
    assert extract_body_font(rules, explicit="Inter, sans-serif") == "Inter, sans-serif"


# ─────────────────────────────────────────────────────────────────────────────
# Bundled example file actually loads
# ─────────────────────────────────────────────────────────────────────────────

def test_ultomiris_ruleset_loads():
    path = REPO_ROOT / "examples" / "style_guide_ruleset_ultomiris.json"
    assert path.exists()
    rules = load_style_guide(path)
    assert has_any_rules(rules)
    assert len(rules["color_scheme_rules"]) >= 8
    palette_text = " ".join(rules["color_scheme_rules"])
    assert "#008578" in palette_text
    assert "#FF6A39" in palette_text
    other_text = " ".join(rules["other_rules"])
    assert "#4B2D8F" in other_text


def test_bundled_example_loads():
    assert EXAMPLE_FILE.exists(), f"example file missing: {EXAMPLE_FILE}"
    rules = load_style_guide(EXAMPLE_FILE)
    assert set(rules.keys()) == set(EMPTY_RULES.keys())
    for k in EMPTY_RULES:
        assert isinstance(rules[k], list)
        for entry in rules[k]:
            assert not entry.startswith("TEMPLATE — "), \
                f"TEMPLATE entry leaked through loader: {entry[:80]!r}"
