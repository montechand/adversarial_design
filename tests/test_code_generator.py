"""
tests/test_code_generator.py

Exercise the pure-Python pieces of the candidate generator (parsers,
fallback helpers, config dispatcher). The actual LLM round-trips are
mocked / not exercised here — they require ANTHROPIC_API_KEY.

Coverage:
  - _parse_full_response handles bare JSON, fenced JSON, prose+JSON,
    and totally malformed inputs, and emits an accurate parse_trace
    (status / missing_section_ids / sections_filled_by_llm) for the
    code-generator's audit artifact persistence.
  - _parse_full_response back-fills sections the LLM forgot with the
    fallback fragment so compile downstream never sees a hole.
  - _normalize_section_artifact self-heals a missing data-section-id by
    splicing it onto the root <mj-section> tag.
  - _fallback_section_artifact preserves headline + copy verbatim.
  - _load_candidate_mode round-trips the config value and clamps unknown
    modes back to the default.
"""

from __future__ import annotations

import json

import pytest

from pipeline.nodes.code_generator import (
    _DEFAULT_CANDIDATE_MODE,
    _VALID_CANDIDATE_MODES,
    _fallback_section_artifact,
    _load_candidate_mode,
    _normalize_section_artifact,
    _parse_full_response,
    _parse_section_artifact,
)


# ─────────────────────────────────────────────────────────────────────────────
# _parse_full_response
# ─────────────────────────────────────────────────────────────────────────────

BLUEPRINT = [
    {"section_id": "intro",    "order": 1, "headline": "Hello", "copy_outline": "Body."},
    {"section_id": "efficacy", "order": 2, "headline": "Wow",   "copy_outline": "Data."},
    {"section_id": "cta",      "order": 3, "headline": "Act",   "copy_outline": "Click."},
]


def _sample_response(sids=("intro", "efficacy", "cta"), reasoning="overall vibe"):
    return {
        "design_reasoning": reasoning,
        "sections": [
            {
                "section_id": sid,
                "design_reasoning": f"{sid} fits the arc",
                "mjml_fragment": f'<mj-section data-section-id="{sid}"><mj-text>{sid}</mj-text></mj-section>',
                "claims_used": [f"{sid}-1"],
            }
            for sid in sids
        ],
    }


def test_parse_full_response_bare_json():
    payload = json.dumps(_sample_response())
    artifacts, reasoning, trace = _parse_full_response(payload, blueprint_dicts=BLUEPRINT)
    assert reasoning == "overall vibe"
    assert set(artifacts.keys()) == {"intro", "efficacy", "cta"}
    for sid, art in artifacts.items():
        assert f'data-section-id="{sid}"' in art["mjml_fragment"]
        assert art["claims_used"] == [f"{sid}-1"]
        assert "fits the arc" in art["design_reasoning"]
    assert trace == {
        "status": "ok",
        "missing_section_ids": [],
        "sections_total": 3,
        "sections_filled_by_llm": 3,
    }


def test_parse_full_response_fenced_json_block():
    payload = "```json\n" + json.dumps(_sample_response()) + "\n```"
    artifacts, reasoning, trace = _parse_full_response(payload, blueprint_dicts=BLUEPRINT)
    assert reasoning == "overall vibe"
    assert len(artifacts) == 3
    assert trace["status"] == "ok"
    assert trace["sections_filled_by_llm"] == 3


def test_parse_full_response_with_prose_around_json():
    payload = (
        "Here is my design:\n\n"
        + json.dumps(_sample_response())
        + "\n\nLet me know if you want changes."
    )
    artifacts, reasoning, trace = _parse_full_response(payload, blueprint_dicts=BLUEPRINT)
    assert reasoning == "overall vibe"
    assert "intro" in artifacts
    assert trace["status"] == "ok"
    assert trace["missing_section_ids"] == []


def test_parse_full_response_backfills_missing_sections():
    # LLM forgot the cta section
    incomplete = _sample_response(sids=("intro", "efficacy"))
    artifacts, _, trace = _parse_full_response(
        json.dumps(incomplete), blueprint_dicts=BLUEPRINT,
    )
    assert set(artifacts.keys()) == {"intro", "efficacy", "cta"}
    # The missing one got a fallback fragment that mentions the headline
    cta_frag = artifacts["cta"]["mjml_fragment"]
    assert 'data-section-id="cta"' in cta_frag
    assert "Act" in cta_frag  # headline preserved
    assert "Click." in cta_frag  # copy preserved
    # And the trace flags the gap so _persist_codegen_artifacts can surface it
    assert trace["status"] == "missing_sections"
    assert trace["missing_section_ids"] == ["cta"]
    assert trace["sections_filled_by_llm"] == 2
    assert trace["sections_total"] == 3


def test_parse_full_response_total_garbage_returns_all_fallbacks():
    artifacts, reasoning, trace = _parse_full_response(
        "this is not JSON at all <<<>>>", blueprint_dicts=BLUEPRINT,
    )
    assert reasoning == ""
    assert set(artifacts.keys()) == {"intro", "efficacy", "cta"}
    for sid in ("intro", "efficacy", "cta"):
        assert f'data-section-id="{sid}"' in artifacts[sid]["mjml_fragment"]
    # This is the exact branch persisted as `parse_status: "non_json"` —
    # raw response should be retained at the call site, every section
    # reported as a fallback, and zero sections marked as LLM-filled.
    assert trace["status"] == "non_json"
    assert set(trace["missing_section_ids"]) == {"intro", "efficacy", "cta"}
    assert trace["sections_filled_by_llm"] == 0
    assert trace["sections_total"] == 3


def test_parse_full_response_handles_response_not_an_object():
    artifacts, reasoning, trace = _parse_full_response(
        "[1, 2, 3]", blueprint_dicts=BLUEPRINT,
    )
    assert reasoning == ""
    assert len(artifacts) == 3  # all fallbacks
    # A JSON-but-not-an-object payload is treated the same as `non_json`:
    # we cannot extract any sections from it, so every section falls back.
    assert trace["status"] == "non_json"
    assert trace["sections_filled_by_llm"] == 0


def test_parse_full_response_ignores_section_without_fragment():
    payload = {
        "design_reasoning": "",
        "sections": [
            {"section_id": "intro", "mjml_fragment": "", "claims_used": []},
            {"section_id": "efficacy",
             "mjml_fragment": '<mj-section data-section-id="efficacy">x</mj-section>'},
            {"section_id": "cta",
             "mjml_fragment": '<mj-section data-section-id="cta">y</mj-section>'},
        ],
    }
    artifacts, _, trace = _parse_full_response(
        json.dumps(payload), blueprint_dicts=BLUEPRINT,
    )
    # intro has empty fragment → falls back
    assert "Body." in artifacts["intro"]["mjml_fragment"]
    # the others come through unchanged
    assert artifacts["efficacy"]["mjml_fragment"].endswith("x</mj-section>")
    assert artifacts["cta"]["mjml_fragment"].endswith("y</mj-section>")
    # Empty-fragment entries should be counted as missing in the trace so
    # the persisted metadata explains the fallback to a human inspector.
    assert trace["status"] == "missing_sections"
    assert trace["missing_section_ids"] == ["intro"]
    assert trace["sections_filled_by_llm"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# _normalize_section_artifact — data-section-id self-healing
# ─────────────────────────────────────────────────────────────────────────────

def test_normalize_section_artifact_injects_missing_data_section_id():
    entry = {
        "section_id": "intro",
        "mjml_fragment": "<mj-section><mj-column><mj-text>hi</mj-text></mj-column></mj-section>",
        "claims_used": ["intro-1"],
        "design_reasoning": "n/a",
    }
    art = _normalize_section_artifact(entry, sid="intro")
    assert 'data-section-id="intro"' in art["mjml_fragment"]
    assert art["claims_used"] == ["intro-1"]


def test_normalize_section_artifact_leaves_existing_attribute_alone():
    frag = '<mj-section data-section-id="intro" background-color="#fff">x</mj-section>'
    art = _normalize_section_artifact(
        {"mjml_fragment": frag, "claims_used": [], "design_reasoning": ""},
        sid="intro",
    )
    # attribute count stays at 1 (not 2)
    assert art["mjml_fragment"].count('data-section-id="intro"') == 1


# ─────────────────────────────────────────────────────────────────────────────
# _fallback_section_artifact
# ─────────────────────────────────────────────────────────────────────────────

def test_fallback_section_artifact_preserves_verbatim_content():
    art = _fallback_section_artifact({
        "section_id": "x",
        "headline": "Critical Claim 3.1 pts",
        "copy_outline": "Sentence one. Sentence two with 30% rate.",
    })
    frag = art["mjml_fragment"]
    assert 'data-section-id="x"' in frag
    assert "Critical Claim 3.1 pts" in frag
    assert "Sentence one. Sentence two with 30% rate." in frag


def test_fallback_section_artifact_handles_empty_inputs():
    art = _fallback_section_artifact({"section_id": "x"})
    assert 'data-section-id="x"' in art["mjml_fragment"]
    assert art["claims_used"] == []


# ─────────────────────────────────────────────────────────────────────────────
# _parse_section_artifact (section-wise mode parser, sanity check)
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_section_artifact_self_heals_data_section_id():
    raw = json.dumps({
        "design_reasoning": "n/a",
        "mjml_fragment": "<mj-section><mj-text>hi</mj-text></mj-section>",
        "claims_used": ["a"],
    })
    art = _parse_section_artifact(raw, sid="intro")
    assert 'data-section-id="intro"' in art["mjml_fragment"]
    assert art["claims_used"] == ["a"]


def test_parse_section_artifact_returns_empty_on_garbage():
    art = _parse_section_artifact("not json", sid="intro")
    assert art == {}


# ─────────────────────────────────────────────────────────────────────────────
# _load_candidate_mode
# ─────────────────────────────────────────────────────────────────────────────

def test_load_candidate_mode_returns_one_of_valid_modes():
    mode = _load_candidate_mode()
    assert mode in _VALID_CANDIDATE_MODES


def test_load_candidate_mode_default_is_full_mjml():
    assert _DEFAULT_CANDIDATE_MODE == "full_mjml"


def test_load_candidate_mode_falls_back_on_unknown(tmp_path, monkeypatch):
    """Patch the YAML path to point at a temporary file with an unknown
    mode and confirm we fall back to the default."""
    bad_cfg = tmp_path / "pipeline_config.yaml"
    bad_cfg.write_text("pipeline:\n  candidate_mode: not_a_real_mode\n")
    import pipeline.nodes.code_generator as cg

    monkeypatch.setattr(cg, "_PIPELINE_CONFIG", str(bad_cfg))
    assert cg._load_candidate_mode() == _DEFAULT_CANDIDATE_MODE


def test_load_candidate_mode_reads_section_wise(tmp_path, monkeypatch):
    cfg = tmp_path / "pipeline_config.yaml"
    cfg.write_text("pipeline:\n  candidate_mode: section_wise\n")
    import pipeline.nodes.code_generator as cg

    monkeypatch.setattr(cg, "_PIPELINE_CONFIG", str(cfg))
    assert cg._load_candidate_mode() == "section_wise"
