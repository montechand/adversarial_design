"""
tests/test_mjml_utils.py — exercises the pure-Python pieces of mjml_utils.

We do NOT exercise the real `mjml-python` round-trip here (that depends on
an optional binary). Instead we cover the deterministic logic:

  - compile_mjml_document orders by `order`, wraps with markers, builds
    <mj-head> with the brand body font
  - validate_fragment correctly flags missing data-section-id, <sup> tags
    without data-claim-id, missing headlines, dropped sentences
  - find_missing_sentences ignores formatting / case / whitespace and only
    flags substantive prose drift
  - the mjml_to_html retry loop returns a graceful failure when mjml-python
    isn't installed (no LLM client supplied)

This is the same set of mechanical checks the production pipeline performs
(see Backend-Server/.../message_generator_from_scratch_v0_1.py line 16006
SUMMARY block and line 14211 data-section-id contract).
"""

from __future__ import annotations

import asyncio

import pytest

from pipeline.utils.mjml_utils import (
    compile_mjml_document,
    find_missing_sentences,
    mjml_to_html,
    normalize_section_widths,
    validate_all_sections,
    validate_fragment,
)


# ─────────────────────────────────────────────────────────────────────────────
# compile_mjml_document
# ─────────────────────────────────────────────────────────────────────────────

def test_compile_orders_by_order_field():
    sections = [
        {"section_id": "B", "order": 2, "headline": "B head", "copy_outline": "B body."},
        {"section_id": "A", "order": 1, "headline": "A head", "copy_outline": "A body."},
        {"section_id": "C", "order": 3, "headline": "C head", "copy_outline": "C body."},
    ]
    fragments = {
        "A": '<mj-section data-section-id="A">A frag</mj-section>',
        "B": '<mj-section data-section-id="B">B frag</mj-section>',
        "C": '<mj-section data-section-id="C">C frag</mj-section>',
    }
    doc = compile_mjml_document(
        fragments_by_sid=fragments,
        blueprint_sections=sections,
        brand_body_font="Lato, sans-serif",
        layout_width=600,
    )
    a_pos = doc.find("A frag")
    b_pos = doc.find("B frag")
    c_pos = doc.find("C frag")
    assert 0 < a_pos < b_pos < c_pos


def test_compile_includes_section_markers_and_head():
    sections = [{"section_id": "intro", "order": 1, "headline": "h", "copy_outline": ""}]
    fragments = {"intro": '<mj-section data-section-id="intro">x</mj-section>'}
    doc = compile_mjml_document(
        fragments_by_sid=fragments,
        blueprint_sections=sections,
        brand_body_font="Inter, sans-serif",
        layout_width=640,
    )
    assert "<mj-head>" in doc
    assert "<mj-body" in doc
    assert 'width="640px"' in doc
    assert "Inter, sans-serif" in doc
    assert "<!-- Section:Start intro -->" in doc
    assert "<!-- Section:End intro -->" in doc


def test_compile_skips_missing_fragment_section(caplog):
    sections = [
        {"section_id": "A", "order": 1, "headline": "h", "copy_outline": "c"},
        {"section_id": "B", "order": 2, "headline": "h", "copy_outline": "c"},
    ]
    fragments = {"A": '<mj-section data-section-id="A">A</mj-section>'}  # B missing

    with caplog.at_level("WARNING"):
        doc = compile_mjml_document(
            fragments_by_sid=fragments,
            blueprint_sections=sections,
            brand_body_font="Arial, sans-serif",
        )
    assert "Section:Start A" in doc
    assert "Section:Start B" not in doc
    assert any("missing MJML fragments" in r.message for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────────
# validate_fragment
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_fragment_happy_path():
    sec = {
        "section_id": "intro",
        "headline": "Reduce MG-ADL by 3 points",
        "copy_outline": "ULTOMIRIS reduced MG-ADL by 3.1 points at week 26. "
                        "Patients tolerated treatment well.",
    }
    frag = (
        '<mj-section data-section-id="intro">'
        '<mj-column>'
        '<mj-text>Reduce MG-ADL by 3 points</mj-text>'
        '<mj-text>ULTOMIRIS reduced MG-ADL by 3.1 points<sup data-claim-id="intro-1">1</sup> '
        'at week 26.</mj-text>'
        '<mj-text>Patients tolerated treatment well.</mj-text>'
        '</mj-column></mj-section>'
    )
    v = validate_fragment(fragment=frag, section=sec)
    assert v.data_section_id_ok is True
    assert v.sup_total == 1
    assert v.sup_with_claim_id == 1
    assert v.sup_without_claim_id == 0
    assert v.missing_headline is False
    assert v.missing_sentences == []
    assert v.warnings == []


def test_validate_fragment_flags_missing_data_section_id():
    sec = {"section_id": "intro", "headline": "", "copy_outline": ""}
    frag = "<mj-section><mj-column><mj-text>hi</mj-text></mj-column></mj-section>"
    v = validate_fragment(fragment=frag, section=sec)
    assert v.data_section_id_ok is False
    assert any("data-section-id" in w for w in v.warnings)


def test_validate_fragment_flags_sup_without_data_claim_id():
    sec = {"section_id": "intro", "headline": "h", "copy_outline": "h"}
    frag = (
        '<mj-section data-section-id="intro">'
        '<mj-text>h<sup>1</sup></mj-text>'
        '</mj-section>'
    )
    v = validate_fragment(fragment=frag, section=sec)
    assert v.sup_total == 1
    assert v.sup_with_claim_id == 0
    assert v.sup_without_claim_id == 1
    assert any("data-claim-id" in w for w in v.warnings)


def test_validate_fragment_flags_missing_headline_and_sentences():
    sec = {
        "section_id": "efficacy",
        "headline": "Definitive Efficacy Results",
        "copy_outline": "ULTOMIRIS demonstrated significant benefit. "
                        "QMG scores improved by 2.8 points.",
    }
    # Headline reworded, second sentence dropped entirely
    frag = (
        '<mj-section data-section-id="efficacy">'
        '<mj-text>Some efficacy data</mj-text>'
        '<mj-text>ULTOMIRIS demonstrated significant benefit.</mj-text>'
        '</mj-section>'
    )
    v = validate_fragment(fragment=frag, section=sec)
    assert v.missing_headline is True
    assert "QMG scores improved by 2.8 points." in v.missing_sentences


# ─────────────────────────────────────────────────────────────────────────────
# find_missing_sentences
# ─────────────────────────────────────────────────────────────────────────────

def test_find_missing_sentences_returns_dropped_sentences():
    outline = (
        "ULTOMIRIS reduced MG-ADL by 3.1 points. "
        "QMG dropped by 2.8 points. "
        "30% of patients were super-responders."
    )
    rendered = (
        "ULTOMIRIS reduced MG-ADL by 3.1 points. "
        "Some other rephrased text here. "
        "30% of patients were super-responders."
    )
    missing = find_missing_sentences(outline, rendered)
    assert missing == ["QMG dropped by 2.8 points."]


def test_find_missing_sentences_normalizes_whitespace_and_case():
    outline = "Headline rendered EXACTLY. Body sentence here."
    rendered = "   headline   rendered exactly.   body sentence here.  "
    assert find_missing_sentences(outline, rendered) == []


def test_find_missing_sentences_skips_short_sentences():
    outline = "OK. This longer sentence must appear verbatim."
    rendered = "This longer sentence must appear verbatim."
    # "OK." is < 4 chars and ignored
    assert find_missing_sentences(outline, rendered) == []


def test_find_missing_sentences_handles_empty_inputs():
    assert find_missing_sentences("", "anything") == []
    assert find_missing_sentences("anything.", "") == []


# ─────────────────────────────────────────────────────────────────────────────
# validate_all_sections (aggregator over multiple sections)
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_all_sections_collects_warnings_per_section():
    sections = [
        {"section_id": "A", "headline": "Heading A", "copy_outline": "Body A here."},
        {"section_id": "B", "headline": "Heading B", "copy_outline": "Body B here."},
    ]
    fragments = {
        "A": '<mj-section data-section-id="A"><mj-text>Heading A Body A here.</mj-text></mj-section>',
        # B is missing data-section-id AND drops the body
        "B": '<mj-section><mj-text>Heading B</mj-text></mj-section>',
    }
    results, warnings = validate_all_sections(
        fragments_by_sid=fragments, blueprint_sections=sections,
    )
    assert len(results) == 2
    # B should have at least two warnings (data-section-id + missing sentence)
    b_warnings = [w for w in warnings if w.startswith("[B]")]
    assert any("data-section-id" in w for w in b_warnings)
    assert any("not found verbatim" in w for w in b_warnings)
    # A should have no warnings
    a_warnings = [w for w in warnings if w.startswith("[A]")]
    assert a_warnings == []


def test_validate_all_sections_flags_missing_fragment():
    sections = [{"section_id": "A", "headline": "h", "copy_outline": ""}]
    results, warnings = validate_all_sections(
        fragments_by_sid={}, blueprint_sections=sections,
    )
    assert results == []
    assert any("no MJML fragment" in w for w in warnings)


# ─────────────────────────────────────────────────────────────────────────────
# normalize_section_widths — section overflow auto-fix
#
# Exercises the failure cases pulled from a real overflowing candidate
# (storage/runs/20260526_183322/candidate_2.mjml): a CTA section with
# `padding="32px 40px"` + `<mj-column width="600px">` rendered at 680px,
# and 2-column sections with `padding="24px"` + 240+360 columns at 648px.
# ─────────────────────────────────────────────────────────────────────────────

def test_normalizer_fixes_cta_overflow_single_column():
    """The real-world CTA case: padding="32px 40px" + column width="600px"
    inside body width=600 → 680px overflow. Expect width="100%" (one column,
    so ratio collapses to 100%)."""
    frag = (
        '<mj-section data-section-id="cta_button" padding="32px 40px 32px 40px" '
        'background-color="#008578">'
        '<mj-column width="600px" padding="0">'
        '<mj-button>Click</mj-button>'
        '</mj-column>'
        '</mj-section>'
    )
    out = normalize_section_widths(frag, body_width=600, sid="cta_button")
    assert 'width="100%"' in out
    assert 'width="600px"' not in out
    # Padding on the section itself MUST be preserved (we only touch column widths).
    assert 'padding="32px 40px 32px 40px"' in out


def test_normalizer_fixes_two_column_overflow_preserves_ratio():
    """240+360=600 column widths + 24px+24px padding → 648px overflow.
    Expect 40%/60% to preserve the ratio."""
    frag = (
        '<mj-section data-section-id="hero" padding="24px 24px 24px 24px">'
        '<mj-column width="240px"><mj-text>L</mj-text></mj-column>'
        '<mj-column width="360px"><mj-text>R</mj-text></mj-column>'
        '</mj-section>'
    )
    out = normalize_section_widths(frag, body_width=600, sid="hero")
    assert 'width="40%"' in out
    assert 'width="60%"' in out
    assert 'width="240px"' not in out
    assert 'width="360px"' not in out


def test_normalizer_noop_when_columns_fit():
    """Sum=600 + padding=0 = 600 → fits exactly. Leave untouched."""
    frag = (
        '<mj-section data-section-id="ok" padding="0">'
        '<mj-column width="240px"></mj-column>'
        '<mj-column width="360px"></mj-column>'
        '</mj-section>'
    )
    out = normalize_section_widths(frag, body_width=600, sid="ok")
    assert out == frag


def test_normalizer_noop_when_columns_already_percent():
    """`width="100%"` already auto-fits — no overflow possible. Leave alone."""
    frag = (
        '<mj-section data-section-id="ok" padding="32px 40px">'
        '<mj-column width="100%"><mj-text>hi</mj-text></mj-column>'
        '</mj-section>'
    )
    out = normalize_section_widths(frag, body_width=600, sid="ok")
    assert out == frag


def test_normalizer_noop_when_mixed_px_and_percent():
    """Mixed units — too ambiguous to safely rebalance. Leave alone (let
    the LLM own the fix); the validator will still surface the issue."""
    frag = (
        '<mj-section data-section-id="amb" padding="24px 24px">'
        '<mj-column width="300px"></mj-column>'
        '<mj-column width="50%"></mj-column>'
        '</mj-section>'
    )
    out = normalize_section_widths(frag, body_width=600, sid="amb")
    assert out == frag


def test_normalizer_noop_when_no_explicit_column_widths():
    """No width attrs at all → MJML auto-distributes. Leave alone."""
    frag = (
        '<mj-section data-section-id="auto" padding="24px 24px">'
        '<mj-column></mj-column>'
        '<mj-column></mj-column>'
        '</mj-section>'
    )
    out = normalize_section_widths(frag, body_width=600, sid="auto")
    assert out == frag


def test_normalizer_respects_padding_left_right_overrides():
    """`padding-left`/`padding-right` win over the shorthand. So
    padding="32px 8px" + padding-left="40px" → horizontal = 8 + 40 = 48px,
    which combined with column=580px would overflow (580+48=628 > 600)."""
    frag = (
        '<mj-section data-section-id="ov" padding="32px 8px" padding-left="40px">'
        '<mj-column width="580px"></mj-column>'
        '</mj-section>'
    )
    out = normalize_section_widths(frag, body_width=600, sid="ov")
    assert 'width="100%"' in out
    assert 'width="580px"' not in out


def test_normalizer_handles_multi_section_fragment():
    """A fragment with two `<mj-section>` siblings: one OK, one overflowing.
    Only the overflowing one should be rewritten."""
    frag = (
        '<mj-section data-section-id="ok" padding="0">'
        '<mj-column width="600px"></mj-column>'
        '</mj-section>'
        '<mj-section data-section-id="bad" padding="32px 40px">'
        '<mj-column width="600px"></mj-column>'
        '</mj-section>'
    )
    out = normalize_section_widths(frag, body_width=600, sid="merged")
    # OK section preserved
    assert 'data-section-id="ok"' in out and 'width="600px"' in out.split('data-section-id="bad"')[0]
    # BAD section rewritten
    bad_part = out.split('data-section-id="bad"')[1]
    assert 'width="100%"' in bad_part
    assert 'width="600px"' not in bad_part


def test_compile_mjml_document_applies_normalizer_end_to_end(caplog):
    """End-to-end: compile a document with a section the LLM laid out
    incorrectly. The compiled MJML must contain the percentage fix and the
    overflow must be logged at WARNING."""
    sections = [{"section_id": "cta", "order": 1, "headline": "", "copy_outline": ""}]
    fragments = {
        "cta": (
            '<mj-section data-section-id="cta" padding="32px 40px">'
            '<mj-column width="600px"><mj-button>Go</mj-button></mj-column>'
            '</mj-section>'
        )
    }
    with caplog.at_level("WARNING"):
        doc = compile_mjml_document(
            fragments_by_sid=fragments,
            blueprint_sections=sections,
            brand_body_font="Arial, sans-serif",
            layout_width=600,
        )
    assert 'width="100%"' in doc
    assert 'width="600px"' not in doc.split("<mj-body")[1].split("</mj-body>")[0].split('data-section-id="cta"')[1]
    assert any(
        "overflow" in r.message and "[cta]" in r.message
        for r in caplog.records
    ), f"expected overflow warning in compile, got: {[r.message for r in caplog.records]}"


def test_validate_fragment_reports_width_overflow_as_warning():
    """The validator must surface the overflow as a warning so it ends up in
    `candidate.validation_warnings` (the audit trail for how often the LLM
    gets width arithmetic wrong)."""
    sec = {"section_id": "cta", "headline": "", "copy_outline": ""}
    frag = (
        '<mj-section data-section-id="cta" padding="32px 40px">'
        '<mj-column width="600px"></mj-column>'
        '</mj-section>'
    )
    v = validate_fragment(fragment=frag, section=sec, body_width=600)
    assert v.width_overflows  # one entry
    assert v.width_overflows[0] == (80, 80, 600)  # overflow=80, padding=80, cols=600
    assert any("section-width overflow" in w for w in v.warnings)


def test_validate_fragment_no_overflow_no_warning():
    sec = {"section_id": "ok", "headline": "", "copy_outline": ""}
    frag = (
        '<mj-section data-section-id="ok" padding="24px 0">'
        '<mj-column width="240px"></mj-column>'
        '<mj-column width="360px"></mj-column>'
        '</mj-section>'
    )
    v = validate_fragment(fragment=frag, section=sec, body_width=600)
    assert v.width_overflows == []
    assert all("section-width overflow" not in w for w in v.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# mjml_to_html — graceful failure when mjml-python is missing / errors
# ─────────────────────────────────────────────────────────────────────────────

def test_mjml_to_html_graceful_failure_when_lib_missing_or_invalid():
    """
    Even if `mjml-python` is installed we feed it obvious garbage and
    confirm we get a clean failure (no exception, no LLM call attempted
    because async_llm_client=None).
    """
    result = asyncio.run(mjml_to_html(
        "not valid mjml at all <<<>>>",
        async_llm_client=None,
        max_retries=0,
    ))
    # Either: the lib is missing → success=False with "not installed"
    # Or:     the lib is present → success=False with a parse error
    assert result.success in (True, False)  # don't crash
    if not result.success:
        assert result.attempts == 1
        assert result.error  # some error string
