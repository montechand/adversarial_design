"""
pipeline/utils/style_guide.py

Load and normalize the style-guide ruleset that the code generator,
creativizer, and compliance check inject as the "Brand Rules (SUPREME —
NON-NEGOTIABLE)" block.

The contract mirrors the backend's `_normalize_brand_tokens` helper exactly:
the source is `design_bible.website.{content_pattern_rules, color_scheme_rules,
design_pattern_rules, other_rules}` and the consumer receives that same
four-list dict, serialized verbatim into the prompt.

Public API consumed by the pipeline:

  - load_style_guide(path)        → normalized 4-list dict (drop-in for prompts)
  - load_raw_style_guide(path)    → full parsed JSON (for inspection)
  - normalize_rules(raw)          → raw payload → 4-list dict
  - format_brand_rules_block(rules) → prompt-ready block string ("" if empty)
  - rules_block_summary(rules)    → compact bullet digest for tight prompts
  - extract_body_font(rules)      → CSS font-family string for the MJML <mj-head>

See:
  - examples/style_guide_ruleset.json (expected shape)
  - Backend-Server/.../message_generator_from_scratch_v0_1.py line 1448
    `_normalize_brand_tokens` (production loader we mirror)
"""

from __future__ import annotations

import json
import re
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EMPTY_RULES: dict[str, list[str]] = {
    "content_pattern_rules": [],
    "color_scheme_rules": [],
    "design_pattern_rules": [],
    "other_rules": [],
}

_RULE_KEYS = tuple(EMPTY_RULES.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_style_guide(path: str | Path | None) -> dict[str, Any]:
    """Read the JSON file as-is. Returns `{}` for missing paths / parse errors."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        logger.warning("[style_guide] file not found: %s — using empty rules", p)
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        logger.error("[style_guide] invalid JSON in %s: %s", p, e)
        return {}


def normalize_rules(style_guide: dict[str, Any]) -> dict[str, list[str]]:
    """
    Extract the four natural-language rule lists from a style-guide payload.

    Mirrors `_normalize_brand_tokens` in the backend:
      - looks for `design_bible.website` first (production shape)
      - falls back to `design_bible` directly
      - falls back to `brand_guide` / `style_guide` keys
      - falls back to top-level keys (`content_pattern_rules`, etc.)
      - returns `EMPTY_RULES` on anything malformed

    Filters out lines beginning with the literal "TEMPLATE — " marker — those
    are documentation placeholders that should never reach an LLM.
    """
    if not isinstance(style_guide, dict):
        return dict(EMPTY_RULES)

    bible = (
        style_guide.get("design_bible")
        or style_guide.get("brand_guide")
        or style_guide.get("style_guide")
        or {}
    )
    if isinstance(bible, dict) and "website" in bible and isinstance(bible["website"], dict):
        tokens = bible["website"]
    elif isinstance(bible, dict):
        tokens = bible
    else:
        tokens = {}

    if not any(k in tokens for k in _RULE_KEYS):
        tokens = style_guide  # last-chance: top-level keys

    return {k: _clean_list(tokens.get(k)) for k in _RULE_KEYS}


def _clean_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for v in values:
        if not isinstance(v, str):
            continue
        v_strip = v.strip()
        if not v_strip:
            continue
        if v_strip.startswith("TEMPLATE — "):
            continue
        out.append(v_strip)
    return out


def load_style_guide(path: str | Path | None) -> dict[str, list[str]]:
    """
    Drop-in loader: read the JSON file → normalized 4-list dict.

    This is what the CLI and pipeline nodes import. Returns `EMPTY_RULES`
    if `path` is None or the file is missing / invalid, so callers can safely
    forward the result into prompts without None-checks.
    """
    return normalize_rules(load_raw_style_guide(path))


# Backwards-compatible alias for `load_and_normalize` (older code paths).
load_and_normalize = load_style_guide


def has_any_rules(rules: dict[str, list[str]] | None) -> bool:
    """True iff at least one of the four rule lists is non-empty."""
    if not rules:
        return False
    return any(rules.get(k) for k in _RULE_KEYS)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt-side helpers
# ─────────────────────────────────────────────────────────────────────────────

_RULES_HEADER = (
    "## Brand Rules (SUPREME — NON-NEGOTIABLE)\n"
    "**These rules override ALL other instructions.** Fonts, colors, design "
    "patterns, content patterns — follow them exactly."
)

_RULES_FOOTER = (
    "Font sizes, spacing, and layout are your creative decisions — but the "
    "font families, colors, and design patterns specified above MUST be used."
)


def format_brand_rules_block(rules: dict[str, list[str]] | None) -> str:
    """
    Format the 4-list rules dict into the same "Brand Rules (SUPREME — NON-
    NEGOTIABLE)" block the backend designer prompt uses (line ~14195 of
    `message_generator_from_scratch_v0_1.py`).

    Returns `""` if `rules` is None or every category is empty, so callers
    can conditionally splice without an extra check.
    """
    if not has_any_rules(rules):
        return ""
    body = json.dumps(rules, ensure_ascii=False, indent=2)
    return f"{_RULES_HEADER}\n```json\n{body}\n```\n\n{_RULES_FOOTER}"


def rules_block_json(rules: dict[str, list[str]] | None) -> str:
    """Raw JSON-serialized rules (no header / footer wrapping). Empty string
    if no rules. Useful for callers that want to splice the JSON into their
    own prompt scaffolding."""
    if not has_any_rules(rules):
        return ""
    return json.dumps(rules, ensure_ascii=False, indent=2)


def rules_block_summary(rules: dict[str, list[str]] | None, max_per_list: int = 4) -> str:
    """Compact bullet-point digest for prompts where space is tight."""
    if not has_any_rules(rules):
        return "(no brand rules provided)"
    lines: list[str] = []
    for k in _RULE_KEYS:
        vs = rules.get(k) or []
        if not vs:
            continue
        lines.append(f"- {k}:")
        for v in vs[:max_per_list]:
            v_trim = v if len(v) < 200 else v[:197] + "..."
            lines.append(f"    • {v_trim}")
        if len(vs) > max_per_list:
            lines.append(f"    • (+{len(vs) - max_per_list} more rules)")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Brand body font extraction (mirrors backend's _extract_body_font_from_brand)
# ─────────────────────────────────────────────────────────────────────────────

_KNOWN_FONTS = (
    "Inter", "Lato", "Roboto", "Open Sans", "Source Sans Pro", "Source Sans",
    "Helvetica Neue", "Helvetica", "Arial", "Georgia", "Times New Roman", "Times",
    "Verdana", "Tahoma", "Trebuchet MS", "Trebuchet",
    "Oswald", "Bebas Neue", "Montserrat", "Poppins", "Raleway",
    "PT Sans", "Noto Sans", "Nunito", "Work Sans",
    "Merriweather", "Playfair Display", "Lora",
)

_FONT_REGEX = re.compile(
    r"(?:body|email|primary|paragraph|copy)[^.\n]{0,80}?"
    r"(?:font(?:-family)?|typeface)[^\n]{0,80}?"
    r"(?P<font>" + "|".join(re.escape(f) for f in _KNOWN_FONTS) + r")",
    re.IGNORECASE,
)

_FALLBACK_FONT = "Arial, Helvetica, sans-serif"


def extract_body_font(
    rules: dict[str, list[str]] | None,
    explicit: str | None = None,
) -> str:
    """
    Pick a CSS font-family string for the email body / <mj-head>.

    Resolution order:
      1. `explicit` argument if provided (e.g. from CLI flag).
      2. A "body" / "email" / "primary" font hint found in any rule list
         (e.g. "Body copy is Lato 14px, 1.5 line-height").
      3. The first known font name mentioned anywhere in the rules.
      4. `Arial, Helvetica, sans-serif` fallback.

    Always returns a string with a safe sans-serif fallback appended.
    """
    if explicit:
        return _ensure_fallback(explicit)

    if not rules:
        return _FALLBACK_FONT

    all_text = "\n".join(v for lst in rules.values() for v in lst)
    if not all_text:
        return _FALLBACK_FONT

    m = _FONT_REGEX.search(all_text)
    if m:
        return _ensure_fallback(m.group("font"))

    for font in _KNOWN_FONTS:
        if re.search(r"\b" + re.escape(font) + r"\b", all_text, re.IGNORECASE):
            return _ensure_fallback(font)

    return _FALLBACK_FONT


def _ensure_fallback(font: str) -> str:
    font = font.strip().strip(",")
    if not font:
        return _FALLBACK_FONT
    lower = font.lower()
    if "sans-serif" in lower or "serif" in lower:
        return font
    return f"{font}, Helvetica, Arial, sans-serif"
