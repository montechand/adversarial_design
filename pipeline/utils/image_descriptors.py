"""
utils/image_descriptors.py

Mine descriptive metadata for the baseline images so the reverse prompter
can reason about how each image was originally used (not just what bytes
it carries).

Two layers of metadata:

  * **intrinsic** — derived from the image bytes themselves
    (decoded width/height/aspect ratio/mime/size).
  * **usage** — derived from how the baseline HTML actually places the
    image (rendered width/height attribute, alt text, parent class
    chain, role hints like "banner" / "hero").

Output schema (per token):

    {
      "intrinsic": {
        "width_px": int,
        "height_px": int,
        "aspect_ratio": "W:H",
        "aspect_ratio_decimal": float,   # width / height
        "mime": "image/png",
        "size_bytes": int,
        "orientation": "landscape" | "portrait" | "square",
      },
      "usage": {
        "alt": str,
        "parent_selector": str,          # e.g. "a.banner" or "div.section.hero"
        "css_role_hint": str,            # one of "banner|hero|moa|efficacy|..." mined from class
        "is_referenced_in_html": bool,
      },
    }

Note: rendered width/height attributes are intentionally not captured.
Most HCP-email `<img>` tags only set `width=` (height derives from aspect
ratio at render time), so the height side was almost always `null`. The
intrinsic block already gives the LLM enough to reason about scaling, and
the layout's intended size is better inferred from the `parent_selector`
(e.g. `.section.hero` vs `.moa-grid > .col`) than from a single attribute.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from typing import Any
from html.parser import HTMLParser
from math import gcd

from PIL import Image

from pipeline.utils.html_compression import (
    PLACEHOLDER_REGEX,
    placeholder_tokens,
    split_data_uri,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Intrinsic dimensions (from base64 bytes)
# ─────────────────────────────────────────────────────────────────────────────

def decode_intrinsic_dimensions(data_uri: str) -> dict[str, Any]:
    """Decode a data URI and return `{width_px, height_px, mime, size_bytes,
    aspect_ratio, aspect_ratio_decimal, orientation}`.

    Returns an empty dict if decoding fails — callers should treat that as
    "intrinsic unknown" and still emit usage data.
    """
    try:
        mime, payload = split_data_uri(data_uri)
    except Exception as e:
        logger.warning("[image_descriptors] not a data URI: %s", e)
        return {}
    try:
        payload_clean = re.sub(r"\s+", "", payload)
        raw = base64.b64decode(payload_clean)
        with Image.open(io.BytesIO(raw)) as img:
            w, h = img.size
    except Exception as e:
        logger.warning("[image_descriptors] failed to decode image bytes: %s", e)
        return {"mime": mime, "size_bytes": 0}

    return {
        "width_px": int(w),
        "height_px": int(h),
        "mime": mime,
        "size_bytes": len(raw),
        "aspect_ratio": _aspect_ratio_str(w, h),
        "aspect_ratio_decimal": round(w / h, 3) if h else 0.0,
        "orientation": _orientation(w, h),
    }


def _aspect_ratio_str(w: int, h: int) -> str:
    if not w or not h:
        return ""
    g = gcd(w, h)
    return f"{w // g}:{h // g}"


def _orientation(w: int, h: int) -> str:
    if w == h:
        return "square"
    return "landscape" if w > h else "portrait"


# ─────────────────────────────────────────────────────────────────────────────
# HTML usage extraction (from the compressed HTML containing tokens)
# ─────────────────────────────────────────────────────────────────────────────

_VOID_TAGS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
})


class _UsageHarvester(HTMLParser):
    """Walk the compressed HTML, recording each `<img>` / `<source>` /
    background-url that references a `[images_base64_N]` token along with
    its width/height attrs, alt text, and ancestor class chain."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._stack: list[tuple[str, dict[str, str]]] = []
        self.findings: dict[str, dict[str, Any]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = {k.lower(): (v or "") for k, v in attrs}

        if tag in ("img", "source"):
            src = attrs_dict.get("src") or attrs_dict.get("srcset", "")
            for token in PLACEHOLDER_REGEX.findall(src or ""):
                self._record(
                    token=f"[images_base64_{token}]",
                    attrs=attrs_dict,
                    tag=tag,
                )

        # CSS background:url('[images_base64_N]') in inline style attribute
        style = attrs_dict.get("style", "")
        for token in PLACEHOLDER_REGEX.findall(style or ""):
            self._record(
                token=f"[images_base64_{token}]",
                attrs=attrs_dict,
                tag=tag,
                via="background",
            )

        # Void elements never close, so don't push them on the ancestor stack.
        if tag not in _VOID_TAGS:
            self._stack.append((tag, attrs_dict))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # XHTML-style `<img ... />` — same logic as a start tag without push.
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _VOID_TAGS:
            return
        # Pop until the matching tag — tolerates unclosed inner elements
        # without letting the stack grow unbounded.
        for idx in range(len(self._stack) - 1, -1, -1):
            if self._stack[idx][0] == tag:
                del self._stack[idx:]
                return

    def _record(
        self,
        *,
        token: str,
        attrs: dict[str, str],
        tag: str,
        via: str = "src",
    ) -> None:
        if token in self.findings:
            return  # first occurrence wins; cleanest usage signal
        parent_selector = self._compose_selector(self._stack)
        # All class tokens in the ancestor chain — useful to mine a role hint.
        all_classes = " ".join(
            (a.get("class") or "") for _, a in self._stack
        ).strip()
        self.findings[token] = {
            "alt": attrs.get("alt", "").strip(),
            "parent_selector": parent_selector,
            "tag": tag,
            "via": via,
            "css_role_hint": _infer_role_hint(all_classes, attrs.get("alt", "")),
            "is_referenced_in_html": True,
        }

    @staticmethod
    def _compose_selector(stack: list[tuple[str, dict[str, str]]]) -> str:
        """Render the ancestor chain as a CSS-like selector. We keep at most
        the two innermost ancestors so the string stays compact."""
        if not stack:
            return ""
        chain: list[str] = []
        for tag, attrs in stack[-3:]:
            cls = (attrs.get("class") or "").strip()
            if cls:
                cls_token = "." + ".".join(cls.split())
                chain.append(f"{tag}{cls_token}")
            else:
                chain.append(tag)
        return " > ".join(chain)


# Coarse semantic labels mined from common email-section class names so the
# LLM gets a hint about what role the image plays even if alt text is sparse.
_ROLE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("banner",   ("banner",)),
    ("hero",     ("hero",)),
    ("logo",     ("logo",)),
    ("moa",      ("moa", "mechanism")),
    ("efficacy", ("efficacy", "kaplan", "km-curve", "dfs", "primary-efficacy")),
    ("safety",   ("safety", "ae-table")),
    ("isi",      ("isi", "important-safety")),
    ("cta",      ("cta", "call-to-action")),
    ("signature",("signature", "sign-off")),
    ("icon",     ("icon",)),
)


def _infer_role_hint(class_chain: str, alt: str) -> str:
    haystack = f"{class_chain} {alt}".lower()
    for role, keywords in _ROLE_KEYWORDS:
        if any(k in haystack for k in keywords):
            return role
    return ""


def extract_html_usage(compressed_html: str) -> dict[str, dict[str, Any]]:
    """Walk the compressed HTML and return `{token: usage_dict}` for every
    `[images_base64_N]` token referenced by an `<img>`, `<source>`, or
    inline-`style` background URL."""
    if not compressed_html:
        return {}
    parser = _UsageHarvester()
    try:
        parser.feed(compressed_html)
    except Exception as e:
        logger.warning("[image_descriptors] HTML parse failed: %s", e)
    return parser.findings


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_descriptors(
    *,
    images_b64: dict[str, str],
    compressed_html: str = "",
    existing: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build a `{token: descriptor}` map for every token in `images_b64`.

    For each token we attempt to:
      1. Reuse any descriptor already supplied by the caller (e.g. loaded
         from JSON sidecar) — wins on every individual field.
      2. Decode intrinsic dimensions from the data URI bytes (PIL).
      3. Mine HTML usage from `compressed_html` (rendered width/height,
         alt, parent_selector, role hint).

    Missing tokens get a stub entry so downstream code doesn't have to
    constantly check for `None`.
    """
    existing = existing or {}
    html_usage = extract_html_usage(compressed_html)
    out: dict[str, dict[str, Any]] = {}
    for token in placeholder_tokens(images_b64):
        prior = existing.get(token) or {}
        data_uri = images_b64.get(token, "")
        intrinsic_mined = decode_intrinsic_dimensions(data_uri) if data_uri else {}
        usage_mined = html_usage.get(token, {
            "alt": "",
            "parent_selector": "",
            "tag": "",
            "via": "",
            "css_role_hint": "",
            "is_referenced_in_html": False,
        })
        out[token] = {
            "intrinsic": {**intrinsic_mined, **(prior.get("intrinsic") or {})},
            "usage":     {**usage_mined,     **(prior.get("usage") or {})},
        }
    return out


def format_descriptor_for_prompt(token: str, descriptor: dict[str, Any]) -> str:
    """Render one descriptor as a compact, LLM-friendly summary string.

    Example output:

        token [images_base64_0] — intrinsic 1360x220 (6.18:1, landscape,
          image/png, 178 KB) — usage <img> in a.banner; role=banner;
          alt: "ULTOMIRIS (ravulizumab-cwvz)..."
    """
    intrinsic = descriptor.get("intrinsic") or {}
    usage = descriptor.get("usage") or {}

    parts: list[str] = [f"token {token}"]

    if intrinsic.get("width_px") and intrinsic.get("height_px"):
        ratio = intrinsic.get("aspect_ratio") or ""
        orientation = intrinsic.get("orientation") or ""
        mime = intrinsic.get("mime") or "image"
        size_kb = (intrinsic.get("size_bytes") or 0) / 1024
        bits = [
            f"intrinsic {intrinsic['width_px']}x{intrinsic['height_px']}",
        ]
        descriptors_inner = []
        if ratio:
            descriptors_inner.append(ratio)
        if orientation:
            descriptors_inner.append(orientation)
        descriptors_inner.append(mime)
        if size_kb >= 1:
            descriptors_inner.append(f"{size_kb:.0f} KB")
        else:
            descriptors_inner.append(f"{int(intrinsic.get('size_bytes') or 0)} B")
        bits[-1] += f" ({', '.join(descriptors_inner)})"
        parts.append("; ".join(bits))
    else:
        parts.append("intrinsic unknown")

    if usage.get("is_referenced_in_html"):
        usage_bits = []
        tag = usage.get("tag") or "img"
        via = usage.get("via") or "src"
        usage_bits.append(f"<{tag}> via {via}")
        sel = usage.get("parent_selector") or ""
        if sel:
            usage_bits.append(f"in {sel}")
        role = usage.get("css_role_hint") or ""
        if role:
            usage_bits.append(f"role={role}")
        alt = usage.get("alt") or ""
        if alt:
            alt_trim = alt if len(alt) <= 160 else alt[:157] + "…"
            usage_bits.append(f'alt: "{alt_trim}"')
        parts.append("usage " + "; ".join(usage_bits))
    else:
        parts.append("usage: not referenced by current baseline HTML")

    return " — ".join(parts)
