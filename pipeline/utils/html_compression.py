r"""
pipeline/utils/html_compression.py

Two-way bridge between *rendered* HTML (with embedded `data:image/...;base64,...`
URIs) and *LLM-cheap* HTML (with `[images_base64_i]` placeholder tokens).

Why this exists
---------------
The backend-server email generator produces self-contained HTML where every
image is embedded as a `data:image/png;base64,XXXXX...` URI. A single email
can easily blow past 50–200 KB of base64 per image. Feeding that into a text
LLM is wasteful and frequently overflows context windows.

The contract used by the adversarial pipeline:

  * LLM **text** calls (reverse_prompter narrative arm, creativizer,
    code_generator, compliance_check) always see the *compressed* HTML where
    each base64 URI has been replaced by a single short token of the form
    `[images_base64_<index>]`.
  * LLM **vision** calls (reverse_prompter image arm, quality_judge,
    anti_ai_judge) and the screenshotter see the *rehydrated* HTML so the
    rendered output is visually faithful.
  * The mapping `{token: raw_b64}` travels alongside the compressed HTML in
    pipeline state.

The compressor accepts and produces token names that match this regex:
    `\[images_base64_(\d+)\]`
"""

from __future__ import annotations

import re
import base64
from typing import Iterable, Any
from pathlib import Path


PLACEHOLDER_TEMPLATE = "[images_base64_{i}]"
PLACEHOLDER_REGEX = re.compile(r"\[images_base64_(\d+)\]")

# Match `data:<mime>;base64,<payload>` URIs anywhere — both inside attributes
# (src="data:..."), inside CSS (background:url('data:...')), and as bare URIs.
# Capture groups:
#   1. mime type (e.g. "image/png")
#   2. base64 payload
DATA_URI_REGEX = re.compile(
    r"data:(image/[A-Za-z0-9.+-]+);base64,([A-Za-z0-9+/=\r\n\s]+?)(?=[\"'\)\s>])",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Compression: rendered HTML → LLM-cheap HTML + sidecar map
# ─────────────────────────────────────────────────────────────────────────────

def compress_html_base64(html: str) -> tuple[str, dict[str, str]]:
    """
    Replace every `data:image/...;base64,XXX` URI with `[images_base64_i]`.

    Returns:
        compressed_html: the HTML with placeholders in place of data URIs.
        images_b64:      `{placeholder_token: original_data_uri}` mapping.
                         The value is the **full** data URI (including
                         `data:image/png;base64,` prefix) so rehydration is
                         lossless.

    Notes:
      - Identical data URIs share a single token (deduped). Saves more tokens
        when the same image (e.g. a logo) recurs.
      - Whitespace inside the base64 payload (line breaks) is preserved in
        the stored value so rehydration produces byte-identical HTML.
    """
    if not html:
        return html, {}

    seen_uri_to_token: dict[str, str] = {}
    images_b64: dict[str, str] = {}
    next_index = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal next_index
        full_uri = match.group(0)
        if full_uri in seen_uri_to_token:
            return seen_uri_to_token[full_uri]
        token = PLACEHOLDER_TEMPLATE.format(i=next_index)
        next_index += 1
        seen_uri_to_token[full_uri] = token
        images_b64[token] = full_uri
        return token

    compressed = DATA_URI_REGEX.sub(_replace, html)
    return compressed, images_b64


# ─────────────────────────────────────────────────────────────────────────────
# Rehydration: LLM-cheap HTML + map → rendered HTML
# ─────────────────────────────────────────────────────────────────────────────

def rehydrate_html(compressed_html: str, images_b64: dict[str, str]) -> str:
    """
    Replace `[images_base64_i]` tokens with their original data URIs, but
    **only** when the token appears in an image-loading context:

      - `<img src="…">` / `<img srcset="…">` (and the same on `<source>`)
      - `<img background="…">` / `<img poster="…">` (legacy attrs)
      - any CSS `url(…)` call (inline `style="…"` or `<style>` block)

    Tokens anywhere else — HTML comments, plain text content, prose inside
    `<p>` etc. — are left untouched. This stops LLM-emitted commentary like
    `<!-- ASSET CHECK: [images_base64_0] must be the gMG hero -->` from
    accidentally inlining a megabyte of base64 into the rendered file.

    Tokens that ARE in an image context but missing from `images_b64` are
    still replaced with a 1×1 transparent PNG so the browser does not 404.
    """
    if not compressed_html:
        return compressed_html

    def _replace_tokens(text: str) -> str:
        def repl(m: re.Match[str]) -> str:
            token = m.group(0)
            if token in images_b64:
                return images_b64[token]
            return TRANSPARENT_PNG_DATA_URI
        return PLACEHOLDER_REGEX.sub(repl, text)

    # Pass 1: replace tokens inside <img>/<source> attribute values. We
    # target the whole tag with the OUTER regex so attribute substitution
    # only runs within tag bounds (never in surrounding text or comments).
    def _img_attr_sub(am: re.Match[str]) -> str:
        attr, quote, value = am.group(1), am.group(2), am.group(3)
        return f"{attr}={quote}{_replace_tokens(value)}{quote}"

    def _img_tag_sub(tm: re.Match[str]) -> str:
        return _IMG_ATTR_RE.sub(_img_attr_sub, tm.group(0))

    out = _IMG_TAG_RE.sub(_img_tag_sub, compressed_html)

    # Pass 2: CSS url(…) anywhere — inline styles and <style> blocks.
    def _url_sub(um: re.Match[str]) -> str:
        quote, value = um.group(1), um.group(2)
        return f"url({quote}{_replace_tokens(value)}{quote})"

    out = _CSS_URL_RE.sub(_url_sub, out)
    return out


# Image-loading tags. We replace tokens ONLY inside these tag bounds, so
# tokens inside HTML comments / arbitrary prose are left literal.
_IMG_TAG_RE = re.compile(r"<(?:img|source)\b[^>]*>", re.IGNORECASE)

# Attribute-value substitution inside an image tag. Covers the two modern
# attrs (`src` / `srcset`) plus the two legacy attrs (`background` / `poster`)
# that some MJML/HTML emitters use for fallbacks.
_IMG_ATTR_RE = re.compile(
    r"\b(src|srcset|background|poster)\s*=\s*(['\"])(.*?)\2",
    re.IGNORECASE | re.DOTALL,
)

# CSS `url(…)` — quoted or unquoted. Captures the optional quote so we can
# preserve whichever style the source used.
_CSS_URL_RE = re.compile(
    r"url\(\s*(['\"]?)([^'\")\s]+)\1\s*\)",
    re.IGNORECASE,
)


# A 1×1 transparent PNG, used as a safe fallback for unknown placeholder
# tokens (e.g. LLM hallucinations) so the browser does not 404.
TRANSPARENT_PNG_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgAAIAAAUAAeImBZsAAAAASUVORK5CYII="
)


# ─────────────────────────────────────────────────────────────────────────────
# Inspection helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_placeholders(text: str) -> list[str]:
    """Return all `[images_base64_i]` tokens present in `text`, in order."""
    return PLACEHOLDER_REGEX.findall(text)


def placeholder_tokens(images_b64: dict[str, str]) -> list[str]:
    """Return the placeholder tokens in numeric order (i=0, i=1, ...)."""
    def _key(token: str) -> int:
        m = PLACEHOLDER_REGEX.match(token)
        return int(m.group(1)) if m else -1
    return sorted(images_b64.keys(), key=_key)


def split_data_uri(data_uri: str) -> tuple[str, str]:
    """
    Split a `data:<mime>;base64,<payload>` URI into `(mime, payload)`.

    Raises:
        ValueError: when the string is not a data URI.
    """
    match = re.match(r"data:([^;]+);base64,(.*)", data_uri, flags=re.DOTALL)
    if not match:
        raise ValueError("Not a base64 data URI")
    return match.group(1), match.group(2)


def decode_to_bytes(data_uri: str) -> bytes:
    """Return raw image bytes for a data URI (whitespace tolerant)."""
    _mime, payload = split_data_uri(data_uri)
    payload = re.sub(r"\s+", "", payload)
    return base64.b64decode(payload)


# ─────────────────────────────────────────────────────────────────────────────
# Side-car JSON I/O (convenience for the CLI)
# ─────────────────────────────────────────────────────────────────────────────

def load_images_b64_json(path: str | Path) -> dict[str, str]:
    """
    Load a `{placeholder_token: raw_b64_or_data_uri}` map from JSON.

    Accepts three on-disk shapes:
      - **legacy string** — `{"[images_base64_0]": "data:image/png;base64,..."}`
      - **legacy raw**    — `{"[images_base64_0]": "iVBORw0KGgoAAAA..."}` (wrapped
        into a `data:image/png;base64,` prefix permissively).
      - **enriched**      — `{"[images_base64_0]": {"data_uri": "...",
        "intrinsic": {...}, "usage": {...}}}` — only the `data_uri` field is
        returned here; call `load_images_b64_with_descriptors` to also pull
        the descriptor metadata.
    """
    images_b64, _descriptors = load_images_b64_with_descriptors(path)
    return images_b64


def load_images_b64_with_descriptors(
    path: str | Path,
) -> tuple[dict[str, str], dict[str, dict]]:
    """
    Same input formats as :func:`load_images_b64_json`, but also returns the
    descriptor metadata when present.

    Returns `(images_b64, descriptors)` where:
      - `images_b64` is `{placeholder_token: data_uri}`
      - `descriptors` is `{placeholder_token: {"intrinsic": {...}, "usage": {...}}}`
        Tokens whose JSON value was a plain string get an empty descriptor
        dict — callers can then mine descriptors themselves at runtime.
    """
    import json
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected JSON object, got {type(raw).__name__}")
    images: dict[str, str] = {}
    descriptors: dict[str, dict] = {}
    for token, value in raw.items():
        if isinstance(value, str):
            images[token] = value if value.startswith("data:") else f"data:image/png;base64,{value.strip()}"
            descriptors[token] = {}
        elif isinstance(value, dict):
            data_uri = value.get("data_uri") or value.get("src") or ""
            if not isinstance(data_uri, str):
                raise ValueError(f"{path}: {token!r}.data_uri must be a string")
            if data_uri and not data_uri.startswith("data:"):
                data_uri = f"data:image/png;base64,{data_uri.strip()}"
            images[token] = data_uri
            descriptors[token] = {
                "intrinsic": dict(value.get("intrinsic") or {}),
                "usage":     dict(value.get("usage") or {}),
            }
        else:
            raise ValueError(
                f"{path}: value for {token!r} must be a string or object, "
                f"got {type(value).__name__}"
            )
    return images, descriptors


def save_images_b64_json(images_b64: dict[str, str], path: str | Path) -> None:
    import json
    Path(path).write_text(json.dumps(images_b64, indent=2))


def save_images_b64_with_descriptors(
    images_b64: dict[str, str],
    descriptors: dict[str, dict],
    path: str | Path,
) -> None:
    """Write the enriched format. Tokens without descriptors are still
    serialized as `{"data_uri": "..."}` for forward compatibility."""
    import json
    payload: dict[str, dict] = {}
    for token, uri in images_b64.items():
        entry: dict[str, Any] = {"data_uri": uri}
        desc = descriptors.get(token) or {}
        if desc.get("intrinsic"):
            entry["intrinsic"] = desc["intrinsic"]
        if desc.get("usage"):
            entry["usage"] = desc["usage"]
        payload[token] = entry
    Path(path).write_text(json.dumps(payload, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Misc utilities used by ingestion / candidates
# ─────────────────────────────────────────────────────────────────────────────

def estimate_compression_savings(html: str) -> dict[str, int]:
    """
    Quick diagnostic returned by ingestion logs.

    Returns sizes in bytes and the number of placeholders that were inserted.
    """
    compressed, mapping = compress_html_base64(html)
    return {
        "original_chars": len(html),
        "compressed_chars": len(compressed),
        "saved_chars": len(html) - len(compressed),
        "image_count": len(mapping),
    }


def ensure_placeholder_format(token: str) -> str:
    """Normalize a token to `[images_base64_<i>]`. Raises on malformed input."""
    m = PLACEHOLDER_REGEX.fullmatch(token)
    if not m:
        raise ValueError(f"Not a valid placeholder token: {token!r}")
    return PLACEHOLDER_TEMPLATE.format(i=int(m.group(1)))


def iter_tokens_with_payloads(
    images_b64: dict[str, str],
) -> Iterable[tuple[str, str, bytes]]:
    """Yield `(token, mime, raw_bytes)` for each entry — used by judges."""
    for token in placeholder_tokens(images_b64):
        data_uri = images_b64[token]
        mime, _ = split_data_uri(data_uri)
        yield token, mime, decode_to_bytes(data_uri)
