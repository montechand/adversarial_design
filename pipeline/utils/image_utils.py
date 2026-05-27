"""
utils/image_utils.py

Encode image files to base64 for Anthropic VLM API input.
Handles PNG and JPEG. Optionally downscales large screenshots
to stay within API limits.
"""
import base64
import os
from pathlib import Path
from PIL import Image
import io

MAX_DIMENSION = 1568   # Anthropic VLM max recommended dimension


def encode_image_b64(image_path: str, max_dim: int = MAX_DIMENSION) -> str:
    """Return base64-encoded PNG string, downscaling if needed."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")
