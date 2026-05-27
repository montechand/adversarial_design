"""
utils/llm_clients.py

LLM/VLM clients used across the pipeline.

There are two layers here:

1. Legacy Anthropic-only helpers (kept for backward compatibility — used by
   the nodes that have not been migrated to the multi-provider router):

     - `get_llm_client()`, `get_vlm_client()`        → sync `Anthropic`
     - `get_async_llm_client()`, `get_async_vlm_client()` → `AsyncAnthropic`

2. Multi-provider router (used by the creativizer and reverse prompter so
   the underlying model can be swapped via `config/pipeline_config.yaml`):

     - `get_async_router(node)` → `LLMRouter` for that node, which exposes
       an Anthropic-shaped `client.messages.create(messages=[...], ...)` API
       on top of Anthropic, OpenAI, or Google Gemini.

The router accepts Anthropic-style content blocks (text + base64 image
blocks) and translates them to the active provider's native shape, so call
sites do NOT need to know which provider is in use.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

import yaml
from anthropic import Anthropic, AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy Anthropic-only helpers (kept for nodes that have not migrated to the
# multi-provider router — code_generator, quality_judge, anti_ai_judge,
# compliance_check).
# ─────────────────────────────────────────────────────────────────────────────

_sync_client: Anthropic | None = None
_async_client: AsyncAnthropic | None = None


def _anthropic_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; cannot create Anthropic client. "
            "Set it in your .env or environment before running the pipeline."
        )
    return key


def get_llm_client() -> Anthropic:
    """Sync Anthropic client (cached singleton)."""
    global _sync_client
    if _sync_client is None:
        _sync_client = Anthropic(api_key=_anthropic_api_key())
    return _sync_client


def get_vlm_client() -> Anthropic:
    """Sync VLM client — same API as the LLM, different models at call site."""
    return get_llm_client()


def get_async_llm_client() -> AsyncAnthropic:
    """Async Anthropic client (cached singleton).

    Use this from `async def` nodes that fan out work with `asyncio.gather`.
    """
    global _async_client
    if _async_client is None:
        _async_client = AsyncAnthropic(api_key=_anthropic_api_key())
    return _async_client


def get_async_vlm_client() -> AsyncAnthropic:
    """Async VLM client — same API as the LLM, different models at call site."""
    return get_async_llm_client()


# ─────────────────────────────────────────────────────────────────────────────
# Multi-provider router
#
# Goal: let `config/pipeline_config.yaml > models.<node>` be either
#
#   models:
#     creativizer: "claude-sonnet-4-6"        # legacy string → auto-detect
#   models:
#     creativizer: "gpt-5.5"                  # auto-detect → openai
#   models:
#     creativizer: "gemini-2.5-pro"           # auto-detect → google
#   models:
#     creativizer:
#       provider: openai
#       model: gpt-5.5
#       temperature: 0.9
#
# ─────────────────────────────────────────────────────────────────────────────

_PIPELINE_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "../../config/pipeline_config.yaml"
)

SUPPORTED_PROVIDERS = ("anthropic", "openai", "google")

# Abstract reasoning-effort vocabulary. Each provider adapter translates to
# its native API shape (Anthropic `output_config.effort`, OpenAI
# `reasoning_effort`, Gemini `thinking_config.thinking_budget`). `"off"` and
# `"none"` are equivalent — both disable thinking on providers that support
# opting out (no-op on providers that cannot disable, e.g. Gemini 2.5 Pro).
THINKING_EFFORTS = ("max", "xhigh", "high", "medium", "low", "minimal", "off")
_DEFAULT_THINKING_EFFORT = "max"


@dataclass(frozen=True)
class ModelSpec:
    """Resolved model selection for one node.

    `provider` is always lower-cased and one of SUPPORTED_PROVIDERS. `model`
    is the provider-native model id (e.g. "claude-sonnet-4-6", "gpt-5.5",
    "gemini-2.5-pro"). `thinking_effort` is the abstract reasoning depth knob
    (one of THINKING_EFFORTS, or `None` to defer to the adapter default).
    `extra` captures any other node-level overrides (e.g. `temperature`)
    declared in the config dict form.
    """

    provider: str
    model: str
    thinking_effort: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _split_explicit_provider_prefix(model: str) -> tuple[str | None, str]:
    """Accept `<provider><sep><model>` shortcuts in the string form.

    Recognised separators are `/`, `:`, and `-` (the last only when followed
    by a model id that has no overlap with the provider's native vocabulary,
    so we don't misread `claude-opus-4-7` as `claude/opus-4-7`).

    Returns `(provider_or_None, remaining_model_id)`. When no explicit
    provider prefix is present, returns `(None, model)` and the caller falls
    back to substring-based detection.
    """
    if not model:
        return None, model
    for sep in ("/", ":"):
        head, _, tail = model.partition(sep)
        if tail and head.lower() in SUPPORTED_PROVIDERS:
            return head.lower(), tail
    # `-` separator: only treat as explicit when the head is `openai`,
    # `anthropic`, or `google`/`gemini` AND the tail starts with a known
    # model family. This avoids mangling `claude-opus-4-7`.
    head, _, tail = model.partition("-")
    head_l = head.lower()
    if not tail:
        return None, model
    aliases = {
        "openai":    "openai",
        "anthropic": "anthropic",
        "google":    "google",
        "gemini":    "google",   # `gemini-gemini-2.5-pro` would be silly;
                                 # this branch only fires for `gemini-...`
                                 # which the substring detector handles below.
    }
    provider = aliases.get(head_l)
    if provider == "google" and head_l == "gemini":
        # Don't eat the gemini- prefix — substring detection wants it.
        return None, model
    if provider is not None:
        return provider, tail
    return None, model


def _detect_provider_from_model(model: str) -> str:
    """Map a bare model id to its provider.

    Supports two shapes:
      - Native model id whose prefix encodes the provider (e.g.
        `claude-sonnet-4-6`, `gpt-5.5`, `gemini-2.5-pro`).
      - Explicit `<provider><sep><model>` shortcuts where `<sep>` is
        `/`, `:`, or `-` (e.g. `openai/gpt-5.5`, `openai:gpt-5.5`,
        `openai-gpt-5.5`, `anthropic-claude-sonnet-4-6`,
        `google-gemini-2.5-pro`).
    """
    raw = (model or "").strip()
    if not raw:
        raise ValueError("Empty model id; cannot detect provider.")

    explicit_provider, _remainder = _split_explicit_provider_prefix(raw)
    if explicit_provider is not None:
        return explicit_provider

    m = raw.lower()
    if m.startswith("claude"):
        return "anthropic"
    if (
        m.startswith("gpt")
        or m.startswith("o1")
        or m.startswith("o3")
        or m.startswith("o4")
        or m.startswith("o5")
        or m.startswith("chatgpt")
    ):
        return "openai"
    if m.startswith("gemini") or m.startswith("models/gemini"):
        return "google"
    raise ValueError(
        f"Cannot auto-detect provider from model id {model!r}. "
        f"Use one of these forms instead:\n"
        f"  - native id: 'claude-sonnet-4-6' | 'gpt-5.5' | 'gemini-2.5-pro'\n"
        f"  - explicit:  'openai/gpt-5.5' | 'openai:gpt-5.5' | 'openai-gpt-5.5'\n"
        f"  - dict form: {{provider: <{'|'.join(SUPPORTED_PROVIDERS)}>, model: ...}}"
    )


def _normalise_thinking_effort(raw: Any, *, where: str) -> str | None:
    """Validate and normalise a `thinking_effort` value (or None)."""
    if raw is None:
        return None
    effort = str(raw).strip().lower()
    if not effort:
        return None
    if effort in ("none",):   # alias for "off"
        return "off"
    if effort not in THINKING_EFFORTS:
        raise ValueError(
            f"{where}.thinking_effort={raw!r} is not one of {THINKING_EFFORTS}"
        )
    return effort


def resolve_model_spec(
    raw: str | dict | None,
    node: str,
    *,
    default_thinking_effort: str | None = None,
) -> ModelSpec:
    """Resolve a `models.<node>` config value into a ModelSpec.

    Accepts:
      - "claude-sonnet-4-6"               (auto-detect provider)
      - {"provider": "openai", "model": "gpt-5.5"}
      - {"provider": "openai", "model": "gpt-5.5", "temperature": 0.5}
      - {"provider": "openai", "model": "gpt-5.5", "thinking_effort": "high"}

    `default_thinking_effort` is used when the entry does not specify one
    (typically wired to `models.thinking_effort_default` by load_model_spec).

    Raises ValueError if the value cannot be parsed.
    """
    if raw is None:
        raise ValueError(f"models.{node} is missing from pipeline_config.yaml")
    if isinstance(raw, str):
        effort = _normalise_thinking_effort(default_thinking_effort, where=f"models.{node}")
        provider = _detect_provider_from_model(raw)
        # Strip the explicit provider prefix (`openai/`, `openai-`, etc.)
        # so the SDK receives only the native model id.
        explicit_provider, remainder = _split_explicit_provider_prefix(raw)
        model_id = remainder if explicit_provider is not None else raw
        return ModelSpec(
            provider=provider,
            model=model_id,
            thinking_effort=effort,
        )
    if isinstance(raw, dict):
        provider = str(raw.get("provider") or "").strip().lower()
        model = str(raw.get("model") or "").strip()
        if not model:
            raise ValueError(f"models.{node} dict is missing 'model'")
        if provider and provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"models.{node}.provider={provider!r} is not one of "
                f"{SUPPORTED_PROVIDERS}"
            )
        if not provider:
            provider = _detect_provider_from_model(model)
        # Strip a redundant explicit prefix if user wrote both
        # provider: openai AND model: openai-gpt-5.5.
        explicit_provider, remainder = _split_explicit_provider_prefix(model)
        if explicit_provider is not None and explicit_provider == provider:
            model = remainder

        effort_raw = raw.get("thinking_effort", default_thinking_effort)
        effort = _normalise_thinking_effort(effort_raw, where=f"models.{node}")

        extra = {
            k: v for k, v in raw.items()
            if k not in ("provider", "model", "thinking_effort")
        }
        return ModelSpec(
            provider=provider, model=model, thinking_effort=effort, extra=extra,
        )
    raise ValueError(
        f"models.{node} must be a string or {{provider, model}} dict, got {type(raw).__name__}"
    )


def load_model_spec(
    node: str,
    *,
    config_path: str = _PIPELINE_CONFIG_PATH,
    default: str | dict | None = None,
) -> ModelSpec:
    """Read `models.<node>` from pipeline_config.yaml and resolve it.

    The top-level `models.thinking_effort_default` (if present) provides the
    effort fallback for entries that don't carry their own `thinking_effort`.
    When that key is also missing, defaults to "max" — reasoning-heavy nodes
    like the creativizer and reverse prompter benefit from maximum thought
    budget on every provider.

    On any read/parse failure, falls back to `default` (if provided) or
    re-raises. Use `default` to make the call site degrade gracefully when
    the config has been wiped or the file is missing in tests.
    """
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        models_cfg = cfg.get("models", {}) if isinstance(cfg, dict) else {}
        default_effort = _normalise_thinking_effort(
            models_cfg.get("thinking_effort_default", _DEFAULT_THINKING_EFFORT),
            where="models.thinking_effort_default",
        )
        raw = models_cfg.get(node, default)
        return resolve_model_spec(raw, node, default_thinking_effort=default_effort)
    except (OSError, ValueError, yaml.YAMLError) as e:
        if default is not None:
            logger.warning(
                "[llm_clients] failed to load models.%s from %s (%s); "
                "falling back to default=%r",
                node, config_path, e, default,
            )
            return resolve_model_spec(
                default, node, default_thinking_effort=_DEFAULT_THINKING_EFFORT,
            )
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Response shim — every adapter returns this so call sites can keep using
# `response.content[0].text`.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _ContentBlock:
    text: str
    type: str = "text"


@dataclass
class LLMResponse:
    content: list[_ContentBlock]
    model: str = ""
    provider: str = ""
    usage: dict[str, int] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Adapter base + concrete providers
# ─────────────────────────────────────────────────────────────────────────────


class _MessagesNamespace:
    """Anthropic-style `client.messages.create(...)` shim attached to each router."""

    def __init__(self, owner: "LLMRouter"):
        self._owner = owner

    async def create(self, **kwargs: Any) -> LLMResponse:
        return await self._owner._create_message(**kwargs)


class LLMRouter:
    """Unified async LLM client routed to one provider for a single node.

    Surface mirrors `AsyncAnthropic`:

        response = await router.messages.create(
            model=...,            # optional — defaults to router.spec.model
            max_tokens=...,
            system=...,
            messages=[
                {"role": "user", "content": "<string OR content blocks>"},
            ],
            temperature=...,      # optional
        )

    Content blocks use the canonical Anthropic shape:
        {"type": "text",  "text": "..."}
        {"type": "image", "source": {"type": "base64",
                                     "media_type": "image/png",
                                     "data": "<base64>"}}

    Adapters translate these to the active provider's native format.
    """

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.messages = _MessagesNamespace(self)

    @property
    def provider(self) -> str:
        return self.spec.provider

    @property
    def model(self) -> str:
        return self.spec.model

    async def _create_message(self, **kwargs: Any) -> LLMResponse:
        raise NotImplementedError


# ── Anthropic ────────────────────────────────────────────────────────────────

# Floor on `max_tokens` when adaptive thinking is enabled at a given effort.
# Anthropic's guidance: start at 32k for high effort and 64k for max/xhigh —
# the model needs room for thinking + visible output (max_tokens caps both).
_ANTHROPIC_MAX_TOKENS_FLOOR: dict[str, int] = {
    "max":     64000,
    "xhigh":   64000,
    "high":    32000,
    "medium":  16000,
    "low":     8000,
    "minimal": 4000,
}

# Map the abstract effort vocabulary to Anthropic's `output_config.effort`.
# Anthropic supports max | xhigh | high | medium | low; we map our "minimal"
# down to "low" and our "off" to no effort (caller-side suppresses thinking).
_ANTHROPIC_EFFORT: dict[str, str] = {
    "max":     "max",
    "xhigh":   "xhigh",
    "high":    "high",
    "medium":  "medium",
    "low":     "low",
    "minimal": "low",
}

# When `max_tokens` exceeds this, switch from `messages.create()` to
# `messages.stream()`. The SDK's internal guard refuses non-streaming
# requests whose expected runtime exceeds ~10 minutes, which formula-wise
# trips at max_tokens > 21,333 on a 128k-context model. We keep a generous
# margin (8000) so we never bump up against it for non-trivial calls.
_ANTHROPIC_STREAM_THRESHOLD: int = 8000


class _AnthropicRouter(LLMRouter):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._client = AsyncAnthropic(api_key=_anthropic_api_key())

    async def _create_message(
        self,
        *,
        messages: list[dict],
        max_tokens: int,
        model: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        thinking_effort: str | None = None,
        **_: Any,
    ) -> LLMResponse:
        effort = (thinking_effort or self.spec.thinking_effort or "").lower() or None
        enable_thinking = effort is not None and effort != "off"

        if enable_thinking:
            # Ensure max_tokens is large enough for thinking + visible output.
            floor = _ANTHROPIC_MAX_TOKENS_FLOOR.get(effort, max_tokens)
            max_tokens = max(int(max_tokens), floor)

        call_kwargs: dict[str, Any] = {
            "model": model or self.spec.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system:
            call_kwargs["system"] = system
        if temperature is not None:
            call_kwargs["temperature"] = temperature

        if enable_thinking:
            call_kwargs["thinking"] = {"type": "adaptive"}
            native_effort = _ANTHROPIC_EFFORT.get(effort, "high")
            call_kwargs["output_config"] = {"effort": native_effort}

        # The Anthropic SDK refuses non-streaming requests whose expected
        # runtime exceeds 10 minutes. Its internal estimate is
        # `expected = 3600 * max_tokens / 128_000`; anything above
        # max_tokens ≈ 21k trips the guard. We also stream whenever
        # adaptive thinking is on, because those responses are commonly
        # long-running. See anthropic/_base_client.py:_calculate_nonstreaming_timeout.
        needs_stream = enable_thinking or max_tokens > _ANTHROPIC_STREAM_THRESHOLD

        try:
            if needs_stream:
                resp = await self._create_via_stream(call_kwargs)
            else:
                resp = await self._client.messages.create(**call_kwargs)
        except TypeError as e:
            # Older anthropic SDKs may not know `output_config` / adaptive
            # thinking — fall back to a non-thinking call so the pipeline
            # keeps running. Log it loudly so the operator notices.
            if enable_thinking and "output_config" in str(e):
                logger.warning(
                    "[llm_clients] Anthropic SDK rejected adaptive-thinking "
                    "params (output_config / thinking={type:adaptive}); "
                    "retrying without them. Upgrade the `anthropic` package "
                    "to enable thinking. Error: %s", e,
                )
                call_kwargs.pop("output_config", None)
                call_kwargs.pop("thinking", None)
                if needs_stream:
                    resp = await self._create_via_stream(call_kwargs)
                else:
                    resp = await self._client.messages.create(**call_kwargs)
            else:
                raise

        text = _extract_anthropic_text(resp)
        usage = {
            "input_tokens": getattr(getattr(resp, "usage", None), "input_tokens", 0) or 0,
            "output_tokens": getattr(getattr(resp, "usage", None), "output_tokens", 0) or 0,
        }
        return LLMResponse(
            content=[_ContentBlock(text=text)],
            model=getattr(resp, "model", call_kwargs["model"]),
            provider="anthropic",
            usage=usage,
        )

    async def _create_via_stream(self, call_kwargs: dict[str, Any]) -> Any:
        """Issue an Anthropic request via `messages.stream()` and return the final message.

        Identical inputs to `messages.create()`. We consume the stream silently
        (no incremental output is exposed upstream) and return the accumulated
        Message object so the rest of the adapter is shape-compatible.
        """
        async with self._client.messages.stream(**call_kwargs) as stream:
            # Drain the stream so the SDK can fully accumulate the message.
            # Iterating events (rather than text) is safe even when the model
            # only emits thinking blocks before timing out.
            async for _event in stream:
                pass
            return await stream.get_final_message()


def _extract_anthropic_text(resp: Any) -> str:
    """Concatenate all visible text blocks in an Anthropic response.

    Adaptive thinking can return interleaved `thinking` / `text` blocks; the
    pipeline only consumes the visible output, never the chain-of-thought,
    so this filter explicitly skips thinking blocks. Falls back to
    `response.content[0].text` behaviour for single-block responses.
    """
    pieces: list[str] = []
    for block in getattr(resp, "content", []) or []:
        btype = getattr(block, "type", None)
        if btype and btype != "text":
            # thinking / tool_use / redacted_thinking / etc. — never echo.
            continue
        text = getattr(block, "text", None)
        if text:
            pieces.append(text)
    return "".join(pieces)


# ── OpenAI ───────────────────────────────────────────────────────────────────


def _openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set; cannot create OpenAI client. "
            "Set it in your .env or environment before running with provider=openai."
        )
    return key


# Map abstract effort vocabulary to OpenAI `reasoning_effort` values.
# Reference: developers.openai.com/api/docs/guides/reasoning — supports
# minimal | low | medium | high | xhigh (xhigh is restricted to
# gpt-5.1-codex-max; we conservatively collapse our `max` and `xhigh` down
# to `high` so the same config string works on every reasoning model).
_OPENAI_EFFORT: dict[str, str] = {
    "max":     "high",
    "xhigh":   "high",
    "high":    "high",
    "medium":  "medium",
    "low":     "low",
    "minimal": "minimal",
}

# Min ceiling on max_completion_tokens when reasoning is on — reasoning
# tokens count toward this budget on OpenAI's chat.completions, so a low
# cap would prematurely truncate the visible answer.
_OPENAI_MAX_TOKENS_FLOOR: dict[str, int] = {
    "max":     32000,
    "xhigh":   32000,
    "high":    16000,
    "medium":  8000,
    "low":     4000,
    "minimal": 2000,
}


class _OpenAIRouter(LLMRouter):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai package is not installed but provider=openai was requested. "
                "Run `pip install 'openai>=1.50.0'` (or `pip install -e '.[dev]'`)."
            ) from e
        self._client = AsyncOpenAI(api_key=_openai_api_key())

    async def _create_message(
        self,
        *,
        messages: list[dict],
        max_tokens: int,
        model: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        thinking_effort: str | None = None,
        **_: Any,
    ) -> LLMResponse:
        effort = (thinking_effort or self.spec.thinking_effort or "").lower() or None
        enable_reasoning = effort is not None and effort != "off"

        if enable_reasoning:
            floor = _OPENAI_MAX_TOKENS_FLOOR.get(effort, max_tokens)
            max_tokens = max(int(max_tokens), floor)

        oai_messages: list[dict[str, Any]] = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        for msg in messages:
            oai_messages.append(_anthropic_msg_to_openai(msg))

        call_kwargs: dict[str, Any] = {
            "model": model or self.spec.model,
            "messages": oai_messages,
            "max_completion_tokens": max_tokens,
        }
        # Reasoning models on OpenAI ignore non-default temperature; only
        # forward temperature on calls that have NOT enabled reasoning.
        if temperature is not None and not enable_reasoning:
            call_kwargs["temperature"] = temperature

        if enable_reasoning:
            call_kwargs["reasoning_effort"] = _OPENAI_EFFORT.get(effort, "high")

        try:
            resp = await self._client.chat.completions.create(**call_kwargs)
        except TypeError as e:
            # Older openai SDK builds may not know `max_completion_tokens`
            # and/or `reasoning_effort` — degrade gracefully.
            err = str(e)
            if "reasoning_effort" in err:
                logger.warning(
                    "[llm_clients] OpenAI SDK rejected `reasoning_effort` — "
                    "retrying without it. Upgrade `openai` to enable reasoning. "
                    "Error: %s", e,
                )
                call_kwargs.pop("reasoning_effort", None)
            if "max_completion_tokens" in err:
                call_kwargs["max_tokens"] = call_kwargs.pop("max_completion_tokens")
            resp = await self._client.chat.completions.create(**call_kwargs)

        text = ""
        if resp.choices:
            text = resp.choices[0].message.content or ""

        usage = {}
        if getattr(resp, "usage", None):
            usage = {
                "input_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
            }
            reasoning_tokens = None
            details = getattr(resp.usage, "completion_tokens_details", None)
            if details is not None:
                reasoning_tokens = getattr(details, "reasoning_tokens", None)
            if reasoning_tokens:
                usage["reasoning_tokens"] = int(reasoning_tokens)
        return LLMResponse(
            content=[_ContentBlock(text=text)],
            model=getattr(resp, "model", call_kwargs["model"]),
            provider="openai",
            usage=usage,
        )


def _anthropic_msg_to_openai(msg: dict) -> dict:
    """Translate a single Anthropic-style message to OpenAI chat format.

    Text-only messages stay as `{"role", "content": "..."}`; mixed
    text/image messages become a content-block list with `image_url` blocks
    that inline the base64 image as a data URI.
    """
    role = msg.get("role", "user")
    content = msg.get("content", "")
    if isinstance(content, str):
        return {"role": role, "content": content}

    out_blocks: list[dict[str, Any]] = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "")
            if text:
                out_blocks.append({"type": "text", "text": text})
        elif btype == "image":
            src = block.get("source", {}) or {}
            if src.get("type") == "base64":
                media_type = src.get("media_type", "image/png")
                data = src.get("data", "")
                if data:
                    out_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{data}"},
                    })
            elif src.get("type") == "url":
                url = src.get("url", "")
                if url:
                    out_blocks.append({"type": "image_url", "image_url": {"url": url}})
        # Unknown block types are dropped silently; provider would reject them anyway.
    # Collapse single-text-block payloads back to a string for cheaper tokenization.
    if len(out_blocks) == 1 and out_blocks[0]["type"] == "text":
        return {"role": role, "content": out_blocks[0]["text"]}
    return {"role": role, "content": out_blocks}


# ── Google Gemini ────────────────────────────────────────────────────────────


def _google_api_key() -> str:
    key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_GENAI_API_KEY")
    )
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY / GOOGLE_API_KEY is not set; cannot create Gemini client. "
            "Set one in your .env before running with provider=google."
        )
    return key


# Gemini 2.5 thinking budgets — `thinking_budget=-1` is "dynamic" (model
# picks budget per request, capped by the model's own ceiling) and is the
# closest analogue to "max effort" since it never under-allocates. Discrete
# levels follow the per-model maxima documented in the Vertex / GenAI
# thinking guide (Pro caps at 32768, Flash at 24576). `0` disables thinking
# on Flash/Flash-Lite; Pro cannot be disabled so we omit the config there.
_GEMINI_THINKING_BUDGET: dict[str, int] = {
    "max":     -1,        # dynamic — model decides up to its native max
    "xhigh":   -1,        # same as max for Gemini (no separate xhigh tier)
    "high":    16384,
    "medium":  8192,
    "low":     2048,
    "minimal": 512,
}

# Per-model dynamic-mode ceiling — used so `max` effort still respects the
# model-specific upper bound when we choose to clamp instead of pass -1.
_GEMINI_MAX_BUDGET_BY_MODEL_PREFIX: list[tuple[str, int]] = [
    ("gemini-2.5-pro",        32768),
    ("gemini-2.5-flash-lite", 24576),
    ("gemini-2.5-flash",      24576),
]


def _gemini_thinking_budget(effort: str, model: str) -> int:
    """Resolve an abstract effort to a concrete thinking_budget int.

    For max/xhigh we return -1 (dynamic). For named tiers we return the
    discrete budget but clamp to the model's documented maximum so we never
    request more than the model accepts.
    """
    raw = _GEMINI_THINKING_BUDGET.get(effort, -1)
    if raw < 0:
        return raw
    model_l = (model or "").lower()
    for prefix, model_cap in _GEMINI_MAX_BUDGET_BY_MODEL_PREFIX:
        if model_l.startswith(prefix):
            return min(raw, model_cap)
    return raw


class _GeminiRouter(LLMRouter):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        try:
            from google import genai  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "google-genai package is not installed but provider=google was requested. "
                "Run `pip install 'google-genai>=1.0.0'` (or `pip install -e '.[dev]'`)."
            ) from e
        self._client = genai.Client(api_key=_google_api_key())

    async def _create_message(
        self,
        *,
        messages: list[dict],
        max_tokens: int,
        model: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        thinking_effort: str | None = None,
        **_: Any,
    ) -> LLMResponse:
        from google.genai import types  # type: ignore

        effort = (thinking_effort or self.spec.thinking_effort or "").lower() or None
        enable_thinking = effort is not None and effort != "off"
        active_model = model or self.spec.model

        contents = []
        for msg in messages:
            role = "user" if msg.get("role", "user") == "user" else "model"
            parts = _anthropic_content_to_gemini_parts(msg.get("content", ""))
            contents.append(types.Content(role=role, parts=parts))

        # Reasoning tokens count against max_output_tokens, so leave plenty
        # of headroom when thinking is enabled (the GenAI docs note that
        # otherwise the response can finish with MAX_TOKENS before any
        # visible text is emitted).
        if enable_thinking:
            floor = {
                "max": 32000, "xhigh": 32000,
                "high": 16000, "medium": 8000, "low": 4000, "minimal": 2000,
            }.get(effort, max_tokens)
            max_tokens = max(int(max_tokens), floor)

        config_kwargs: dict[str, Any] = {"max_output_tokens": max_tokens}
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if system:
            config_kwargs["system_instruction"] = system

        if enable_thinking:
            budget = _gemini_thinking_budget(effort, active_model)
            try:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=budget,
                )
            except (TypeError, ValueError) as e:
                logger.warning(
                    "[llm_clients] Gemini SDK rejected ThinkingConfig(budget=%s) "
                    "for model %s — proceeding without explicit thinking config. "
                    "Error: %s", budget, active_model, e,
                )

        config = types.GenerateContentConfig(**config_kwargs)

        resp = await self._client.aio.models.generate_content(
            model=active_model,
            contents=contents,
            config=config,
        )
        text = getattr(resp, "text", None) or ""
        usage: dict[str, int] = {}
        meta = getattr(resp, "usage_metadata", None)
        if meta:
            usage = {
                "input_tokens": getattr(meta, "prompt_token_count", 0) or 0,
                "output_tokens": getattr(meta, "candidates_token_count", 0) or 0,
            }
            thoughts = getattr(meta, "thoughts_token_count", None)
            if thoughts:
                usage["reasoning_tokens"] = int(thoughts)
        return LLMResponse(
            content=[_ContentBlock(text=text)],
            model=active_model,
            provider="google",
            usage=usage,
        )


def _anthropic_content_to_gemini_parts(content: str | list) -> list:
    """Translate an Anthropic-style message content (string or block list)
    into a list of `google.genai.types.Part` objects."""
    from google.genai import types  # type: ignore

    if isinstance(content, str):
        return [types.Part.from_text(text=content)]

    parts: list = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "")
            if text:
                parts.append(types.Part.from_text(text=text))
        elif btype == "image":
            src = block.get("source", {}) or {}
            if src.get("type") == "base64":
                data_b64 = src.get("data", "")
                if not data_b64:
                    continue
                try:
                    raw = base64.b64decode(data_b64)
                except (ValueError, TypeError):
                    continue
                parts.append(types.Part.from_bytes(
                    data=raw,
                    mime_type=src.get("media_type", "image/png"),
                ))
            elif src.get("type") == "url":
                url = src.get("url", "")
                if url:
                    parts.append(types.Part.from_uri(
                        file_uri=url,
                        mime_type=src.get("media_type", "image/png"),
                    ))
    return parts


# ─────────────────────────────────────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────────────────────────────────────

_router_cache: dict[tuple[str, str], LLMRouter] = {}


def _build_router(spec: ModelSpec) -> LLMRouter:
    if spec.provider == "anthropic":
        return _AnthropicRouter(spec)
    if spec.provider == "openai":
        return _OpenAIRouter(spec)
    if spec.provider == "google":
        return _GeminiRouter(spec)
    raise ValueError(
        f"Unsupported provider {spec.provider!r}; expected one of {SUPPORTED_PROVIDERS}"
    )


def get_async_router(
    node: str,
    *,
    default: str | dict | None = None,
    config_path: str = _PIPELINE_CONFIG_PATH,
) -> LLMRouter:
    """Return a cached async LLMRouter for the given pipeline node.

    Reads `models.<node>` from pipeline_config.yaml and constructs the
    appropriate provider client. Subsequent calls with the same (provider,
    model) pair return the same router instance — fine because every
    adapter is thread/async-safe.

    `default` is forwarded to `load_model_spec` so the call site can declare
    a sensible fallback (used by tests and as defensive fallback if the
    config row is missing).
    """
    spec = load_model_spec(node, config_path=config_path, default=default)
    key = (spec.provider, spec.model)
    cached = _router_cache.get(key)
    if cached is not None:
        return cached
    router = _build_router(spec)
    _router_cache[key] = router
    return router


def reset_router_cache() -> None:
    """Used by tests when monkeypatching env / config to force fresh wiring."""
    _router_cache.clear()


__all__ = [
    "ModelSpec",
    "LLMResponse",
    "LLMRouter",
    "SUPPORTED_PROVIDERS",
    "THINKING_EFFORTS",
    "resolve_model_spec",
    "load_model_spec",
    "get_async_router",
    "reset_router_cache",
    "get_llm_client",
    "get_vlm_client",
    "get_async_llm_client",
    "get_async_vlm_client",
]
