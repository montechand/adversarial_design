"""
tests/test_llm_clients.py

Unit tests for the multi-provider LLM router in `pipeline/utils/llm_clients.py`.

These tests cover the pure-Python parts (config resolution + Anthropic→native
message translation). They do NOT make real API calls — the adapters
themselves are exercised via a small `_create_message` monkeypatch.

Run:
    cd adversarial_design
    PYTHONPATH=. pytest tests/test_llm_clients.py -v
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import pytest
import yaml

from pipeline.utils import llm_clients
from pipeline.utils.llm_clients import (
    LLMResponse,
    ModelSpec,
    SUPPORTED_PROVIDERS,
    THINKING_EFFORTS,
    _ContentBlock,
    _anthropic_content_to_gemini_parts,
    _anthropic_msg_to_openai,
    _detect_provider_from_model,
    _gemini_thinking_budget,
    _normalise_thinking_effort,
    get_async_router,
    load_model_spec,
    reset_router_cache,
    resolve_model_spec,
)


# ─────────────────────────────────────────────────────────────────────────────
# provider auto-detection
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-sonnet-4-6", "anthropic"),
        ("Claude-Opus-4-6", "anthropic"),
        ("gpt-5.5", "openai"),
        ("gpt-4o-mini", "openai"),
        ("o3-mini", "openai"),
        ("o4-mini", "openai"),
        ("chatgpt-latest", "openai"),
        ("gemini-2.5-pro", "google"),
        ("gemini-1.5-flash", "google"),
        ("models/gemini-2.0-flash-001", "google"),
    ],
)
def test_detect_provider_known_prefixes(model: str, expected: str) -> None:
    assert _detect_provider_from_model(model) == expected


def test_detect_provider_unknown_raises() -> None:
    with pytest.raises(ValueError, match="auto-detect"):
        _detect_provider_from_model("mistral-large-latest")


def test_detect_provider_empty_raises() -> None:
    with pytest.raises(ValueError):
        _detect_provider_from_model("")


# Explicit `<provider><sep><model>` shortcuts — `/`, `:`, and `-` all work.
# This was the bug surfaced when the user wrote `openai-gpt-5.5` in their
# YAML and the auto-detector silently fell back to a different provider.
@pytest.mark.parametrize(
    "model,expected_provider",
    [
        # Slash separator (most conventional)
        ("openai/gpt-5.5",                 "openai"),
        ("anthropic/claude-sonnet-4-6",    "anthropic"),
        ("google/gemini-2.5-pro",          "google"),
        # Colon separator
        ("openai:gpt-5.5",                 "openai"),
        ("anthropic:claude-sonnet-4-6",    "anthropic"),
        # Hyphen separator — the exact form the user wrote.
        ("openai-gpt-5.5",                 "openai"),
        ("anthropic-claude-sonnet-4-6",    "anthropic"),
        ("google-gemini-2.5-pro",          "google"),
        # Case-insensitivity in the provider segment.
        ("OpenAI/gpt-5.5",                 "openai"),
        ("ANTHROPIC/claude-opus-4-7",      "anthropic"),
    ],
)
def test_detect_provider_explicit_prefix(model: str, expected_provider: str) -> None:
    assert _detect_provider_from_model(model) == expected_provider


def test_native_claude_id_not_mangled_by_hyphen_prefix_parsing() -> None:
    # Bare `claude-opus-4-7` must NOT be parsed as `{provider: claude, model: opus-4-7}`.
    # `claude` is not in SUPPORTED_PROVIDERS so the hyphen-prefix path must skip it.
    assert _detect_provider_from_model("claude-opus-4-7") == "anthropic"


def test_resolve_model_spec_strips_explicit_prefix_from_model_id() -> None:
    """The SDK should receive the native model id, not `openai-gpt-5.5`."""
    spec = resolve_model_spec("openai-gpt-5.5", node="creativizer")
    assert spec.provider == "openai"
    assert spec.model == "gpt-5.5"

    spec = resolve_model_spec("openai/gpt-5.5", node="creativizer")
    assert spec.provider == "openai"
    assert spec.model == "gpt-5.5"

    spec = resolve_model_spec("google-gemini-2.5-pro", node="reverse_prompter")
    assert spec.provider == "google"
    assert spec.model == "gemini-2.5-pro"


def test_resolve_model_spec_native_claude_id_preserved() -> None:
    """`claude-opus-4-7` must not be stripped down to `opus-4-7`."""
    spec = resolve_model_spec("claude-opus-4-7", node="creativizer")
    assert spec.provider == "anthropic"
    assert spec.model == "claude-opus-4-7"


def test_resolve_model_spec_dict_drops_redundant_explicit_prefix() -> None:
    """{provider: openai, model: openai-gpt-5.5} → model becomes gpt-5.5."""
    spec = resolve_model_spec(
        {"provider": "openai", "model": "openai-gpt-5.5"}, node="creativizer",
    )
    assert spec.provider == "openai"
    assert spec.model == "gpt-5.5"


def test_load_model_spec_handles_explicit_prefix_string_entry(tmp_path: Path) -> None:
    """End-to-end: YAML with the form the user originally wrote should now load."""
    cfg = _write_config(tmp_path, {"models": {"creativizer": "openai-gpt-5.5"}})
    spec = load_model_spec("creativizer", config_path=str(cfg))
    assert spec.provider == "openai"
    assert spec.model == "gpt-5.5"


def test_supported_providers_set() -> None:
    assert set(SUPPORTED_PROVIDERS) == {"anthropic", "openai", "google"}


# ─────────────────────────────────────────────────────────────────────────────
# resolve_model_spec
# ─────────────────────────────────────────────────────────────────────────────


def test_resolve_model_spec_from_string() -> None:
    spec = resolve_model_spec("claude-sonnet-4-6", node="creativizer")
    assert spec == ModelSpec(provider="anthropic", model="claude-sonnet-4-6")
    assert spec.extra == {}


def test_resolve_model_spec_from_dict_explicit_provider() -> None:
    spec = resolve_model_spec(
        {"provider": "openai", "model": "gpt-5.5", "temperature": 0.4},
        node="creativizer",
    )
    assert spec.provider == "openai"
    assert spec.model == "gpt-5.5"
    assert spec.extra == {"temperature": 0.4}


def test_resolve_model_spec_dict_autodetects_provider() -> None:
    spec = resolve_model_spec({"model": "gemini-2.5-pro"}, node="reverse_prompter")
    assert spec.provider == "google"
    assert spec.model == "gemini-2.5-pro"


def test_resolve_model_spec_normalises_provider_case() -> None:
    spec = resolve_model_spec(
        {"provider": "OpenAI", "model": "gpt-5.5"},
        node="creativizer",
    )
    assert spec.provider == "openai"


def test_resolve_model_spec_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="provider"):
        resolve_model_spec(
            {"provider": "mistral", "model": "mistral-large"},
            node="creativizer",
        )


def test_resolve_model_spec_rejects_missing_model() -> None:
    with pytest.raises(ValueError, match="model"):
        resolve_model_spec({"provider": "openai"}, node="creativizer")


def test_resolve_model_spec_rejects_invalid_type() -> None:
    with pytest.raises(ValueError):
        resolve_model_spec(42, node="creativizer")  # type: ignore[arg-type]


def test_resolve_model_spec_rejects_none() -> None:
    with pytest.raises(ValueError, match="missing"):
        resolve_model_spec(None, node="creativizer")


# ─────────────────────────────────────────────────────────────────────────────
# load_model_spec from a YAML file
# ─────────────────────────────────────────────────────────────────────────────


def _write_config(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "pipeline_config.yaml"
    p.write_text(yaml.safe_dump(payload))
    return p


def test_load_model_spec_string_entry(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, {"models": {"creativizer": "gpt-5.5"}})
    spec = load_model_spec("creativizer", config_path=str(cfg))
    assert spec.provider == "openai"
    assert spec.model == "gpt-5.5"


def test_load_model_spec_dict_entry(tmp_path: Path) -> None:
    cfg = _write_config(
        tmp_path,
        {"models": {"reverse_prompter": {"provider": "google", "model": "gemini-2.5-pro"}}},
    )
    spec = load_model_spec("reverse_prompter", config_path=str(cfg))
    assert spec.provider == "google"
    assert spec.model == "gemini-2.5-pro"


def test_load_model_spec_falls_back_to_default_when_node_missing(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, {"models": {"creativizer": "claude-sonnet-4-6"}})
    spec = load_model_spec(
        "reverse_prompter", config_path=str(cfg), default="claude-opus-4-6"
    )
    assert spec.provider == "anthropic"
    assert spec.model == "claude-opus-4-6"


def test_load_model_spec_falls_back_when_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    spec = load_model_spec(
        "creativizer", config_path=str(missing), default="gpt-5.5"
    )
    assert spec.provider == "openai"
    assert spec.model == "gpt-5.5"


def test_load_model_spec_raises_without_default_when_node_missing(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, {"models": {"creativizer": "claude-sonnet-4-6"}})
    with pytest.raises(ValueError):
        load_model_spec("reverse_prompter", config_path=str(cfg))


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic → OpenAI message translation
# ─────────────────────────────────────────────────────────────────────────────


def test_openai_translation_text_only_message() -> None:
    msg = {"role": "user", "content": "hello world"}
    out = _anthropic_msg_to_openai(msg)
    assert out == {"role": "user", "content": "hello world"}


def test_openai_translation_single_text_block_collapses_to_string() -> None:
    msg = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    out = _anthropic_msg_to_openai(msg)
    # Collapsed for cheaper tokenization
    assert out == {"role": "user", "content": "hi"}


def test_openai_translation_text_and_image_blocks() -> None:
    img_b64 = base64.b64encode(b"\x89PNG\r\n").decode("ascii")
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "describe this:"},
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": img_b64,
            }},
        ],
    }
    out = _anthropic_msg_to_openai(msg)
    assert out["role"] == "user"
    blocks = out["content"]
    assert isinstance(blocks, list) and len(blocks) == 2
    assert blocks[0] == {"type": "text", "text": "describe this:"}
    assert blocks[1]["type"] == "image_url"
    assert blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert blocks[1]["image_url"]["url"].endswith(img_b64)


def test_openai_translation_drops_unknown_block_types() -> None:
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "keep me"},
            {"type": "tool_use", "id": "ignored"},
        ],
    }
    out = _anthropic_msg_to_openai(msg)
    assert out == {"role": "user", "content": "keep me"}


def test_openai_translation_handles_url_image_source() -> None:
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "look:"},
            {"type": "image", "source": {"type": "url", "url": "https://example.com/x.png"}},
        ],
    }
    out = _anthropic_msg_to_openai(msg)
    blocks = out["content"]
    assert isinstance(blocks, list)
    assert blocks[-1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/x.png"},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic → Gemini message translation
# ─────────────────────────────────────────────────────────────────────────────


def test_gemini_translation_text_string() -> None:
    parts = _anthropic_content_to_gemini_parts("hello")
    assert len(parts) == 1
    # Gemini Part objects expose .text via .text attribute on the Part
    assert getattr(parts[0], "text", None) == "hello"


def test_gemini_translation_text_blocks_only() -> None:
    parts = _anthropic_content_to_gemini_parts([
        {"type": "text", "text": "alpha"},
        {"type": "text", "text": "beta"},
    ])
    assert len(parts) == 2
    assert parts[0].text == "alpha"
    assert parts[1].text == "beta"


def test_gemini_translation_text_and_base64_image() -> None:
    raw_bytes = b"\x89PNG\r\nabc"
    img_b64 = base64.b64encode(raw_bytes).decode("ascii")
    parts = _anthropic_content_to_gemini_parts([
        {"type": "text", "text": "see:"},
        {"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": img_b64,
        }},
    ])
    assert len(parts) == 2
    assert parts[0].text == "see:"
    # Image part should round-trip back to the same raw bytes via inline_data
    inline = parts[1].inline_data
    assert inline is not None
    assert inline.mime_type == "image/png"
    assert inline.data == raw_bytes


def test_gemini_translation_skips_undecodable_base64() -> None:
    parts = _anthropic_content_to_gemini_parts([
        {"type": "text", "text": "ok"},
        {"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": "@@@not_b64@@@",
        }},
    ])
    # Bad base64 silently dropped; text part survives
    assert len(parts) == 1
    assert parts[0].text == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# get_async_router end-to-end with monkeypatched adapters
# ─────────────────────────────────────────────────────────────────────────────


class _StubRouter(llm_clients.LLMRouter):
    """In-memory router that records the spec it was built with and the
    last call kwargs — used to verify config wiring without doing IO."""

    instances: list["_StubRouter"] = []
    last_call: dict | None = None

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        _StubRouter.instances.append(self)

    async def _create_message(self, **kwargs):
        _StubRouter.last_call = {"spec": self.spec, **kwargs}
        return LLMResponse(
            content=[_ContentBlock(text="stub-ok")],
            model=self.spec.model,
            provider=self.spec.provider,
        )


@pytest.fixture
def stub_routers(monkeypatch):
    """Replace the adapter factory so we can assert routing without HTTP."""
    _StubRouter.instances = []
    _StubRouter.last_call = None
    reset_router_cache()
    monkeypatch.setattr(llm_clients, "_build_router", _StubRouter)
    yield _StubRouter
    reset_router_cache()


def test_get_async_router_uses_string_config(tmp_path: Path, stub_routers) -> None:
    cfg = _write_config(tmp_path, {"models": {"creativizer": "gpt-5.5"}})
    router = get_async_router("creativizer", config_path=str(cfg))
    assert router.provider == "openai"
    assert router.model == "gpt-5.5"


def test_get_async_router_uses_dict_config(tmp_path: Path, stub_routers) -> None:
    cfg = _write_config(
        tmp_path,
        {"models": {"reverse_prompter": {"provider": "google", "model": "gemini-2.5-pro"}}},
    )
    router = get_async_router("reverse_prompter", config_path=str(cfg))
    assert router.provider == "google"
    assert router.model == "gemini-2.5-pro"


def test_get_async_router_caches_same_spec(tmp_path: Path, stub_routers) -> None:
    cfg = _write_config(tmp_path, {"models": {"creativizer": "claude-sonnet-4-6"}})
    a = get_async_router("creativizer", config_path=str(cfg))
    b = get_async_router("creativizer", config_path=str(cfg))
    assert a is b
    assert len(stub_routers.instances) == 1


def test_get_async_router_distinct_specs_get_distinct_clients(
    tmp_path: Path, stub_routers
) -> None:
    cfg = _write_config(
        tmp_path,
        {
            "models": {
                "creativizer": "gpt-5.5",
                "reverse_prompter": "gemini-2.5-pro",
            }
        },
    )
    a = get_async_router("creativizer", config_path=str(cfg))
    b = get_async_router("reverse_prompter", config_path=str(cfg))
    assert a is not b
    assert a.provider == "openai"
    assert b.provider == "google"
    assert len(stub_routers.instances) == 2


def test_router_messages_create_returns_response_shim(
    tmp_path: Path, stub_routers
) -> None:
    cfg = _write_config(tmp_path, {"models": {"creativizer": "gpt-5.5"}})
    router = get_async_router("creativizer", config_path=str(cfg))

    async def _go():
        return await router.messages.create(
            max_tokens=100,
            system="be brief",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.5,
        )

    resp = asyncio.run(_go())
    assert resp.content[0].text == "stub-ok"
    assert stub_routers.last_call is not None
    assert stub_routers.last_call["max_tokens"] == 100
    assert stub_routers.last_call["system"] == "be brief"
    assert stub_routers.last_call["temperature"] == 0.5
    assert stub_routers.last_call["messages"] == [
        {"role": "user", "content": "hi"}
    ]


# ─────────────────────────────────────────────────────────────────────────────
# thinking_effort: vocabulary + ModelSpec / config wiring
# ─────────────────────────────────────────────────────────────────────────────


def test_thinking_efforts_vocabulary() -> None:
    assert set(THINKING_EFFORTS) == {
        "max", "xhigh", "high", "medium", "low", "minimal", "off",
    }


@pytest.mark.parametrize("raw,expected", [
    ("max", "max"),
    ("HIGH", "high"),
    ("  Medium ", "medium"),
    ("none", "off"),       # alias
    ("off", "off"),
    (None, None),
    ("", None),
])
def test_normalise_thinking_effort_accepted_values(raw, expected) -> None:
    assert _normalise_thinking_effort(raw, where="x") == expected


def test_normalise_thinking_effort_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="thinking_effort"):
        _normalise_thinking_effort("ultra", where="models.creativizer")


def test_resolve_model_spec_string_inherits_default_effort() -> None:
    spec = resolve_model_spec(
        "gpt-5.5", node="creativizer", default_thinking_effort="max",
    )
    assert spec.thinking_effort == "max"


def test_resolve_model_spec_dict_per_node_effort_wins_over_default() -> None:
    spec = resolve_model_spec(
        {"provider": "openai", "model": "gpt-5.5", "thinking_effort": "low"},
        node="creativizer",
        default_thinking_effort="max",
    )
    assert spec.thinking_effort == "low"
    # `thinking_effort` is NOT leaked into `extra`
    assert "thinking_effort" not in spec.extra


def test_resolve_model_spec_dict_rejects_invalid_effort() -> None:
    with pytest.raises(ValueError, match="thinking_effort"):
        resolve_model_spec(
            {"provider": "openai", "model": "gpt-5.5", "thinking_effort": "ultra"},
            node="creativizer",
        )


def test_load_model_spec_uses_top_level_thinking_effort_default(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, {
        "models": {
            "thinking_effort_default": "max",
            "creativizer": "claude-sonnet-4-6",
            "reverse_prompter": {
                "provider": "openai", "model": "gpt-5.5",
                "thinking_effort": "medium",
            },
        }
    })
    cv = load_model_spec("creativizer", config_path=str(cfg))
    rp = load_model_spec("reverse_prompter", config_path=str(cfg))
    # string entry inherits top-level default
    assert cv.thinking_effort == "max"
    # dict entry's per-node override takes precedence over the default
    assert rp.thinking_effort == "medium"


def test_load_model_spec_defaults_to_max_when_no_top_level_default(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, {"models": {"creativizer": "claude-sonnet-4-6"}})
    spec = load_model_spec("creativizer", config_path=str(cfg))
    # When neither models.thinking_effort_default nor a per-node entry is set,
    # the module-level default ("max") applies — reasoning-heavy nodes
    # benefit from maximum thinking by default.
    assert spec.thinking_effort == "max"


# ─────────────────────────────────────────────────────────────────────────────
# Gemini thinking-budget resolver
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("effort,model,expected", [
    # dynamic mode is preserved for max/xhigh regardless of model
    ("max",     "gemini-2.5-pro",        -1),
    ("max",     "gemini-2.5-flash",      -1),
    ("xhigh",   "gemini-2.5-pro",        -1),
    # discrete tiers respect per-model cap
    ("high",    "gemini-2.5-pro",        16384),
    ("high",    "gemini-2.5-flash",      16384),
    ("medium",  "gemini-2.5-pro",        8192),
    ("low",     "gemini-2.5-flash-lite", 2048),
    ("minimal", "gemini-2.5-pro",        512),
])
def test_gemini_thinking_budget_resolution(effort, model, expected) -> None:
    assert _gemini_thinking_budget(effort, model) == expected


def test_gemini_thinking_budget_clamps_to_model_cap() -> None:
    # Force a fake effort value that would exceed Pro's documented 32768 cap.
    # Easiest way to assert clamping: monkey the lookup table indirectly by
    # checking that the resolver never returns a value higher than the cap
    # for known model prefixes.
    cap_pro = 32768
    for effort in ("max", "high", "medium", "low", "minimal"):
        v = _gemini_thinking_budget(effort, "gemini-2.5-pro")
        assert v == -1 or v <= cap_pro


# ─────────────────────────────────────────────────────────────────────────────
# Adapter-level translation: thinking params reach the underlying SDK
# ─────────────────────────────────────────────────────────────────────────────


def _build_anthropic_response(text: str = "ok") -> object:
    """Build a MagicMock that quacks like an Anthropic Message response."""
    from unittest.mock import MagicMock
    return MagicMock(
        content=[MagicMock(type="text", text=text)],
        model="claude-sonnet-4-6",
        usage=MagicMock(input_tokens=10, output_tokens=5),
    )


class _FakeAnthropicStream:
    """Async-context-manager double for `client.messages.stream(...)`.

    Mirrors the small surface our adapter touches:
      - `async with client.messages.stream(...) as stream:`
      - `async for _event in stream:` (yields nothing in tests)
      - `await stream.get_final_message()`
    """

    def __init__(self, final_message: object, recorder: dict | None = None,
                 kwargs: dict | None = None):
        self._final = final_message
        if recorder is not None:
            recorder["stream_kwargs"] = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def get_final_message(self):
        return self._final


def _install_fake_anthropic(monkeypatch, *, final_message: object | None = None) -> dict:
    """Install a fake AsyncAnthropic that records both create() and stream() calls.

    Returns a dict recorder so individual tests can inspect which path
    (`create` vs `stream`) was taken and with what kwargs.
    """
    from unittest.mock import AsyncMock, MagicMock

    final = final_message if final_message is not None else _build_anthropic_response()
    recorder: dict = {"create_kwargs": None, "stream_kwargs": None}

    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=final)

    def _stream(**kwargs):
        recorder["stream_kwargs"] = kwargs
        return _FakeAnthropicStream(final, recorder=None, kwargs=None)

    fake_client.messages.stream = _stream

    # Wrap create so we also record its kwargs
    real_create = fake_client.messages.create
    async def _create(**kwargs):
        recorder["create_kwargs"] = kwargs
        return await real_create(**kwargs)
    fake_client.messages.create = _create

    monkeypatch.setattr(llm_clients, "AsyncAnthropic", MagicMock(return_value=fake_client))
    return recorder


def test_anthropic_adapter_sends_adaptive_thinking_at_max_effort(monkeypatch) -> None:
    """Spec with thinking_effort='max' should produce an adaptive thinking
    config + output_config.effort='max' on the underlying Anthropic call,
    and auto-bump max_tokens to the documented floor (64k for max).

    At max effort with max_tokens auto-bumped to 64000 the SDK's >10-min
    guard would trip on `messages.create()`, so the adapter must dispatch
    through `messages.stream()` instead."""
    reset_router_cache()
    rec = _install_fake_anthropic(monkeypatch)

    spec = ModelSpec(
        provider="anthropic", model="claude-sonnet-4-6", thinking_effort="max",
    )
    router = llm_clients._build_router(spec)

    async def _go():
        return await router.messages.create(
            max_tokens=8000, system="be brief",
            messages=[{"role": "user", "content": "hi"}],
        )

    resp = asyncio.run(_go())
    assert resp.content[0].text == "ok"

    # Thinking + 64k max_tokens MUST route through the streaming helper.
    assert rec["stream_kwargs"] is not None, "expected messages.stream() to be used"
    assert rec["create_kwargs"] is None, "messages.create() should be skipped"

    call = rec["stream_kwargs"]
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "max"}
    # Caller passed max_tokens=8000 but `max` effort needs ≥ 64k room.
    assert call["max_tokens"] == 64000
    assert call["system"] == "be brief"


def test_anthropic_adapter_omits_thinking_when_effort_off(monkeypatch) -> None:
    reset_router_cache()
    rec = _install_fake_anthropic(monkeypatch)

    spec = ModelSpec(
        provider="anthropic", model="claude-sonnet-4-6", thinking_effort="off",
    )
    router = llm_clients._build_router(spec)

    async def _go():
        return await router.messages.create(
            max_tokens=8000,
            messages=[{"role": "user", "content": "hi"}],
        )

    asyncio.run(_go())
    # No thinking, max_tokens at threshold (8000) → non-streaming create() path.
    assert rec["create_kwargs"] is not None
    assert rec["stream_kwargs"] is None
    call = rec["create_kwargs"]
    assert "thinking" not in call
    assert "output_config" not in call
    # max_tokens left untouched when thinking is disabled
    assert call["max_tokens"] == 8000


def test_anthropic_adapter_streams_when_max_tokens_above_threshold(monkeypatch) -> None:
    """Even without thinking, large max_tokens (> 8000) must go through
    `messages.stream()` to avoid the SDK's >10-min non-streaming guard."""
    reset_router_cache()
    rec = _install_fake_anthropic(monkeypatch)

    spec = ModelSpec(
        provider="anthropic", model="claude-sonnet-4-6", thinking_effort="off",
    )
    router = llm_clients._build_router(spec)

    async def _go():
        return await router.messages.create(
            max_tokens=32000,
            messages=[{"role": "user", "content": "hi"}],
        )

    asyncio.run(_go())
    assert rec["stream_kwargs"] is not None, "expected messages.stream() for large max_tokens"
    assert rec["create_kwargs"] is None
    assert rec["stream_kwargs"]["max_tokens"] == 32000


def test_anthropic_adapter_uses_create_below_threshold_without_thinking(monkeypatch) -> None:
    """Below the streaming threshold AND thinking off → simple create() path."""
    reset_router_cache()
    rec = _install_fake_anthropic(monkeypatch)

    spec = ModelSpec(
        provider="anthropic", model="claude-sonnet-4-6", thinking_effort="off",
    )
    router = llm_clients._build_router(spec)

    async def _go():
        return await router.messages.create(
            max_tokens=4000,
            messages=[{"role": "user", "content": "hi"}],
        )

    asyncio.run(_go())
    assert rec["create_kwargs"] is not None
    assert rec["stream_kwargs"] is None


def test_anthropic_adapter_streams_when_thinking_enabled_even_with_small_max_tokens(
    monkeypatch,
) -> None:
    """Thinking-enabled responses are commonly long-running; always stream."""
    reset_router_cache()
    rec = _install_fake_anthropic(monkeypatch)

    spec = ModelSpec(
        provider="anthropic", model="claude-sonnet-4-6", thinking_effort="low",
    )
    router = llm_clients._build_router(spec)

    async def _go():
        return await router.messages.create(
            max_tokens=2000,
            messages=[{"role": "user", "content": "hi"}],
        )

    asyncio.run(_go())
    assert rec["stream_kwargs"] is not None, "thinking-enabled call must stream"
    assert rec["create_kwargs"] is None
    # `low` effort floor is 8000, so max_tokens is bumped up from 2000.
    assert rec["stream_kwargs"]["max_tokens"] == 8000


def test_anthropic_adapter_skips_thinking_blocks_in_extracted_text(monkeypatch) -> None:
    """Adaptive thinking can return interleaved `thinking` blocks; the
    router must only surface visible `text` blocks to the caller."""
    from unittest.mock import MagicMock
    reset_router_cache()
    fake_final = MagicMock(
        content=[
            MagicMock(type="thinking", text="HIDDEN_COT_DO_NOT_LEAK"),
            MagicMock(type="text",     text="visible answer"),
        ],
        model="claude-sonnet-4-6",
        usage=MagicMock(input_tokens=1, output_tokens=1),
    )
    _install_fake_anthropic(monkeypatch, final_message=fake_final)

    spec = ModelSpec(
        provider="anthropic", model="claude-sonnet-4-6", thinking_effort="high",
    )
    router = llm_clients._build_router(spec)

    async def _go():
        return await router.messages.create(
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
        )

    resp = asyncio.run(_go())
    assert resp.content[0].text == "visible answer"
    assert "HIDDEN_COT_DO_NOT_LEAK" not in resp.content[0].text


def test_openai_adapter_sends_reasoning_effort_high_for_max(monkeypatch) -> None:
    """`max` effort → OpenAI `reasoning_effort='high'` (xhigh is restricted
    to gpt-5.1-codex-max; we collapse to the universally-supported max)."""
    from unittest.mock import AsyncMock, MagicMock
    reset_router_cache()
    import openai

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))],
        model="gpt-5.5",
        usage=MagicMock(
            prompt_tokens=12, completion_tokens=7,
            completion_tokens_details=MagicMock(reasoning_tokens=128),
        ),
    ))
    monkeypatch.setattr(openai, "AsyncOpenAI", MagicMock(return_value=fake_client))

    spec = ModelSpec(provider="openai", model="gpt-5.5", thinking_effort="max")
    router = llm_clients._build_router(spec)

    async def _go():
        return await router.messages.create(
            max_tokens=4000, temperature=0.9,
            messages=[{"role": "user", "content": "hi"}],
        )

    resp = asyncio.run(_go())
    assert resp.content[0].text == "ok"
    # Reasoning token usage is surfaced when present
    assert resp.usage.get("reasoning_tokens") == 128

    call = fake_client.chat.completions.create.await_args.kwargs
    assert call["reasoning_effort"] == "high"
    # max_completion_tokens floored to 32k for max effort
    assert call["max_completion_tokens"] == 32000
    # Reasoning models ignore non-default temperature — router suppresses it.
    assert "temperature" not in call


def test_openai_adapter_keeps_temperature_when_effort_off(monkeypatch) -> None:
    from unittest.mock import AsyncMock, MagicMock
    reset_router_cache()
    import openai

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))],
        model="gpt-5.5",
        usage=MagicMock(prompt_tokens=1, completion_tokens=1),
    ))
    monkeypatch.setattr(openai, "AsyncOpenAI", MagicMock(return_value=fake_client))

    spec = ModelSpec(provider="openai", model="gpt-4o", thinking_effort="off")
    router = llm_clients._build_router(spec)

    async def _go():
        return await router.messages.create(
            max_tokens=500, temperature=0.7,
            messages=[{"role": "user", "content": "hi"}],
        )

    asyncio.run(_go())
    call = fake_client.chat.completions.create.await_args.kwargs
    assert "reasoning_effort" not in call
    assert call["temperature"] == 0.7
    assert call["max_completion_tokens"] == 500


def test_gemini_adapter_sends_dynamic_thinking_budget_at_max(monkeypatch) -> None:
    """`max` effort on a Gemini model should translate to
    `ThinkingConfig(thinking_budget=-1)` (dynamic / model-decided)."""
    from unittest.mock import AsyncMock, MagicMock
    reset_router_cache()
    from google import genai
    from google.genai import types

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = AsyncMock(return_value=MagicMock(
        text="ok",
        usage_metadata=MagicMock(
            prompt_token_count=5, candidates_token_count=3, thoughts_token_count=64,
        ),
    ))
    monkeypatch.setattr(genai, "Client", MagicMock(return_value=fake_client))

    spec = ModelSpec(provider="google", model="gemini-2.5-pro", thinking_effort="max")
    router = llm_clients._build_router(spec)

    async def _go():
        return await router.messages.create(
            max_tokens=4000,
            system="be brief",
            messages=[{"role": "user", "content": "hi"}],
        )

    resp = asyncio.run(_go())
    assert resp.content[0].text == "ok"
    # Thinking-token usage is surfaced when present
    assert resp.usage.get("reasoning_tokens") == 64

    call = fake_client.aio.models.generate_content.await_args.kwargs
    assert call["model"] == "gemini-2.5-pro"
    cfg = call["config"]
    assert isinstance(cfg.thinking_config, types.ThinkingConfig)
    assert cfg.thinking_config.thinking_budget == -1
    assert cfg.system_instruction == "be brief"
    # max_output_tokens floored to 32k for max effort
    assert cfg.max_output_tokens == 32000


def test_gemini_adapter_omits_thinking_config_when_off(monkeypatch) -> None:
    from unittest.mock import AsyncMock, MagicMock
    reset_router_cache()
    from google import genai

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = AsyncMock(return_value=MagicMock(
        text="ok", usage_metadata=MagicMock(prompt_token_count=1, candidates_token_count=1),
    ))
    monkeypatch.setattr(genai, "Client", MagicMock(return_value=fake_client))

    spec = ModelSpec(provider="google", model="gemini-2.5-pro", thinking_effort="off")
    router = llm_clients._build_router(spec)

    async def _go():
        return await router.messages.create(
            max_tokens=512,
            messages=[{"role": "user", "content": "hi"}],
        )

    asyncio.run(_go())
    cfg = fake_client.aio.models.generate_content.await_args.kwargs["config"]
    assert cfg.thinking_config is None or getattr(
        cfg.thinking_config, "thinking_budget", None,
    ) is None
    # max_output_tokens preserved as caller passed it
    assert cfg.max_output_tokens == 512
