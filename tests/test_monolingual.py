from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "monolingual_testpkg"


def load_plugin_package() -> ModuleType:
    """Load the plugin root as a package so relative imports work in tests."""
    if PACKAGE_NAME in sys.modules:
        return sys.modules[PACKAGE_NAME]
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module


plugin = load_plugin_package()
config_mod = sys.modules[f"{PACKAGE_NAME}.config"]
detector = sys.modules[f"{PACKAGE_NAME}.detector"]
translator = sys.modules[f"{PACKAGE_NAME}.translator"]


@pytest.fixture(autouse=True)
def reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    detector.detect_language.cache_clear()
    detector._detect_lingua.cache_clear()
    with plugin._profiles_lock:
        plugin._session_profiles.clear()
    config_mod._cached_config = None
    config_mod._cached_config_loaded_at = 0.0
    monkeypatch.setattr(config_mod.MonolingualConfig, "from_hermes_config", classmethod(lambda cls: cls()))


def test_basic_unicode_detection_zh_en_and_mixed() -> None:
    assert detector.detect_language("这是一段中文内容，用来测试语言检测。", method="unicode") == "zh"
    assert detector.detect_language("This is a normal English sentence for detection.", method="unicode") == "en"
    assert detector.detect_language("This answer contains 中文 mixed in.", method="unicode") == "zh"


def test_detect_user_language_uses_recent_user_messages_and_override() -> None:
    history = [
        {"role": "assistant", "content": "ignored"},
        {"role": "user", "content": "This is a long enough English user message."},
        {"role": "user", "content": "这是一段足够长的中文用户消息。"},
    ]
    assert detector.detect_user_language(history, "", min_sample_length=5, method="unicode") == "en"
    assert detector.detect_user_language(history, "", target_override="ja") == "ja"


@pytest.mark.parametrize(
    ("response", "target", "expected", "source"),
    [
        ("这是一段中文回复，需要翻译。", "en", True, "zh"),
        ("This is already English.", "en", False, "en"),
        ("This is English with 一个 Chinese character.", "en", True, "zh"),
        ("这是一段中文，but contains several Latin words.", "zh", True, "zh"),
        ("```python\nprint('中文')\n```", "en", False, None),
        ("Hola mundo", "en", False, "en"),
    ],
)
def test_needs_translation_branches(response: str, target: str, expected: bool, source: str | None) -> None:
    should_translate, detected_source = detector.needs_translation(
        response,
        target,
        method="unicode",
        min_mix_ratio=0.15,
    )
    assert should_translate is expected
    assert detected_source == source


def test_unicode_fallback_when_lingua_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector, "_detect_lingua", lambda text: None)
    assert detector.detect_language("中文 fallback", method="lingua") == "zh"


def test_cjk_char_scan_counts_supplementary_plane_with_uppercase_escape() -> None:
    ext_b_char = "\U00020000"
    assert detector._count_non_target_script_chars(ext_b_char, "en") == 1
    assert detector._count_non_target_script_chars("中文", "en") == 2
    assert detector._count_non_target_script_chars("API HTTP URL", "zh") == 0
    assert detector._count_non_target_script_chars("three Latin words", "zh") == 3


def test_code_block_stripping_removes_fenced_and_inline_code() -> None:
    text = "Before\n```python\nprint('中文')\n```\nAfter `中文 inline` done"
    stripped = detector._strip_code_blocks(text)
    assert "print" not in stripped
    assert "inline" not in stripped
    assert stripped == "Before\n\nAfter  done"


def test_config_defaults_and_coercion() -> None:
    cfg = config_mod.MonolingualConfig(
        {
            "disabled": "yes",
            "sample_window": "0",
            "min_sample_length": "bad",
            "min_mix_ratio": "0.25",
            "detector": "bad",
            "translation_timeout": "2",
            "max_translation_chars": "123",
            "target_language": " EN ",
        }
    )
    assert cfg.disabled is True
    assert cfg.sample_window == 1
    assert cfg.min_sample_length == config_mod.DEFAULT_MIN_SAMPLE_LENGTH
    assert cfg.min_mix_ratio == 0.25
    assert cfg.detector == config_mod.DEFAULT_DETECTOR
    assert cfg.translation_timeout == 2
    assert cfg.max_translation_chars == 123
    assert cfg.target_language == "en"


def test_config_loading_defaults_when_hermes_config_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    original = config_mod.MonolingualConfig.from_hermes_config.__func__

    def raise_import(cls: type[Any]) -> Any:
        raise RuntimeError("missing hermes config")

    monkeypatch.setattr(config_mod.MonolingualConfig, "from_hermes_config", classmethod(raise_import))
    assert isinstance(config_mod.get_config(), config_mod.MonolingualConfig)
    monkeypatch.setattr(config_mod.MonolingualConfig, "from_hermes_config", classmethod(original))
    defaults = config_mod.MonolingualConfig.from_hermes_config()
    assert defaults.sample_window == config_mod.DEFAULT_SAMPLE_WINDOW


class DummyCtx:
    def __init__(self, llm_result: Any = None, raise_llm: bool = False) -> None:
        self.hooks: dict[str, Any] = {}
        self.llm_calls: list[dict[str, Any]] = []
        self._llm_result = llm_result or SimpleNamespace(text="Translated text")
        self._raise_llm = raise_llm
        self.llm = SimpleNamespace(complete=self.complete)

    def register_hook(self, name: str, callback: Any) -> None:
        self.hooks[name] = callback

    def complete(self, **kwargs: Any) -> Any:
        self.llm_calls.append(kwargs)
        if self._raise_llm:
            raise RuntimeError("llm failed")
        return self._llm_result


def test_translation_uses_ctx_llm_and_preserves_timeout() -> None:
    cfg = config_mod.MonolingualConfig({"translation_timeout": 7})
    ctx = DummyCtx(SimpleNamespace(text="你好"))
    assert translator.translate(ctx, "hello", "en", "zh", cfg) == "你好"
    assert ctx.llm_calls[0]["purpose"] == "monolingual_translation"
    assert ctx.llm_calls[0]["timeout"] == 7
    system_prompt = ctx.llm_calls[0]["messages"][0]["content"]
    assert "Preserve the original structure exactly" in system_prompt


def test_translation_fail_open_when_llm_and_auxiliary_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(translator, "_load_auxiliary_call_llm", lambda: None)
    assert translator.translate(DummyCtx(raise_llm=True), "original", "zh", "en") == "original"


def test_translation_skips_oversized_text() -> None:
    cfg = config_mod.MonolingualConfig({"max_translation_chars": 5})
    ctx = DummyCtx(SimpleNamespace(text="translated"))
    assert translator.translate(ctx, "too long", "zh", "en", cfg) == "too long"
    assert ctx.llm_calls == []


def test_plugin_hooks_translate_and_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = DummyCtx(SimpleNamespace(text="Translated"))
    plugin.register(ctx)

    ctx.hooks["pre_llm_call"](
        session_id="s1",
        user_message="This is a long enough English message.",
        conversation_history=[],
    )
    output = ctx.hooks["transform_llm_output"](
        session_id="s1",
        response_text="这是一段中文回复。",
    )
    assert output == "[translated from Chinese by GwynTel/monolingual]\n\nTranslated"

    monkeypatch.setattr(plugin, "needs_translation", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    assert ctx.hooks["transform_llm_output"](session_id="s1", response_text="中文") is None

    ctx.hooks["on_session_end"](session_id="s1")
    assert plugin._get_profile("s1") is None


def test_plugin_disabled_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_mod.MonolingualConfig, "from_hermes_config", classmethod(lambda cls: cls({"disabled": True})))
    cfg = config_mod.reload_config()
    assert cfg.disabled is True
    ctx = DummyCtx(SimpleNamespace(text="Translated"))
    plugin.register(ctx)
    assert ctx.hooks["pre_llm_call"](session_id="s1", user_message="English message long enough") is None
    assert plugin._get_profile("s1") is None