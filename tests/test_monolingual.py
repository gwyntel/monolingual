from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from conftest import config_mod, detector, plugin, translator


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

def test_detect_mixing_config_option() -> None:
    assert config_mod.MonolingualConfig({}).detect_mixing is True
    assert config_mod.MonolingualConfig({"detect_mixing": False}).detect_mixing is False
    assert config_mod.MonolingualConfig({"detect_mixing": "no"}).detect_mixing is False
    assert config_mod.MonolingualConfig({"detect_mixing": "yes"}).detect_mixing is True


def test_needs_translation_respects_detect_mixing_flag() -> None:
    # Single CJK char in English text: whole-text detection says "en",
    # so only the mixing check can trigger translation.
    response = "This is English with 一 Chinese character."
    should_on, _ = detector.needs_translation(response, "en", method="unicode", detect_mixing=True)
    should_off, _ = detector.needs_translation(response, "en", method="unicode", detect_mixing=False)
    assert should_on is True
    assert should_off is False


class _FakeLinguaLanguage:
    """Minimal stand-in for a lingua ``Language`` enum member."""

    def __init__(self, iso_code: str) -> None:
        self.iso_code_639_1 = SimpleNamespace(name=iso_code.upper())


class _FakeDetectionResult:
    """Stand-in for lingua 2.x ``DetectionResult``.

    The real class exposes ``start_index`` / ``end_index`` / ``word_count`` /
    ``language`` — character offsets into the analysed text — and has no
    ``.words`` attribute at all (accessing it raises AttributeError). The
    pre-fix detector read ``span.words``, the AttributeError was swallowed by
    the blanket ``except``, and the lingua mixing layer silently returned
    ``(False, 0.0)`` forever. Mirroring the missing attribute here means a
    regression to ``span.words`` makes the assertions below fail instead of
    quietly passing.
    """

    def __init__(self, start_index: int, end_index: int, iso_code: str) -> None:
        self.start_index = start_index
        self.end_index = end_index
        self.word_count = 1
        self.language = _FakeLinguaLanguage(iso_code)

    @property
    def words(self) -> Any:  # pragma: no cover - attribute intentionally absent
        raise AttributeError("'DetectionResult' object has no attribute 'words'")


def _install_fake_lingua(monkeypatch: pytest.MonkeyPatch, spans: list[_FakeDetectionResult]) -> None:
    """Point the detector's lingua accessor at a deterministic span list."""
    stub = SimpleNamespace(detect_multiple_languages_of=lambda text: list(spans))
    monkeypatch.setattr(detector, "_get_lingua_detector", lambda: stub)


def test_lingua_mixing_detection_uses_char_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    # Deterministic replacement for the real-package test: no native lingua
    # dependency, so this runs in every environment (CI included).
    english = "The quarterly report shows strong growth across all major product lines. "
    chinese = "本季度报告显示所有主要产品线均实现了强劲增长。"
    en_block, zh_block = english * 6, chinese * 2
    text = en_block + zh_block
    spans = [
        _FakeDetectionResult(0, len(en_block), "en"),
        _FakeDetectionResult(len(en_block), len(text), "zh"),
    ]
    _install_fake_lingua(monkeypatch, spans)

    # The mock must reproduce lingua's contract, not the buggy assumption.
    with pytest.raises(AttributeError):
        spans[1].words  # type: ignore[attr-defined]

    is_mixing, ratio = detector._detect_lingua_mixing(text, "en")
    assert is_mixing is True
    # Span characters are counted, not word counts: ratio is exactly the
    # minority span's share of the characters covered by all spans.
    assert ratio == pytest.approx(len(zh_block) / len(text), rel=0, abs=1e-12)

    # Exact 70/30 split: 30 of 100 covered characters are non-target.
    mixed = "x" * 70 + "中" * 30
    _install_fake_lingua(
        monkeypatch,
        [_FakeDetectionResult(0, 70, "en"), _FakeDetectionResult(70, 100, "zh")],
    )
    is_mixing, ratio = detector._detect_lingua_mixing(mixed, "en")
    assert is_mixing is True
    assert ratio == pytest.approx(0.30, rel=0, abs=1e-12)

    # A 2% minority span is below the 0.05 mixing floor.
    mostly_english = "x" * 98 + "中" * 2
    _install_fake_lingua(
        monkeypatch,
        [_FakeDetectionResult(0, 98, "en"), _FakeDetectionResult(98, 100, "zh")],
    )
    is_mixing, ratio = detector._detect_lingua_mixing(mostly_english, "en")
    assert is_mixing is False
    assert ratio == pytest.approx(0.02, rel=0, abs=1e-12)

    # No spans / no detector → no mixing, zero ratio (fail-open).
    _install_fake_lingua(monkeypatch, [])
    assert detector._detect_lingua_mixing(text, "en") == (False, 0.0)
    monkeypatch.setattr(detector, "_get_lingua_detector", lambda: None)
    assert detector._detect_lingua_mixing(text, "en") == (False, 0.0)


def test_lingua_mixing_detection_real_package_integration() -> None:
    # Integration coverage against the real native package. Skips when lingua
    # is absent; the deterministic test above is the always-on regression gate.
    pytest.importorskip("lingua")
    ling = detector._get_lingua_detector()
    if ling is None:  # pragma: no cover - lingua import races
        pytest.skip("lingua detector unavailable")
    assert ling is not None  # narrows the Optional for type checkers

    english = "The quarterly report shows strong growth across all major product lines. " * 8
    chinese = "本季度报告显示所有主要产品线均实现了强劲增长。" * 8
    text = english + chinese
    spans = ling.detect_multiple_languages_of(text)
    assert spans, "lingua returned no language spans"
    for span in spans:
        assert isinstance(span.start_index, int)
        assert isinstance(span.end_index, int)
        # lingua 2.x contract: char offsets, no word list.
        assert not hasattr(span, "words")

    langs = {detector._lang_from_lingua(span.language) for span in spans}
    if len(langs) < 2:
        pytest.skip("lingua did not return multiple language spans for mixed text")

    covered = sum(span.end_index - span.start_index for span in spans)
    non_target = sum(
        span.end_index - span.start_index
        for span in spans
        if detector._lang_from_lingua(span.language) != "en"
    )
    is_mixing, ratio = detector._detect_lingua_mixing(text, "en")
    assert is_mixing is True
    assert ratio == pytest.approx(non_target / covered, rel=0, abs=1e-12)


def test_detect_mixing_false_skips_mixed_text_with_two_cjk_chars() -> None:
    # Greptile P1 regression: with detect_mixing=False, an English response
    # containing exactly two CJK chars ("中文") must NOT be translated. The
    # unicode detector classifies the whole text as zh, but the target script
    # dominates, so this is mixing — not a wrong-language response.
    response = "This is an English response containing 中文 characters."
    should_off, detected = detector.needs_translation(
        response, "en", method="unicode", detect_mixing=False)
    assert should_off is False
    assert detected == "zh"
    should_on, _ = detector.needs_translation(
        response, "en", method="unicode", detect_mixing=True)
    assert should_on is True


def test_fully_wrong_language_still_translates_with_detect_mixing_false() -> None:
    # A genuinely Chinese response to an English user is *not* mixing —
    # it must still be translated even when detect_mixing is off.
    response = "这是一段中文回复，需要翻译成英文。"
    should_translate, detected = detector.needs_translation(
        response, "en", method="unicode", detect_mixing=False)
    assert should_translate is True
    assert detected == "zh"


def test_non_latin_target_translates_foreign_reply_carrying_latin_terms() -> None:
    # Greptile P1 regression: a Chinese reply to a *Russian* user must be
    # translated even with detect_mixing=False. The old heuristic counted only
    # ASCII letters, so a CJK-dominant reply whose product names push the ASCII
    # share under half was called "mixed" — skipping the translation entirely
    # even though not one Cyrillic character was present in the text.
    response = (
        "这是一段很长的中文说明文字，介绍了 OpenAI 和 ChatGPT 以及 Proxmox "
        "这几个工具的安装方法、配置要点、常见问题以及注意事项，请仔细阅读。"
    )
    ascii_letters = sum(1 for ch in response if ch.isascii() and ch.isalpha())
    assert ascii_letters / len(response) < 0.5, "fixture must be CJK-dominant"
    assert not any("\u0400" <= ch <= "\u04ff" for ch in response)

    should_translate, detected = detector.needs_translation(
        response, "ru", method="unicode", detect_mixing=False)
    assert should_translate is True
    assert detected == "zh"

    # Same shape for an Arabic target with an ASCII-heavy foreign reply.
    should_translate_ar, _ = detector.needs_translation(
        response, "ar", method="unicode", detect_mixing=False)
    assert should_translate_ar is True


def test_non_latin_target_still_honors_detect_mixing_for_mixed_text() -> None:
    # The original mixed-text behaviour must survive: a Russian reply carrying
    # enough Latin technical terms is mixing, so detect_mixing=False skips it
    # while detect_mixing=True translates it.
    response = "Это ответ на русском языке про Tailscale, Proxmox и OpenAI."
    should_off, _ = detector.needs_translation(
        response, "ru", method="unicode", detect_mixing=False)
    assert should_off is False

    should_on, _ = detector.needs_translation(
        response, "ru", method="unicode", detect_mixing=True)
    assert should_on is True


def test_plugin_hooks_honor_detect_mixing_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config → register() → needs_translation wiring for detect_mixing.

    Covers the plumbing end to end: a ``detect_mixing: false`` config entry
    must reach ``needs_translation`` through the hook, so an English reply
    carrying two CJK characters is passed through untouched, while the default
    (``true``) still translates it.
    """
    mixed = "This is an English response containing 中文 characters."

    def use_config(mixing: bool) -> None:
        def from_hermes_config(cls: type[Any]) -> Any:
            return cls({"detector": "unicode", "detect_mixing": mixing})

        monkeypatch.setattr(
            config_mod.MonolingualConfig,
            "from_hermes_config",
            classmethod(from_hermes_config),
        )
        config_mod.reload_config()

    use_config(False)
    ctx = DummyCtx(SimpleNamespace(text="Translated"))
    plugin.register(ctx)
    ctx.hooks["pre_llm_call"](
        session_id="mix",
        user_message="This is a long enough English message.",
        conversation_history=[],
    )
    assert plugin._get_profile("mix") == "en"

    assert ctx.hooks["transform_llm_output"](session_id="mix", response_text=mixed) is None
    assert ctx.llm_calls == []

    use_config(True)
    output = ctx.hooks["transform_llm_output"](session_id="mix", response_text=mixed)
    assert output == "[translated from Chinese by GwynTel/monolingual]\n\nTranslated"
    assert len(ctx.llm_calls) == 1
