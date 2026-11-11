"""Regression tests for script coverage and target-language handling.

Two opposite failure modes are pinned here, because fixing one has repeatedly
broken the other:

* a reply written entirely in a foreign script but carrying Latin product names
  was called "mixed" (ASCII share < half) and skipped when ``detect_mixing`` was
  off, so the user never got the translation;
* a long reply in the user's *own* language was rewritten because the active
  detector cannot report that language at all — the unicode heuristic names one
  code per script (all Cyrillic → ``ru``, all Latin → ``en``) and lingua only
  knows its candidate list.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from conftest import config_mod, detector, plugin


# ---------------------------------------------------------------------------
# Fixtures: foreign-script replies carrying Latin product names
# ---------------------------------------------------------------------------

# Long enough that no length-based guard can explain the outcome. Latin product
# names push the ASCII share under half, which is exactly what the removed
# ASCII-minority heuristic mistook for "mixing".
CHINESE_WITH_PRODUCTS = (
    "这是一段很长的中文说明文字，介绍了 OpenAI 和 ChatGPT 以及 Proxmox "
    "这几个工具的安装方法、配置要点、常见问题以及注意事项，请仔细阅读。"
)
ARABIC_WITH_PRODUCTS = (
    "هذا نص عربي طويل يشرح تثبيت OpenAI وChatGPT وProxmox "
    "مع التفاصيل والملاحظات المهمة للمستخدم."
)
UKRAINIAN_WITH_PRODUCTS = (
    "Це довга відповідь українською про встановлення OpenAI, ChatGPT та "
    "Proxmox, з подробицями та поясненнями для користувача."
)
HEBREW_WITH_PRODUCTS = (
    "זוהי תשובה ארוכה בעברית על התקנת OpenAI ו-ChatGPT ו-Proxmox, "
    "עם פרטים והסברים נוספים למשתמש."
)
GREEK_WITH_PRODUCTS = (
    "Αυτή είναι μια μακροσκελής απάντηση στα ελληνικά για την εγκατάσταση "
    "των OpenAI, ChatGPT και Proxmox, με λεπτομέρειες."
)

UKRAINIAN_PLAIN = (
    "Це звичайна відповідь українською мовою без жодних технічних назв."
)
HEBREW_PLAIN = "זוהי תשובה רגילה בעברית ללא שמות טכניים כלשהם."
GREEK_PLAIN = "Αυτή είναι μια κανονική απάντηση στα ελληνικά χωρίς τεχνικά ονόματα."

TAMIL_PLAIN = "இது OpenAI மற்றும் ChatGPT மற்றும் Proxmox பற்றிய நீண்ட தமிழ் பதில் ஆகும்."

# ---------------------------------------------------------------------------
# Fixtures: long replies already in the user's language
# ---------------------------------------------------------------------------

# Every one of these is longer than SAME_SCRIPT_MIN_CHARS by a wide margin, so
# the "long response" branch is reached and only the target-language gate can
# keep them intact.
UKRAINIAN_LONG = (
    "Це звичайна відповідь українською мовою без технічних назв та без посилань на інструменти. "
    "У ній немає жодного слова іншою мовою, тому перекладати її не потрібно. "
) * 3
RUSSIAN_LONG = (
    "Это обычный ответ на русском языке без технических названий и без ссылок на инструменты. "
    "В нём нет ни одного слова на другом языке, поэтому переводить его не нужно. "
) * 3
HEBREW_LONG = (
    "זוהי תשובה רגילה בעברית ללא שמות טכניים וללא קישורים לכלים חיצוניים. "
    "אין בה אף מילה בשפה אחרת ולכן אין צורך לתרגם אותה. "
) * 4
GREEK_LONG = (
    "Αυτή είναι μια κανονική απάντηση στα ελληνικά χωρίς τεχνικά ονόματα και χωρίς συνδέσμους. "
    "Δεν περιέχει καμία λέξη σε άλλη γλώσσα, επομένως δεν χρειάζεται μετάφραση. "
) * 3
# Finnish is Latin-script but outside both detectors' vocabulary: lingua has no
# Finnish candidate and the unicode heuristic calls every Latin script "en".
FINNISH_LONG = (
    "Tämä on tavallinen suomenkielinen vastaus ilman teknisiä nimiä ja ilman linkkejä työkaluihin. "
    "Siinä ei ole yhtään sanaa toisella kielellä, joten sitä ei tarvitse kääntää. "
) * 3


@pytest.fixture(autouse=True)
def reset_detection_caches() -> None:
    detector.detect_language.cache_clear()
    detector._detect_lingua.cache_clear()
    with plugin._profiles_lock:
        plugin._session_profiles.clear()
    config_mod._cached_config = None
    config_mod._cached_config_loaded_at = 0.0


def _is_cyrillic(text: str) -> bool:
    return any("\u0400" <= ch <= "\u04ff" for ch in text)


def _is_hebrew(text: str) -> bool:
    return any("\u0590" <= ch <= "\u05ff" for ch in text)


def _is_greek(text: str) -> bool:
    return any("\u0370" <= ch <= "\u03ff" for ch in text)


def test_long_fixtures_clear_the_length_threshold() -> None:
    # Guard the fixtures: only text well past SAME_SCRIPT_MIN_CHARS exercises the
    # length branch that used to rewrite the user's own language. 135 chars of
    # slack keeps them meaningful whatever the exact threshold is.
    for reply in (UKRAINIAN_LONG, RUSSIAN_LONG, HEBREW_LONG, GREEK_LONG, FINNISH_LONG):
        assert len(reply) > detector.SAME_SCRIPT_MIN_CHARS + 135


def test_fixtures_are_foreign_script_dominant_with_ascii_minority() -> None:
    """Guard the fixtures: they must reproduce the old heuristic's failure."""
    for text, script_check in (
        (CHINESE_WITH_PRODUCTS, _is_cyrillic),
        (ARABIC_WITH_PRODUCTS, _is_hebrew),
    ):
        ascii_letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
        assert ascii_letters / len(text) < 0.5
        assert script_check(text) is False


# ---------------------------------------------------------------------------
# Foreign-script replies must translate for every supported target script
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", ["uk", "bg", "sr", "mk"])
def test_cyrillic_targets_translate_foreign_script_reply(target: str) -> None:
    # Regression: uk/bg/sr/mk had no entry in _TARGET_SCRIPT_RANGES, so the
    # ASCII-minority fallback called this Chinese reply "mixed" and
    # detect_mixing=False skipped a translation the user needed.
    assert not _is_cyrillic(CHINESE_WITH_PRODUCTS)
    assert detector._is_mixed_text(CHINESE_WITH_PRODUCTS, target) is False

    should_off, source = detector.needs_translation(
        CHINESE_WITH_PRODUCTS, target, method="unicode", detect_mixing=False)
    assert should_off is True
    assert source == "zh"

    should_on, _ = detector.needs_translation(
        CHINESE_WITH_PRODUCTS, target, method="unicode", detect_mixing=True)
    assert should_on is True


@pytest.mark.parametrize(("target", "reply", "source"), [
    ("he", CHINESE_WITH_PRODUCTS, "zh"),
    ("he", ARABIC_WITH_PRODUCTS, "ar"),
    ("el", CHINESE_WITH_PRODUCTS, "zh"),
    ("el", ARABIC_WITH_PRODUCTS, "ar"),
])
def test_hebrew_and_greek_targets_translate_foreign_script_reply(
        target: str, reply: str, source: str) -> None:
    # Same regression for Hebrew and Greek targets: a Chinese or Arabic reply
    # contains none of the target script, so it is wrong-language text, not
    # mixing.
    target_check = _is_hebrew if target == "he" else _is_greek
    assert target_check(reply) is False
    assert detector._is_mixed_text(reply, target) is False
    should_translate, detected = detector.needs_translation(
        reply, target, method="unicode", detect_mixing=False)
    assert should_translate is True
    assert detected == source


def test_foreign_script_reply_without_latin_words_translates() -> None:
    # The old non-Latin path gated on "3+ Latin words" before anything else, so
    # a pure foreign-script reply with no Latin at all was skipped unless it was
    # over 200 chars. Script evidence alone must be enough.
    reply = "这是一段完全没有拉丁字母的中文回复，内容很长但仍然不需要任何英文单词。"
    assert detector._count_non_target_script_chars(reply, "uk") == 0
    assert detector._is_mixed_text(reply, "uk") is False
    should_translate, source = detector.needs_translation(
        reply, "uk", method="unicode", detect_mixing=False)
    assert should_translate is True
    assert source == "zh"


# ---------------------------------------------------------------------------
# Target-majority mixing keeps its documented behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("target", "reply"),
    [("uk", UKRAINIAN_WITH_PRODUCTS), ("he", HEBREW_WITH_PRODUCTS), ("el", GREEK_WITH_PRODUCTS)],
)
def test_target_majority_mixing_still_honors_detect_mixing(target: str, reply: str) -> None:
    # A reply already in the user's language that carries Latin product names
    # is genuine mixing: skipped when detect_mixing is off, rewritten when on.
    assert detector._is_mixed_text(reply, target) is True

    should_off, _ = detector.needs_translation(
        reply, target, method="unicode", detect_mixing=False)
    assert should_off is False

    should_on, _ = detector.needs_translation(
        reply, target, method="unicode", detect_mixing=True)
    assert should_on is True


@pytest.mark.parametrize(
    ("target", "reply"),
    [("uk", UKRAINIAN_PLAIN), ("he", HEBREW_PLAIN), ("el", GREEK_PLAIN)],
)
def test_plain_target_language_reply_is_left_alone(target: str, reply: str) -> None:
    # No foreign content at all: never rewritten, whatever detect_mixing says.
    # (uk stays "ru" for the unicode heuristic — all Cyrillic maps to ru — but
    # the absence of Latin words keeps it from being classed as mixing.)
    for detect_mixing in (False, True):
        should_translate, _ = detector.needs_translation(
            reply, target, method="unicode", detect_mixing=detect_mixing)
        assert should_translate is False


# ---------------------------------------------------------------------------
# Long replies already in the target language must never be rewritten
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["unicode", "lingua"])
@pytest.mark.parametrize(
    ("target", "reply"),
    [("uk", UKRAINIAN_LONG), ("he", HEBREW_LONG), ("el", GREEK_LONG)],
)
def test_long_target_language_reply_is_not_rewritten(
        target: str, reply: str, method: str) -> None:
    # The reported blocker: with target "uk" and the unicode method, a 390-char
    # Ukrainian reply is detected as "ru" — the method cannot emit "uk" at all —
    # and the length branch rewrote the user's own language.
    for detect_mixing in (False, True):
        should_translate, _ = detector.needs_translation(
            reply, target, method=method, detect_mixing=detect_mixing)
        assert should_translate is False


@pytest.mark.parametrize("method", ["unicode", "lingua"])
def test_long_latin_target_reply_is_not_rewritten(method: str) -> None:
    # Finnish is a Latin-script target outside both detectors' vocabulary: a
    # long Finnish reply was detected as "it"/"en" and rewritten.
    assert "fi" in detector.LATIN_TARGET_LANGUAGES
    assert "fi" not in detector._detectable_languages(method)

    for detect_mixing in (False, True):
        should_translate, _ = detector.needs_translation(
            FINNISH_LONG, "fi", method=method, detect_mixing=detect_mixing)
        assert should_translate is False

    # A genuinely foreign reply to a Finnish user still translates.
    should_translate, detected = detector.needs_translation(
        CHINESE_WITH_PRODUCTS, "fi", method=method, detect_mixing=False)
    assert should_translate is True
    assert detected == "zh"


def test_missing_lingua_judges_same_script_by_unicode_vocabulary(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: with lingua unavailable, needs_translation(method="lingua")
    # falls back to the unicode heuristic, which reports every Cyrillic text as
    # "ru". The same-script length gate looked at the *requested* method's
    # vocabulary (uk is a lingua candidate), so a long valid Ukrainian reply was
    # rewritten into itself. The gate must use the detector that actually ran.
    monkeypatch.setattr(detector, "_get_lingua_detector", lambda: None)
    detector.detect_language.cache_clear()
    detector._detect_lingua.cache_clear()
    try:
        assert detector._effective_method("lingua") == "unicode"
        assert detector._effective_method("unicode") == "unicode"
        assert "uk" not in detector._detectable_languages(
            detector._effective_method("lingua"))
        # The trap really is armed: the "lingua" verdict is the script guess.
        assert detector.detect_language(UKRAINIAN_LONG, method="lingua") == "ru"

        for detect_mixing in (False, True):
            assert detector.needs_translation(
                UKRAINIAN_LONG, "uk", method="lingua",
                detect_mixing=detect_mixing) == (False, "ru")

        # Foreign-script replies still translate through the same fallback.
        for detect_mixing in (False, True):
            should_translate, source = detector.needs_translation(
                CHINESE_WITH_PRODUCTS, "uk", method="lingua",
                detect_mixing=detect_mixing)
            assert should_translate is True
            assert source == "zh"
    finally:
        detector.detect_language.cache_clear()
        detector._detect_lingua.cache_clear()


@pytest.mark.parametrize(("reply", "target", "source"), [
    (RUSSIAN_LONG, "uk", "ru"),
    (UKRAINIAN_LONG, "ru", "uk"),
])
def test_lingua_translates_between_cyrillic_languages(
        reply: str, target: str, source: str) -> None:
    # uk/bg/sr are lingua candidates now, so a real same-script mismatch is
    # distinguishable from the user's own language and must translate.
    pytest.importorskip("lingua")
    assert detector._get_lingua_detector() is not None
    assert detector.detect_language(reply, method="lingua") == source

    for detect_mixing in (False, True):
        assert detector.needs_translation(
            reply, target, method="lingua", detect_mixing=detect_mixing) == (True, source)


@pytest.mark.parametrize(("reply", "target"), [
    (RUSSIAN_LONG, "uk"),     # uk is not in the unicode heuristic's vocabulary
    (RUSSIAN_LONG, "be"),
    (RUSSIAN_LONG, "kk"),
    (RUSSIAN_LONG, "mk"),
    (UKRAINIAN_LONG, "ru"),   # ...and ru cannot be told apart from uk
])
def test_shared_script_ambiguity_falls_through(reply: str, target: str) -> None:
    # The unicode heuristic reports one code per script. A Cyrillic reply whose
    # language it cannot name must fall through rather than be rewritten: for
    # uk/be/kk/mk the detector's "ru" verdict is indistinguishable from the
    # target, so there is no evidence of a wrong-language reply at all.
    for detect_mixing in (False, True):
        should_translate, _ = detector.needs_translation(
            reply, target, method="unicode", detect_mixing=detect_mixing)
        assert should_translate is False


def test_unicode_cannot_disambiguate_ru_from_uk() -> None:
    # Documents the residual limitation instead of pretending it is fixed: the
    # unicode fallback maps both languages to "ru", so a Russian reply to a
    # Ukrainian user is only translated by the lingua detector (see
    # test_lingua_translates_between_cyrillic_languages). The lingua assertion
    # therefore requires the real package: without it the same call runs the
    # unicode fallback and correctly reports no evidence.
    assert detector.detect_language(UKRAINIAN_LONG, method="unicode") == "ru"
    assert detector.detect_language(RUSSIAN_LONG, method="unicode") == "ru"
    assert detector.needs_translation(
        RUSSIAN_LONG, "uk", method="unicode", detect_mixing=False) == (False, "ru")

    pytest.importorskip("lingua")
    assert detector.detect_language(RUSSIAN_LONG, method="lingua") == "ru"
    assert detector.needs_translation(
        RUSSIAN_LONG, "uk", method="lingua", detect_mixing=False) == (True, "ru")


# ---------------------------------------------------------------------------
# Targets whose script the plugin cannot name stay conservative
# ---------------------------------------------------------------------------

def test_unknown_target_script_never_reports_mixing() -> None:
    # "am" (Amharic/Ethiopic) has no script entry. The old fallback reported
    # "mixed" purely from an ASCII minority and skipped the translation.
    assert detector._is_mixed_text(CHINESE_WITH_PRODUCTS, "am") is False

    should_translate, source = detector.needs_translation(
        CHINESE_WITH_PRODUCTS, "am", method="unicode", detect_mixing=False)
    assert should_translate is True
    assert source == "zh"


def test_unknown_target_script_does_not_rewrite_short_latin_fragments() -> None:
    # Conservative in both directions: unknown target + short Latin fragment is
    # not enough evidence to call a rewrite, but a real foreign sentence is.
    for fragment in ("OK", "Hello world"):
        should_translate, _ = detector.needs_translation(
            fragment, "am", method="unicode", detect_mixing=False)
        assert should_translate is False

    should_translate, _ = detector.needs_translation(
        "The deployment finished successfully.", "am", method="unicode", detect_mixing=False)
    assert should_translate is True


# ---------------------------------------------------------------------------
# Script table plumbing
# ---------------------------------------------------------------------------

def test_target_script_table_covers_all_non_latin_targets_in_use() -> None:
    for code in ("uk", "bg", "sr", "mk", "be", "kk", "mn", "he", "el"):
        assert code in detector._TARGET_SCRIPT_RANGES, code
    # Latin targets stay out of the table on purpose.
    assert detector._TARGET_SCRIPT_RANGES.keys().isdisjoint(detector.LATIN_TARGET_LANGUAGES)


def _lingua_non_latin_codes() -> set[str]:
    """Non-Latin codes lingua knows, from lingua's own script classification."""
    lingua = pytest.importorskip("lingua")
    latin = set(lingua.Language.all_with_latin_script())
    return {
        language.iso_code_639_1.name.lower()
        for language in lingua.Language.all()
        if language not in latin and language.iso_code_639_1 is not None
    }


def test_every_non_latin_lingua_language_has_a_target_script() -> None:
    # The original table covered only seven scripts; every other non-Latin
    # language lingua can detect fell back to the ASCII heuristic that this PR
    # removed. Derive the set from lingua instead of hardcoding it, so a lingua
    # upgrade cannot silently widen the gap.
    non_latin = _lingua_non_latin_codes()
    assert len(non_latin) >= 26
    missing = non_latin - set(detector._TARGET_SCRIPTS)
    assert missing == set()


def test_newly_supported_non_latin_languages_are_lingua_candidates() -> None:
    # uk/bg/sr/he/el were added to the scoped candidate list so same-script
    # mismatches are real verdicts rather than coverage artifacts.
    non_latin = _lingua_non_latin_codes()
    for code in ("uk", "bg", "sr", "he", "el"):
        assert code in non_latin, code
        assert code in detector.LINGUA_DETECTABLE_LANGUAGES, code

    samples = {
        "uk": UKRAINIAN_PLAIN,
        "bg": "Това е обикновен отговор на български език без технически имена.",
        "sr": "Ово је обичан одговор на српском језику без техничких назива.",
        "he": HEBREW_PLAIN,
        "el": GREEK_PLAIN,
    }
    assert detector._get_lingua_detector() is not None
    for code, sample in samples.items():
        assert detector.detect_language(sample, method="lingua") == code, code


def test_detectable_code_sets_are_method_specific() -> None:
    assert detector._detectable_languages("unicode") == detector.UNICODE_DETECTABLE_LANGUAGES
    assert detector._detectable_languages("lingua") == detector.LINGUA_DETECTABLE_LANGUAGES
    # Unknown method names fall back to the lingua path in detect_language.
    assert detector._detectable_languages("bogus") == detector.LINGUA_DETECTABLE_LANGUAGES

    assert "ru" in detector.UNICODE_DETECTABLE_LANGUAGES
    assert "uk" not in detector.UNICODE_DETECTABLE_LANGUAGES
    assert "fi" not in detector.UNICODE_DETECTABLE_LANGUAGES
    assert {"uk", "bg", "sr", "he", "el"} <= detector.LINGUA_DETECTABLE_LANGUAGES
    assert "fi" not in detector.LINGUA_DETECTABLE_LANGUAGES


UNICODE_SAMPLES = {
    "en": "This is plain English text.",
    "zh": "这是一段中文文本。",
    "ja": "これは日本語のテキストです。",
    "ko": "한국어 텍스트입니다.",
    "ru": "Это русский текст.",
    "ar": "هذا نص عربي.",
    "hi": "यह हिंदी पाठ है।",
    "th": "นี่คือข้อความภาษาไทย",
    "he": "זהו טקסט בעברית.",
    "el": "Αυτό είναι ελληνικό κείμενο.",
    "bn": "এটি বাংলা পাঠ।",
    "pa": "ਇਹ ਪੰਜਾਬੀ ਟੈਕਸਟ ਹੈ।",
    "gu": "આ ગુજરાતી ટેક્સ્ટ છે.",
    "ta": "இது தமிழ் உரை.",
    "te": "ఇది తెలుగు వచనం.",
    "ka": "ეს არის ქართული ტექსტი.",
    "hy": "Սա հայերեն տեքստ է։",
}


@pytest.mark.parametrize(("code", "sample"), sorted(UNICODE_SAMPLES.items()))
def test_every_unicode_detectable_code_is_actually_emitted(code: str, sample: str) -> None:
    # The declared set must match reality: a code listed here but never returned
    # would let the length gate trust a verdict the method cannot produce.
    assert code in detector.UNICODE_DETECTABLE_LANGUAGES
    assert detector.detect_language(sample, method="unicode") == code


def test_unicode_detectable_set_is_fully_covered_by_samples() -> None:
    assert set(UNICODE_SAMPLES) == set(detector.UNICODE_DETECTABLE_LANGUAGES)


def test_count_foreign_script_chars_ignores_only_the_target_script() -> None:
    assert detector._count_foreign_script_chars(CHINESE_WITH_PRODUCTS, "uk") > 0
    assert detector._count_foreign_script_chars(CHINESE_WITH_PRODUCTS, "zh") == 0
    assert detector._count_foreign_script_chars(ARABIC_WITH_PRODUCTS, "ar") == 0
    assert detector._count_foreign_script_chars(ARABIC_WITH_PRODUCTS, "he") > 0
    assert detector._count_foreign_script_chars(GREEK_PLAIN, "el") == 0
    assert detector._count_foreign_script_chars(GREEK_PLAIN, "ru") > 0
    assert detector._count_foreign_script_chars(HEBREW_PLAIN, "he") == 0
    assert detector._count_foreign_script_chars(HEBREW_PLAIN, "en") > 0


def test_named_dominance_threshold_marks_half_as_mixed() -> None:
    half_greek = "ααααα" + "abcde"          # 5 / 10 target-script chars
    minority_greek = "αααα" + "abcdef"      # 4 / 10 target-script chars
    assert detector._is_mixed_text(half_greek, "el") is True
    assert detector._is_mixed_text(minority_greek, "el") is False


# ---------------------------------------------------------------------------
# Scripts newly recognised as foreign for Latin targets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("reply", "source"),
    [(HEBREW_WITH_PRODUCTS, "he"), (TAMIL_PLAIN, "ta"), (GREEK_PLAIN, "el")],
)
def test_hebrew_greek_and_tamil_replies_translate_for_english_target(
        reply: str, source: str) -> None:
    # Hebrew, Greek and Tamil were missing from both the unicode counters and
    # the foreign-script ranges, so a fully foreign reply to an English user was
    # detected as "en" and skipped when detect_mixing was off.
    assert detector.detect_language(reply, method="unicode") == source
    should_translate, detected = detector.needs_translation(
        reply, "en", method="unicode", detect_mixing=False)
    assert should_translate is True
    assert detected == source


def test_greek_symbols_in_english_prose_are_not_treated_as_greek_text() -> None:
    # α/β/θ are ordinary notation: neither the detector nor the foreign-script
    # counter may turn an English technical reply into a Greek one.
    prose = "The angle theta is small; we write α and β for the two coefficients."
    assert detector.detect_language(prose, method="unicode") == "en"
    assert detector._count_non_target_script_chars(prose, "en") == 0
    for detect_mixing in (False, True):
        should_translate, _ = detector.needs_translation(
            prose, "en", method="unicode", detect_mixing=detect_mixing)
        assert should_translate is False


def test_greek_heavy_text_still_counts_as_greek() -> None:
    # The ratio gate must not hide real Greek text in mixed technical replies.
    reply = "Η συνάρτηση χρησιμοποιεί τις μεταβλητές α και β για την είσοδο."
    assert detector.detect_language(reply, method="unicode") == "el"
    should_translate, source = detector.needs_translation(
        reply, "en", method="unicode", detect_mixing=False)
    assert should_translate is True
    assert source == "el"


# ---------------------------------------------------------------------------
# Default behaviour for pre-existing targets is unchanged
# ---------------------------------------------------------------------------

def test_default_behaviour_for_existing_targets_is_unchanged() -> None:
    cjk_char = "This is English with 一 Chinese character."
    assert detector.needs_translation(cjk_char, "en", method="unicode") == (True, "en")
    assert detector.needs_translation(
        cjk_char, "en", method="unicode", detect_mixing=False) == (False, "en")

    assert detector.needs_translation(
        "This is already English.", "en", method="unicode") == (False, "en")
    assert detector.needs_translation(
        "Hola mundo", "en", method="unicode") == (False, "en")

    chinese_reply = "这是一段中文回复，需要翻译成英文。"
    assert detector.needs_translation(
        chinese_reply, "en", method="unicode", detect_mixing=False) == (True, "zh")

    mixed_zh = "这是一段中文，but contains several Latin words."
    assert detector.needs_translation(
        mixed_zh, "zh", method="unicode", detect_mixing=False) == (False, "zh")
    assert detector.needs_translation(
        mixed_zh, "zh", method="unicode", detect_mixing=True) == (True, "zh")

    russian_mixed = "Это ответ на русском языке про Tailscale, Proxmox и OpenAI."
    assert detector.needs_translation(
        russian_mixed, "ru", method="unicode", detect_mixing=False) == (False, "ru")
    assert detector.needs_translation(
        russian_mixed, "ru", method="unicode", detect_mixing=True) == (True, "ru")

    assert detector.needs_translation(
        "```python\nprint('中文')\n```", "en", method="unicode") == (False, None)


# ---------------------------------------------------------------------------
# Real lingua, where it is installed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", ["uk", "bg", "sr", "he", "el"])
def test_real_lingua_translates_foreign_reply_for_new_targets(target: str) -> None:
    pytest.importorskip("lingua")
    assert detector._get_lingua_detector() is not None
    assert detector.detect_language(CHINESE_WITH_PRODUCTS, method="lingua") == "zh"

    for detect_mixing in (False, True):
        should_translate, source = detector.needs_translation(
            CHINESE_WITH_PRODUCTS, target, method="lingua", detect_mixing=detect_mixing)
        assert should_translate is True
        assert source == "zh"


def test_real_lingua_leaves_plain_target_language_reply_alone() -> None:
    pytest.importorskip("lingua")
    # Hebrew and Greek are lingua candidates and are detected outright, so a
    # plain target-language reply never even reaches the mismatch branch.
    for target, reply in (("he", HEBREW_PLAIN), ("el", GREEK_PLAIN)):
        assert detector.detect_language(reply, method="lingua") == target
        assert detector._is_mixed_text(reply, target) is True
        should_translate, _ = detector.needs_translation(
            reply, target, method="lingua", detect_mixing=False)
        assert should_translate is False


def test_latin_target_set_matches_lingua() -> None:
    # Pins LATIN_TARGET_LANGUAGES to lingua's own script classification, so a
    # Latin-script target can never drift into the unknown-script path.
    lingua = pytest.importorskip("lingua")
    latin_codes = {
        language.iso_code_639_1.name.lower()
        for language in lingua.Language.all_with_latin_script()
    }
    assert detector.LATIN_TARGET_LANGUAGES == latin_codes


# ---------------------------------------------------------------------------
# Plugin hook wiring: the reported false rewrite, end to end
# ---------------------------------------------------------------------------

class _DummyCtx:
    """Minimal plugin context: records ``ctx.llm.complete`` calls."""

    def __init__(self) -> None:
        self.hooks: dict[str, Any] = {}
        self.llm_calls: list[dict[str, Any]] = []
        self.llm = SimpleNamespace(complete=self.complete)

    def register_hook(self, name: str, callback: Any) -> None:
        self.hooks[name] = callback

    def complete(self, **kwargs: Any) -> Any:
        self.llm_calls.append(kwargs)
        return SimpleNamespace(text="Translated text")


def test_plugin_hook_leaves_long_ukrainian_reply_untouched(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression via the public entry point: target uk + unicode detector, a
    # long Ukrainian reply must not reach the LLM at all.
    monkeypatch.setattr(
        config_mod.MonolingualConfig,
        "from_hermes_config",
        classmethod(lambda cls: cls({"detector": "unicode", "target_language": "uk"})),
    )
    config_mod.reload_config()

    ctx = _DummyCtx()
    plugin.register(ctx)
    ctx.hooks["pre_llm_call"](
        session_id="uk", user_message=UKRAINIAN_LONG, conversation_history=[])
    assert plugin._get_profile("uk") == "uk"

    assert ctx.hooks["transform_llm_output"](
        session_id="uk", response_text=UKRAINIAN_LONG) is None
    assert ctx.llm_calls == []

    # A foreign reply still gets translated through the same wiring.
    output = ctx.hooks["transform_llm_output"](
        session_id="uk", response_text=CHINESE_WITH_PRODUCTS)
    assert output == "[translated from Chinese by GwynTel/monolingual]\n\nTranslated text"
    assert len(ctx.llm_calls) == 1
