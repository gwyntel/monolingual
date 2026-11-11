"""Language detection for the monolingual plugin.

Detects the user's preferred language from their conversation history,
then determines whether the LLM response needs translation.

Two detection strategies:
1. **lingua** (default) — Rust-backed, accurate, supports mixed-language detection.
2. **unicode** (fallback) — Zero-dependency script heuristic. No pip install
   needed. It reports *scripts*, not languages: every Cyrillic language comes
   back as ``ru`` and every Latin one as ``en``, so see
   :data:`UNICODE_DETECTABLE_LANGUAGES` for what it can actually name.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("hermes.plugins.monolingual")

# ---------------------------------------------------------------------------
# ISO 639-1 ↔ human-readable names (common languages only)
# ---------------------------------------------------------------------------

LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "cs": "Czech",
    "sv": "Swedish",
    "uk": "Ukrainian",
    "bg": "Bulgarian",
    "sr": "Serbian",
    "he": "Hebrew",
    "el": "Greek",
    "bn": "Bengali",
    "pa": "Punjabi",
    "gu": "Gujarati",
    "ta": "Tamil",
    "te": "Telugu",
    "ka": "Georgian",
    "hy": "Armenian",
}

# Every language lingua 2.x classifies as Latin-script (``Language.all_with_latin_script``).
# Targets sharing the Latin script are handled by the same-script rules — the
# plugin cannot tell, say, Finnish from English by script, and the unicode
# fallback reports all of them as "en". Kept complete so that a Latin-script
# ``target_language`` from config is never mistaken for a script-unknown
# language; ``test_latin_target_set_matches_lingua`` pins it to lingua's own
# classification (the non-Latin remainder of lingua's 75 languages is
# deliberately out of scope here).
LATIN_TARGET_LANGUAGES = frozenset({
    "af", "az", "bs", "ca", "cs", "cy", "da", "de", "en", "eo", "es", "et",
    "eu", "fi", "fr", "ga", "hr", "hu", "id", "is", "it", "la", "lg", "lt",
    "lv", "mi", "ms", "nb", "nl", "nn", "pl", "pt", "ro", "sk", "sl", "sn",
    "so", "sq", "st", "sv", "sw", "tl", "tn", "tr", "ts", "vi", "xh", "yo",
    "zu",
})

# Lingua confidence below this point is often ambiguous on short technical text.
# We use unicode script heuristics as a tiebreaker instead of blindly trusting it.
LINGUA_CONFIDENCE_THRESHOLD = 0.65

# Minimum response length before a *same-script* mismatch is trusted as a
# wrong-language reply. Below this, detector noise (lingua calling "Hello
# world" Vietnamese) outweighs the signal. Only applies when the detector that
# actually ran can emit the target language — see _effective_method and
# _detectable_languages.
SAME_SCRIPT_MIN_CHARS = 200

# Share of a response that must be written in the target's own script before the
# text counts as language *mixing* rather than a fully wrong-language reply.
# Half is deliberately loose: a target-language reply that carries foreign
# product names or a quoted phrase must still be classified as mixed (and
# therefore be skipped when ``detect_mixing`` is off), while a reply in another
# script has no target-script characters at all.
TARGET_SCRIPT_DOMINANCE_RATIO = 0.5

# Greek letters double as mathematical symbols (α, β, θ, λ), so counting Greek
# *characters* alone would classify English technical prose as Greek text. Greek
# only counts as text when Greek letters are a meaningful share of all letters.
GREEK_TEXT_MIN_LETTER_RATIO = 0.25


def language_name(code: str) -> str:
    """Return human-readable name for an ISO 639-1 code."""
    return LANGUAGE_NAMES.get(code, code)


# ---------------------------------------------------------------------------
# Language code normalization
# ---------------------------------------------------------------------------

def _normalize_lang(code: str) -> str:
    """Normalize language codes. lingua returns 'zh' for Chinese, etc."""
    if code is None:
        return "unknown"
    code = code.lower().strip()
    # Map common variants
    mappings = {
        "zho": "zh", "chi": "zh", "cmn": "zh",
        "jpn": "ja", "kor": "ko",
    }
    return mappings.get(code, code)


def _lang_from_lingua(language: Any) -> Optional[str]:
    """Return an ISO 639-1 code from a lingua Language object."""
    try:
        iso = language.iso_code_639_1
        if iso is None:
            return None
        return _normalize_lang(iso.name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Unicode-range heuristic (zero-dependency fallback)
# ---------------------------------------------------------------------------

def _cjk_ratio(text: str) -> float:
    """Fraction of characters that fall in CJK Unicode ranges."""
    if not text:
        return 0.0
    cjk_count = 0
    for ch in text:
        cp = ord(ch)
        if (_is_cjk_unified(cp) or       # CJK Unified Ideographs + extensions
            0x3040 <= cp <= 0x309F or   # Hiragana
            0x30A0 <= cp <= 0x30FF or   # Katakana
            0xAC00 <= cp <= 0xD7AF or   # Hangul Syllables
            0x0600 <= cp <= 0x06FF or   # Arabic
            0x0400 <= cp <= 0x04FF or   # Cyrillic
            0x0900 <= cp <= 0x097F or   # Devanagari
            0x0E00 <= cp <= 0x0E7F):    # Thai
            cjk_count += 1
    return cjk_count / len(text)


def _is_cjk_unified(cp: int) -> bool:
    """Return True for CJK unified ideograph blocks, including supplements."""
    return (
        0x4E00 <= cp <= 0x9FFF or     # CJK Unified Ideographs
        0x3400 <= cp <= 0x4DBF or     # Extension A
        0x20000 <= cp <= 0x2A6DF or   # Extension B
        0x2A700 <= cp <= 0x2B73F or   # Extension C
        0x2B740 <= cp <= 0x2B81F or   # Extension D
        0x2B820 <= cp <= 0x2CEAF or   # Extension E
        0x2CEB0 <= cp <= 0x2EBEF or   # Extension F and adjacent blocks
        0x30000 <= cp <= 0x3134F      # Extension G
    )


# Which script counts, and with what weight, for each language the unicode
# heuristic can name. Single source of truth for the scores built in
# :func:`_detect_unicode` and for :data:`UNICODE_DETECTABLE_LANGUAGES`.
_UNICODE_SCORE_WEIGHTS: Dict[str, Tuple[Tuple[str, float], ...]] = {
    "zh": (("han", 1.0),),
    "ja": (("kana", 1.0), ("han", 0.5)),   # kanji-only Japanese reads as Chinese
    "ko": (("hangul", 1.0),),
    "ru": (("cyrillic", 1.0),),
    "ar": (("arabic", 1.0),),
    "hi": (("devanagari", 1.0),),
    "th": (("thai", 1.0),),
    "he": (("hebrew", 1.0),),
    "el": (("greek", 1.0),),
    "bn": (("bengali", 1.0),),
    "pa": (("gurmukhi", 1.0),),
    "gu": (("gujarati", 1.0),),
    "ta": (("tamil", 1.0),),
    "te": (("telugu", 1.0),),
    "ka": (("georgian", 1.0),),
    "hy": (("armenian", 1.0),),
}

# Language codes the unicode heuristic can return. Text in any other script —
# or in Latin script — is reported as "en", so a target outside this set can
# never equal this detector's verdict; a mismatch for such a target is a
# coverage artifact, not evidence of a wrong-language reply.
UNICODE_DETECTABLE_LANGUAGES = frozenset(_UNICODE_SCORE_WEIGHTS) | {"en"}


def _detect_unicode(text: str) -> Optional[str]:
    """Crude unicode-range heuristic. Returns ISO 639-1 or None.

    Covers every script in :data:`_SCRIPT_RANGES`. The result names the
    *script*: languages sharing one are not told apart (Ukrainian, Bulgarian
    and Serbian all come back as ``ru``; any Latin-script language comes back
    as ``en``). Lingua is the detector that distinguishes them — see
    :data:`UNICODE_DETECTABLE_LANGUAGES` for the full set of codes this can
    return.
    """
    if not text:
        return None
    counts = {
        script: len(pattern.findall(text))
        for script, pattern in _SCRIPT_RANGES.items()
    }
    # Greek letters inside Latin prose are usually notation, not Greek text.
    letters = sum(1 for ch in text if ch.isalpha())
    greek_score = counts["greek"]
    if letters and greek_score / letters < GREEK_TEXT_MIN_LETTER_RATIO:
        greek_score = 0

    counts["greek"] = greek_score
    scores = {
        code: sum(counts[script] * weight for script, weight in weights)
        for code, weights in _UNICODE_SCORE_WEIGHTS.items()
    }
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best] < 2:
        return "en"  # assume Latin = English
    return best


# ---------------------------------------------------------------------------
# Lingua detector (primary) — lazy-loaded, scoped languages
# ---------------------------------------------------------------------------

# Language codes handed to lingua as candidates, mapped to their lingua
# ``Language`` member names. Lingua can only ever return one of these codes, so
# the table doubles as :data:`LINGUA_DETECTABLE_LANGUAGES`.
#
# The list is deliberately narrow — lingua is more accurate on short text with
# fewer candidates — but it must contain every language whose script the plugin
# claims to distinguish. Cyrillic languages (uk/bg/sr) and Hebrew/Greek are in
# here because the unicode fallback collapses all Cyrillic to ``ru`` and cannot
# name Hebrew or Greek text reliably.
_LINGUA_CANDIDATES: Dict[str, str] = {
    "en": "ENGLISH",
    "zh": "CHINESE",
    "ja": "JAPANESE",
    "ko": "KOREAN",
    "es": "SPANISH",
    "fr": "FRENCH",
    "de": "GERMAN",
    "ru": "RUSSIAN",
    "ar": "ARABIC",
    "hi": "HINDI",
    "it": "ITALIAN",
    "nl": "DUTCH",
    "pl": "POLISH",
    "tr": "TURKISH",
    "vi": "VIETNAMESE",
    "th": "THAI",
    "pt": "PORTUGUESE",
    "cs": "CZECH",
    "sv": "SWEDISH",
    "id": "INDONESIAN",
    "uk": "UKRAINIAN",
    "bg": "BULGARIAN",
    "sr": "SERBIAN",
    "he": "HEBREW",
    "el": "GREEK",
}

LINGUA_DETECTABLE_LANGUAGES = (
    frozenset(_LINGUA_CANDIDATES) | UNICODE_DETECTABLE_LANGUAGES
)
# The lingua path also consults the unicode heuristic when confidence is low
# (see _detect_lingua), so it can return any code listed there as well.

_lingua_detector = None


def _detectable_languages(method: str) -> frozenset[str]:
    """Return the language codes the given detection method can report.

    A target outside this set can never equal the method's verdict, so a
    "different language" result for it carries no information: it is a coverage
    gap (unicode reports one code per script; lingua only knows its candidates),
    not evidence the reply is in the wrong language.
    """
    if (method or "lingua").lower() == "unicode":
        return UNICODE_DETECTABLE_LANGUAGES
    return LINGUA_DETECTABLE_LANGUAGES


def _effective_method(method: str) -> str:
    """Return the detection method that actually runs for ``method``.

    :func:`detect_language` falls back to the unicode heuristic whenever lingua
    is unavailable, so asking for ``"lingua"`` does not guarantee lingua's
    vocabulary: the verdict can be a script code instead. Callers weighing a
    same-script mismatch must consult *this* method, not the requested one —
    otherwise a target lingua knows but unicode cannot name (``uk``, ``bg``,
    ``sr``, ...) is "detected" as ``ru`` and a long reply in the user's own
    language is rewritten.
    """
    if (method or "lingua").lower() == "unicode":
        return "unicode"
    return "lingua" if _get_lingua_detector() is not None else "unicode"


def _get_lingua_detector():
    """Lazy-init lingua detector.

    Uses a scoped set of languages supported by this plugin's script heuristics.
    Lingua recommends narrowing candidates when the expected language set is
    known; this improves accuracy on short/technical text. Models are preloaded
    because hooks call detection repeatedly during a Hermes session.
    """
    global _lingua_detector
    if _lingua_detector is not None:
        return _lingua_detector
    try:
        from lingua import Language, LanguageDetectorBuilder
        candidates = [
            getattr(Language, attribute) for attribute in _LINGUA_CANDIDATES.values()
        ]
        _lingua_detector = (
            LanguageDetectorBuilder
            .from_languages(*candidates)
            .with_preloaded_language_models()
            .build()
        )
        return _lingua_detector
    except ImportError:
        return None
    except Exception as exc:
        logger.warning("lingua detector init failed: %s", exc)
        return None


@lru_cache(maxsize=1024)
def _detect_lingua(text: str) -> Optional[str]:
    """Detect language using lingua with confidence-weighted result.

    If the top result has low confidence, falls back to unicode heuristic.
    """
    detector = _get_lingua_detector()
    if detector is None:
        return None
    try:
        confidence_values = detector.compute_language_confidence_values(text)
        if not confidence_values:
            return None
        top = confidence_values[0]
        top_lang = _lang_from_lingua(top.language)
        if not top_lang:
            return _detect_unicode(text)
        top_conf = top.value

        # If lingua is very confident, trust it
        if top_conf >= LINGUA_CONFIDENCE_THRESHOLD:
            return top_lang

        # Low confidence — if top candidate is a "close" language (de/nl for en),
        # the text is probably English with technical jargon. Use unicode as tiebreaker.
        logger.debug(
            "lingua low confidence: %s=%.3f, falling back to unicode. text=%.80s",
            top_lang, top_conf, text[:80],
        )
        unicode_result = _detect_unicode(text)
        return unicode_result or top_lang
    except Exception as exc:
        logger.debug("lingua detection failed: %s", exc)
        return None


def _detect_lingua_mixing(text: str, target: str) -> Tuple[bool, float]:
    """Check if text contains significant non-target language mixing.

    Returns (is_mixing, non_target_ratio).

    Note: lingua's ``DetectionResult`` exposes character offsets
    (``start_index``/``end_index``), not word lists — span length is measured
    in characters so multi-sentence minority spans are weighed correctly.
    """
    detector = _get_lingua_detector()
    if detector is None:
        return False, 0.0
    try:
        results = detector.detect_multiple_languages_of(text)
        if not results:
            return False, 0.0
        total_chars = 0
        non_target_chars = 0
        for span in results:
            span_len = max(span.end_index - span.start_index, 1)
            lang_code = _lang_from_lingua(span.language)
            total_chars += span_len
            if lang_code and lang_code != target:
                non_target_chars += span_len
        ratio = non_target_chars / total_chars if total_chars > 0 else 0.0
        return ratio > 0.05, ratio
    except Exception as exc:
        logger.debug("lingua mixing detection failed: %s", exc)
        return False, 0.0


# ---------------------------------------------------------------------------
# Unified detection API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2048)
def detect_language(text: str, method: str = "lingua") -> Optional[str]:
    """Detect the dominant language of text.

    Args:
        text: The text to classify.
        method: "lingua" (default, accurate) or "unicode" (fast, zero-dep).

    Returns:
        ISO 639-1 code (e.g. "en", "zh") or None if detection fails.
    """
    normalized_method = (method or "lingua").lower()
    if normalized_method == "unicode":
        return _detect_unicode(text)
    # Try lingua first, fall back to unicode
    result = _detect_lingua(text)
    if result is None:
        logger.debug("lingua unavailable, falling back to unicode heuristic")
        return _detect_unicode(text)
    return result


def detect_user_language(
    conversation_history: list,
    user_message: str,
    sample_window: int = 10,
    min_sample_length: int = 20,
    method: str = "lingua",
    target_override: Optional[str] = None,
) -> Optional[str]:
    """Detect the user's preferred language from their recent messages.

    Samples the last N user-role messages from conversation_history, plus the
    current user_message. Runs language detection on each, accumulates
    length-weighted scores (longer messages count more), and returns the
    language with the highest total score.

    Args:
        conversation_history: List of message dicts with 'role' and 'content'.
        user_message: The current user message.
        sample_window: How many recent user messages to sample.
        min_sample_length: Skip messages shorter than this (unreliable detection).
        method: Detection method ("lingua" or "unicode").
        target_override: If set, skip detection and return this directly.

    Returns:
        ISO 639-1 code for the detected language, or None if no samples.
    """
    if target_override:
        return target_override

    # Collect recent user messages
    user_texts: List[str] = []
    for msg in reversed(conversation_history or []):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle multimodal content — extract text parts
                content = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            if len(content.strip()) >= min_sample_length:
                user_texts.append(content.strip())
            if len(user_texts) >= sample_window:
                break

    # Add current message
    if user_message and len(user_message.strip()) >= min_sample_length:
        user_texts.insert(0, user_message.strip())

    if not user_texts:
        return None

    # Detect language per message with length weighting
    lang_scores: Dict[str, float] = defaultdict(float)
    for text in user_texts:
        lang = detect_language(text, method=method)
        if lang:
            # Weight by message length — longer messages are more reliable
            lang_scores[lang] += len(text)

    if not lang_scores:
        return None

    best_lang = max(lang_scores, key=lang_scores.get)  # type: ignore[arg-type]
    return best_lang


def _is_mixed_text(text: str, target_language: str) -> bool:
    """Return True if the target script is present and dominant in the text.

    Distinguishes genuine language mixing from a wrong-language reply: a
    target-language response carrying foreign product names stays "mixed" and
    honours ``detect_mixing``, while a reply with none of the target script is
    the wrong language and must be translated. A target whose script this
    plugin cannot name is never reported as mixed — without a target script
    there is no evidence the reply is in the user's language, and guessing
    "mixed" is what silently skipped translations.
    """
    if not text:
        return False
    if target_language in LATIN_TARGET_LANGUAGES:
        foreign_share = _count_foreign_chars(text) / len(text)
        return foreign_share < TARGET_SCRIPT_DOMINANCE_RATIO
    target_range = _TARGET_SCRIPT_RANGES.get(target_language)
    if target_range is None:
        return False
    target_share = len(target_range.findall(text)) / len(text)
    return target_share >= TARGET_SCRIPT_DOMINANCE_RATIO


def needs_translation(
    response_text: str,
    target_language: str,
    method: str = "lingua",
    min_mix_ratio: float = 0.15,
    detect_mixing: bool = True,
) -> Tuple[bool, Optional[str]]:
    """Determine whether response_text needs to be translated into target_language.

    Args:
        response_text: The LLM output to check.
        target_language: ISO 639-1 code for the user's language.
        method: Detection method.
        min_mix_ratio: Fraction of non-target text to trigger mixing translation.
        detect_mixing: Whether to also check for intra-sentence mixing.

    Returns:
        (needs_translation, detected_source_language)

    A mismatch is only trusted when it carries information. ``detected`` can
    never equal a target the detector that actually ran cannot emit, so for such
    targets (unicode with any code but the ones it names, lingua with a language
    outside its candidate list) a same-script mismatch is left alone instead of
    rewriting the user's own language; see :func:`_effective_method` and
    :func:`_detectable_languages`.
    """
    if not response_text or not response_text.strip():
        return False, None

    # Strip code blocks before checking — they're language-agnostic
    text_no_code = _strip_code_blocks(response_text)
    if not text_no_code.strip():
        return False, None  # only code, nothing to translate

    detected = detect_language(text_no_code, method=method)
    # Same-script evidence is only meaningful when the detector that actually
    # ran could have reported the target language at all. Otherwise every reply
    # in the user's language is a guaranteed "mismatch" (unicode reports every
    # Cyrillic language as ru and every Latin one as en), and a long one would
    # be rewritten on no evidence. `method` alone is not enough: a request for
    # lingua that falls back to unicode must be judged by unicode's vocabulary.
    same_script_reliable = target_language in _detectable_languages(
        _effective_method(method))

    # Case 1: response is entirely in a different language
    if detected and detected != target_language:
        # For Latin targets this counts foreign-script characters; for non-Latin
        # targets it counts Latin words (see _count_non_target_script_chars).
        foreign_count = _count_non_target_script_chars(text_no_code, target_language)
        is_latin_target = target_language in LATIN_TARGET_LANGUAGES

        if is_latin_target:
            if foreign_count == 0:
                # Same script as the target: lingua misclassifies very short
                # Latin text (e.g. "Hello world" → vi), so only trust the
                # mismatch on a long response the method could have named.
                if same_script_reliable and len(text_no_code) > SAME_SCRIPT_MIN_CHARS:
                    return True, detected
                # Short and same script: not enough evidence to rewrite.
            elif _is_mixed_text(text_no_code, target_language):
                # The target script still dominates: this is language *mixing*,
                # not a fully wrong-language response — honor detect_mixing.
                return detect_mixing, detected
            else:
                # Foreign-script characters dominate: translate.
                return True, detected
        else:
            foreign_script_count = _count_foreign_script_chars(
                text_no_code, target_language)
            mixed = _is_mixed_text(text_no_code, target_language)
            if foreign_script_count > 0 and not mixed:
                # No character of the target script, but another script is
                # present: the whole reply is in a foreign language. Translate
                # it even with detect_mixing disabled.
                return True, detected
            if foreign_count == 0:
                # Same script family (or no Latin at all): the "foreign"
                # verdict may just be same-script confusion — the unicode
                # heuristic reports every Cyrillic language as ru — so only
                # trust a long response the method could have named.
                if same_script_reliable and len(text_no_code) > SAME_SCRIPT_MIN_CHARS:
                    return True, detected
            elif mixed:
                # Target script present and dominant: language mixing, not a
                # fully wrong-language reply. Honor detect_mixing.
                return detect_mixing, detected
            elif foreign_count >= 3:
                # Latin-script reply carrying 3+ Latin words: a real foreign
                # reply, not a product name.
                return True, detected

    # Case 2: response looks like target language, but check for mixing
    if detect_mixing and detected == target_language:
        # Try lingua's multi-language detection first (catches larger mixed spans)
        if (method or "lingua").lower() == "lingua":
            is_mixing, ratio = _detect_lingua_mixing(text_no_code, target_language)
            if is_mixing and ratio >= min_mix_ratio:
                return True, detected

        # Fallback: unicode character scan. Lingua's per-word detection ignores
        # single CJK characters embedded in Latin text — this catches them.
        non_target_chars = _count_non_target_script_chars(text_no_code, target_language)
        if target_language in LATIN_TARGET_LANGUAGES:
            # For Latin targets, even 1 non-Latin char is suspicious
            if non_target_chars > 0:
                return True, detected
        else:
            # For non-Latin targets, require 3+ Latin words to avoid
            # false-positives on technical terms like "Tailscale", "Proxmox"
            if non_target_chars >= 3:
                return True, detected

    return False, detected


# ---------------------------------------------------------------------------
# Script coverage
# ---------------------------------------------------------------------------

# Character-class bodies for every script this plugin recognises. Ranges are
# kept as fragments so the per-language target ranges and the "any non-Latin
# script" range are all built from one source of truth.
_SCRIPT_CLASS_BODIES: Dict[str, str] = {
    "han": (
        r"\u4e00-\u9fff"          # CJK Unified Ideographs
        r"\u3400-\u4dbf"          # CJK Extension A
        r"\U00020000-\U0002a6df"  # CJK Extension B
        r"\U0002a700-\U0002b73f"  # CJK Extension C
        r"\U0002b740-\U0002b81f"  # CJK Extension D
        r"\U0002b820-\U0002ceaf"  # CJK Extension E
        r"\U0002ceb0-\U0002ebef"  # CJK Extension F and adjacent blocks
        r"\U00030000-\U0003134f"  # CJK Extension G
    ),
    "kana": (
        r"\u3040-\u309f"          # Hiragana
        r"\u30a0-\u30ff"          # Katakana
        r"\uff66-\uff9d"          # Halfwidth Katakana
    ),
    "hangul": (
        r"\uac00-\ud7af"          # Hangul Syllables
        r"\u1100-\u11ff"          # Hangul Jamo
        r"\u3130-\u318f"          # Hangul Compatibility Jamo
    ),
    "cyrillic": r"\u0400-\u04ff",
    "hebrew": r"\u0590-\u05ff",
    "greek": r"\u0370-\u03ff\u1f00-\u1fff",
    "arabic": r"\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff",
    "devanagari": r"\u0900-\u097f",
    "bengali": r"\u0980-\u09ff",
    "gurmukhi": r"\u0a00-\u0a7f",
    "gujarati": r"\u0a80-\u0aff",
    "tamil": r"\u0b80-\u0bff",
    "telugu": r"\u0c00-\u0c7f",
    "thai": r"\u0e00-\u0e7f",
    "georgian": r"\u10a0-\u10ff\u1c90-\u1cbf",
    "armenian": r"\u0530-\u058f",
}


def _script_range(*scripts: str) -> re.Pattern[str]:
    """Compile one character class covering the given script(s)."""
    return re.compile("[" + "".join(_SCRIPT_CLASS_BODIES[s] for s in scripts) + "]")


_SCRIPT_RANGES: Dict[str, re.Pattern[str]] = {
    script: _script_range(script) for script in _SCRIPT_CLASS_BODIES
}

# Every non-Latin script the plugin recognises, used to catch single foreign
# characters that lingua's word-level mixing detector misses. Greek is excluded
# from this blanket range because bare Greek letters (θ, α, λ) are ordinary
# notation in technical prose; Greek is counted separately, and only when it
# dominates the text's letters (see _count_foreign_chars), so a Greek reply is
# still detected while an equation is not rewritten.
_NON_LATIN_RANGES = _script_range(
    *(script for script in _SCRIPT_CLASS_BODIES if script != "greek")
)

# Script(s) each non-Latin target language is written in. Cyrillic languages
# share one range: the unicode heuristic cannot tell them apart, but
# _is_mixed_text only needs "is the target script present and dominant".
# Languages outside this table are treated as script-unknown; see
# _is_mixed_text and _count_foreign_script_chars.
_TARGET_SCRIPTS: Dict[str, Tuple[str, ...]] = {
    "zh": ("han",),
    "ja": ("han", "kana"),
    "ko": ("hangul",),
    "ru": ("cyrillic",),
    "uk": ("cyrillic",),
    "bg": ("cyrillic",),
    "sr": ("cyrillic",),
    "mk": ("cyrillic",),
    "be": ("cyrillic",),
    "kk": ("cyrillic",),
    "mn": ("cyrillic",),
    "he": ("hebrew",),
    "el": ("greek",),
    "ar": ("arabic",),
    "fa": ("arabic",),
    "ur": ("arabic",),
    "hi": ("devanagari",),
    "mr": ("devanagari",),
    "ne": ("devanagari",),
    "bn": ("bengali",),
    "pa": ("gurmukhi",),
    "gu": ("gujarati",),
    "ta": ("tamil",),
    "te": ("telugu",),
    "th": ("thai",),
    "ka": ("georgian",),
    "hy": ("armenian",),
}

# Script range for the target itself, used to decide whether a non-Latin reply is
# "mixed" (target script present and dominant) or simply the wrong language.
_TARGET_SCRIPT_RANGES: Dict[str, re.Pattern[str]] = {
    code: _script_range(*scripts) for code, scripts in _TARGET_SCRIPTS.items()
}


def _count_foreign_script_chars(text: str, target_language: str) -> int:
    """Count characters in a recognised script that is not the target's.

    Tells "reply written in a completely different script" (a wrong-language
    response that must be translated) apart from "reply in the target script"
    (mixing, governed by ``detect_mixing``). Targets whose script is unknown
    count every recognised script as foreign, which is the conservative
    choice: a named foreign script is evidence the text is not the target
    language, while silence would skip the translation.
    """
    if not text:
        return 0
    target_scripts = _TARGET_SCRIPTS.get(target_language, ())
    return sum(
        len(pattern.findall(text))
        for script, pattern in _SCRIPT_RANGES.items()
        if script not in target_scripts
    )


def _count_foreign_chars(text: str) -> int:
    """Count non-Latin-script characters in text written for a Latin target.

    Greek is special-cased: a Greek reply is foreign, but ``α``/``β`` inside
    English prose is notation. Greek characters only count when they make up a
    meaningful share of the text's letters.
    """
    foreign = len(_NON_LATIN_RANGES.findall(text))
    greek = len(_SCRIPT_RANGES["greek"].findall(text))
    if greek:
        letters = sum(1 for ch in text if ch.isalpha())
        if letters and greek / letters >= GREEK_TEXT_MIN_LETTER_RATIO:
            foreign += greek
    return foreign


def _count_latin_words(text: str) -> int:
    """Count Latin-alphabet words that look like prose, not product names.

    URLs, filesystem paths, e-mail addresses and ALL-CAPS abbreviations are
    stripped first so technical tokens do not masquerade as foreign words.
    """
    cleaned = re.sub(r"https?://\S+|www\.\S+|\b[\w.-]+@[\w.-]+", " ", text)
    cleaned = re.sub(r"\b[A-Za-z]:[\\/][^\s]+|(?:[./~]|/)[^\s]+", " ", cleaned)
    cleaned = re.sub(r"\b[A-Z_][A-Z0-9_]{2,}\b", " ", cleaned)
    return len(re.findall(r"[A-Za-z]{3,}", cleaned))


def _count_non_target_script_chars(text: str, target_language: str) -> int:
    """Count characters whose script doesn't belong to target_language.

    For Latin-script targets (en, es, fr, de, etc.), counts characters in a
    recognised non-Latin script — even single chars count.

    For non-Latin targets (zh, ja, ko, ar, ru, he, el, ...), counts Latin
    words that appear outside of URLs, paths, and ALL-CAPS abbreviations.
    Returns the raw count so the caller can decide the threshold.

    Returns 0 if no foreign-script characters found.
    """
    if not text:
        return 0

    if target_language in LATIN_TARGET_LANGUAGES:
        # Count non-Latin script chars — even a single one is suspicious
        return _count_foreign_chars(text)
    return _count_latin_words(text)


def _strip_code_blocks(text: str) -> str:
    """Remove fenced code blocks (```...```) and inline code from text."""
    # Remove fenced Markdown blocks with backticks or tildes.
    text = re.sub(r'(?ms)^\s*(```|~~~).*?^\s*\1\s*$', '', text)
    # Remove inline code spans, including multi-backtick spans.
    text = re.sub(r'`+[^`]+`+', '', text)
    return text.strip()
