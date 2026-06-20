"""Language detection for the monolingual plugin.

Detects the user's preferred language from their conversation history,
then determines whether the LLM response needs translation.

Two detection strategies:
1. **lingua** (default) — Rust-backed, accurate, supports mixed-language detection.
2. **unicode** (fallback) — Zero-dependency heuristic for CJK vs Latin mixing.
   No pip install needed. Good enough for the zh↔en mixing case.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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
}


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
        if (0x4E00 <= cp <= 0x9FFF or   # CJK Unified Ideographs
            0x3400 <= cp <= 0x4DBF or   # CJK Extension A
            0x20000 <= cp <= 0x2A6DF or  # CJK Extension B
            0x3040 <= cp <= 0x309F or   # Hiragana
            0x30A0 <= cp <= 0x30FF or   # Katakana
            0xAC00 <= cp <= 0xD7AF or   # Hangul Syllables
            0x0600 <= cp <= 0x06FF or   # Arabic
            0x0400 <= cp <= 0x04FF or   # Cyrillic
            0x0900 <= cp <= 0x097F or   # Devanagari
            0x0E00 <= cp <= 0x0E7F):    # Thai
            cjk_count += 1
    return cjk_count / len(text)


def _detect_unicode(text: str) -> Optional[str]:
    """Crude unicode-range heuristic. Returns ISO 639-1 or None."""
    if not text:
        return None
    # Script counters
    zh_count = sum(1 for ch in text if 0x4E00 <= ord(ch) <= 0x9FFF or 0x3400 <= ord(ch) <= 0x4DBF or 0x20000 <= ord(ch) <= 0x2A6DF)
    ja_hira = sum(1 for ch in text if 0x3040 <= ord(ch) <= 0x309F)
    ja_kata = sum(1 for ch in text if 0x30A0 <= ord(ch) <= 0x30FF)
    ko_count = sum(1 for ch in text if 0xAC00 <= ord(ch) <= 0xD7AF)
    ar_count = sum(1 for ch in text if 0x0600 <= ord(ch) <= 0x06FF)
    cyrillic = sum(1 for ch in text if 0x0400 <= ord(ch) <= 0x04FF)
    deva = sum(1 for ch in text if 0x0900 <= ord(ch) <= 0x097F)
    thai = sum(1 for ch in text if 0x0E00 <= ord(ch) <= 0x0E7F)

    scores = {
        "zh": zh_count,
        "ja": ja_hira + ja_kata + zh_count * 0.5,
        "ko": ko_count,
        "ar": ar_count,
        "ru": cyrillic,
        "hi": deva,
        "th": thai,
    }
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best] < 2:
        return "en"  # assume Latin = English
    return best


# ---------------------------------------------------------------------------
# Lingua detector (primary) — lazy-loaded, scoped languages
# ---------------------------------------------------------------------------

_lingua_detector = None


def _get_lingua_detector():
    """Lazy-init lingua detector.

    Uses a scoped set of common languages instead of all 75 — much more
    accurate on short/technical text that would otherwise confuse the detector.
    """
    global _lingua_detector
    if _lingua_detector is not None:
        return _lingua_detector
    try:
        from lingua import Language, LanguageDetectorBuilder
        _lingua_detector = (
            LanguageDetectorBuilder
            .from_languages(
                Language.ENGLISH,
                Language.CHINESE,
                Language.JAPANESE,
                Language.KOREAN,
                Language.SPANISH,
                Language.FRENCH,
                Language.GERMAN,
                Language.RUSSIAN,
                Language.ARABIC,
                Language.HINDI,
                Language.ITALIAN,
                Language.DUTCH,
                Language.POLISH,
                Language.TURKISH,
                Language.VIETNAMESE,
                Language.THAI,
                Language.PORTUGUESE,
                Language.CZECH,
                Language.SWEDISH,
                Language.INDONESIAN,
            )
            .with_low_accuracy_mode()
            .build()
        )
        return _lingua_detector
    except ImportError:
        return None
    except Exception as exc:
        logger.warning("lingua detector init failed: %s", exc)
        return None


def _detect_lingua(text: str) -> Optional[str]:
    """Detect language using lingua with confidence-weighted result.

    If the top result has low confidence (< 0.5), falls back to unicode heuristic.
    """
    detector = _get_lingua_detector()
    if detector is None:
        return None
    try:
        confidence_values = detector.compute_language_confidence_values(text)
        if not confidence_values:
            return None
        top = confidence_values[0]
        top_lang = top.language.iso_code_639_1.name.lower()
        top_conf = top.value

        # If lingua is very confident, trust it
        if top_conf >= 0.5:
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
            span_text = " ".join(w.word if hasattr(w, 'word') else str(w) for w in (span.words or []))
            span_len = max(len(span_text), 1)
            lang_code = span.language.iso_code_639_1.name.lower() if hasattr(span.language, 'iso_code_639_1') else None
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

def detect_language(text: str, method: str = "lingua") -> Optional[str]:
    """Detect the dominant language of text.

    Args:
        text: The text to classify.
        method: "lingua" (default, accurate) or "unicode" (fast, zero-dep).

    Returns:
        ISO 639-1 code (e.g. "en", "zh") or None if detection fails.
    """
    if method == "unicode":
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
    confidence-weighted scores, and returns the language with the highest
    total confidence score.

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

    # Detect language per message with confidence weighting
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
    """
    if not response_text or not response_text.strip():
        return False, None

    # Strip code blocks before checking — they're language-agnostic
    text_no_code = _strip_code_blocks(response_text)
    if not text_no_code.strip():
        return False, None  # only code, nothing to translate

    detected = detect_language(text_no_code, method=method)

    # Case 1: response is entirely in a different language
    # But only trigger if there are actually foreign-script characters present.
    # Lingua misclassifies very short Latin text (e.g. "Hello world" → vi),
    # so we guard with a script check.
    if detected and detected != target_language:
        foreign_count = _count_non_target_script_chars(text_no_code, target_language)
        if foreign_count == 0:
            # Same script as target, different language detected —
            # likely short-text misclassification. Only trust if long.
            if len(text_no_code) > 200:
                return True, detected
            # Short text, same script — skip. Fall through to mixing check.
        else:
            # Foreign-script chars present. For Latin targets (en, es, etc.),
            # even 1 CJK char is suspicious. For non-Latin targets (zh, ja, etc.),
            # require 3+ Latin words to avoid false-positives on tech terms.
            latin_targets = {"en", "es", "fr", "de", "pt", "it", "nl", "pl", "tr", "sv", "cs", "id", "vi"}
            if target_language in latin_targets or foreign_count >= 3:
                return True, detected

    # Case 2: response looks like target language, but check for mixing
    if detect_mixing and detected == target_language:
        # Try lingua's multi-language detection first (catches larger mixed spans)
        if method == "lingua":
            is_mixing, ratio = _detect_lingua_mixing(text_no_code, target_language)
            if is_mixing and ratio >= min_mix_ratio:
                return True, detected

        # Fallback: unicode character scan. Lingua's per-word detection ignores
        # single CJK characters embedded in Latin text — this catches them.
        non_target_chars = _count_non_target_script_chars(text_no_code, target_language)
        latin_targets = {"en", "es", "fr", "de", "pt", "it", "nl", "pl", "tr", "sv", "cs", "id", "vi"}
        if target_language in latin_targets:
            # For Latin targets, even 1 non-Latin char is suspicious
            if non_target_chars > 0:
                return True, detected
        else:
            # For non-Latin targets, require 3+ Latin words to avoid
            # false-positives on technical terms like "Tailscale", "Proxmox"
            if non_target_chars >= 3:
                return True, detected

    return False, detected


# Non-target script character ranges. Used to catch single CJK chars that
# lingua's word-level mixing detector misses.
_NON_LATIN_RANGES = re.compile(
    "[" +
    r"\u4e00-\u9fff"     # CJK Unified Ideographs
    r"\u3400-\u4dbf"     # CJK Extension A
    r"\U00020000-\U0002a6df"  # CJK Extension B (needs \U, not \u)
    r"\u3040-\u309f"     # Hiragana
    r"\u30a0-\u30ff"     # Katakana
    r"\uac00-\ud7af"     # Hangul Syllables
    r"\u0600-\u06ff"     # Arabic
    r"\u0400-\u04ff"     # Cyrillic
    r"\u0900-\u097f"     # Devanagari
    r"\u0e00-\u0e7f"     # Thai
    "]"
)

_LATIN_LETTER_RE = re.compile(r"[a-zA-Z]")


def _count_non_target_script_chars(text: str, target_language: str) -> int:
    """Count characters whose script doesn't belong to target_language.

    For Latin-script targets (en, es, fr, de, etc.), counts CJK/Arabic/
    Cyrillic/Devanagari/Thai characters — even single chars count.

    For non-Latin targets (zh, ja, ko, ar, ru, hi, th), counts Latin
    characters that appear outside of URLs, numbers, and common technical
    terms. Returns the raw count so the caller can decide the threshold.

    Returns 0 if no foreign-script characters found.
    """
    if not text:
        return 0

    latin_targets = {"en", "es", "fr", "de", "pt", "it", "nl", "pl", "tr", "sv", "cs", "id", "vi"}
    if target_language in latin_targets:
        # Count non-Latin script chars — even a single one is suspicious
        return len(_NON_LATIN_RANGES.findall(text))
    else:
        # For non-Latin targets, count sequences of Latin letters that look
        # like actual words (3+ consecutive letters, not just abbreviations).
        latin_words = re.findall(r'[a-zA-Z]{3,}', text)
        return len(latin_words)


def _strip_code_blocks(text: str) -> str:
    """Remove fenced code blocks (```...```) and inline code from text."""
    # Remove fenced blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    # Remove inline code
    text = re.sub(r'`[^`]+`', '', text)
    return text.strip()
