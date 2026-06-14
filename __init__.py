"""GwynTel Monolingual — Hermes Agent plugin for auto language-filtering.

Detects when the LLM outputs text in the wrong language (or mixes languages)
and rewrites it into the user's detected preferred language. Zero config —
learns the user's language from their own messages.

Hooks:
  - pre_llm_call — samples user messages to detect preferred language
  - transform_llm_output — detects language issues and triggers translation
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from .config import MonolingualConfig, get_config
from .detector import (
    detect_user_language,
    language_name,
    needs_translation,
)
from .translator import translate

logger = logging.getLogger("hermes.plugins.monolingual")

# ---------------------------------------------------------------------------
# Session-scoped language profiles
# ---------------------------------------------------------------------------

# {session_id: DetectedLanguage}
# Thread-safe via lock since hooks may fire from different threads.
_session_profiles: Dict[str, str] = {}
_profiles_lock = threading.Lock()


def _set_profile(session_id: str, lang: str) -> None:
    with _profiles_lock:
        _session_profiles[session_id] = lang


def _get_profile(session_id: str) -> Optional[str]:
    with _profiles_lock:
        return _session_profiles.get(session_id)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Called by the Hermes plugin loader. Registers hooks."""

    def on_pre_llm_call(**kwargs):
        """Sample user messages to detect their preferred language.

        Stores the result in _session_profiles for use by transform_llm_output.
        """
        config = get_config()
        conversation_history = kwargs.get("conversation_history", [])
        user_message = kwargs.get("user_message", "")
        session_id = kwargs.get("session_id", "")

        if not session_id:
            return None

        lang = detect_user_language(
            conversation_history=conversation_history,
            user_message=user_message or "",
            sample_window=config.sample_window,
            min_sample_length=config.min_sample_length,
            method=config.detector,
            target_override=config.target_language,
        )

        if lang:
            existing = _get_profile(session_id)
            if existing != lang:
                logger.info(
                    "monolingual: session %s detected user language: %s",
                    session_id, language_name(lang),
                )
            _set_profile(session_id, lang)

        # pre_llm_call can only inject context, not block.
        # We don't need to inject anything — just updating the profile.
        return None

    def on_transform_llm_output(**kwargs) -> Optional[str]:
        """Check if the LLM output needs translation, and translate if so.

        Returns the translated text (with flag prefix) or None to pass through.
        """
        response_text = kwargs.get("response_text", "")
        session_id = kwargs.get("session_id", "")

        if not response_text or not response_text.strip():
            return None

        # Get the user's detected language for this session
        target_lang = _get_profile(session_id)
        if not target_lang:
            # First turn or no user history — can't determine target, pass through
            return None

        config = get_config()
        should_translate, source_lang = needs_translation(
            response_text=response_text,
            target_language=target_lang,
            method=config.detector,
            min_mix_ratio=config.min_mix_ratio,
        )

        if not should_translate:
            return None

        source_display = language_name(source_lang) if source_lang else source_lang or "unknown"
        target_display = language_name(target_lang)

        logger.info(
            "monolingual: translating %s → %s (session %s, %d chars)",
            source_display, target_display, session_id, len(response_text),
        )

        # Translate
        translated = translate(
            ctx=ctx,
            text=response_text,
            source_lang=source_lang or "unknown",
            target_lang=target_lang,
            config=config,
        )

        if translated == response_text:
            # Translation failed or returned identical text — no flag needed
            return None

        # Prepend the translation flag
        flag = config.flag_template.format(
            source=source_display,
            target=target_display,
        )
        return f"{flag}\n\n{translated}"

    # Register both hooks
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("transform_llm_output", on_transform_llm_output)

    logger.info("monolingual plugin registered (language detection from user messages)")
