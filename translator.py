"""Translation engine for the monolingual plugin.

Uses the host's LLM (via ctx.llm PluginLlm facade) to translate text
while preserving technical terms, code, and formatting.

Fallback chain:
1. ctx.llm.complete() — uses the host's active model + auth
2. agent.auxiliary_client.call_llm(task="title_generation") — yoinks the
   title_generation aux config if the host has one set up
3. Pass-through (original text, no translation) — fail-open
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .config import MonolingualConfig, get_config
from .detector import language_name

logger = logging.getLogger("hermes.plugins.monolingual")

TRANSLATION_SYSTEM_PROMPT = (
    "You are a translation engine. Translate the following text to {target_lang_name}. "
    "Rules:\n"
    "1. Preserve all technical terms that are standard in {target_lang_name} "
    "(e.g. 'database', 'API', 'kernel' stay in English for English output).\n"
    "2. Do NOT translate: code blocks, inline code, URLs, file paths, "
    "command names, environment variables, package names.\n"
    "3. Preserve the original tone and register — informal stays informal, "
    "technical stays technical.\n"
    "4. Preserve all formatting: markdown, bullet points, numbered lists, headers.\n"
    "5. If a term has no standard {target_lang_name} equivalent, keep the original.\n"
    "6. Output ONLY the translated text. No explanations, no notes, no preamble."
)


def _build_messages(
    text: str,
    source_lang: str,
    target_lang: str,
) -> list:
    """Build the chat messages for a translation LLM call."""
    target_name = language_name(target_lang)
    system = TRANSLATION_SYSTEM_PROMPT.format(
        target_lang_name=target_name,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]


def translate_via_ctx_llm(
    ctx: Any,
    text: str,
    source_lang: str,
    target_lang: str,
    config: Optional[MonolingualConfig] = None,
) -> Optional[str]:
    """Translate text using the plugin's ctx.llm facade.

    Returns the translated string, or None on failure.
    """
    config = config or get_config()
    messages = _build_messages(text, source_lang, target_lang)

    try:
        result = ctx.llm.complete(
            messages=messages,
            temperature=0.1,  # low temp for faithful translation
            max_tokens=4096,
            timeout=config.translation_timeout,
            purpose="monolingual_translation",
        )
        if result and result.text and result.text.strip():
            return result.text.strip()
        logger.warning("ctx.llm.complete returned empty result")
        return None
    except Exception as exc:
        logger.warning("ctx.llm translation failed: %s", exc)
        return None


def translate_via_auxiliary(
    text: str,
    source_lang: str,
    target_lang: str,
    config: Optional[MonolingualConfig] = None,
) -> Optional[str]:
    """Fallback: translate using auxiliary_client.call_llm with title_generation config.

    This 'yoinks' the title_generation aux config — if the user has a working
    LLM endpoint for title generation, we reuse it for translation too.
    """
    config = config or get_config()
    messages = _build_messages(text, source_lang, target_lang)

    try:
        from agent.auxiliary_client import call_llm
        response = call_llm(
            task="title_generation",
            messages=messages,
            temperature=0.1,
            max_tokens=4096,
            timeout=config.translation_timeout,
        )
        content = response.choices[0].message.content if response else None
        if content and content.strip():
            return content.strip()
        return None
    except Exception as exc:
        logger.debug("auxiliary_client translation fallback failed: %s", exc)
        return None


def translate(
    ctx: Any,
    text: str,
    source_lang: str,
    target_lang: str,
    config: Optional[MonolingualConfig] = None,
) -> str:
    """Translate text with fallback chain. Always returns a string.

    Returns the translated text on success, or the original text on failure
    (fail-open — never block output).
    """
    config = config or get_config()

    # Primary: ctx.llm (host's active model)
    result = translate_via_ctx_llm(ctx, text, source_lang, target_lang, config)
    if result is not None:
        return result

    # Fallback: auxiliary_client with title_generation config
    result = translate_via_auxiliary(text, source_lang, target_lang, config)
    if result is not None:
        return result

    # Fail-open: return original text unchanged
    logger.warning(
        "All translation methods failed — returning original text. "
        "source=%s target=%s len=%d",
        source_lang, target_lang, len(text),
    )
    return text
