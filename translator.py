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
from typing import Any, Callable, Optional

from .config import MonolingualConfig, get_config
from .detector import language_name

logger = logging.getLogger("hermes.plugins.monolingual")

TRANSLATION_SYSTEM_PROMPT = (
    "You are a precise translation engine. Translate the user's text from "
    "{source_lang_name} to {target_lang_name}.\n\n"
    "Rules:\n"
    "1. Output ONLY the translated text. Do not add explanations, notes, "
    "preambles, or markdown fences around the whole answer.\n"
    "2. Preserve the original structure exactly: markdown headings, lists, "
    "tables, blockquotes, blank lines, indentation, punctuation style, and emoji.\n"
    "3. Do NOT translate code blocks, inline code, URLs, file paths, command "
    "names, CLI flags, environment variables, package names, identifiers, logs, "
    "or quoted data formats.\n"
    "4. Preserve placeholders and templating syntax such as {{braces}}, "
    "${{variables}}, %s, and {{{{ double_braces }}}}.\n"
    "5. Preserve technical terms when they are normally used untranslated in "
    "{target_lang_name}; otherwise use the standard {target_lang_name} term.\n"
    "6. Keep the same tone and register; informal stays informal and technical "
    "stays technical."
)


def _build_messages(
    text: str,
    source_lang: str,
    target_lang: str,
) -> list:
    """Build the chat messages for a translation LLM call."""
    target_name = language_name(target_lang)
    source_name = language_name(source_lang)
    system = TRANSLATION_SYSTEM_PROMPT.format(
        source_lang_name=source_name,
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
        llm = getattr(ctx, "llm", None)
        if llm is None or not hasattr(llm, "complete"):
            logger.debug("ctx.llm.complete is unavailable")
            return None
        result = llm.complete(
            messages=messages,
            temperature=0.1,  # low temp for faithful translation
            max_tokens=4096,
            timeout=config.translation_timeout,
            purpose="monolingual_translation",
        )
        content = getattr(result, "text", None)
        if isinstance(result, str):
            content = result
        if content and str(content).strip():
            return str(content).strip()
        logger.warning("ctx.llm.complete returned empty result")
        return None
    except Exception as exc:
        logger.warning("ctx.llm translation failed: %s", exc)
        return None


def _load_auxiliary_call_llm() -> Optional[Callable[..., Any]]:
    """Return Hermes' internal auxiliary LLM helper if it is importable.

    ``ctx.llm`` is the supported plugin API. This fallback is deliberately best
    effort because Hermes' internal module path has changed across versions.
    """
    import_paths = (
        "agent.auxiliary_client",
        "hermes_cli.agent.auxiliary_client",
    )
    for module_name in import_paths:
        try:
            module = __import__(module_name, fromlist=["call_llm"])
            call_llm = getattr(module, "call_llm", None)
            if callable(call_llm):
                return call_llm
        except ImportError as exc:
            logger.debug("auxiliary_client import failed for %s: %s", module_name, exc)
        except Exception as exc:
            logger.debug("auxiliary_client import error for %s: %s", module_name, exc)
    return None


def _extract_auxiliary_content(response: Any) -> Optional[str]:
    """Extract message content from common OpenAI-like response shapes."""
    if isinstance(response, str):
        return response.strip() or None
    try:
        content = response.choices[0].message.content if response else None
    except Exception:
        content = None
    if not content and isinstance(response, dict):
        try:
            content = response["choices"][0]["message"]["content"]
        except Exception:
            content = None
    return str(content).strip() if content and str(content).strip() else None


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

    call_llm = _load_auxiliary_call_llm()
    if call_llm is None:
        return None

    try:
        response = call_llm(
            task="title_generation",
            messages=messages,
            temperature=0.1,
            max_tokens=4096,
            timeout=config.translation_timeout,
        )
        return _extract_auxiliary_content(response)
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

    if config.max_translation_chars and len(text) > config.max_translation_chars:
        logger.warning(
            "Skipping translation for oversized text (%d chars > max_translation_chars=%d)",
            len(text), config.max_translation_chars,
        )
        return text

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
