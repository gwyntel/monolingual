# GwynTel/Monolingual

**Auto-detect language mixing in LLM output and rewrite into the user's language.**

Zero config. Installs in 30 seconds. Works with any Hermes Agent instance.

## The Problem

Multi-lingual capable models (GLM, Qwen, Kimi, etc.) sometimes respond in the wrong language — Chinese when you wanted English, or a confusing mix of both. This is especially common with models trained primarily on Chinese data.

## The Solution

Monolingual is a Hermes Agent plugin that:

1. **Watches your messages** — samples your last N messages to detect your preferred language
2. **Checks every LLM response** — if the response is in the wrong language or mixes languages, it triggers a translation
3. **Rewrites via the host's LLM** — uses the same model+auth that's already configured, no extra keys needed
4. **Flags the translation** — prepends `[translated from Chinese by GwynTel/monolingual]` so you know it happened

## Installation

```bash
# Install the lingua detection library
pip install lingua-language-detector

# Clone the plugin into Hermes plugins directory
git clone https://github.com/GwynTel/monolingual.git ~/.hermes/plugins/monolingual

# Enable the plugin
hermes plugins enable monolingual

# Restart your session (or /reset in an active session)
```

That's it. No configuration required.

## How It Works

### Language Detection

On every LLM call, the `pre_llm_call` hook samples your recent messages and runs language detection using [lingua](https://github.com/pemistahl/lingua-py) — the most accurate Python language detection library, with explicit support for mixed-language text. Detection results are cached in memory to avoid re-processing repeated messages.

The detected language is stored per-session in memory. It adapts as you switch languages — it's a rolling window, not a fixed setting.

### Translation Trigger

On every LLM response, the `transform_llm_output` hook checks:

1. **Primary check:** Is the response entirely in a different language than yours?
2. **Secondary check:** Does the response mix your language with another language? (Only if `detect_mixing` is on.)

If either check triggers, the plugin calls the host's `ctx.llm.complete()` to translate the response, then replaces the original text with the translation.

Very large responses are skipped by default (`max_translation_chars: 20000`) so an accidental huge dump does not trigger an expensive translation call.

### Translation Execution

The translation uses your Hermes Agent's **already configured** LLM model and auth. No extra API keys, no provider setup. It goes through the `PluginLlm` facade — the supported way for plugins to make LLM calls.

**Fallback chain:**
1. `ctx.llm.complete()` — host's active model
2. `auxiliary_client.call_llm(task="title_generation")` — reuses title generation's LLM config
3. Pass-through (original text unchanged) — fail-open, never blocks output

### Flag

Translated responses are prefixed with:
```
[translated from Chinese by GwynTel/monolingual]
```

This makes it clear the response was translated, not original output.

## Configuration (All Optional)

Everything has sane defaults. Config goes in `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - monolingual
  entries:
    monolingual:
      # Soft-disable the plugin without removing it from plugins.enabled
      disabled: false                  # default: false

      # Override auto-detection: force a specific target language
      # If empty (default), the plugin learns from your messages
      target_language: ""              # e.g. "en", "zh", "ja", "ko", "es"

      # How many recent user messages to sample for language detection
      sample_window: 10               # default: 10

      # Skip messages shorter than this (unreliable for detection)
      min_sample_length: 20           # default: 20

      # Fraction of non-target text to trigger mixing translation
      min_mix_ratio: 0.15             # default: 0.15

      # Detection method: "lingua" (accurate) or "unicode" (fast, zero-dep)
      detector: "lingua"              # default: "lingua"

      # Translation flag format. {source} and {target} are language names
      flag_template: "[translated from {source} by GwynTel/monolingual]"

      # Timeout in seconds for the translation LLM call
      translation_timeout: 30         # default: 30

      # Do not translate responses larger than this many characters
      # Set to 0 to disable the size guard
      max_translation_chars: 20000     # default: 20000

      # How often to refresh config from disk while Hermes is running
      # Set to 0 to keep the first loaded config until process restart
      config_reload_seconds: 5         # default: 5
```

## Detection Methods

### lingua (default)

[lingua-language-detector](https://pypi.org/project/lingua-language-detector/) — Rust-backed, highly accurate, supports mixed-language detection across 75 languages. This is the recommended method. Monolingual narrows Lingua to the plugin's supported common languages for better short-text accuracy, preloads language models for repeated plugin use, and falls back to Unicode script heuristics when confidence is low.

Install: `pip install lingua-language-detector`

### unicode (fallback)

A zero-dependency heuristic that checks Unicode codepoint ranges for CJK, Arabic, Cyrillic, Devanagari, Thai, and Hangul. No pip install needed. Good enough for the common zh↔en mixing case. Falls back to this automatically if lingua isn't installed.

## Design Decisions

- **Fail-open:** Any error in detection or translation → original text passes through. Never blocks output.
- **Session-scoped profiles:** Language detection is per-session, rebuilt from conversation history each turn, and cleared on `on_session_end`. No persistent state needed.
- **`transform_llm_output` hook:** The right Hermes hook for this job — returns a replacement string. `post_llm_call` is observer-only.
- **`ctx.llm` for translation:** Uses the host's PluginLlm facade — supported, no internal API imports needed.
- **Best-effort fallback:** If available, an auxiliary LLM client may be used after `ctx.llm`; otherwise output passes through unchanged.
- **Translate entire response, not spans:** Simpler, preserves coherence. Span-by-span translation would create jarring switches.
- **First turn is a free pass:** No user history → no detection → no translation. Kicks in from the second turn onward.

## File Structure

```
monolingual/
├── plugin.yaml          # Plugin manifest
├── __init__.py          # register(ctx) — hook wiring
├── config.py            # Config reader (all optional)
├── detector.py          # Language detection + mixing detection
├── translator.py        # LLM-based rewrite with fallback chain
├── tests/               # pytest coverage for detection/config/translation
└── README.md            # This file
```

## Requirements

- Hermes Agent ≥ 0.15.0 (for `transform_llm_output` hook and `ctx.llm` facade)
- `lingua-language-detector` (recommended) or Python 3.11+

## License

Apache 2.0

---

Built by [GwynTel](https://github.com/GwynTel) because GLM-5.1 kept responding in Chinese and we got tired of it.
