[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/gwyntel/monolingual)

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
git clone https://github.com/gwyntel/monolingual.git ~/.hermes/plugins/monolingual

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

Same-script mismatches are only trusted on longer text, and only for languages the active detector can actually report: if the detected language differs from yours but uses the same script (e.g. a short French sentence for an English user), the response must be over 200 characters before it counts as wrong-language. Shorter text is treated as detector noise and falls through to the mixing check. When the active detector cannot name your language at all — the `unicode` fallback reports one code per script, lingua only knows its candidate list — the same-script check is skipped entirely: a mismatch there is a coverage gap, not evidence, and rewriting a long reply on it would translate your own language into itself.

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
      target_language: ""              # e.g. "en", "zh", "ja", "ko", "es", "uk", "he"

      # How many recent user messages to sample for language detection
      sample_window: 10               # default: 10

      # Skip messages shorter than this (unreliable for detection)
      min_sample_length: 20           # default: 20

      # Fraction of non-target lingua spans needed to trigger mixing
      # translation. Only applies to the lingua span check; the unicode
      # fallback uses fixed script thresholds instead (1 foreign char for
      # Latin targets, 3 Latin words for non-Latin targets).
      min_mix_ratio: 0.15             # default: 0.15

      # Also translate responses that mix your language with another one.
      # Disable if you only want full wrong-language responses rewritten.
      detect_mixing: true             # default: true

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

[lingua-language-detector](https://pypi.org/project/lingua-language-detector/) — Rust-backed, highly accurate, supports mixed-language detection across 75 languages. This is the recommended method.

Monolingual narrows lingua to a candidate list for better short-text accuracy, preloads the language models for repeated plugin use, and falls back to Unicode script heuristics when confidence is low.

Candidates: English, Chinese, Japanese, Korean, Spanish, French, German, Portuguese, Italian, Dutch, Polish, Czech, Swedish, Turkish, Vietnamese, Indonesian, Russian, Ukrainian, Bulgarian, Serbian, Hebrew, Greek, Arabic, Hindi, Thai.

Ukrainian, Bulgarian, Serbian, Hebrew and Greek are included because the Unicode fallback cannot distinguish them — it reports every Cyrillic language as `ru` and cannot name Hebrew or Greek text reliably. A target outside this list (Finnish, Hungarian, …) is not distinguishable by either detector, so a same-script reply in that language is never treated as a wrong-language response; it passes through untouched.

Install: `pip install lingua-language-detector`

### unicode (fallback)

A zero-dependency heuristic that reads Unicode codepoint ranges. It names *scripts*, not languages — good enough for the common zh↔en mixing case, no pip install needed.

Scripts recognised: Han (including extension ranges through U+3134F), Hiragana/Katakana, Hangul, Cyrillic, Arabic, Devanagari, Bengali, Gurmukhi, Gujarati, Tamil, Telugu, Thai, Hebrew, Greek, Georgian, Armenian. Greek only counts as Greek text when Greek letters make up at least 25% of all letters, so `α`/`β`/`θ` in technical prose stay mathematics. Text in any other script — including every Latin-script language — is reported as English.

Limits of the method:

- One code per script. Ukrainian, Bulgarian, Serbian, Belarusian, Kazakh, Mongolian and Macedonian all report as `ru`; Finnish, Polish, Spanish and every other Latin-script language report as `en`.
- Mixing *between* languages that share a script cannot be detected, and a reply in a same-script language the heuristic cannot name is left alone rather than translated — it may be the user's own language.
- Use lingua if you need those distinctions; `detector: "lingua"` is the default for exactly this reason.

## Design Decisions

- **Fail-open:** Any error in detection or translation → original text passes through. Never blocks output.
- **Method-aware trust:** a same-script mismatch only counts as a wrong-language signal when the active detector can actually report the user's language; otherwise the reply passes through untouched. Coverage gaps must never become translations of the user's own words.
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
