"""Configuration reader for the monolingual plugin.

Reads from ``plugins.entries.monolingual`` in ``~/.hermes/config.yaml``.
All settings are optional — the plugin works with zero config.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes.plugins.monolingual")

# Defaults
DEFAULT_SAMPLE_WINDOW = 10
DEFAULT_MIN_SAMPLE_LENGTH = 20
DEFAULT_MIN_MIX_RATIO = 0.15
DEFAULT_FLAG_TEMPLATE = "[translated from {source} by GwynTel/monolingual]"
DEFAULT_DETECTOR = "lingua"
DEFAULT_TRANSLATION_TIMEOUT = 30
DEFAULT_MAX_TRANSLATION_CHARS = 20_000
DEFAULT_CONFIG_RELOAD_SECONDS = 5.0


def _as_bool(value: Any, default: bool = False) -> bool:
    """Coerce common config values to bool without surprising truthiness."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


def _as_int(value: Any, default: int, *, minimum: int = 0) -> int:
    """Coerce an integer config value with a lower bound."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(parsed, minimum)


def _as_float(value: Any, default: float, *, minimum: float = 0.0) -> float:
    """Coerce a float config value with a lower bound."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(parsed, minimum)


class MonolingualConfig:
    """Parsed configuration for the monolingual plugin."""

    def __init__(self, raw: Optional[Dict[str, Any]] = None) -> None:
        raw = raw or {}
        self.disabled: bool = _as_bool(raw.get("disabled"), False)
        self.sample_window: int = _as_int(raw.get("sample_window"), DEFAULT_SAMPLE_WINDOW, minimum=1)
        self.min_sample_length: int = _as_int(raw.get("min_sample_length"), DEFAULT_MIN_SAMPLE_LENGTH, minimum=0)
        self.min_mix_ratio: float = _as_float(raw.get("min_mix_ratio"), DEFAULT_MIN_MIX_RATIO, minimum=0.0)
        self.flag_template: str = str(raw.get("flag_template") or DEFAULT_FLAG_TEMPLATE)
        detector = str(raw.get("detector") or DEFAULT_DETECTOR).strip().lower()
        self.detector: str = detector if detector in {"lingua", "unicode"} else DEFAULT_DETECTOR
        self.translation_timeout: int = _as_int(raw.get("translation_timeout"), DEFAULT_TRANSLATION_TIMEOUT, minimum=1)
        self.max_translation_chars: int = _as_int(
            raw.get("max_translation_chars"),
            DEFAULT_MAX_TRANSLATION_CHARS,
            minimum=0,
        )
        self.config_reload_seconds: float = _as_float(
            raw.get("config_reload_seconds"),
            DEFAULT_CONFIG_RELOAD_SECONDS,
            minimum=0.0,
        )
        target = str(raw.get("target_language") or "").strip().lower()
        self.target_language: Optional[str] = target or None

    @classmethod
    def from_hermes_config(cls) -> "MonolingualConfig":
        """Load config from ``~/.hermes/config.yaml`` → plugins.entries.monolingual."""
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            plugins_cfg = cfg.get("plugins", {})
            entries = plugins_cfg.get("entries", {}) if isinstance(plugins_cfg, dict) else {}
            mono_cfg = entries.get("monolingual", {}) if isinstance(entries, dict) else {}
            if not isinstance(mono_cfg, dict):
                logger.debug("monolingual config entry is not a mapping; using defaults")
                mono_cfg = {}
            return cls(mono_cfg)
        except Exception as exc:
            logger.debug("monolingual config load fallback to defaults: %s", exc)
            return cls()


# Module-level singleton — reloaded periodically so config edits take effect
# without process restart while still avoiding disk reads on every hook call.
_cached_config: Optional[MonolingualConfig] = None
_cached_config_loaded_at = 0.0
_config_lock = threading.Lock()


def get_config() -> MonolingualConfig:
    """Return cached config, reloading after ``config_reload_seconds``."""
    global _cached_config, _cached_config_loaded_at
    now = time.monotonic()
    with _config_lock:
        should_reload = _cached_config is None
        if _cached_config is not None and _cached_config.config_reload_seconds > 0:
            should_reload = now - _cached_config_loaded_at >= _cached_config.config_reload_seconds
        if should_reload:
            try:
                _cached_config = MonolingualConfig.from_hermes_config()
            except Exception as exc:
                logger.debug("monolingual config reload failed; using defaults: %s", exc)
                _cached_config = MonolingualConfig()
            _cached_config_loaded_at = now
        return _cached_config


def reload_config() -> MonolingualConfig:
    """Force a config reload (useful after config changes)."""
    global _cached_config, _cached_config_loaded_at
    with _config_lock:
        try:
            _cached_config = MonolingualConfig.from_hermes_config()
        except Exception as exc:
            logger.debug("monolingual forced config reload failed; using defaults: %s", exc)
            _cached_config = MonolingualConfig()
        _cached_config_loaded_at = time.monotonic()
        return _cached_config
