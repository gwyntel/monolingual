"""Configuration reader for the monolingual plugin.

Reads from ``plugins.entries.monolingual`` in ``~/.hermes/config.yaml``.
All settings are optional — the plugin works with zero config.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes.plugins.monolingual")

# Defaults
DEFAULT_SAMPLE_WINDOW = 10
DEFAULT_MIN_SAMPLE_LENGTH = 20
DEFAULT_MIN_MIX_RATIO = 0.15
DEFAULT_FLAG_TEMPLATE = "[translated from {source} by GwynTel/monolingual]"
DEFAULT_DETECTOR = "lingua"
DEFAULT_TRANSLATION_TIMEOUT = 30


class MonolingualConfig:
    """Parsed configuration for the monolingual plugin."""

    def __init__(self, raw: Optional[Dict[str, Any]] = None):
        raw = raw or {}
        self.sample_window: int = raw.get("sample_window", DEFAULT_SAMPLE_WINDOW)
        self.min_sample_length: int = raw.get("min_sample_length", DEFAULT_MIN_SAMPLE_LENGTH)
        self.min_mix_ratio: float = raw.get("min_mix_ratio", DEFAULT_MIN_MIX_RATIO)
        self.flag_template: str = raw.get("flag_template", DEFAULT_FLAG_TEMPLATE)
        self.detector: str = raw.get("detector", DEFAULT_DETECTOR)
        self.translation_timeout: int = raw.get("translation_timeout", DEFAULT_TRANSLATION_TIMEOUT)
        self.target_language: Optional[str] = raw.get("target_language") or None

    @classmethod
    def from_hermes_config(cls) -> "MonolingualConfig":
        """Load config from ``~/.hermes/config.yaml`` → plugins.entries.monolingual."""
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            plugins_cfg = cfg.get("plugins", {})
            entries = plugins_cfg.get("entries", {}) if isinstance(plugins_cfg, dict) else {}
            mono_cfg = entries.get("monolingual", {}) if isinstance(entries, dict) else {}
            return cls(mono_cfg)
        except Exception as exc:
            logger.debug("monolingual config load fallback to defaults: %s", exc)
            return cls()


# Module-level singleton — loaded once, reused across hook invocations.
_cached_config: Optional[MonolingualConfig] = None


def get_config() -> MonolingualConfig:
    """Return cached config, loading on first call."""
    global _cached_config
    if _cached_config is None:
        _cached_config = MonolingualConfig.from_hermes_config()
    return _cached_config


def reload_config() -> MonolingualConfig:
    """Force a config reload (useful after config changes)."""
    global _cached_config
    _cached_config = MonolingualConfig.from_hermes_config()
    return _cached_config
