"""Shared pytest support for the monolingual test suite.

The plugin ships as a single directory whose name (``monolingual``, or a
hyphenated checkout directory such as ``monolingual-pr2``) is not an importable
Python identifier, so the package is loaded explicitly from ``__init__.py`` with
its own directory as the search path. Both test modules import the loaded
objects from here, so they always exercise the same module objects.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "monolingual_testpkg"


def load_plugin_package() -> ModuleType:
    """Load the plugin root as a package so relative imports work in tests."""
    if PACKAGE_NAME in sys.modules:
        return sys.modules[PACKAGE_NAME]
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module


plugin = load_plugin_package()
config_mod = sys.modules[f"{PACKAGE_NAME}.config"]
detector = sys.modules[f"{PACKAGE_NAME}.detector"]
translator = sys.modules[f"{PACKAGE_NAME}.translator"]
