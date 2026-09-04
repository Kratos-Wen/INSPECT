"""Stable public entry point for the INSPECT runtime."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


_RUNTIME_PACKAGE = "inspect_runtime"


def load_runtime_package() -> ModuleType:
    """Load the repository runtime under a checkout-independent package name."""

    loaded = sys.modules.get(_RUNTIME_PACKAGE)
    if loaded is not None:
        return loaded

    package_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        _RUNTIME_PACKAGE,
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load the INSPECT runtime from {package_root}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_RUNTIME_PACKAGE] = module
    spec.loader.exec_module(module)
    return module


def _load_runtime_package() -> None:
    """Backward-compatible loader used by existing command-line integrations."""

    load_runtime_package()


def __getattr__(name: str):
    runtime = load_runtime_package()
    try:
        return getattr(runtime, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


__all__ = ["load_runtime_package"]
