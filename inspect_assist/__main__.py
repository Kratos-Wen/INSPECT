"""Run the INSPECT command-line interface from any checkout directory."""

from __future__ import annotations

import importlib
from . import load_runtime_package


def _load_runtime_package() -> None:
    """Compatibility alias for older integrations."""

    load_runtime_package()


def main() -> None:
    load_runtime_package()
    cli = importlib.import_module("inspect_runtime.cli")
    cli.main()


if __name__ == "__main__":
    main()
