"""INSPECT: assistance-derived supervision for robot state verification.

The package intentionally keeps its top-level import lightweight. Import
runtime components from their concrete modules, for example
``inspect_system.active_view`` or ``inspect_system.verifier``. This prevents
tooling scripts from loading camera, robot, or training dependencies just to
access one submodule.
"""

SYSTEM_NAME = "INSPECT"
SYSTEM_FULL_NAME = "Interactive Supervision for Procedural Evidence and Cross-view Task Verification"

__all__ = ["SYSTEM_FULL_NAME", "SYSTEM_NAME"]
