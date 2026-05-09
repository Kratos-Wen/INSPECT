"""Typed ops-state snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class OpsSnapshot:
    """Current runtime state for dashboards or post-run inspection."""

    status: str
    frame_index: int
    step_id: str
    reason: str
    review_action: str = ""
    counts: Dict[str, int] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)
