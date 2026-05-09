"""Typed sparse events written by the runtime control plane."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PipelineEvent:
    """One sparse runtime event for replay and observability."""

    event_type: str
    timestamp: float
    frame_index: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)
