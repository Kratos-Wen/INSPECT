"""Typed payloads for sparse review decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from ..memory.types import MemoryObservation, MemoryRecallResult
from ..types import FeedbackEvent, FusionResult, StepPrediction


@dataclass(frozen=True)
class ReviewRequest:
    """Structured snapshot passed into the sparse reviewer."""

    frame_index: int
    prev_step: Optional[str]
    stable: bool
    fusion_result: FusionResult
    expert_predictions: Dict[str, StepPrediction]
    observation: MemoryObservation
    memory_recall: MemoryRecallResult
    recent_corrections: int = 0

    def fused_margin(self) -> float:
        """Return the top-1 vs top-2 margin for the fused scores."""

        ordered = sorted(self.fusion_result.scores.values(), reverse=True)
        if len(ordered) < 2:
            return float(ordered[0]) if ordered else 0.0
        return float(ordered[0] - ordered[1])

    def memory_step(self) -> str:
        """Return the recalled step identifier, if any."""

        return str(self.memory_recall.prediction.step_id).strip().upper()

    def memory_confidence(self) -> float:
        """Return the recalled memory confidence."""

        return float(self.memory_recall.prediction.confidence)


@dataclass(frozen=True)
class ReviewTrigger:
    """Review policy output before the reviewer runs."""

    should_review: bool
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ReviewDecision:
    """Reviewer output used by the runtime control plane."""

    action: str
    label: Optional[str]
    reason: str
    trigger_reasons: Tuple[str, ...] = ()
    should_prompt_human: bool = False
    weak_feedback: Optional[FeedbackEvent] = None
    extras: Dict[str, object] = field(default_factory=dict)
