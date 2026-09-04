"""Stateful orchestration for sparse review and recent-correction tracking."""

from __future__ import annotations

from typing import Dict, Optional

from ..memory.types import MemoryObservation, MemoryRecallResult
from ..core_types import FeedbackEvent, FusionResult, StepPrediction
from .agent import SparseReviewerAgent
from .policy import SparseReviewPolicy
from .types import ReviewDecision, ReviewRequest


class ReviewManager:
    """Coordinate sparse review while keeping the hot path bounded."""

    def __init__(
        self,
        policy: SparseReviewPolicy,
        reviewer: SparseReviewerAgent,
        correction_window: int = 48,
    ) -> None:
        self.policy = policy
        self.reviewer = reviewer
        self.correction_window = int(correction_window)
        self.last_review_frame: Optional[int] = None
        self.correction_frames: list[int] = []

    def consider(
        self,
        frame_index: int,
        prev_step: Optional[str],
        stable: bool,
        fusion_result: FusionResult,
        expert_predictions: Dict[str, StepPrediction],
        observation: MemoryObservation,
        memory_recall: MemoryRecallResult,
    ) -> Optional[ReviewDecision]:
        """Run policy + reviewer and return a sparse decision when warranted."""

        self._prune(frame_index)
        request = ReviewRequest(
            frame_index=int(frame_index),
            prev_step=str(prev_step).strip().upper() if prev_step else None,
            stable=bool(stable),
            fusion_result=fusion_result,
            expert_predictions=expert_predictions,
            observation=observation,
            memory_recall=memory_recall,
            recent_corrections=len(self.correction_frames),
        )
        trigger = self.policy.decide(request, self.last_review_frame)
        if not trigger.should_review:
            return None
        self.last_review_frame = frame_index
        return self.reviewer.review(request, trigger)

    def record_feedback(self, frame_index: int, feedback: FeedbackEvent) -> None:
        """Track recent human corrections for future review triggers."""

        if str(feedback.source).strip().lower() == "reviewer":
            return
        if feedback.accepted:
            self._prune(frame_index)
            return
        self.correction_frames.append(int(frame_index))
        self._prune(frame_index)

    def _prune(self, frame_index: int) -> None:
        cutoff = int(frame_index) - self.correction_window
        self.correction_frames = [value for value in self.correction_frames if value >= cutoff]
