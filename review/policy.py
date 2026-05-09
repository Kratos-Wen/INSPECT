"""Sparse gating policy for deciding when review is worth the latency."""

from __future__ import annotations

from .types import ReviewRequest, ReviewTrigger


class SparseReviewPolicy:
    """Trigger review only for stable but suspicious decisions."""

    def __init__(
        self,
        enabled: bool = True,
        low_confidence: float = 0.62,
        margin_threshold: float = 0.12,
        disagreement_threshold: float = 0.34,
        memory_confidence_threshold: float = 0.55,
        correction_streak: int = 2,
        cooldown_frames: int = 6,
    ) -> None:
        self.enabled = bool(enabled)
        self.low_confidence = float(low_confidence)
        self.margin_threshold = float(margin_threshold)
        self.disagreement_threshold = float(disagreement_threshold)
        self.memory_confidence_threshold = float(memory_confidence_threshold)
        self.correction_streak = int(correction_streak)
        self.cooldown_frames = int(cooldown_frames)

    def decide(self, request: ReviewRequest, last_review_frame: int | None) -> ReviewTrigger:
        """Return whether review should run for the current stable window."""

        if not self.enabled or not request.stable:
            return ReviewTrigger(False)
        if last_review_frame is not None and (request.frame_index - last_review_frame) < self.cooldown_frames:
            return ReviewTrigger(False)

        reasons: list[str] = []
        if request.fusion_result.confidence <= self.low_confidence:
            reasons.append("low_confidence")
        if request.fused_margin() <= self.margin_threshold:
            reasons.append("low_margin")
        if request.observation.expert_disagreement >= self.disagreement_threshold:
            reasons.append("expert_disagreement")

        memory_step = request.memory_step()
        if (
            memory_step
            and memory_step != request.fusion_result.step_id
            and request.memory_confidence() >= self.memory_confidence_threshold
        ):
            reasons.append("memory_conflict")

        if request.recent_corrections >= self.correction_streak:
            reasons.append("recent_corrections")

        return ReviewTrigger(bool(reasons), tuple(reasons))
