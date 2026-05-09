"""Policies that decide when memory recall is worth the latency budget."""

from __future__ import annotations

from dataclasses import dataclass

from .types import MemoryObservation


@dataclass(frozen=True)
class RecallDecision:
    """Decision returned by the recall policy."""

    should_recall: bool
    reason: str


class UncertaintyRecallPolicy:
    """Trigger recall only on uncertain, disagreeing, or transitional states."""

    def __init__(
        self,
        margin_threshold: float = 0.18,
        disagreement_threshold: float = 0.30,
        cooldown_frames: int = 2,
        recall_on_transition: bool = True,
    ) -> None:
        self.margin_threshold = float(margin_threshold)
        self.disagreement_threshold = float(disagreement_threshold)
        self.cooldown_frames = max(0, int(cooldown_frames))
        self.recall_on_transition = bool(recall_on_transition)

    def decide(
        self,
        observation: MemoryObservation,
        last_recall_frame: int | None,
        has_session: bool,
        has_long_term: bool,
    ) -> RecallDecision:
        """Return whether the current observation should trigger recall."""

        if not has_session and not has_long_term:
            return RecallDecision(False, "empty_memory")

        if not observation.has_visual_evidence:
            return RecallDecision(False, "no_visual_evidence")

        if last_recall_frame is not None and observation.frame_index - last_recall_frame < self.cooldown_frames:
            return RecallDecision(False, "cooldown")

        if observation.ensemble_margin <= self.margin_threshold:
            return RecallDecision(True, "low_margin")

        if observation.expert_disagreement >= self.disagreement_threshold:
            return RecallDecision(True, "expert_disagreement")

        if (
            self.recall_on_transition
            and has_session
            and observation.prev_step
            and observation.ensemble_step
            and observation.ensemble_step != observation.prev_step
        ):
            return RecallDecision(True, "transition")

        return RecallDecision(False, "confident_no_recall")
