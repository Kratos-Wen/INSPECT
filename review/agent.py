"""Lightweight reviewer that arbitrates only uncertain stable windows."""

from __future__ import annotations

from typing import Dict, Tuple

from ..types import FeedbackEvent, StepPrediction
from .types import ReviewDecision, ReviewRequest, ReviewTrigger


class SparseReviewerAgent:
    """Cheap reviewer that vetoes or weakly re-routes uncertain outputs."""

    def __init__(
        self,
        prefer_vote_count: int = 2,
        prefer_confidence: float = 0.58,
        prefer_margin: float = 0.10,
        hold_confidence: float = 0.48,
        request_human_confidence: float = 0.42,
        reviewer_feedback_strength: float = 0.35,
    ) -> None:
        self.prefer_vote_count = int(prefer_vote_count)
        self.prefer_confidence = float(prefer_confidence)
        self.prefer_margin = float(prefer_margin)
        self.hold_confidence = float(hold_confidence)
        self.request_human_confidence = float(request_human_confidence)
        self.reviewer_feedback_strength = float(reviewer_feedback_strength)

    def review(self, request: ReviewRequest, trigger: ReviewTrigger) -> ReviewDecision:
        """Return a sparse governance action for the current stable window."""

        fused_step = str(request.fusion_result.step_id).strip().upper()
        fused_margin = request.fused_margin()
        best_step, best_count, best_avg_conf, supporters = self._best_consensus(request.expert_predictions)
        memory_step = request.memory_step()
        memory_conf = request.memory_confidence()

        if request.recent_corrections >= 2:
            return ReviewDecision(
                action="request_human",
                label=fused_step,
                reason="recent_corrections",
                trigger_reasons=trigger.reasons,
                should_prompt_human=True,
                extras={"candidate": best_step, "supporters": supporters},
            )

        if (
            best_step
            and best_step != fused_step
            and best_count >= self.prefer_vote_count
            and best_avg_conf >= self.prefer_confidence
            and fused_margin <= self.prefer_margin
        ):
            return ReviewDecision(
                action="prefer_candidate",
                label=best_step,
                reason="multi_expert_consensus",
                trigger_reasons=trigger.reasons,
                weak_feedback=FeedbackEvent(
                    label=best_step,
                    strength=self.reviewer_feedback_strength,
                    accepted=False,
                    source="reviewer",
                    note="sparse_review_prefer_candidate",
                ),
                extras={
                    "candidate": best_step,
                    "supporters": supporters,
                    "candidate_vote_count": best_count,
                    "candidate_avg_confidence": best_avg_conf,
                },
            )

        if memory_step and memory_step != fused_step and memory_conf >= self.request_human_confidence:
            return ReviewDecision(
                action="request_human",
                label=fused_step,
                reason="memory_conflict",
                trigger_reasons=trigger.reasons,
                should_prompt_human=True,
                extras={
                    "candidate": best_step,
                    "memory_step": memory_step,
                    "memory_confidence": memory_conf,
                },
            )

        if request.fusion_result.confidence <= self.request_human_confidence:
            return ReviewDecision(
                action="request_human",
                label=fused_step,
                reason="very_low_confidence",
                trigger_reasons=trigger.reasons,
                should_prompt_human=True,
                extras={"candidate": best_step},
            )

        if request.fusion_result.confidence <= self.hold_confidence or fused_margin <= self.prefer_margin:
            return ReviewDecision(
                action="hold",
                label=fused_step,
                reason="review_hold",
                trigger_reasons=trigger.reasons,
                extras={"candidate": best_step, "supporters": supporters},
            )

        return ReviewDecision(
            action="approve",
            label=fused_step,
            reason="review_approve",
            trigger_reasons=trigger.reasons,
            extras={"candidate": best_step, "supporters": supporters},
        )

    @staticmethod
    def _best_consensus(expert_predictions: Dict[str, StepPrediction]) -> Tuple[str, int, float, list[str]]:
        votes: dict[str, dict[str, object]] = {}
        for expert_name, prediction in expert_predictions.items():
            step_id = str(prediction.step_id).strip().upper()
            entry = votes.setdefault(step_id, {"count": 0, "conf_sum": 0.0, "supporters": []})
            entry["count"] = int(entry["count"]) + 1
            entry["conf_sum"] = float(entry["conf_sum"]) + float(prediction.confidence)
            entry["supporters"] = list(entry["supporters"]) + [str(expert_name)]

        if not votes:
            return "", 0, 0.0, []

        ordered = sorted(
            votes.items(),
            key=lambda item: (int(item[1]["count"]), float(item[1]["conf_sum"])),
            reverse=True,
        )
        step_id, payload = ordered[0]
        count = int(payload["count"])
        conf_sum = float(payload["conf_sum"])
        supporters = list(payload["supporters"])
        avg_conf = conf_sum / max(1, count)
        return str(step_id), count, avg_conf, supporters
