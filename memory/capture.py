"""Policies for turning verified outcomes into episodic memories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..types import FeedbackEvent, FusionResult
from .types import MemoryObservation


@dataclass(frozen=True)
class CaptureDecision:
    """Whether and how a verified event should be stored."""

    to_session: bool
    to_long_term: bool
    source: str
    trust: float
    note: str = ""


class VerifiedOutcomeCapturePolicy:
    """Capture corrected or trustworthy stable events while avoiding noise."""

    def __init__(
        self,
        min_auto_capture_confidence: float = 0.78,
        accepted_long_term_margin: float = 0.22,
        accepted_long_term_disagreement: float = 0.30,
    ) -> None:
        self.min_auto_capture_confidence = float(min_auto_capture_confidence)
        self.accepted_long_term_margin = float(accepted_long_term_margin)
        self.accepted_long_term_disagreement = float(accepted_long_term_disagreement)

    def decide_feedback(
        self,
        observation: MemoryObservation,
        feedback: FeedbackEvent,
    ) -> CaptureDecision:
        """Capture user-verified outcomes with different strength by supervision type."""

        if str(feedback.source).strip().lower() == "reviewer":
            return CaptureDecision(
                to_session=True,
                to_long_term=False,
                source="reviewer_hint",
                trust=min(0.80, max(0.55, 0.45 + 0.35 * float(feedback.strength))),
                note="weak_reviewer_preference",
            )

        if feedback.accepted:
            to_long_term = (
                observation.ensemble_margin <= self.accepted_long_term_margin
                or observation.expert_disagreement >= self.accepted_long_term_disagreement
            )
            return CaptureDecision(
                to_session=True,
                to_long_term=to_long_term,
                source="feedback_accept",
                trust=0.90,
                note="accepted_by_operator",
            )

        return CaptureDecision(
            to_session=True,
            to_long_term=True,
            source="feedback_corrected",
            trust=1.00,
            note="corrected_by_operator",
        )

    def decide_auto(
        self,
        observation: MemoryObservation,
        fusion_result: FusionResult,
        stable: bool,
    ) -> Optional[CaptureDecision]:
        """Capture stable automatic outcomes into session memory only."""

        if not stable:
            return None
        if not observation.has_visual_evidence:
            return None
        if fusion_result.confidence < self.min_auto_capture_confidence:
            return None
        if fusion_result.step_id != observation.ensemble_step:
            return None
        return CaptureDecision(
            to_session=True,
            to_long_term=False,
            source="stable_auto",
            trust=min(0.85, max(0.55, fusion_result.confidence)),
            note="stable_auto",
        )
