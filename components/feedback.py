"""Feedback providers for online adaptation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from ..core_types import EvidenceToken, FeedbackEvent, FusionResult, StepPrediction


class TimelineFeedbackProvider:
    """Programmatic feedback provider backed by ego-video state annotations.

    This is used for offline experiments where spoken state annotations are
    treated as simulated human corrections. Labels outside the legal step set
    are handled as no-step / invalid negative supervision.
    """

    request_on_unstable = True

    def __init__(
        self,
        segments: Iterable[dict[str, object]],
        steps: Iterable[str],
        no_step_labels: Iterable[str] = ("WRONG", "INVALID", "NO_STEP", "NEGATIVE"),
        min_interval_frames: int = 15,
        accept_strength: float = 0.35,
        correction_strength: float = 1.0,
        no_step_strength: float = 1.0,
    ) -> None:
        self.segments = [
            {
                "state": str(item.get("state", "")).strip().upper(),
                "start_frame": int(item.get("start_frame", 0) or 0),
                "end_frame": item.get("end_frame", None),
            }
            for item in segments
            if str(item.get("state", "")).strip()
        ]
        self.steps = {str(step).strip().upper() for step in steps}
        self.no_step_labels = {str(label).strip().upper() for label in no_step_labels}
        self.min_interval_frames = max(1, int(min_interval_frames))
        self.accept_strength = float(accept_strength)
        self.correction_strength = float(correction_strength)
        self.no_step_strength = float(no_step_strength)
        self._last_feedback_frame: Optional[int] = None

    def request(
        self,
        fusion_result: FusionResult,
        expert_predictions: dict[str, StepPrediction],
        evidence_token: Optional[EvidenceToken] = None,
        review_decision: Optional[object] = None,
    ) -> Optional[FeedbackEvent]:
        frame_index = int(getattr(evidence_token, "frame_index", 0) if evidence_token is not None else 0)
        if self._last_feedback_frame is not None and frame_index - self._last_feedback_frame < self.min_interval_frames:
            return None
        truth = self._truth_for_frame(frame_index)
        if not truth:
            return None
        predicted = str(fusion_result.step_id).strip().upper()
        self._last_feedback_frame = frame_index
        if truth in self.steps:
            accepted = predicted == truth
            return FeedbackEvent(
                label=truth,
                strength=self.accept_strength if accepted else self.correction_strength,
                accepted=accepted,
                source="gt_timeline",
                note=f"simulated_human_feedback;truth={truth}",
                extras={
                    "truth": truth,
                    "simulated_feedback": True,
                    "evidence_feedback": {
                        "status": "verified" if accepted else "corrected",
                        "reason": "",
                        "hypothesis": predicted,
                        "review_action": getattr(review_decision, "action", "") if review_decision is not None else "",
                    },
                },
            )
        if truth in self.no_step_labels or truth.startswith("WRONG") or truth.startswith("INVALID"):
            return FeedbackEvent(
                label="INVALID",
                strength=self.no_step_strength,
                accepted=False,
                source="gt_timeline",
                note=f"simulated_no_step_feedback;truth={truth}",
                extras={
                    "truth": truth,
                    "no_step": True,
                    "simulated_feedback": True,
                    "evidence_feedback": {
                        "status": "rejected",
                        "reason": "wrong_assembly",
                        "hypothesis": predicted,
                        "review_action": getattr(review_decision, "action", "") if review_decision is not None else "",
                    },
                },
            )
        return None

    def _truth_for_frame(self, frame_index: int) -> str:
        for item in self.segments:
            start = int(item["start_frame"])
            end_value = item.get("end_frame")
            end = None if end_value is None else int(end_value)
            if frame_index < start:
                continue
            if end is None or frame_index <= end:
                return str(item["state"]).strip().upper()
        return ""


class ConsoleFeedbackProvider:
    """Simple console feedback provider in English."""

    def __init__(self, evidence_prompt_enabled: bool = True) -> None:
        self.evidence_prompt_enabled = bool(evidence_prompt_enabled)

    def request(
        self,
        fusion_result: FusionResult,
        expert_predictions: dict[str, StepPrediction],
        evidence_token: Optional[EvidenceToken] = None,
        review_decision: Optional[object] = None,
    ) -> Optional[FeedbackEvent]:
        expert_summary = ", ".join(
            f"{name}={prediction.step_id}/{prediction.confidence:.2f}"
            for name, prediction in sorted(expert_predictions.items())
        )
        message = f"[Feedback] stable prediction: fused={fusion_result.step_id} ({fusion_result.confidence:.2f}), {expert_summary}"
        print(message)
        user_input = input(
            "[Feedback] ENTER accept | 1/2/3/4 correct step | s skip: "
        ).strip()
        if not user_input:
            return self._with_evidence_prompt(
                FeedbackEvent(label=fusion_result.step_id, accepted=True, source="console"),
                fusion_result=fusion_result,
                evidence_token=evidence_token,
                review_decision=review_decision,
            )
        if user_input.lower() in {"s", "skip"}:
            return None
        normalized_step = self._normalize_step(user_input)
        return self._with_evidence_prompt(
            FeedbackEvent(label=normalized_step, accepted=False, source="console"),
            fusion_result=fusion_result,
            evidence_token=evidence_token,
            review_decision=review_decision,
        )

    def _with_evidence_prompt(
        self,
        feedback: FeedbackEvent,
        fusion_result: FusionResult,
        evidence_token: Optional[EvidenceToken],
        review_decision: Optional[object],
    ) -> FeedbackEvent:
        if not self.evidence_prompt_enabled:
            return feedback
        evidence_summary = self._evidence_summary(evidence_token)
        if not evidence_summary:
            return feedback
        print("[Evidence] observed candidates:")
        for line in evidence_summary[:6]:
            print(f"  - {line}")
        answer = input(
            "[Evidence] verify postcondition? [y] yes / [n] no / [o] occluded / ENTER unknown: "
        ).strip().lower()
        evidence_status = {"y": "verified", "n": "rejected", "o": "occluded", "": "unknown"}.get(answer, "unknown")
        reason = ""
        if evidence_status in {"rejected", "occluded"}:
            reason = input(
                "[Evidence] reason: 1 partial / 2 wrong_object / 3 wrong_order / 4 missing / 5 occluded / text: "
            ).strip()
            reason = self._normalize_reason(reason)
        extras: Dict[str, Any] = dict(feedback.extras)
        extras["evidence_feedback"] = {
            "status": evidence_status,
            "reason": reason,
            "hypothesis": str(fusion_result.step_id).strip().upper(),
            "observed_evidence": list(evidence_summary),
            "review_action": getattr(review_decision, "action", "") if review_decision is not None else "",
            "review_reason": getattr(review_decision, "reason", "") if review_decision is not None else "",
        }
        note_parts = [feedback.note] if feedback.note else []
        note_parts.append(f"evidence_status={evidence_status}")
        if reason:
            note_parts.append(f"evidence_reason={reason}")
        return FeedbackEvent(
            label=feedback.label,
            strength=feedback.strength,
            accepted=feedback.accepted,
            source=feedback.source,
            note=";".join(note_parts),
            extras=extras,
        )

    @staticmethod
    def _normalize_reason(value: str) -> str:
        mapping = {
            "1": "partial",
            "2": "wrong_object",
            "3": "wrong_order",
            "4": "missing",
            "5": "occluded",
        }
        return mapping.get(value.strip().lower(), value.strip().lower().replace(" ", "_"))

    @staticmethod
    def _normalize_step(value: str) -> str:
        text = value.strip().upper()
        if text.isdigit():
            return f"S{text}"
        if len(text) == 1 and text in {"A", "B", "C", "D"}:
            return f"S{ord(text) - ord('A') + 1}"
        return text

    @staticmethod
    def _evidence_summary(token: Optional[EvidenceToken]) -> list[str]:
        if token is None:
            return []
        items: list[str] = []
        for subject, predicate, obj in token.relation_facts[:6]:
            items.append(f"relation:{subject}:{predicate}:{obj}")
        for subject, predicate, obj in token.contact_facts[:4]:
            items.append(f"contact:{subject}:{predicate}:{obj}")
        if token.active_object:
            items.append(f"active_object:{token.active_object}")
        if token.interaction_target:
            items.append(f"interaction_target:{token.interaction_target}")
        return items
