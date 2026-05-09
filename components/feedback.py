"""Feedback providers for online adaptation."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..types import EvidenceToken, FeedbackEvent, FusionResult, StepPrediction


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
            "[Feedback] press ENTER to accept, type S1/S2/... to correct, or 'skip' to ignore: "
        ).strip()
        if not user_input:
            return self._with_evidence_prompt(
                FeedbackEvent(label=fusion_result.step_id, accepted=True, source="console"),
                fusion_result=fusion_result,
                evidence_token=evidence_token,
                review_decision=review_decision,
            )
        if user_input.lower() == "skip":
            return None
        return self._with_evidence_prompt(
            FeedbackEvent(label=user_input.upper(), accepted=False, source="console"),
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
