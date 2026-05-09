"""Encode current frame artifacts into compact evidence tokens for temporal reasoning."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional

from ..types import Detection, EvidenceToken, InteractionEvidence, SceneGraphFrame, StepPrediction


class StructuredEvidenceEncoder:
    """Build reusable temporal evidence tokens from already-computed frame artifacts."""

    def __init__(self, steps: Iterable[str]) -> None:
        self.steps = [str(step).strip().upper() for step in steps]

    def encode(
        self,
        frame_index: int,
        prev_step: Optional[str],
        detections: List[Detection],
        relevant_detections: List[Detection],
        scene_graph: SceneGraphFrame,
        state_prediction: StepPrediction,
        retrieval_prediction: StepPrediction,
        memory_prediction: StepPrediction,
        has_visual_evidence: bool,
        review_action: str = "",
        review_reason: str = "",
        interaction_evidence: Optional[InteractionEvidence] = None,
    ) -> EvidenceToken:
        """Construct one compact evidence token from existing perception outputs."""

        relation_facts = [
            (
                str(relation.subject_name).strip().lower(),
                str(relation.predicate).strip().lower(),
                str(relation.object_name).strip().lower(),
            )
            for relation in getattr(scene_graph, "relations", [])[:16]
        ]
        relation_counts = Counter(predicate for _, predicate, _ in relation_facts)
        interaction = interaction_evidence or InteractionEvidence(contacts=[])
        return EvidenceToken(
            frame_index=int(frame_index),
            prev_step=str(prev_step).strip().upper() if prev_step else None,
            visible_counts=self._count_names(detections),
            relevant_counts=self._count_names(relevant_detections),
            relation_counts={str(key): int(value) for key, value in relation_counts.items()},
            relation_facts=relation_facts,
            state_scores=self._dense_scores(state_prediction),
            retrieval_scores=self._dense_scores(retrieval_prediction),
            memory_scores=self._dense_scores(memory_prediction),
            state_confidence=float(state_prediction.confidence),
            retrieval_confidence=float(retrieval_prediction.confidence),
            memory_confidence=float(memory_prediction.confidence),
            memory_active=bool(memory_prediction.extras.get("active", False)),
            memory_reason=str(memory_prediction.extras.get("reason", "")).strip(),
            review_action=str(review_action).strip().lower(),
            review_reason=str(review_reason).strip(),
            has_visual_evidence=bool(has_visual_evidence),
            hand_object_contacts=[asdict(contact) for contact in interaction.contacts[:16]],
            contact_counts={str(key): int(value) for key, value in interaction.contact_counts.items()},
            contact_facts=[
                (str(subject).strip().lower(), str(predicate).strip().lower(), str(obj).strip().lower())
                for subject, predicate, obj in interaction.contact_facts[:16]
            ],
            active_object=str(interaction.active_object).strip().lower(),
            interaction_target=str(interaction.interaction_target).strip().lower(),
            contact_phase=str(interaction.contact_phase).strip().lower(),
            transition_likelihood=float(interaction.transition_likelihood),
        )

    def _dense_scores(self, prediction: StepPrediction) -> Dict[str, float]:
        return {step_id: float(prediction.scores.get(step_id, 0.0)) for step_id in self.steps}

    @staticmethod
    def _count_names(detections: Iterable[Detection]) -> Dict[str, int]:
        counts = Counter(str(detection.name).strip().lower() for detection in detections if str(detection.name).strip())
        return {str(key): int(value) for key, value in counts.items()}
