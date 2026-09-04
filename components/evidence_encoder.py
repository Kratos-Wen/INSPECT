"""Encode current frame artifacts into compact evidence tokens for temporal reasoning."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional

from ..core_types import (
    Detection,
    EvidenceToken,
    InteractionEvidence,
    SceneEvidenceFrame,
    SceneGraphFrame,
    StepPrediction,
    TrackEvidenceFrame,
)


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
        track_evidence: Optional[TrackEvidenceFrame] = None,
        scene_evidence: Optional[SceneEvidenceFrame] = None,
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
        tracks = track_evidence or TrackEvidenceFrame(objects=[])
        scene = scene_evidence or SceneEvidenceFrame()
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
            track_counts={str(key): int(value) for key, value in tracks.track_counts.items()},
            stable_track_counts={str(key): int(value) for key, value in tracks.stable_track_counts.items()},
            track_ids={str(key): [int(item) for item in value] for key, value in tracks.track_ids.items()},
            track_confidences={str(key): float(value) for key, value in tracks.track_confidences.items()},
            track_hits={str(key): int(value) for key, value in tracks.track_hits.items()},
            track_ages={str(key): int(value) for key, value in tracks.track_ages.items()},
            track_motion={str(key): float(value) for key, value in tracks.track_motion.items()},
            track_evidence_keys=[str(item) for item in tracks.evidence_keys],
            track_objects=[dict(item) for item in tracks.objects[:24]],
            scene_evidence_keys=[str(item) for item in scene.evidence_keys],
            scene_visible_objects=[str(item) for item in scene.visible_objects],
            scene_stable_objects=[str(item) for item in scene.stable_objects],
            scene_active_objects=[str(item) for item in scene.active_objects],
            scene_moving_objects=[str(item) for item in scene.moving_objects],
            scene_relation_keys=[str(item) for item in scene.relation_keys],
            scene_relation_change_keys=[str(item) for item in scene.relation_change_keys],
            scene_transition_keys=[str(item) for item in scene.transition_keys],
        )

    def _dense_scores(self, prediction: StepPrediction) -> Dict[str, float]:
        return {step_id: float(prediction.scores.get(step_id, 0.0)) for step_id in self.steps}

    @staticmethod
    def _count_names(detections: Iterable[Detection]) -> Dict[str, int]:
        counts = Counter(str(detection.name).strip().lower() for detection in detections if str(detection.name).strip())
        return {str(key): int(value) for key, value in counts.items()}
