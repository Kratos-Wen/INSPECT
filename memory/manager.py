"""Lifecycle manager for episodic memory recall and capture."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from ..config import MemoryConfig
from ..types import Detection, FeedbackEvent, FusionResult, GeometryFrame, SceneGraphFrame, StepPrediction
from ..components.visual_embedding import SharedVisualEncoder
from .capture import CaptureDecision, VerifiedOutcomeCapturePolicy
from .features import (
    StructuredMemoryEncoder,
    detection_signature,
    relation_signature,
    summarize_geometry,
    summarize_step_consensus,
)
from .prior import WeightedMemoryPrior
from .recall_policy import UncertaintyRecallPolicy
from .retrieval import EventMemoryRetriever
from .store import InMemoryEventStore, JsonlEventStore
from .types import MemoryObservation, MemoryRecallResult, MemoryRecord


class MemoryLifecycleManager:
    """Own episodic recall and capture without coupling it to the hot path logic."""

    def __init__(
        self,
        steps: List[str],
        component_names: List[str],
        expert_names: List[str],
        config: MemoryConfig,
        long_term_path: Path,
        visual_encoder: Optional[SharedVisualEncoder] = None,
    ) -> None:
        self.steps = [str(step).strip().upper() for step in steps]
        self.component_names = [str(name).strip().lower() for name in component_names]
        self.expert_names = [str(name).strip().lower() for name in expert_names]
        self.config = config
        self.long_term_path = Path(long_term_path)
        self.visual_encoder = visual_encoder
        self.encoder = StructuredMemoryEncoder(self.steps, self.component_names, self.expert_names)
        self.recall_policy = UncertaintyRecallPolicy(
            margin_threshold=config.recall_margin,
            disagreement_threshold=config.recall_disagreement,
            cooldown_frames=config.recall_cooldown,
            recall_on_transition=config.recall_on_transition,
        )
        self.capture_policy = VerifiedOutcomeCapturePolicy(
            min_auto_capture_confidence=config.min_auto_capture_confidence,
            accepted_long_term_margin=config.accepted_long_term_margin,
            accepted_long_term_disagreement=config.accepted_long_term_disagreement,
        )
        self.retriever = EventMemoryRetriever(
            topk=config.topk,
            max_per_source=config.max_per_source,
            vector_weight=config.vector_weight,
            token_weight=config.token_weight,
            prev_step_bonus=config.prev_step_bonus,
            recency_half_life_sec=config.recency_half_life_sec,
            source_weights={
                "session": config.session_source_weight,
                "long_term": config.long_term_source_weight,
            },
        )
        self.prior_builder = WeightedMemoryPrior(
            self.steps,
            corrected_source_gain=config.corrected_source_gain,
            accepted_source_gain=config.accepted_source_gain,
            auto_source_gain=config.auto_source_gain,
        )
        self.long_term_store = (
            JsonlEventStore(self.long_term_path, dedup_similarity=config.dedup_similarity)
            if config.enabled and config.long_term_enabled
            else None
        )
        self.session_store = InMemoryEventStore(dedup_similarity=config.dedup_similarity)
        self.run_id = "unbound"
        self.last_recall_frame: Optional[int] = None

    def attach_run(self, run_dir: Path) -> None:
        """Attach a concrete run directory for session-memory persistence."""

        self.run_id = run_dir.name
        if self.config.enabled and self.config.session_enabled:
            self.session_store.close()
            self.session_store = JsonlEventStore(
                run_dir / "session_memory.jsonl",
                dedup_similarity=self.config.dedup_similarity,
            )
        else:
            self.session_store = InMemoryEventStore(dedup_similarity=self.config.dedup_similarity)

    def build_observation(
        self,
        frame_index: int,
        prev_step: Optional[str],
        detections: List[Detection],
        relevant_detections: List[Detection],
        nearest_index: Optional[int],
        geometry: GeometryFrame,
        frame_bgr,
        scene_graph: Optional[SceneGraphFrame],
        expert_predictions: Dict[str, StepPrediction],
    ) -> MemoryObservation:
        """Build a structured observation from current perception evidence."""

        filtered_predictions = {
            name: prediction for name, prediction in expert_predictions.items() if name in self.expert_names
        }
        focus_detection = detections[nearest_index] if nearest_index is not None and 0 <= nearest_index < len(detections) else None
        has_visual_evidence = bool(detections or relevant_detections or relation_signature(scene_graph))
        visual_embedding: List[float] = []
        if self.visual_encoder is not None and has_visual_evidence:
            visual_embedding = [
                float(value)
                for value in self.visual_encoder.encode_query(frame_bgr, focus_detection=focus_detection).tolist()
            ]
        consensus = summarize_step_consensus(filtered_predictions, self.steps)
        return MemoryObservation(
            frame_index=int(frame_index),
            prev_step=str(prev_step).strip().upper() if prev_step else None,
            signature=detection_signature(detections),
            relevant_signature=detection_signature(relevant_detections),
            relation_signature=relation_signature(scene_graph),
            geometry_stats=summarize_geometry(geometry, detections, relevant_detections, nearest_index),
            scene_graph_stats=dict(scene_graph.stats) if scene_graph is not None else {},
            expert_steps={
                name: prediction.step_id.strip().upper()
                for name, prediction in filtered_predictions.items()
            },
            expert_confidences={
                name: float(prediction.confidence)
                for name, prediction in filtered_predictions.items()
            },
            expert_scores={
                name: {step_id: float(prediction.scores.get(step_id, 0.0)) for step_id in self.steps}
                for name, prediction in filtered_predictions.items()
            },
            ensemble_step=str(consensus["step_id"]),
            ensemble_confidence=float(consensus["confidence"]),
            ensemble_margin=float(consensus["margin"]),
            expert_disagreement=float(consensus["disagreement"]),
            ensemble_scores={str(key): float(value) for key, value in dict(consensus["scores"]).items()},
            num_detections=len(detections),
            num_relevant=len(relevant_detections),
            has_visual_evidence=has_visual_evidence,
            visual_embedding=visual_embedding,
        )

    def recall(self, observation: MemoryObservation) -> MemoryRecallResult:
        """Recall similar past events and convert them into a dense memory prior."""

        fallback = self._empty_prediction(observation.ensemble_step, reason="disabled")
        if not self.config.enabled:
            return MemoryRecallResult(prediction=fallback, recalled=False, reason="disabled")

        has_session = self.config.session_enabled and self.session_store.count() > 0
        has_long_term = self.long_term_store is not None and self.long_term_store.count() > 0
        decision = self.recall_policy.decide(
            observation,
            last_recall_frame=self.last_recall_frame,
            has_session=has_session,
            has_long_term=has_long_term,
        )
        if not decision.should_recall:
            return MemoryRecallResult(
                prediction=self._empty_prediction(observation.ensemble_step, reason=decision.reason),
                recalled=False,
                reason=decision.reason,
            )

        stores = {}
        if has_session:
            stores["session"] = self.session_store
        if has_long_term and self.long_term_store is not None:
            stores["long_term"] = self.long_term_store

        query_vector = self.encoder.encode(observation)
        query_tokens = self.encoder.tokens(observation)
        matches = self.retriever.search(observation, query_vector, query_tokens, stores=stores)
        prediction = self.prior_builder.build(matches, fallback_step=observation.ensemble_step, reason=decision.reason)
        self.last_recall_frame = observation.frame_index
        return MemoryRecallResult(
            prediction=prediction,
            matches=matches,
            recalled=bool(matches),
            reason=decision.reason,
        )

    def record_feedback(
        self,
        observation: MemoryObservation,
        feedback: FeedbackEvent,
    ) -> None:
        """Persist a verified feedback event into the configured memory stores."""

        if not self.config.enabled:
            return
        decision = self.capture_policy.decide_feedback(observation, feedback)
        self._capture(observation, label=feedback.label, decision=decision, accepted=feedback.accepted)

    def record_auto(
        self,
        observation: MemoryObservation,
        fusion_result: FusionResult,
        stable: bool,
    ) -> None:
        """Persist a stable automatic event into session memory when appropriate."""

        if not self.config.enabled:
            return
        if not self.config.auto_capture_enabled:
            return
        decision = self.capture_policy.decide_auto(observation, fusion_result, stable=stable)
        if decision is None:
            return
        self._capture(observation, label=fusion_result.step_id, decision=decision, accepted=False)

    def close(self) -> None:
        """Flush and close all backing stores."""

        self.session_store.close()
        if self.long_term_store is not None:
            self.long_term_store.close()

    def _capture(
        self,
        observation: MemoryObservation,
        label: str,
        decision: CaptureDecision,
        accepted: bool,
    ) -> None:
        record = self._build_record(
            observation=observation,
            label=label,
            source=decision.source,
            trust=decision.trust,
            note=decision.note,
            accepted=accepted,
        )
        if decision.to_session and self.config.session_enabled:
            self.session_store.append(record)
        if decision.to_long_term and self.long_term_store is not None:
            self.long_term_store.append(record)

    def _build_record(
        self,
        observation: MemoryObservation,
        label: str,
        source: str,
        trust: float,
        note: str,
        accepted: bool,
    ) -> MemoryRecord:
        vector = self.encoder.encode(observation)
        tokens = self.encoder.tokens(observation)
        return MemoryRecord(
            record_id=uuid4().hex,
            run_id=self.run_id,
            frame_index=int(observation.frame_index),
            timestamp=time.time(),
            step_id=str(label).strip().upper(),
            prev_step=observation.prev_step,
            source=source,
            trust=float(trust),
            accepted=bool(accepted),
            note=note,
            signature=observation.signature,
            relevant_signature=observation.relevant_signature,
            relation_signature=observation.relation_signature,
            geometry_stats=dict(observation.geometry_stats),
            scene_graph_stats=dict(observation.scene_graph_stats),
            expert_steps=dict(observation.expert_steps),
            expert_confidences=dict(observation.expert_confidences),
            expert_scores={expert: dict(scores) for expert, scores in observation.expert_scores.items()},
            num_detections=int(observation.num_detections),
            num_relevant=int(observation.num_relevant),
            has_visual_evidence=bool(observation.has_visual_evidence),
            visual_embedding=list(observation.visual_embedding),
            vector=[float(value) for value in vector.tolist()],
            tokens=list(tokens),
        )

    def _empty_prediction(self, fallback_step: str, reason: str) -> StepPrediction:
        return StepPrediction(
            step_id=fallback_step,
            confidence=0.0,
            scores={step_id: 0.0 for step_id in self.steps},
            extras={"active": False, "reason": reason, "matches": []},
        )
