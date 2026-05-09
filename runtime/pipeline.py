"""Pipeline assembly and runtime loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..components import (
    ByteTrackLiteFusion,
    CausalTemporalStepExpert,
    CompiledStepGraphPrior,
    ConsoleFeedbackProvider,
    DepthContextSelector,
    DummyRetrievalExpert,
    DualStabilityTracker,
    EvidenceTimelineStore,
    GeometryAwareSceneGraphBuilder,
    GradientGeometryProvider,
    HandObjectContactEstimator,
    IdentityStepCalibrator,
    JsonlCsvLogger,
    KnowledgeBase,
    LinearMarginFusion,
    MoGeGeometryProvider,
    RuleBasedStepExpert,
    SharedVisualEncoder,
    StreamingGRUTemporalStepExpert,
    StructuredEvidenceEncoder,
    TemporalEvidenceVectorizer,
    WindowIoUFusion,
    YOLODetectorComponent,
    build_segmentation_backend,
)
from ..components.retrieval import GalleryRetrievalExpert
from ..config import AppConfig
from ..events import JsonlEventBus
from ..interfaces import (
    ContextSelector,
    Detector,
    FeedbackProvider,
    GeometryProvider,
    SceneGraphBuilder,
    StabilityPolicy,
    StepCalibrator,
    StepExpert,
    StepFusion,
    TemporalFusion,
)
from ..memory import MemoryLifecycleManager
from ..memory.types import MemoryObservation, MemoryRecallResult
from ..ops import OpsStateTracker
from ..review import ReviewDecision, ReviewManager, SparseReviewPolicy, SparseReviewerAgent
from ..types import (
    Detection,
    EvidenceToken,
    FeedbackEvent,
    FusionResult,
    GeometryFrame,
    InteractionEvidence,
    RetrievalInput,
    SceneGraphFrame,
    StepPrediction,
)


def _serialize_detection(detection: Detection) -> dict[str, object]:
    return {"name": detection.name, "xyxy": [float(v) for v in detection.xyxy], "confidence": float(detection.confidence)}


def _serialize_evidence_token(token: EvidenceToken) -> dict[str, object]:
    return {
        "frame_index": int(token.frame_index),
        "prev_step": token.prev_step,
        "visible_counts": {str(key): int(value) for key, value in token.visible_counts.items()},
        "relevant_counts": {str(key): int(value) for key, value in token.relevant_counts.items()},
        "relation_counts": {str(key): int(value) for key, value in token.relation_counts.items()},
        "state_scores": {str(key): float(value) for key, value in token.state_scores.items()},
        "retrieval_scores": {str(key): float(value) for key, value in token.retrieval_scores.items()},
        "memory_scores": {str(key): float(value) for key, value in token.memory_scores.items()},
        "state_confidence": float(token.state_confidence),
        "retrieval_confidence": float(token.retrieval_confidence),
        "memory_confidence": float(token.memory_confidence),
        "memory_active": bool(token.memory_active),
        "memory_reason": str(token.memory_reason),
        "review_action": str(token.review_action),
        "review_reason": str(token.review_reason),
        "has_visual_evidence": bool(token.has_visual_evidence),
        "hand_object_contacts": list(token.hand_object_contacts),
        "contact_counts": {str(key): int(value) for key, value in token.contact_counts.items()},
        "contact_facts": [list(item) for item in token.contact_facts],
        "active_object": str(token.active_object),
        "interaction_target": str(token.interaction_target),
        "contact_phase": str(token.contact_phase),
        "transition_likelihood": float(token.transition_likelihood),
    }


def _candidate_gallery_roots() -> list[Path]:
    package_dir = Path(__file__).resolve().parents[1]
    workspace_root = package_dir.parent
    return [package_dir / "gallery", workspace_root / "test" / "after_online", workspace_root / "test" / "before_online"]


def _looks_like_gallery(root: Path, steps: list[str], exts: list[str]) -> bool:
    if not root.exists() or not root.is_dir():
        return False
    ext_set = {str(ext).lower() for ext in exts}
    step_set = {str(step).strip().upper() for step in steps}
    for child in root.iterdir():
        if not child.is_dir():
            continue
        upper = child.name.strip().upper()
        if upper in step_set or upper.startswith("STEP"):
            if any(path.is_file() and path.suffix.lower() in ext_set for path in child.rglob("*")):
                return True
    return False


def _resolve_gallery_root(config: AppConfig, steps: list[str]) -> Optional[str]:
    explicit = str(config.experts.gallery_root).strip()
    if explicit:
        path = Path(explicit)
        return str(path) if _looks_like_gallery(path, steps, config.experts.gallery_exts) else None
    for candidate in _candidate_gallery_roots():
        if _looks_like_gallery(candidate, steps, config.experts.gallery_exts):
            return str(candidate)
    return None


def _draw_overlay(frame_bgr: np.ndarray, detections: list[Detection], step_text: str) -> np.ndarray:
    canvas = frame_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection.xyxy]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f"{detection.name} {detection.confidence:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(canvas, step_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 200, 255), 2, cv2.LINE_AA)
    return canvas


def _feedback_log_payload(
    processed_index: int,
    frame_index: int,
    feedback: FeedbackEvent,
    fusion_result,
    memory_step: str,
    memory_conf: float,
) -> dict[str, object]:
    return {
        "iter": processed_index,
        "frame_index": frame_index,
        "feedback": {
            "label": feedback.label,
            "accepted": feedback.accepted,
            "strength": feedback.strength,
            "source": feedback.source,
            "note": feedback.note,
            "extras": dict(feedback.extras),
        },
        "before": {
            "fused_step": fusion_result.step_id,
            "fused_conf": fusion_result.confidence,
            "gate_state": float(fusion_result.gates.get("state", 0.0)),
            "gate_temporal": float(fusion_result.gates.get("temporal", 0.0)),
            "gate_retrieval": float(fusion_result.gates.get("retrieval", 0.0)),
            "gate_memory": float(fusion_result.gates.get("memory", 0.0)),
            "memory_step": memory_step,
            "memory_conf": memory_conf,
        },
    }


@dataclass(frozen=True)
class ProcessedFrame:
    """Shared perception and fusion artifacts for one processed frame."""

    frame_index: int
    raw_detections: list[Detection]
    fused_detections: list[Detection]
    relevant_detections: list[Detection]
    nearest_index: Optional[int]
    signature: tuple[tuple[str, int], ...]
    stable: bool
    geometry: GeometryFrame
    scene_graph: SceneGraphFrame
    evidence_token: EvidenceToken
    state_prediction: StepPrediction
    temporal_prediction: StepPrediction
    retrieval_prediction: StepPrediction
    memory_observation: MemoryObservation
    memory_recall: MemoryRecallResult
    interaction_evidence: InteractionEvidence
    expert_predictions: dict[str, StepPrediction]
    fusion_result: FusionResult
    review_decision: Optional[ReviewDecision]


class StepPipeline:
    """End-to-end modular step-recognition pipeline."""

    def __init__(
        self,
        config: AppConfig,
        steps: list[str],
        detector: Detector,
        geometry_provider: GeometryProvider,
        temporal_fusion: TemporalFusion,
        context_selector: ContextSelector,
        scene_graph_builder: SceneGraphBuilder,
        evidence_encoder,
        state_expert: StepExpert,
        temporal_expert: StepExpert,
        retrieval_expert: StepExpert,
        state_calibrator: StepCalibrator,
        temporal_calibrator: StepCalibrator,
        retrieval_calibrator: StepCalibrator,
        fusion_head: StepFusion,
        stability_policy: StabilityPolicy,
        feedback_provider: Optional[FeedbackProvider],
        runlog_root: str,
        kb: KnowledgeBase,
        memory_manager: MemoryLifecycleManager,
        review_manager: ReviewManager,
        interaction_estimator: HandObjectContactEstimator,
        segmentation_backend,
        event_bus: JsonlEventBus,
        ops_tracker: OpsStateTracker,
        component_status: Optional[dict[str, object]] = None,
    ) -> None:
        self.config = config
        self.steps = [str(step).strip().upper() for step in steps]
        self.detector = detector
        self.geometry_provider = geometry_provider
        self.temporal_fusion = temporal_fusion
        self.context_selector = context_selector
        self.scene_graph_builder = scene_graph_builder
        self.evidence_encoder = evidence_encoder
        self.state_expert = state_expert
        self.temporal_expert = temporal_expert
        self.retrieval_expert = retrieval_expert
        self.state_calibrator = state_calibrator
        self.temporal_calibrator = temporal_calibrator
        self.retrieval_calibrator = retrieval_calibrator
        self.fusion_head = fusion_head
        self.stability_policy = stability_policy
        self.feedback_provider = feedback_provider
        self.runlog_root = runlog_root
        self.kb = kb
        self.memory_manager = memory_manager
        self.review_manager = review_manager
        self.interaction_estimator = interaction_estimator
        self.segmentation_backend = segmentation_backend
        self.event_bus = event_bus
        self.ops_tracker = ops_tracker
        self.component_status = dict(component_status or {})

    def prepare_run(self, logger: JsonlCsvLogger, source_uri: str) -> None:
        """Attach per-run sinks and persist run metadata."""

        logger.log_meta({"video_path": source_uri, "config": asdict(self.config), "components": self.component_status})
        reset_temporal = getattr(self.temporal_expert, "reset", None)
        if callable(reset_temporal):
            reset_temporal()
        self.memory_manager.attach_run(logger.run_dir)
        self.event_bus.attach_run(logger.run_dir)
        self.ops_tracker.attach_run(logger.run_dir)
        self.event_bus.emit("run.started", {"video_path": source_uri, "run_dir": str(logger.run_dir)})
        if self.component_status:
            self.event_bus.emit("components.ready", dict(self.component_status))
        self.ops_tracker.update(status="Doing", frame_index=0, step_id="", reason="run_started", force=True)
        if getattr(self.fusion_head, "state_path", None) is None:
            self.fusion_head.state_path = logger.run_dir / "online_fusion_state.json"  # type: ignore[attr-defined]

    def finalize_run(self, logger: JsonlCsvLogger, source_uri: str) -> None:
        """Close all run-bound resources and persist final state."""

        self.fusion_head.save(self.fusion_head.state_path)  # type: ignore[arg-type, attr-defined]
        self.memory_manager.close()
        self.event_bus.emit("run.finished", {"video_path": source_uri, "run_dir": str(logger.run_dir)})
        self.event_bus.close()
        self.ops_tracker.close()
        logger.close()

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        prev_step_for_transition: Optional[str],
        source_kind: str = "video",
    ) -> ProcessedFrame:
        """Run the shared perception, memory, and fusion stack on one frame."""

        det_conf, det_tta = self._detection_params(source_kind)
        raw_detections = self.detector.detect(frame_bgr, conf=det_conf, tta=det_tta)
        canonical_detections = self.kb.canonicalize(raw_detections)
        fused_detections = self.temporal_fusion.update(canonical_detections)
        geometry = self.geometry_provider.infer(frame_bgr)
        relevant_detections, nearest_index = self.context_selector.select(fused_detections, geometry)
        scene_graph = self.scene_graph_builder.build(
            detections=fused_detections,
            geometry=geometry,
            relevant_detections=relevant_detections,
            nearest_index=nearest_index,
        )
        segmentation_prompts = sorted(
            {
                *[str(detection.name).strip().lower() for detection in fused_detections if str(detection.name).strip()],
                *[str(name).strip().lower() for name in self.config.segmentation.prompts if str(name).strip()],
            }
        )
        segmentation_masks = self.segmentation_backend.segment(frame_bgr, segmentation_prompts)
        interaction_evidence = self.interaction_estimator.estimate(
            detections=fused_detections,
            geometry=geometry,
            scene_graph=scene_graph,
            relevant_detections=relevant_detections,
            segmentation_masks=segmentation_masks,
        )

        has_visual_evidence = bool(fused_detections or relevant_detections or scene_graph.relations)
        state_input = relevant_detections if (self.config.depth_context.use_relevant_for_rules and relevant_detections) else fused_detections
        state_prediction = self.state_calibrator.calibrate(self.state_expert.predict(state_input))
        focus_detection = fused_detections[nearest_index] if nearest_index is not None and 0 <= nearest_index < len(fused_detections) else None
        if has_visual_evidence:
            retrieval_prediction = self.retrieval_calibrator.calibrate(
                self.retrieval_expert.predict(
                    RetrievalInput(
                        frame_bgr=frame_bgr,
                        focus_detection=focus_detection,
                    )
                )
            )
        else:
            retrieval_prediction = self._empty_prediction(reason="no_visual_evidence", fallback_step=prev_step_for_transition)

        base_predictions = {"state": state_prediction, "retrieval": retrieval_prediction}
        memory_observation = self.memory_manager.build_observation(
            frame_index=frame_index,
            prev_step=prev_step_for_transition,
            detections=fused_detections,
            relevant_detections=relevant_detections,
            nearest_index=nearest_index,
            geometry=geometry,
            frame_bgr=frame_bgr,
            scene_graph=scene_graph,
            expert_predictions=base_predictions,
        )
        memory_recall = self.memory_manager.recall(memory_observation)
        if memory_recall.recalled:
            self.event_bus.emit(
                "memory.recalled",
                {
                    "reason": memory_recall.reason,
                    "step_id": memory_recall.prediction.step_id,
                    "confidence": memory_recall.prediction.confidence,
                    "matches": [
                        {"store": match.store_name, "step_id": match.record.step_id, "score": match.total_score}
                        for match in memory_recall.matches[:3]
                    ],
                },
                frame_index=frame_index,
            )

        evidence_token = self.evidence_encoder.encode(
            frame_index=frame_index,
            prev_step=prev_step_for_transition,
            detections=fused_detections,
            relevant_detections=relevant_detections,
            scene_graph=scene_graph,
            state_prediction=state_prediction,
            retrieval_prediction=retrieval_prediction,
            memory_prediction=memory_recall.prediction,
            has_visual_evidence=memory_observation.has_visual_evidence,
            review_action="",
            review_reason="",
            interaction_evidence=interaction_evidence,
        )
        temporal_prediction = self.temporal_calibrator.calibrate(self.temporal_expert.predict(evidence_token))

        expert_predictions = {
            **base_predictions,
            "temporal": temporal_prediction,
            "memory": memory_recall.prediction,
        }
        fusion_result = self.fusion_head.fuse(
            expert_predictions=expert_predictions,
            prev_step=prev_step_for_transition,
            token=evidence_token,
        )
        stable, signature = self.stability_policy.update(fused_detections, fusion_result.step_id)
        if not memory_observation.has_visual_evidence:
            stable = False

        review_decision: Optional[ReviewDecision] = None
        if stable and memory_observation.has_visual_evidence:
            review_decision = self.review_manager.consider(
                frame_index=frame_index,
                prev_step=prev_step_for_transition,
                stable=stable,
                fusion_result=fusion_result,
                expert_predictions=expert_predictions,
                observation=memory_observation,
                memory_recall=memory_recall,
            )
            if review_decision is not None:
                self.ops_tracker.record_review(review_decision.action)
                self.event_bus.emit(
                    "review.requested",
                    {
                        "fused_step": fusion_result.step_id,
                        "fused_conf": fusion_result.confidence,
                        "trigger_reasons": list(review_decision.trigger_reasons),
                    },
                    frame_index=frame_index,
                )
                self.event_bus.emit(
                    "review.decision",
                    {
                        "action": review_decision.action,
                        "label": review_decision.label,
                        "reason": review_decision.reason,
                        "trigger_reasons": list(review_decision.trigger_reasons),
                        "extras": dict(review_decision.extras),
                    },
                    frame_index=frame_index,
                )

        return ProcessedFrame(
            frame_index=frame_index,
            raw_detections=raw_detections,
            fused_detections=fused_detections,
            relevant_detections=relevant_detections,
            nearest_index=nearest_index,
            signature=signature,
            stable=stable,
            geometry=geometry,
            scene_graph=scene_graph,
            evidence_token=evidence_token,
            state_prediction=state_prediction,
            temporal_prediction=temporal_prediction,
            retrieval_prediction=retrieval_prediction,
            memory_observation=memory_observation,
            memory_recall=memory_recall,
            interaction_evidence=interaction_evidence,
            expert_predictions=expert_predictions,
            fusion_result=fusion_result,
            review_decision=review_decision,
        )

    def build_iteration_payload(
        self,
        processed_index: int,
        frame: ProcessedFrame,
        source_kind: str = "video",
    ) -> dict[str, object]:
        """Build one structured iteration payload with configurable detail."""

        detail_level = self._iteration_detail_level(source_kind)
        review_decision = frame.review_decision
        payload: dict[str, object] = {
            "iter": processed_index,
            "frame_index": frame.frame_index,
            "source_kind": source_kind,
            "num_raw": len(frame.raw_detections),
            "num_fused": len(frame.fused_detections),
            "num_relevant": len(frame.relevant_detections),
            "stable": bool(frame.stable),
            "has_visual_evidence": bool(frame.memory_observation.has_visual_evidence),
            "nearest_index": frame.nearest_index,
            "signature": list(frame.signature),
            "state_step": frame.state_prediction.step_id,
            "state_conf": frame.state_prediction.confidence,
            "temporal_step": frame.temporal_prediction.step_id,
            "temporal_conf": frame.temporal_prediction.confidence,
            "retrieval_step": frame.retrieval_prediction.step_id,
            "retrieval_conf": frame.retrieval_prediction.confidence,
            "memory_step": frame.memory_recall.prediction.step_id,
            "memory_conf": frame.memory_recall.prediction.confidence,
            "memory_active": bool(frame.memory_recall.prediction.extras.get("active", False)),
            "memory_reason": frame.memory_recall.reason,
            "memory_recalled": bool(frame.memory_recall.recalled),
            "ensemble_step": frame.memory_observation.ensemble_step,
            "ensemble_conf": frame.memory_observation.ensemble_confidence,
            "ensemble_margin": frame.memory_observation.ensemble_margin,
            "expert_disagreement": frame.memory_observation.expert_disagreement,
            "fused_step": frame.fusion_result.step_id,
            "fused_conf": frame.fusion_result.confidence,
            "fused_runner_up": frame.fusion_result.runner_up,
            "gate_state": float(frame.fusion_result.gates.get("state", 0.0)),
            "gate_temporal": float(frame.fusion_result.gates.get("temporal", 0.0)),
            "gate_retrieval": float(frame.fusion_result.gates.get("retrieval", 0.0)),
            "gate_memory": float(frame.fusion_result.gates.get("memory", 0.0)),
            "gates": dict(frame.fusion_result.gates),
            "review_action": review_decision.action if review_decision is not None else "",
            "review_label": review_decision.label if review_decision is not None else "",
            "review_reason": review_decision.reason if review_decision is not None else "",
            "review_triggers": list(review_decision.trigger_reasons) if review_decision is not None else [],
        }
        if bool(self.config.runlog.persist_temporal_tokens):
            payload["evidence_token"] = _serialize_evidence_token(frame.evidence_token)
        if detail_level != "sparse":
            payload.update(
                {
                    "geometry": dict(frame.geometry.extras),
                    "scene_graph_stats": dict(frame.scene_graph.stats),
                    "memory_matches": [
                        {
                            "store": match.store_name,
                            "step_id": match.record.step_id,
                            "source": match.record.source,
                            "score": match.total_score,
                        }
                        for match in frame.memory_recall.matches[:3]
                    ],
                    "state_scores": frame.state_prediction.scores,
                    "temporal_scores": frame.temporal_prediction.scores,
                    "retrieval_scores": frame.retrieval_prediction.scores,
                    "memory_scores": frame.memory_recall.prediction.scores,
                    "fusion_scores": frame.fusion_result.scores,
                    "interaction_evidence": {
                        "active_object": frame.interaction_evidence.active_object,
                        "interaction_target": frame.interaction_evidence.interaction_target,
                        "contact_phase": frame.interaction_evidence.contact_phase,
                        "transition_likelihood": frame.interaction_evidence.transition_likelihood,
                        "contact_counts": dict(frame.interaction_evidence.contact_counts),
                        "segmentation_backend": getattr(self.segmentation_backend, "backend_name", "unknown"),
                    },
                    "base_gates": dict(frame.fusion_result.extras.get("base_gates", {})),
                    "context_gate_features": dict(frame.fusion_result.extras.get("context_gate_features", {})),
                }
            )
        if detail_level == "debug":
            payload.update(
                {
                    "scene_graph_relations": [
                        {
                            "subject": relation.subject_name,
                            "predicate": relation.predicate,
                            "object": relation.object_name,
                            "score": relation.score,
                        }
                        for relation in frame.scene_graph.relations[:8]
                    ],
                    "raw_detections": [_serialize_detection(item) for item in frame.raw_detections],
                    "fused_detections": [_serialize_detection(item) for item in frame.fused_detections],
                    "relevant_detections": [_serialize_detection(item) for item in frame.relevant_detections],
                    "hand_object_contacts": [asdict(item) for item in frame.interaction_evidence.contacts[:8]],
                }
            )
        return payload

    def display_step(self, frame: ProcessedFrame, prev_step_for_transition: Optional[str]) -> str:
        """Return the user-facing step label for the current frame."""

        if frame.memory_observation.has_visual_evidence:
            return frame.fusion_result.step_id
        if prev_step_for_transition:
            return prev_step_for_transition
        return "HOLD"

    def run(self, video_path: Path) -> Path:
        logger = JsonlCsvLogger(self.runlog_root, video_path)
        self.prepare_run(logger, source_uri=str(video_path))
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            self.memory_manager.close()
            self.event_bus.close()
            self.ops_tracker.close()
            logger.close()
            raise RuntimeError(f"Failed to open video: {video_path}")
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        output_fps = self.config.video.output_fps or source_fps
        writer = None
        if self.config.video.write_annotated:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
            annotated_path = logger.run_dir / f"{video_path.stem}_annotated.mp4"
            writer = cv2.VideoWriter(
                str(annotated_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(output_fps),
                (width, height),
            )
        frame_index = 0
        processed_index = 0
        prev_step_for_transition: Optional[str] = None
        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                frame_index += 1
                if frame_index % self.config.video.stride != 0:
                    continue

                processed_index += 1
                processed = self.process_frame(
                    frame_bgr=frame_bgr,
                    frame_index=frame_index,
                    prev_step_for_transition=prev_step_for_transition,
                    source_kind="video",
                )
                logger.log_iteration(self.build_iteration_payload(processed_index, processed, source_kind="video"))

                if not processed.memory_observation.has_visual_evidence:
                    display_step = self.display_step(processed, prev_step_for_transition)
                    self.ops_tracker.update(
                        status="Doing",
                        frame_index=frame_index,
                        step_id=prev_step_for_transition or "",
                        reason="no_visual_evidence",
                    )
                    if writer is not None:
                        writer.write(
                            _draw_overlay(
                                frame_bgr,
                                processed.fused_detections,
                                f"STEP: {display_step} ({processed.fusion_result.confidence:.2f})",
                            )
                        )
                    continue

                fusion_result = processed.fusion_result
                review_decision = processed.review_decision
                expert_predictions = processed.expert_predictions
                memory_recall = processed.memory_recall
                memory_observation = processed.memory_observation

                display_step = fusion_result.step_id
                allow_auto_capture = True
                transition_step: Optional[str] = None
                pending_reason = "stable_accept"
                if review_decision is not None:
                    pending_reason = review_decision.reason
                    if review_decision.action in {"hold", "request_human", "prefer_candidate"}:
                        allow_auto_capture = False
                    if review_decision.action == "prefer_candidate" and review_decision.weak_feedback is not None and self.feedback_provider is None:
                        reviewer_feedback = review_decision.weak_feedback
                        self.fusion_head.apply_feedback(
                            feedback=reviewer_feedback,
                            expert_predictions=expert_predictions,
                            fusion_result=fusion_result,
                        )
                        self.memory_manager.record_feedback(memory_observation, reviewer_feedback)
                        self.review_manager.record_feedback(frame_index, reviewer_feedback)
                        self.ops_tracker.record_feedback(reviewer_feedback)
                        self.event_bus.emit(
                            "feedback.applied",
                            {
                                "source": reviewer_feedback.source,
                                "label": reviewer_feedback.label,
                                "accepted": reviewer_feedback.accepted,
                                "strength": reviewer_feedback.strength,
                                "note": reviewer_feedback.note,
                            },
                            frame_index=frame_index,
                        )
                        logger.log_feedback(
                            _feedback_log_payload(
                                processed_index=processed_index,
                                frame_index=frame_index,
                                feedback=reviewer_feedback,
                                fusion_result=fusion_result,
                                memory_step=memory_recall.prediction.step_id,
                                memory_conf=memory_recall.prediction.confidence,
                            )
                        )
                        display_step = reviewer_feedback.label
                        transition_step = reviewer_feedback.label
                if processed.stable and self.feedback_provider is not None:
                    feedback = self.feedback_provider.request(
                        fusion_result=fusion_result,
                        expert_predictions=expert_predictions,
                        evidence_token=processed.evidence_token,
                        review_decision=review_decision,
                    )
                    if feedback is not None:
                        self.fusion_head.apply_feedback(
                            feedback=feedback,
                            expert_predictions=expert_predictions,
                            fusion_result=fusion_result,
                        )
                        self.memory_manager.record_feedback(memory_observation, feedback)
                        self.review_manager.record_feedback(frame_index, feedback)
                        self.ops_tracker.record_feedback(feedback)
                        self.event_bus.emit(
                            "feedback.applied",
                            {
                                "source": feedback.source,
                                "label": feedback.label,
                                "accepted": feedback.accepted,
                                "strength": feedback.strength,
                                "note": feedback.note,
                            },
                            frame_index=frame_index,
                        )
                        display_step = feedback.label
                        prev_step_for_transition = feedback.label
                        logger.log_feedback(
                            _feedback_log_payload(
                                processed_index=processed_index,
                                frame_index=frame_index,
                                feedback=feedback,
                                fusion_result=fusion_result,
                                memory_step=memory_recall.prediction.step_id,
                                memory_conf=memory_recall.prediction.confidence,
                            )
                        )
                        self.ops_tracker.update(
                            status="Next",
                            frame_index=frame_index,
                            step_id=feedback.label,
                            reason="human_accept" if feedback.accepted else "human_feedback",
                            review_action=review_decision.action if review_decision is not None else "",
                        )
                    else:
                        if transition_step is not None:
                            prev_step_for_transition = transition_step
                            self.ops_tracker.update(
                                status="Next",
                                frame_index=frame_index,
                                step_id=transition_step,
                                reason=pending_reason,
                                review_action=review_decision.action if review_decision is not None else "",
                            )
                        elif allow_auto_capture:
                            prev_step_for_transition = fusion_result.step_id
                            self.ops_tracker.update(
                                status="Next",
                                frame_index=frame_index,
                                step_id=fusion_result.step_id,
                                reason=pending_reason,
                                review_action=review_decision.action if review_decision is not None else "",
                            )
                        else:
                            self.ops_tracker.update(
                                status="Blocked" if review_decision and review_decision.should_prompt_human else "Review",
                                frame_index=frame_index,
                                step_id=fusion_result.step_id,
                                reason=pending_reason,
                                review_action=review_decision.action if review_decision is not None else "",
                            )
                else:
                    if allow_auto_capture:
                        self.memory_manager.record_auto(memory_observation, fusion_result=fusion_result, stable=processed.stable)
                    if transition_step is not None:
                        prev_step_for_transition = transition_step
                        self.ops_tracker.update(
                            status="Next",
                            frame_index=frame_index,
                            step_id=transition_step,
                            reason=pending_reason,
                            review_action=review_decision.action if review_decision is not None else "",
                        )
                    elif allow_auto_capture:
                        prev_step_for_transition = fusion_result.step_id
                        self.ops_tracker.update(
                            status="Next" if processed.stable else "Doing",
                            frame_index=frame_index,
                            step_id=fusion_result.step_id,
                            reason="stable_auto" if processed.stable else "tracking",
                            review_action=review_decision.action if review_decision is not None else "",
                        )
                    else:
                        self.ops_tracker.update(
                            status="Blocked" if review_decision and review_decision.should_prompt_human else "Review",
                            frame_index=frame_index,
                            step_id=fusion_result.step_id,
                            reason=pending_reason,
                            review_action=review_decision.action if review_decision is not None else "",
                        )
                if writer is not None:
                    writer.write(_draw_overlay(frame_bgr, processed.fused_detections, f"STEP: {display_step} ({fusion_result.confidence:.2f})"))
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            self.finalize_run(logger, source_uri=str(video_path))
        return logger.run_dir

    def _detection_params(self, source_kind: str) -> tuple[float, bool]:
        if source_kind == "camera":
            conf_override = self.config.camera.detection_conf
            tta_override = self.config.camera.detection_tta
            conf = float(self.config.detection.conf if conf_override is None else conf_override)
            tta = bool(self.config.detection.tta if tta_override is None else tta_override)
            return conf, tta
        return float(self.config.detection.conf), bool(self.config.detection.tta)

    def _iteration_detail_level(self, source_kind: str) -> str:
        if source_kind == "camera":
            return str(self.config.runlog.live_detail_level).strip().lower() or "sparse"
        return str(self.config.runlog.detail_level).strip().lower() or "debug"

    def _empty_prediction(self, reason: str, fallback_step: Optional[str] = None) -> StepPrediction:
        step_id = str(fallback_step).strip().upper() if fallback_step else (self.steps[0] if self.steps else "")
        return StepPrediction(
            step_id=step_id,
            confidence=0.0,
            scores={step: 0.0 for step in self.steps},
            extras={"active": False, "reason": reason},
        )


def build_default_pipeline(
    config: AppConfig,
    kb_path: str,
    yolo_weights: str,
    device: str,
    state_path: Optional[Path],
    interactive: bool,
) -> StepPipeline:
    kb = KnowledgeBase.from_path(kb_path)
    steps = kb.workflow_steps(config.experts.steps)
    component_names = kb.component_names()
    visual_encoder = SharedVisualEncoder(mode=config.experts.gallery_embed, device=device)
    gallery_root = _resolve_gallery_root(config, steps)
    detector = YOLODetectorComponent(
        weights=yolo_weights,
        device=device,
        nms_iou=config.detection.nms_iou,
        use_builtin_tta=config.detection.use_builtin_tta,
        min_box_area_ratio=config.detection.min_box_area_ratio,
        max_box_area_ratio=config.detection.max_box_area_ratio,
        max_aspect_ratio=config.detection.max_aspect_ratio,
        reject_multi_border_boxes=config.detection.reject_multi_border_boxes,
        border_margin_px=config.detection.border_margin_px,
        max_per_class=config.detection.max_per_class,
        dedupe_iou=config.detection.dedupe_iou,
    )
    geometry_backend = config.geometry.backend.strip().lower()
    if geometry_backend in {"moge", "moge2", "mo-ge-2"}:
        geometry_provider = MoGeGeometryProvider(
            model_name=config.geometry.moge_model,
            device=device,
            use_fp16=config.geometry.use_fp16,
            resolution_level=config.geometry.resolution_level,
            apply_mask=config.geometry.apply_mask,
        )
    else:
        geometry_provider = GradientGeometryProvider(device=device)
    temporal_backend = str(config.temporal.backend).strip().lower()
    if temporal_backend == "window_iou":
        temporal_fusion = WindowIoUFusion(window=config.temporal.window, iou_thr=config.temporal.iou_thr)
    else:
        temporal_fusion = ByteTrackLiteFusion(
            track_high_thresh=config.temporal.track_high_thresh,
            track_low_thresh=config.temporal.track_low_thresh,
            new_track_thresh=config.temporal.new_track_thresh,
            match_iou_thr=config.temporal.match_iou_thr,
            lost_buffer=config.temporal.lost_buffer,
            min_confirmed_hits=config.temporal.min_confirmed_hits,
            smooth_alpha=config.temporal.smooth_alpha,
        )
    context_selector = DepthContextSelector(tau_p=config.depth_context.tau_p, tau_d=config.depth_context.tau_d)
    scene_graph_builder = GeometryAwareSceneGraphBuilder(
        enabled=config.scene_graph.enabled,
        depth_margin=config.scene_graph.depth_margin,
        contact_pixel_gap=config.scene_graph.contact_pixel_gap,
        contact_depth_gap=config.scene_graph.contact_depth_gap,
        support_vertical_gap=config.scene_graph.support_vertical_gap,
        support_overlap_ratio=config.scene_graph.support_overlap_ratio,
        max_relations=config.scene_graph.max_relations,
    )
    interaction_estimator = HandObjectContactEstimator(
        enabled=config.interaction.enabled,
        hand_names=config.interaction.hand_names,
        tool_names=config.interaction.tool_names,
        exclude_object_names=config.interaction.exclude_object_names,
        contact_margin_px=config.interaction.contact_margin_px,
        near_margin_px=config.interaction.near_margin_px,
        depth_contact_gap=config.interaction.depth_contact_gap,
        min_contact_score=config.interaction.min_contact_score,
        history=config.interaction.history,
    )
    try:
        segmentation_backend = build_segmentation_backend(
            backend=config.segmentation.backend,
            model_name=config.segmentation.model_name,
            device=config.segmentation.device if config.segmentation.device else device,
            prompts=config.segmentation.prompts,
            score_threshold=config.segmentation.score_threshold,
            max_masks=config.segmentation.max_masks,
        )
    except Exception:
        if config.segmentation.fail_on_unavailable:
            raise
        from ..components import NullSegmentationBackend

        segmentation_backend = NullSegmentationBackend()
    evidence_encoder = StructuredEvidenceEncoder(steps=steps)
    temporal_timeline = EvidenceTimelineStore(maxlen=config.temporal_step.history_size)
    step_graph_prior = CompiledStepGraphPrior(
        kb=kb,
        steps=steps,
        transitions=config.online_fusion.transitions,
        transition_penalty=config.temporal_step.graph_transition_penalty,
        requirement_bonus=config.temporal_step.graph_requirement_bonus,
        requirement_penalty=config.temporal_step.graph_requirement_penalty,
        forbid_penalty=config.temporal_step.graph_forbid_penalty,
        relation_bonus=config.temporal_step.graph_relation_bonus,
        relation_penalty=config.temporal_step.graph_relation_penalty,
    )
    temporal_vectorizer = TemporalEvidenceVectorizer(
        steps=steps,
        component_names=component_names,
        relation_types=getattr(scene_graph_builder, "RELATION_TYPES", None),
        state_score_weight=config.temporal_step.state_score_weight,
        retrieval_score_weight=config.temporal_step.retrieval_score_weight,
        memory_score_weight=config.temporal_step.memory_score_weight,
    )
    state_expert = RuleBasedStepExpert(kb=kb, steps=steps)
    temporal_backend_name = str(config.temporal_step.backend).strip().lower()
    if temporal_backend_name == "gru_stream":
        temporal_expert = StreamingGRUTemporalStepExpert(
            steps=steps,
            timeline=temporal_timeline,
            graph_prior=step_graph_prior,
            vectorizer=temporal_vectorizer,
            ema_alpha=config.temporal_step.ema_alpha,
            hidden_size=config.temporal_step.gru_hidden_size,
            device=config.temporal_step.device,
            checkpoint_path=config.temporal_step.checkpoint_path,
            learned_token_aggregation=config.temporal_step.learned_token_aggregation,
            token_hidden_size=config.temporal_step.token_hidden_size,
            token_output_size=config.temporal_step.token_output_size,
            enabled=config.temporal_step.enabled,
        )
    else:
        temporal_backend_name = "ema_graph"
        temporal_expert = CausalTemporalStepExpert(
            steps=steps,
            timeline=temporal_timeline,
            graph_prior=step_graph_prior,
            vectorizer=temporal_vectorizer,
            ema_alpha=config.temporal_step.ema_alpha,
            enabled=config.temporal_step.enabled,
        )

    retrieval_status: dict[str, object] = {
        "backend": "dummy",
        "gallery_root": gallery_root or "",
        "gallery_items": 0,
        "reason": "gallery_not_found" if not gallery_root else "",
    }
    if gallery_root:
        retrieval_expert = GalleryRetrievalExpert(
            root=gallery_root,
            steps=steps,
            exts=config.experts.gallery_exts,
            embed_mode=config.experts.gallery_embed,
            topk=config.experts.retrieval_topk,
            encoder=visual_encoder,
        )
        try:
            gallery_items = retrieval_expert.build()
            if gallery_items <= 0:
                raise RuntimeError("gallery_index_empty")
            retrieval_status = {
                "backend": "gallery",
                "gallery_root": gallery_root,
                "gallery_items": gallery_items,
                "embed_mode": config.experts.gallery_embed,
                "topk": config.experts.retrieval_topk,
            }
        except Exception as exc:
            if config.experts.strict_gallery:
                raise
            retrieval_expert = DummyRetrievalExpert(steps=steps)
            retrieval_status = {
                "backend": "dummy",
                "gallery_root": gallery_root,
                "gallery_items": 0,
                "reason": "gallery_build_failed",
                "error": str(exc)[:240],
            }
    else:
        retrieval_expert = DummyRetrievalExpert(steps=steps)

    state_calibrator = IdentityStepCalibrator()
    temporal_calibrator = IdentityStepCalibrator()
    retrieval_calibrator = IdentityStepCalibrator()
    memory_long_term_path = (
        Path(config.memory.long_term_path)
        if config.memory.long_term_path
        else Path(config.runlog.save_dir) / "memory_bank" / "long_term_memory.jsonl"
    )
    memory_manager = MemoryLifecycleManager(
        steps=steps,
        component_names=component_names,
        expert_names=["state", "retrieval"],
        config=config.memory,
        long_term_path=memory_long_term_path,
        visual_encoder=visual_encoder,
    )
    review_manager = ReviewManager(
        policy=SparseReviewPolicy(
            enabled=config.review.enabled,
            low_confidence=config.review.low_confidence,
            margin_threshold=config.review.margin_threshold,
            disagreement_threshold=config.review.disagreement_threshold,
            memory_confidence_threshold=config.review.memory_confidence_threshold,
            correction_streak=config.review.correction_streak,
            cooldown_frames=config.review.cooldown_frames,
        ),
        reviewer=SparseReviewerAgent(
            prefer_vote_count=config.review.prefer_vote_count,
            prefer_confidence=config.review.prefer_confidence,
            prefer_margin=config.review.prefer_margin,
            hold_confidence=config.review.hold_confidence,
            request_human_confidence=config.review.request_human_confidence,
            reviewer_feedback_strength=config.review.reviewer_feedback_strength,
        ),
        correction_window=config.review.correction_window,
    )
    event_bus = JsonlEventBus(enabled=config.events.enabled)
    ops_tracker = OpsStateTracker(enabled=config.ops.enabled, persist_history=config.ops.persist_history)
    fusion_head = LinearMarginFusion(
        steps=steps,
        expert_names=["state", "temporal", "retrieval", "memory"],
        state_gate=config.online_fusion.state_gate,
        temporal_gate=config.online_fusion.temporal_gate,
        retrieval_gate=config.online_fusion.retrieval_gate,
        memory_gate=config.online_fusion.memory_gate,
        leak_state=config.online_fusion.leak_state,
        leak_temporal=config.online_fusion.leak_temporal,
        leak_retrieval=config.online_fusion.leak_retrieval,
        leak_memory=config.online_fusion.leak_memory,
        clamp_lo=config.online_fusion.clamp_lo,
        clamp_hi=config.online_fusion.clamp_hi,
        floor_per_class=config.online_fusion.floor_per_class,
        bias_cap=config.online_fusion.bias_cap,
        eta=config.online_fusion.eta,
        gate_eta=config.online_fusion.gate_eta,
        margin=config.online_fusion.margin,
        positive_margin=config.online_fusion.positive_margin,
        positive_scale=config.online_fusion.positive_scale,
        hit_gamma=config.online_fusion.hit_gamma,
        error_gamma=config.online_fusion.error_gamma,
        freeze_confidence=config.online_fusion.freeze_confidence,
        exposure_rho=config.online_fusion.exposure_rho,
        balance_window=config.online_fusion.balance_window,
        balance_tau=config.online_fusion.balance_tau,
        lambda_transition=config.online_fusion.lambda_transition,
        context_gate_enabled=config.online_fusion.context_gate_enabled,
        context_gate_scale=config.online_fusion.context_gate_scale,
        context_gate_eta=config.online_fusion.context_gate_eta,
        context_gate_path=(Path(config.online_fusion.context_gate_checkpoint) if config.online_fusion.context_gate_checkpoint else None),
        transitions=config.online_fusion.transitions,
        state_path=state_path,
    )
    stability_policy = DualStabilityTracker(
        stable_n=config.stability.stable_n,
        require_same_step=config.stability.require_same_step,
    )
    feedback_provider: Optional[FeedbackProvider]
    feedback_provider = ConsoleFeedbackProvider(evidence_prompt_enabled=config.review.evidence_prompt_enabled) if interactive else None
    component_status = {
        "detector": {
            "backend": "legacy_yolo",
            "weights": str(yolo_weights),
            "device": str(device),
            "default_conf": float(config.detection.conf),
            "default_tta": bool(config.detection.tta),
            "camera_conf": config.camera.detection_conf,
            "camera_tta": config.camera.detection_tta,
            "min_box_area_ratio": float(config.detection.min_box_area_ratio),
            "max_box_area_ratio": float(config.detection.max_box_area_ratio),
            "max_aspect_ratio": float(config.detection.max_aspect_ratio),
            "max_per_class": int(config.detection.max_per_class),
            "dedupe_iou": float(config.detection.dedupe_iou),
        },
        "tracker": {
            "backend": temporal_backend,
            "window": int(config.temporal.window),
            "iou_thr": float(config.temporal.iou_thr),
            "track_high_thresh": float(config.temporal.track_high_thresh),
            "track_low_thresh": float(config.temporal.track_low_thresh),
            "new_track_thresh": float(config.temporal.new_track_thresh),
            "match_iou_thr": float(config.temporal.match_iou_thr),
            "lost_buffer": int(config.temporal.lost_buffer),
            "min_confirmed_hits": int(config.temporal.min_confirmed_hits),
            "smooth_alpha": float(config.temporal.smooth_alpha),
        },
        "geometry": {
            "backend": geometry_backend,
            "model": config.geometry.moge_model if geometry_backend in {"moge", "moge2", "mo-ge-2"} else "gradient",
        },
        "interaction": {
            "enabled": bool(config.interaction.enabled),
            "hand_names": list(config.interaction.hand_names),
            "tool_names": list(config.interaction.tool_names),
            "contact_margin_px": float(config.interaction.contact_margin_px),
            "near_margin_px": float(config.interaction.near_margin_px),
            "depth_contact_gap": float(config.interaction.depth_contact_gap),
            "min_contact_score": float(config.interaction.min_contact_score),
            "history": int(config.interaction.history),
        },
        "segmentation": {
            "backend": str(config.segmentation.backend),
            "resolved_backend": getattr(segmentation_backend, "backend_name", "unknown"),
            "model_name": str(config.segmentation.model_name),
            "device": str(config.segmentation.device),
            "prompts": list(config.segmentation.prompts),
            "score_threshold": float(config.segmentation.score_threshold),
            "max_masks": int(config.segmentation.max_masks),
        },
        "temporal_step": {
            "enabled": bool(config.temporal_step.enabled),
            "backend": temporal_backend_name,
            "history_size": int(config.temporal_step.history_size),
            "ema_alpha": float(config.temporal_step.ema_alpha),
            "device": str(config.temporal_step.device),
            "checkpoint_path": str(config.temporal_step.checkpoint_path),
            "gru_hidden_size": int(config.temporal_step.gru_hidden_size),
            "learned_token_aggregation": bool(getattr(temporal_expert, "token_aggregator", None) is not None),
            "token_hidden_size": int(config.temporal_step.token_hidden_size),
            "token_output_size": int(config.temporal_step.token_output_size),
            "trained": bool(getattr(temporal_expert, "trained", False)),
            "state_score_weight": float(config.temporal_step.state_score_weight),
            "retrieval_score_weight": float(config.temporal_step.retrieval_score_weight),
            "memory_score_weight": float(config.temporal_step.memory_score_weight),
        },
        "retrieval": retrieval_status,
        "fusion": {
            "context_gate_enabled": bool(config.online_fusion.context_gate_enabled),
            "context_gate_scale": float(config.online_fusion.context_gate_scale),
            "context_gate_eta": float(config.online_fusion.context_gate_eta),
            "context_gate_checkpoint": str(config.online_fusion.context_gate_checkpoint),
            "context_gate_loaded": bool(getattr(fusion_head, "context_gate_loaded", False)),
        },
    }
    return StepPipeline(
        config=config,
        steps=steps,
        detector=detector,
        geometry_provider=geometry_provider,
        temporal_fusion=temporal_fusion,
        context_selector=context_selector,
        scene_graph_builder=scene_graph_builder,
        evidence_encoder=evidence_encoder,
        state_expert=state_expert,
        temporal_expert=temporal_expert,
        retrieval_expert=retrieval_expert,
        state_calibrator=state_calibrator,
        temporal_calibrator=temporal_calibrator,
        retrieval_calibrator=retrieval_calibrator,
        fusion_head=fusion_head,
        stability_policy=stability_policy,
        feedback_provider=feedback_provider,
        runlog_root=config.runlog.save_dir,
        kb=kb,
        memory_manager=memory_manager,
        review_manager=review_manager,
        interaction_estimator=interaction_estimator,
        segmentation_backend=segmentation_backend,
        event_bus=event_bus,
        ops_tracker=ops_tracker,
        component_status=component_status,
    )
