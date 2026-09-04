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
    IdentityTemporalFusion,
    IdentityStepCalibrator,
    JsonlCsvLogger,
    KnowledgeBase,
    LinearMarginFusion,
    MoGeGeometryProvider,
    ProceduralSceneEvidenceTracker,
    RuleBasedStepExpert,
    SharedVisualEncoder,
    StreamingGRUTemporalStepExpert,
    StructuredEvidenceEncoder,
    TemporalEvidenceVectorizer,
    TrackEvidenceAggregator,
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
from ..inspect_system.evidence_scorer import PrototypeEvidenceScorer
from ..inspect_system.causal_evidence_bank import CausalEvidenceBank
from ..inspect_system.live_claim_verifier import (
    CLAIM_BY_STEP,
    LiveClaimDecision,
    LiveClaimVerifier,
)
from ..core_types import (
    Detection,
    EvidenceToken,
    FeedbackEvent,
    FusionResult,
    GeometryFrame,
    InteractionEvidence,
    RetrievalInput,
    SceneEvidenceFrame,
    SceneGraphFrame,
    StepPrediction,
    TrackEvidenceFrame,
)
from .edge_runtime import ComputeContext, ComputePlan, EpistemicComputeRouter, StageProfiler


def _serialize_detection(detection: Detection) -> dict[str, object]:
    return {
        "name": detection.name,
        "xyxy": [float(v) for v in detection.xyxy],
        "confidence": float(detection.confidence),
        "meta": dict(detection.meta),
    }


def _serialize_evidence_token(token: EvidenceToken) -> dict[str, object]:
    return {
        "frame_index": int(token.frame_index),
        "prev_step": token.prev_step,
        "visible_counts": {str(key): int(value) for key, value in token.visible_counts.items()},
        "relevant_counts": {str(key): int(value) for key, value in token.relevant_counts.items()},
        "track_counts": {str(key): int(value) for key, value in token.track_counts.items()},
        "stable_track_counts": {str(key): int(value) for key, value in token.stable_track_counts.items()},
        "track_ids": {str(key): [int(item) for item in value] for key, value in token.track_ids.items()},
        "track_confidences": {str(key): float(value) for key, value in token.track_confidences.items()},
        "track_hits": {str(key): int(value) for key, value in token.track_hits.items()},
        "track_ages": {str(key): int(value) for key, value in token.track_ages.items()},
        "track_motion": {str(key): float(value) for key, value in token.track_motion.items()},
        "track_evidence_keys": [str(item) for item in token.track_evidence_keys],
        "track_objects": [dict(item) for item in token.track_objects],
        "scene_evidence_keys": [str(item) for item in token.scene_evidence_keys],
        "scene_visible_objects": [str(item) for item in token.scene_visible_objects],
        "scene_stable_objects": [str(item) for item in token.scene_stable_objects],
        "scene_active_objects": [str(item) for item in token.scene_active_objects],
        "scene_moving_objects": [str(item) for item in token.scene_moving_objects],
        "scene_relation_keys": [str(item) for item in token.scene_relation_keys],
        "scene_relation_change_keys": [str(item) for item in token.scene_relation_change_keys],
        "scene_transition_keys": [str(item) for item in token.scene_transition_keys],
        "relation_counts": {str(key): int(value) for key, value in token.relation_counts.items()},
        "relation_facts": [list(item) for item in token.relation_facts],
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
    track_evidence: TrackEvidenceFrame
    scene_evidence: SceneEvidenceFrame
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
    claim_decision: Optional[LiveClaimDecision]
    review_decision: Optional[ReviewDecision]
    runtime_ms: dict[str, float]
    compute_plan: dict[str, object]


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
        track_evidence_builder: TrackEvidenceAggregator,
        scene_evidence_tracker: ProceduralSceneEvidenceTracker,
        segmentation_backend,
        event_bus: JsonlEventBus,
        ops_tracker: OpsStateTracker,
        claim_verifier: Optional[LiveClaimVerifier] = None,
        causal_evidence_bank: Optional[CausalEvidenceBank] = None,
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
        self.track_evidence_builder = track_evidence_builder
        self.scene_evidence_tracker = scene_evidence_tracker
        self.segmentation_backend = segmentation_backend
        self.event_bus = event_bus
        self.ops_tracker = ops_tracker
        self.claim_verifier = claim_verifier
        self.causal_evidence_bank = causal_evidence_bank
        self.component_status = dict(component_status or {})
        self.edge_router = EpistemicComputeRouter(config.edge_runtime)
        self._edge_processed_index = 0
        self._edge_previous_probe: Optional[np.ndarray] = None
        self._edge_low_change_streak = 0
        self._edge_last_geometry: Optional[GeometryFrame] = None
        self._edge_geometry_age = 0
        self._edge_last_raw_detections: Optional[list[Detection]] = None
        self._edge_last_canonical_detections: Optional[list[Detection]] = None
        self._edge_last_segmentation_masks: Optional[list[object]] = None
        self._edge_last_retrieval_prediction: Optional[StepPrediction] = None
        self._edge_last_visual_embedding: Optional[list[float]] = None
        self._edge_memory_embedding_age = 0
        self._edge_last_stable = False
        self._edge_last_step_confidence = 0.0
        self._edge_last_interaction_transition = False
        self._edge_claim_state = "insufficient"
        self._edge_last_claim_id = ""
        self._edge_last_resolution_confidence = 0.0
        self._edge_epistemic_gain = 1.0
        self._edge_missing_roles: tuple[str, ...] = ()
        self._edge_missing_role_scores: tuple[tuple[str, float], ...] = ()
        self._edge_query_intent = ""

    def prepare_run(self, logger: JsonlCsvLogger, source_uri: str) -> None:
        """Attach per-run sinks and persist run metadata."""

        logger.log_meta({"video_path": source_uri, "config": asdict(self.config), "components": self.component_status})
        reset_temporal = getattr(self.temporal_expert, "reset", None)
        if callable(reset_temporal):
            reset_temporal()
        reset_scene = getattr(self.scene_evidence_tracker, "reset", None)
        if callable(reset_scene):
            reset_scene()
        reset_detector = getattr(self.detector, "reset", None)
        if callable(reset_detector):
            reset_detector()
        reset_claim_verifier = getattr(self.claim_verifier, "reset", None)
        if callable(reset_claim_verifier):
            reset_claim_verifier()
        reset_evidence_bank = getattr(self.causal_evidence_bank, "reset", None)
        if callable(reset_evidence_bank):
            reset_evidence_bank()
        self._reset_edge_runtime()
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

    def update_edge_request_context(
        self,
        *,
        claim_state: str = "insufficient",
        missing_roles: tuple[str, ...] = (),
        query_intent: str = "",
    ) -> None:
        """Expose verifier/query needs to the next perception scheduling decision."""

        self._edge_claim_state = str(claim_state or "insufficient").strip().lower()
        self._edge_missing_roles = tuple(str(role).strip().lower() for role in missing_roles if str(role).strip())
        self._edge_query_intent = str(query_intent or "").strip().lower()

    def _reset_edge_runtime(self) -> None:
        self._edge_processed_index = 0
        self._edge_previous_probe = None
        self._edge_low_change_streak = 0
        self._edge_last_geometry = None
        self._edge_geometry_age = 0
        self._edge_last_raw_detections = None
        self._edge_last_canonical_detections = None
        self._edge_last_segmentation_masks = None
        self._edge_last_retrieval_prediction = None
        self._edge_last_visual_embedding = None
        self._edge_memory_embedding_age = 0
        self._edge_last_stable = False
        self._edge_last_step_confidence = 0.0
        self._edge_last_interaction_transition = False
        self._edge_claim_state = "insufficient"
        self._edge_last_claim_id = ""
        self._edge_last_resolution_confidence = 0.0
        self._edge_epistemic_gain = 1.0
        self._edge_missing_roles = ()
        self._edge_missing_role_scores = ()
        self._edge_query_intent = ""
        self.edge_router.reset()

    def _edge_visual_change(self, frame_bgr: np.ndarray) -> tuple[float, bool]:
        width = max(32, int(self.config.edge_runtime.visual_probe_width))
        height, source_width = frame_bgr.shape[:2]
        probe_height = max(24, int(round(height * width / max(1, source_width))))
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        probe = cv2.resize(gray, (width, probe_height), interpolation=cv2.INTER_AREA)
        previous = self._edge_previous_probe
        self._edge_previous_probe = probe
        if previous is None or previous.shape != probe.shape:
            self._edge_low_change_streak = 0
            return 1.0, True
        score = float(np.mean(cv2.absdiff(probe, previous)) / 255.0)
        changed = score >= max(0.0, float(self.config.edge_runtime.visual_change_threshold))
        self._edge_low_change_streak = 0 if changed else self._edge_low_change_streak + 1
        return score, bool(changed)

    def _edge_compute_plan(self, *, visual_change: bool) -> ComputePlan:
        if not bool(self.config.edge_runtime.enabled):
            return ComputePlan(
                run_detection=True,
                run_geometry=True,
                run_segmentation=True,
                run_retrieval=True,
                run_memory_embedding=True,
                response_tier="none",
                target_latency_ms=float(self.config.edge_runtime.target_perception_ms),
                reasons=("edge_runtime_disabled",),
            )
        return self.edge_router.plan(
            ComputeContext(
                frame_index=int(self._edge_processed_index),
                claim_state=self._edge_claim_state,
                missing_roles=self._edge_missing_roles,
                missing_role_scores=self._edge_missing_role_scores,
                query_intent=self._edge_query_intent,
                step_confidence=float(self._edge_last_step_confidence),
                stable_tracks=bool(self._edge_last_stable or self._edge_low_change_streak >= 2),
                visual_change=bool(visual_change),
                interaction_transition=bool(self._edge_last_interaction_transition),
                epistemic_gain=float(self._edge_epistemic_gain),
            )
        )

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        prev_step_for_transition: Optional[str],
        source_kind: str = "video",
    ) -> ProcessedFrame:
        """Run the shared perception, memory, and fusion stack on one frame."""

        profiler = StageProfiler(
            enabled=bool(self.config.edge_runtime.profiling_enabled),
            synchronize_cuda=bool(self.config.edge_runtime.synchronize_cuda_for_timing),
        )
        self._edge_processed_index += 1
        visual_change_score, visual_change = self._edge_visual_change(frame_bgr)
        compute_plan = self._edge_compute_plan(visual_change=visual_change)
        det_conf, det_tta = self._detection_params(source_kind)
        detection_reused = bool(
            self.config.edge_runtime.enabled
            and not compute_plan.run_detection
            and self._edge_last_raw_detections is not None
            and self._edge_last_canonical_detections is not None
        )
        with profiler.measure("detector"):
            if detection_reused:
                raw_detections = list(self._edge_last_raw_detections or [])
                canonical_detections = list(self._edge_last_canonical_detections or [])
            else:
                raw_detections = self.detector.detect(frame_bgr, conf=det_conf, tta=det_tta)
                canonical_detections = self.kb.canonicalize(raw_detections)
                self._edge_last_raw_detections = list(raw_detections)
                self._edge_last_canonical_detections = list(canonical_detections)
        with profiler.measure("tracking"):
            fused_detections = self.temporal_fusion.update(canonical_detections)
            track_evidence = self.track_evidence_builder.build(fused_detections)
        geometry_reused = False
        with profiler.measure("geometry"):
            can_reuse_geometry = bool(
                self.config.edge_runtime.enabled
                and not compute_plan.run_geometry
                and self._edge_last_geometry is not None
                and self._edge_geometry_age < max(1, int(self.config.edge_runtime.max_geometry_age))
            )
            if can_reuse_geometry:
                geometry = self._edge_last_geometry
                self._edge_geometry_age += 1
                geometry_reused = True
            else:
                geometry = self.geometry_provider.infer(frame_bgr)
                self._edge_last_geometry = geometry
                self._edge_geometry_age = 0
        with profiler.measure("scene_graph"):
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
        segmentation_reused = bool(
            self.config.edge_runtime.enabled
            and not compute_plan.run_segmentation
            and self._edge_last_segmentation_masks is not None
        )
        with profiler.measure("segmentation"):
            if segmentation_reused:
                segmentation_masks = list(self._edge_last_segmentation_masks or [])
            else:
                segmentation_masks = self.segmentation_backend.segment(frame_bgr, segmentation_prompts)
                self._edge_last_segmentation_masks = list(segmentation_masks)
        with profiler.measure("interaction"):
            interaction_evidence = self.interaction_estimator.estimate(
                detections=fused_detections,
                geometry=geometry,
                scene_graph=scene_graph,
                relevant_detections=relevant_detections,
                segmentation_masks=segmentation_masks,
            )
            scene_evidence = self.scene_evidence_tracker.build(
                detections=fused_detections,
                scene_graph=scene_graph,
                track_evidence=track_evidence,
                interaction_evidence=interaction_evidence,
            )

        has_visual_evidence = bool(fused_detections or relevant_detections or scene_graph.relations)
        state_input = {
            "detections": self._state_input_detections(fused_detections, relevant_detections),
            "relations": list(scene_graph.relations),
        }
        with profiler.measure("state_expert"):
            raw_state_prediction = self.state_expert.predict(state_input)
            state_prediction = self.state_calibrator.calibrate(raw_state_prediction)
        focus_detection = fused_detections[nearest_index] if nearest_index is not None and 0 <= nearest_index < len(fused_detections) else None
        retrieval_reused = bool(
            self.config.edge_runtime.enabled
            and not compute_plan.run_retrieval
            and self._edge_last_retrieval_prediction is not None
        )
        with profiler.measure("retrieval_expert"):
            if retrieval_reused:
                retrieval_prediction = self._edge_last_retrieval_prediction
            elif has_visual_evidence:
                retrieval_prediction = self.retrieval_calibrator.calibrate(
                    self.retrieval_expert.predict(
                        RetrievalInput(
                            frame_bgr=frame_bgr,
                            focus_detection=focus_detection,
                        )
                    )
                )
                self._edge_last_retrieval_prediction = retrieval_prediction
            else:
                retrieval_prediction = self._empty_prediction(reason="no_visual_evidence", fallback_step=prev_step_for_transition)

        base_predictions = {"state": state_prediction, "retrieval": retrieval_prediction}
        memory_embedding_reused = False
        cached_embedding = None
        can_reuse_embedding = bool(
            self.config.edge_runtime.enabled
            and not compute_plan.run_memory_embedding
            and self._edge_last_visual_embedding is not None
            and self._edge_memory_embedding_age
            < max(1, int(self.config.edge_runtime.max_memory_embedding_age))
        )
        if can_reuse_embedding:
            cached_embedding = list(self._edge_last_visual_embedding or [])
            self._edge_memory_embedding_age += 1
            memory_embedding_reused = True
        with profiler.measure("memory_observation"):
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
                visual_embedding_override=cached_embedding,
            )
            if not memory_embedding_reused:
                self._edge_last_visual_embedding = list(memory_observation.visual_embedding)
                self._edge_memory_embedding_age = 0
        with profiler.measure("memory_recall"):
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

        with profiler.measure("evidence_temporal"):
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
                track_evidence=track_evidence,
                scene_evidence=scene_evidence,
            )
            temporal_prediction = self.temporal_calibrator.calibrate(self.temporal_expert.predict(evidence_token))

        expert_predictions = {
            **base_predictions,
            "temporal": temporal_prediction,
            "memory": memory_recall.prediction,
        }
        if self.causal_evidence_bank is not None:
            with profiler.measure("causal_evidence_proposal"):
                fusion_result = self.causal_evidence_bank.propose(
                    frame_bgr,
                    fused_detections,
                    scene_graph,
                    evidence_token,
                    raw_state_prediction,
                )
        else:
            with profiler.measure("fusion"):
                fusion_result = self.fusion_head.fuse(
                    expert_predictions=expert_predictions,
                    prev_step=prev_step_for_transition,
                    token=evidence_token,
                )
        with profiler.measure("stability"):
            stability_detections = self._stability_detections(fused_detections)
            stable, signature = self.stability_policy.update(stability_detections, fusion_result.step_id)
            if not memory_observation.has_visual_evidence:
                stable = False

        claim_decision: Optional[LiveClaimDecision] = None
        with profiler.measure("claim_verifier"):
            if self.claim_verifier is not None:
                product_family = self.claim_verifier.infer_product_family(
                    fused_detections
                )
                learned_triage = None
                step_id = str(fusion_result.step_id).strip().upper()
                if (
                    self.causal_evidence_bank is not None
                    and step_id in {"S2", "S3", "S4"}
                ):
                    learned_triage = self.causal_evidence_bank.triage(
                        fused_detections,
                        scene_graph,
                        evidence_token,
                        fusion_result,
                        claim=CLAIM_BY_STEP.get(step_id, "component_present"),
                        step=step_id,
                        product=product_family,
                        image_shape=tuple(int(value) for value in frame_bgr.shape[:2]),
                    )
                claim_decision = self.claim_verifier.verify(
                    fused_detections,
                    frame_index=frame_index,
                    proposed_step=fusion_result.step_id,
                    proposal_confidence=fusion_result.confidence,
                    image_shape=tuple(int(value) for value in frame_bgr.shape[:2]),
                    learned_triage=learned_triage,
                    product_family=product_family,
                )

        review_decision: Optional[ReviewDecision] = None
        with profiler.measure("review"):
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

        self._edge_last_stable = bool(stable)
        self._edge_last_step_confidence = float(fusion_result.confidence)
        self._edge_last_interaction_transition = bool(interaction_evidence.transition_likelihood >= 0.55)
        if claim_decision is not None:
            self._edge_claim_state = str(claim_decision.state)
            resolution_confidence = max(
                float(claim_decision.support_score),
                float(claim_decision.contradiction_score),
            )
            if claim_decision.claim_id == self._edge_last_claim_id:
                self._edge_epistemic_gain = max(
                    0.0,
                    resolution_confidence - self._edge_last_resolution_confidence,
                )
            else:
                self._edge_epistemic_gain = 1.0
            self._edge_last_claim_id = str(claim_decision.claim_id)
            self._edge_last_resolution_confidence = resolution_confidence
            if str(claim_decision.state) == "insufficient":
                self._edge_missing_roles = tuple(claim_decision.missing_roles)
                evidence_deficit = max(
                    0.0,
                    min(1.0, 1.0 - float(claim_decision.visibility_score)),
                )
                self._edge_missing_role_scores = tuple(
                    (str(role), evidence_deficit)
                    for role in claim_decision.missing_roles
                )
            else:
                self._edge_missing_roles = ()
                self._edge_missing_role_scores = ()
        return ProcessedFrame(
            frame_index=frame_index,
            raw_detections=raw_detections,
            fused_detections=fused_detections,
            track_evidence=track_evidence,
            scene_evidence=scene_evidence,
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
            claim_decision=claim_decision,
            review_decision=review_decision,
            runtime_ms=profiler.finish(),
            compute_plan={
                "enabled": bool(self.config.edge_runtime.enabled),
                "visual_change": bool(visual_change),
                "visual_change_score": round(float(visual_change_score), 6),
                "run_detection": not detection_reused,
                "detection_reused": bool(detection_reused),
                "causal_evidence_bank": (
                    dict(self.causal_evidence_bank.diagnostics)
                    if self.causal_evidence_bank is not None
                    else {}
                ),
                "run_geometry": not geometry_reused,
                "geometry_reused": bool(geometry_reused),
                "run_segmentation": not segmentation_reused,
                "segmentation_reused": bool(segmentation_reused),
                "run_retrieval": not retrieval_reused,
                "retrieval_reused": bool(retrieval_reused),
                "run_memory_embedding": not memory_embedding_reused,
                "memory_embedding_reused": bool(memory_embedding_reused),
                "response_tier": compute_plan.response_tier,
                "acquisition_mode": compute_plan.acquisition_mode,
                "external_observation_recommended": bool(compute_plan.external_observation_recommended),
                "stage_evidence_values": dict(compute_plan.stage_evidence_values),
                "reasons": list(compute_plan.reasons),
            },
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
            "num_tracks": len(frame.track_evidence.objects),
            "num_scene_evidence": len(frame.scene_evidence.evidence_keys),
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
            "decision_step": self.decision_step(frame),
            "claim_id": frame.claim_decision.claim_id if frame.claim_decision is not None else "",
            "claim_state": frame.claim_decision.state if frame.claim_decision is not None else "",
            "claim_support": frame.claim_decision.support_score if frame.claim_decision is not None else 0.0,
            "claim_contradiction": frame.claim_decision.contradiction_score if frame.claim_decision is not None else 0.0,
            "claim_margin": frame.claim_decision.counterfactual_margin if frame.claim_decision is not None else 0.0,
            "claim_visibility": frame.claim_decision.visibility_score if frame.claim_decision is not None else 0.0,
            "claim_admissible": frame.claim_decision.admissible if frame.claim_decision is not None else True,
            "claim_memory_ready": frame.claim_decision.memory_ready if frame.claim_decision is not None else True,
            "claim_product_family": frame.claim_decision.product_family if frame.claim_decision is not None else "",
            "claim_proposed_step": frame.claim_decision.proposed_step if frame.claim_decision is not None else "",
            "claim_missing_roles": list(frame.claim_decision.missing_roles) if frame.claim_decision is not None else [],
            "claim_features": dict(frame.claim_decision.features) if frame.claim_decision is not None else {},
            "committed_step": frame.claim_decision.committed_step if frame.claim_decision is not None else "",
            "gate_state": float(frame.fusion_result.gates.get("state", 0.0)),
            "gate_temporal": float(frame.fusion_result.gates.get("temporal", 0.0)),
            "gate_retrieval": float(frame.fusion_result.gates.get("retrieval", 0.0)),
            "gate_memory": float(frame.fusion_result.gates.get("memory", 0.0)),
            "gates": dict(frame.fusion_result.gates),
            "review_action": review_decision.action if review_decision is not None else "",
            "review_label": review_decision.label if review_decision is not None else "",
            "review_reason": review_decision.reason if review_decision is not None else "",
            "review_triggers": list(review_decision.trigger_reasons) if review_decision is not None else [],
            "runtime_ms": dict(frame.runtime_ms),
            "runtime_target_ms": float(self.config.edge_runtime.target_perception_ms),
            "runtime_deadline_miss": bool(
                frame.runtime_ms
                and float(frame.runtime_ms.get("total", 0.0)) > float(self.config.edge_runtime.target_perception_ms)
            ),
            "compute_plan": dict(frame.compute_plan),
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
                    "retrieval_extras": dict(frame.retrieval_prediction.extras),
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
                    "track_evidence": {
                        "track_counts": dict(frame.track_evidence.track_counts),
                        "stable_track_counts": dict(frame.track_evidence.stable_track_counts),
                        "track_evidence_keys": list(frame.track_evidence.evidence_keys),
                        "num_tracks": len(frame.track_evidence.objects),
                    },
                    "scene_evidence": {
                        "evidence_keys": list(frame.scene_evidence.evidence_keys),
                        "visible_objects": list(frame.scene_evidence.visible_objects),
                        "stable_objects": list(frame.scene_evidence.stable_objects),
                        "active_objects": list(frame.scene_evidence.active_objects),
                        "moving_objects": list(frame.scene_evidence.moving_objects),
                        "relation_keys": list(frame.scene_evidence.relation_keys),
                        "relation_change_keys": list(frame.scene_evidence.relation_change_keys),
                        "transition_keys": list(frame.scene_evidence.transition_keys),
                        "extras": dict(frame.scene_evidence.extras),
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
                    "track_objects": [dict(item) for item in frame.track_evidence.objects],
                    "scene_objects": [dict(item) for item in frame.scene_evidence.objects],
                    "scene_relations": [dict(item) for item in frame.scene_evidence.relations],
                    "hand_object_contacts": [asdict(item) for item in frame.interaction_evidence.contacts[:8]],
                }
            )
        return payload

    def _state_input_detections(
        self,
        fused_detections: list[Detection],
        relevant_detections: list[Detection],
    ) -> list[Detection]:
        candidates = relevant_detections if (self.config.depth_context.use_relevant_for_rules and relevant_detections) else fused_detections
        if not bool(self.config.track_evidence.prefer_stable_tracks_for_rules):
            return candidates
        stable = self.track_evidence_builder.stable_detections(candidates)
        if stable:
            return stable
        return candidates if bool(self.config.track_evidence.fallback_to_confirmed_tracks) else []

    def _stability_detections(self, fused_detections: list[Detection]) -> list[Detection]:
        if not bool(self.config.track_evidence.prefer_stable_tracks_for_stability):
            return fused_detections
        stable = self.track_evidence_builder.stable_detections(fused_detections)
        if stable:
            return stable
        return fused_detections if bool(self.config.track_evidence.fallback_to_confirmed_tracks) else []

    def display_step(self, frame: ProcessedFrame, prev_step_for_transition: Optional[str]) -> str:
        """Return the user-facing step label for the current frame."""

        if frame.claim_decision is not None:
            if frame.claim_decision.committed_step:
                return frame.claim_decision.committed_step
            if prev_step_for_transition:
                return str(prev_step_for_transition).strip().upper()
            return "HOLD"

        decision = self.decision_step(frame)
        if decision in {"INVALID", "HOLD"}:
            return decision
        if frame.memory_observation.has_visual_evidence:
            return frame.fusion_result.step_id
        if prev_step_for_transition:
            return prev_step_for_transition
        return "HOLD"

    def decision_step(self, frame: ProcessedFrame) -> str:
        """Return the assistance decision label, including invalid/hold states."""

        if frame.claim_decision is not None:
            return frame.claim_decision.committed_step or "HOLD"

        if not frame.memory_observation.has_visual_evidence:
            return "HOLD"
        extras = dict(frame.retrieval_prediction.extras or {})
        if bool(extras.get("invalid_like", False)):
            return "INVALID"
        scores = frame.retrieval_prediction.scores
        if (
            int(extras.get("num_negative_items", 0) or 0) > 0
            and frame.retrieval_prediction.confidence <= 1e-6
            and scores
            and all(abs(float(value)) <= 1e-6 for value in scores.values())
        ):
            return "INVALID"
        return frame.fusion_result.step_id

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

                display_step = self.display_step(processed, prev_step_for_transition)
                claim_supported = processed.claim_decision is None or processed.claim_decision.state == "supported"
                allow_auto_capture = bool(claim_supported)
                transition_step: Optional[str] = None
                pending_reason = (
                    f"claim_{processed.claim_decision.state}"
                    if processed.claim_decision is not None
                    else "stable_accept"
                )
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
                        transition_step = reviewer_feedback.label if claim_supported else None
                should_request_feedback = self.feedback_provider is not None and (
                    processed.stable or bool(getattr(self.feedback_provider, "request_on_unstable", False))
                )
                if should_request_feedback:
                    feedback = self.feedback_provider.request(
                        fusion_result=fusion_result,
                        expert_predictions=expert_predictions,
                        evidence_token=processed.evidence_token,
                        review_decision=review_decision,
                    )
                    if feedback is not None:
                        is_no_step = bool(dict(feedback.extras).get("no_step", False))
                        if is_no_step and hasattr(self.fusion_head, "apply_no_step_feedback"):
                            self.fusion_head.apply_no_step_feedback(  # type: ignore[attr-defined]
                                feedback=feedback,
                                expert_predictions=expert_predictions,
                                fusion_result=fusion_result,
                            )
                        else:
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
                                "no_step": bool(is_no_step),
                            },
                            frame_index=frame_index,
                        )
                        display_step = "INVALID" if is_no_step else feedback.label
                        if not is_no_step:
                            prev_step_for_transition = feedback.label
                            if self.claim_verifier is not None:
                                self.claim_verifier.confirm_step(feedback.label)
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
                        if is_no_step:
                            self.ops_tracker.update(
                                status="Blocked",
                                frame_index=frame_index,
                                step_id=prev_step_for_transition or "",
                                reason="no_step_feedback",
                                review_action=review_decision.action if review_decision is not None else "",
                            )
                        else:
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
                        elif allow_auto_capture and processed.stable:
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
                    if allow_auto_capture and processed.stable:
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
                    elif allow_auto_capture and processed.stable:
                        prev_step_for_transition = fusion_result.step_id
                        self.ops_tracker.update(
                            status="Next",
                            frame_index=frame_index,
                            step_id=fusion_result.step_id,
                            reason="stable_auto",
                            review_action=review_decision.action if review_decision is not None else "",
                        )
                    elif allow_auto_capture:
                        self.ops_tracker.update(
                            status="Doing",
                            frame_index=frame_index,
                            step_id=fusion_result.step_id,
                            reason="tracking_candidate",
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
    feedback_provider_override: Optional[FeedbackProvider] = None,
) -> StepPipeline:
    kb = KnowledgeBase.from_path(kb_path)
    steps = kb.workflow_steps(config.experts.steps)
    component_names = kb.component_names()
    visual_encoder = SharedVisualEncoder(mode=config.experts.gallery_embed, device=device)
    gallery_root = _resolve_gallery_root(config, steps)
    temporal_backend = str(config.temporal.backend).strip().lower()
    use_ultralytics_tracker = temporal_backend in {"ultralytics_botsort", "ultralytics_bytetrack", "botsort", "bytetrack"}
    tracker_config = str(config.temporal.tracker_config or "").strip()
    if temporal_backend in {"ultralytics_bytetrack", "bytetrack"} and not tracker_config:
        tracker_config = "bytetrack.yaml"
    if not tracker_config:
        tracker_config = "botsort.yaml"
    detector = YOLODetectorComponent(
        weights=yolo_weights,
        device=device,
        nms_iou=config.detection.nms_iou,
        end2end=config.detection.end2end,
        use_builtin_tta=config.detection.use_builtin_tta,
        min_box_area_ratio=config.detection.min_box_area_ratio,
        max_box_area_ratio=config.detection.max_box_area_ratio,
        max_aspect_ratio=config.detection.max_aspect_ratio,
        reject_multi_border_boxes=config.detection.reject_multi_border_boxes,
        border_margin_px=config.detection.border_margin_px,
        max_per_class=config.detection.max_per_class,
        dedupe_iou=config.detection.dedupe_iou,
        track_with_model=use_ultralytics_tracker,
        tracker_config=tracker_config,
        track_class_smoothing_alpha=config.detection.track_class_smoothing_alpha,
        track_class_switch_margin=config.detection.track_class_switch_margin,
        identity_commit_conf=config.detection.identity_commit_conf,
        identity_commit_track_margin=config.detection.identity_commit_track_margin,
        role_bridge_enabled=config.detection.role_bridge_enabled,
        role_bridge_max_gap=config.detection.role_bridge_max_gap,
        role_bridge_confidence_decay=config.detection.role_bridge_confidence_decay,
        role_bridge_min_confidence=config.detection.role_bridge_min_confidence,
        role_bridge_max_width=config.detection.role_bridge_max_width,
        role_bridge_min_points=config.detection.role_bridge_min_points,
        role_bridge_fb_error=config.detection.role_bridge_fb_error,
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
    if use_ultralytics_tracker:
        temporal_fusion = IdentityTemporalFusion()
    elif temporal_backend == "window_iou":
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
            identity_groups=config.temporal.identity_groups,
            identity_smoothing_alpha=config.detection.track_class_smoothing_alpha,
            identity_switch_margin=config.detection.track_class_switch_margin,
            identity_commit_conf=config.detection.identity_commit_conf,
            identity_commit_margin=config.detection.identity_commit_track_margin,
            cross_identity_match_penalty=config.temporal.cross_identity_match_penalty,
            preserve_current_detections=config.temporal.preserve_current_detections,
        )
    context_selector = DepthContextSelector(tau_p=config.depth_context.tau_p, tau_d=config.depth_context.tau_d)
    scene_graph_builder = GeometryAwareSceneGraphBuilder(
        enabled=config.scene_graph.enabled,
        visibility_calibration_enabled=config.scene_graph.visibility_calibration_enabled,
        depth_margin=config.scene_graph.depth_margin,
        depth_inner_ratio=config.scene_graph.depth_inner_ratio,
        min_depth_valid_fraction=config.scene_graph.min_depth_valid_fraction,
        max_relative_depth_mad=config.scene_graph.max_relative_depth_mad,
        contact_pixel_gap=config.scene_graph.contact_pixel_gap,
        contact_depth_gap=config.scene_graph.contact_depth_gap,
        support_vertical_gap=config.scene_graph.support_vertical_gap,
        support_overlap_ratio=config.scene_graph.support_overlap_ratio,
        hard_pair_dedupe_iou=config.scene_graph.hard_pair_dedupe_iou,
        min_relation_score=config.scene_graph.min_relation_score,
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
    track_evidence_builder = TrackEvidenceAggregator(
        enabled=config.track_evidence.enabled,
        stable_hits=config.track_evidence.stable_hits,
        stable_confidence=config.track_evidence.stable_confidence,
        role_track_min_quality=config.track_evidence.role_track_min_quality,
        motion_px_threshold=config.track_evidence.motion_px_threshold,
        max_tracks=config.track_evidence.max_tracks,
    )
    scene_evidence_tracker = ProceduralSceneEvidenceTracker(
        enabled=config.scene_evidence.enabled,
        relation_change_memory=config.scene_evidence.relation_change_memory,
        max_objects=config.scene_evidence.max_objects,
        max_relations=config.scene_evidence.max_relations,
        max_changes=config.scene_evidence.max_changes,
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
                "positive_items": len(getattr(retrieval_expert, "items", [])),
                "negative_items": len(getattr(retrieval_expert, "negative_items", [])),
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
    feedback_provider = (
        feedback_provider_override
        if feedback_provider_override is not None
        else (ConsoleFeedbackProvider(evidence_prompt_enabled=config.review.evidence_prompt_enabled) if interactive else None)
    )
    component_status = {
        "detector": {
            "backend": "ultralytics_yolo",
            "weights": str(yolo_weights),
            "device": str(device),
            "default_conf": float(config.detection.conf),
            "identity_commit_conf": float(config.detection.identity_commit_conf),
            "identity_commit_track_margin": float(config.detection.identity_commit_track_margin),
            "default_tta": bool(config.detection.tta),
            "camera_conf": config.camera.detection_conf,
            "camera_tta": config.camera.detection_tta,
            "track_with_model": bool(use_ultralytics_tracker),
            "tracker_config": tracker_config,
            "min_box_area_ratio": float(config.detection.min_box_area_ratio),
            "max_box_area_ratio": float(config.detection.max_box_area_ratio),
            "max_aspect_ratio": float(config.detection.max_aspect_ratio),
            "max_per_class": int(config.detection.max_per_class),
            "dedupe_iou": float(config.detection.dedupe_iou),
        },
        "tracker": {
            "backend": temporal_backend,
            "tracker_config": tracker_config,
            "model_track_enabled": bool(use_ultralytics_tracker),
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
        "track_evidence": {
            "enabled": bool(config.track_evidence.enabled),
            "stable_hits": int(config.track_evidence.stable_hits),
            "stable_confidence": float(config.track_evidence.stable_confidence),
            "motion_px_threshold": float(config.track_evidence.motion_px_threshold),
            "max_tracks": int(config.track_evidence.max_tracks),
            "prefer_stable_tracks_for_rules": bool(config.track_evidence.prefer_stable_tracks_for_rules),
            "prefer_stable_tracks_for_stability": bool(config.track_evidence.prefer_stable_tracks_for_stability),
            "fallback_to_confirmed_tracks": bool(config.track_evidence.fallback_to_confirmed_tracks),
        },
        "scene_evidence": {
            "enabled": bool(config.scene_evidence.enabled),
            "relation_change_memory": int(config.scene_evidence.relation_change_memory),
            "max_objects": int(config.scene_evidence.max_objects),
            "max_relations": int(config.scene_evidence.max_relations),
            "max_changes": int(config.scene_evidence.max_changes),
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
    causal_evidence_bank: Optional[CausalEvidenceBank] = None
    causal_evidence_bank_status: dict[str, object] = {"enabled": False}
    if bool(config.causal_evidence_bank.enabled):
        package_root = Path(__file__).resolve().parents[1]

        def resolve_deployment_path(raw_path: str) -> Path:
            path = Path(str(raw_path).strip()).expanduser()
            return path if path.is_absolute() else package_root / path

        proposal_model_path = resolve_deployment_path(
            config.causal_evidence_bank.proposal_model_path
        )
        triage_model_path = resolve_deployment_path(
            config.causal_evidence_bank.triage_model_path
        )
        dinov2_repo = resolve_deployment_path(
            config.causal_evidence_bank.dinov2_repo
        )
        for description, path in (
            ("step proposal", proposal_model_path),
            ("claim triage", triage_model_path),
            ("DINOv2 repository", dinov2_repo),
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"Causal evidence bank {description} not found: {path}"
                )
        causal_evidence_bank = CausalEvidenceBank(
            proposal_model_path,
            triage_model_path,
            dinov2_repo=dinov2_repo,
            device=config.causal_evidence_bank.device,
            load_encoder_on_start=config.causal_evidence_bank.load_encoder_on_start,
        )
        causal_evidence_bank_status = {
            "enabled": True,
            "proposal_model_path": str(proposal_model_path),
            "triage_model_path": str(triage_model_path),
            "encoder": causal_evidence_bank.model_name,
            "device": str(causal_evidence_bank.device),
            "refresh_stride": int(causal_evidence_bank.refresh_stride),
            "robot_view_labels_used": False,
            "candidate_robot_images_used": False,
            "ground_truth_boxes_used": False,
            "future_frames_used": False,
        }
    component_status["causal_evidence_bank"] = causal_evidence_bank_status

    claim_verifier: Optional[LiveClaimVerifier] = None
    claim_verifier_status: dict[str, object] = {"enabled": False}
    if bool(config.claim_verifier.enabled):
        scorer: Optional[PrototypeEvidenceScorer] = None
        scorer_path_text = str(config.claim_verifier.evidence_scorer_path).strip()
        scorer_path: Optional[Path] = None
        if scorer_path_text:
            scorer_path = Path(scorer_path_text).expanduser()
            if not scorer_path.is_absolute():
                scorer_path = Path(__file__).resolve().parents[1] / scorer_path
            if scorer_path.exists():
                scorer = PrototypeEvidenceScorer.load(scorer_path)
            elif causal_evidence_bank is None:
                raise FileNotFoundError(
                    f"Claim evidence scorer not found: {scorer_path}"
                )
        if scorer is None and causal_evidence_bank is None:
            raise RuntimeError(
                "Claim verifier requires either an evidence scorer or the "
                "causal evidence bank"
            )
        claim_verifier = LiveClaimVerifier(
            scorer,
            support_threshold=config.claim_verifier.support_threshold,
            contradiction_threshold=config.claim_verifier.contradiction_threshold,
            counterfactual_margin=config.claim_verifier.counterfactual_margin,
            admissibility_gate_enabled=config.claim_verifier.admissibility_gate_enabled,
            memory_gate_enabled=config.claim_verifier.memory_gate_enabled,
            specialized_counterfactual_enabled=config.claim_verifier.specialized_counterfactual_enabled,
            prerequisite_bootstrap_enabled=config.claim_verifier.prerequisite_bootstrap_enabled,
            prerequisite_confirmation_frames=config.claim_verifier.prerequisite_confirmation_frames,
            ema_decay=config.claim_verifier.ema_decay,
            require_step_match_for_support=config.claim_verifier.require_step_match_for_support,
            product_family=config.claim_verifier.product_family,
            family_min_confidence=config.claim_verifier.family_min_confidence,
            family_margin=config.claim_verifier.family_margin,
            family_confirmation_frames=config.claim_verifier.family_confirmation_frames,
        )
        claim_verifier_status = {
            "enabled": True,
            "backend": (
                "causal_evidence_triage"
                if causal_evidence_bank is not None
                else "prototype_evidence_scorer"
            ),
            "evidence_scorer_path": str(scorer_path or ""),
            "fallback_scorer_loaded": bool(scorer is not None),
            "training_source": (
                scorer.metadata.get("training_source", "")
                if scorer is not None
                else "assistant_replay_only"
            ),
            "uses_robot_view_training": (
                bool(scorer.metadata.get("uses_robot_view_training", False))
                if scorer is not None
                else False
            ),
            "support_threshold": float(config.claim_verifier.support_threshold),
            "contradiction_threshold": float(config.claim_verifier.contradiction_threshold),
            "counterfactual_margin": float(config.claim_verifier.counterfactual_margin),
            "admissibility_gate_enabled": bool(config.claim_verifier.admissibility_gate_enabled),
            "memory_gate_enabled": bool(config.claim_verifier.memory_gate_enabled),
            "specialized_counterfactual_enabled": bool(config.claim_verifier.specialized_counterfactual_enabled),
            "prerequisite_bootstrap_enabled": bool(config.claim_verifier.prerequisite_bootstrap_enabled),
            "prerequisite_confirmation_frames": int(config.claim_verifier.prerequisite_confirmation_frames),
            "ema_decay": float(config.claim_verifier.ema_decay),
        }
    component_status["claim_verifier"] = claim_verifier_status
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
        track_evidence_builder=track_evidence_builder,
        scene_evidence_tracker=scene_evidence_tracker,
        segmentation_backend=segmentation_backend,
        event_bus=event_bus,
        ops_tracker=ops_tracker,
        claim_verifier=claim_verifier,
        causal_evidence_bank=causal_evidence_bank,
        component_status=component_status,
    )
