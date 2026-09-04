"""Build robot-view INSPECT observations from RGB frames.

This module is the robot-side counterpart of the INSPECT Trace Engine. It turns
robot or active-camera images into the same lightweight evidence vocabulary used
by the procedural state verifier, so robot experiments do not depend on
hand-written ``RobotObservation`` JSON.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2

from ..components import (
    DepthContextSelector,
    GeometryAwareSceneGraphBuilder,
    GradientGeometryProvider,
    HandObjectContactEstimator,
    IdentityTemporalFusion,
    MoGeGeometryProvider,
    NullSegmentationBackend,
    ByteTrackLiteFusion,
    ProceduralSceneEvidenceTracker,
    TrackEvidenceAggregator,
    WindowIoUFusion,
    YOLODetectorComponent,
    build_segmentation_backend,
)
from ..config import AppConfig
from ..core_types import Detection, GeometryFrame, SceneGraphFrame
from .state_spec import load_state_specs
from .types import RobotObservation, StateSpec


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

_DETECTION_META_KEYS = (
    "raw_name",
    "source",
    "tracked",
    "track_id",
    "tracker",
    "track_hits",
    "track_age",
    "track_class_posterior",
    "track_class_margin",
    "identity_safe",
    "identity_commit_conf",
    "identity_commit_track_margin",
    "proposal_only",
)


def _checkpoint_identity(path: str) -> Dict[str, str]:
    checkpoint = Path(path)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"name": checkpoint.name, "sha256": digest.hexdigest()}


def _serialize_detection(detection: Detection) -> Dict[str, Any]:
    meta = dict(getattr(detection, "meta", {}) or {})
    serialized_meta = {
        key: meta[key]
        for key in _DETECTION_META_KEYS
        if key in meta and isinstance(meta[key], (str, int, float, bool, type(None)))
    }
    return {
        "name": _slug(detection.name),
        "xyxy": [float(v) for v in detection.xyxy],
        "confidence": float(detection.confidence),
        "meta": serialized_meta,
    }

def _slug(text: object) -> str:
    value = str(text or "").strip().lower().replace("-", "_").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value).strip("_")


def _box_area(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))


def _center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def _edge_gap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(ax1 - bx2, bx1 - ax2, 0.0)
    dy = max(ay1 - by2, by1 - ay2, 0.0)
    return float((dx * dx + dy * dy) ** 0.5)


@dataclass(frozen=True)
class RobotFrameRecord:
    """One robot-view frame plus optional evaluation metadata."""

    image_path: Path
    observation_id: str = ""
    frame_index: Optional[int] = None
    view_id: str = ""
    episode_id: str = ""
    reset_tracker: bool = False
    prev_state: str = ""
    candidate_states: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized_metadata(self) -> Dict[str, Any]:
        return dict(self.metadata or {})


def _load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return [dict(item) for item in json.loads(text)]
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(dict(json.loads(line)))
    return records


def _candidate_states_from_text(value: object) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip().upper() for item in value if str(item).strip())
    return tuple(item.strip().upper() for item in str(value).split(",") if item.strip())


def _manifest_record(payload: Dict[str, Any], root: Path) -> RobotFrameRecord:
    image_value = payload.get("image_path") or payload.get("rgb_path") or payload.get("path") or payload.get("file")
    if not image_value:
        raise ValueError(f"Robot frame manifest record is missing an image path: {payload}")
    image_path = Path(str(image_value))
    if not image_path.is_absolute():
        image_path = root / image_path
    metadata = dict(payload.get("metadata") or {})
    for key in (
        "ground_truth_state",
        "verified_state",
        "label",
        "state",
        "current_state",
        "postcondition_verified",
        "next_step_admissible",
        "next_step_allowed",
        "valid_next_step",
        "anomaly",
        "failure",
        "episode_id",
        "sequence_id",
        "run_id",
        "frame_index",
        "timestamp",
        "camera_pose",
        "robot_state",
    ):
        if key in payload and key not in metadata:
            metadata[key] = payload[key]
    frame_index: Optional[int] = None
    frame_value = payload.get("frame_index", payload.get("frame"))
    if frame_value is not None:
        try:
            frame_index = int(frame_value)
        except (TypeError, ValueError):
            frame_index = None
    episode_id = str(
        payload.get("episode_id")
        or payload.get("episode")
        or payload.get("sequence_id")
        or payload.get("run_id")
        or ""
    )
    return RobotFrameRecord(
        image_path=image_path,
        observation_id=str(payload.get("observation_id") or payload.get("id") or image_path.stem),
        frame_index=frame_index,
        view_id=str(payload.get("view_id") or payload.get("view") or ""),
        episode_id=episode_id,
        reset_tracker=bool(payload.get("reset_tracker", False)),
        prev_state=str(payload.get("prev_state") or payload.get("prev_step") or "").upper(),
        candidate_states=_candidate_states_from_text(payload.get("candidate_states")),
        metadata=metadata,
    )


def iter_robot_frame_records(
    frames_path: Path,
    manifest_path: Optional[Path] = None,
    default_view_id: str = "",
    default_prev_state: str = "",
    default_candidate_states: Sequence[str] = (),
    max_frames: Optional[int] = None,
) -> Iterator[RobotFrameRecord]:
    """Yield robot frame records from a directory, image file, or manifest."""

    frames_path = Path(frames_path)
    records: List[RobotFrameRecord] = []
    if manifest_path is not None:
        root = frames_path if frames_path.is_dir() else frames_path.parent
        records = [_manifest_record(payload, root=root) for payload in _load_json_or_jsonl(Path(manifest_path))]
    elif frames_path.suffix.lower() in {".json", ".jsonl"}:
        records = [_manifest_record(payload, root=frames_path.parent) for payload in _load_json_or_jsonl(frames_path)]
    elif frames_path.is_dir():
        image_paths = [
            path
            for path in sorted(frames_path.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        records = [RobotFrameRecord(image_path=path, observation_id=path.stem) for path in image_paths]
    elif frames_path.is_file() and frames_path.suffix.lower() in IMAGE_EXTENSIONS:
        records = [RobotFrameRecord(image_path=frames_path, observation_id=frames_path.stem)]
    else:
        raise FileNotFoundError(f"No robot frames found at {frames_path}")

    default_candidates = tuple(str(item).upper() for item in default_candidate_states if str(item).strip())
    limit = len(records) if max_frames is None else max(0, min(len(records), int(max_frames)))
    for record in records[:limit]:
        yield RobotFrameRecord(
            image_path=record.image_path,
            observation_id=record.observation_id or record.image_path.stem,
            frame_index=record.frame_index,
            view_id=record.view_id or default_view_id,
            episode_id=record.episode_id,
            reset_tracker=record.reset_tracker,
            prev_state=record.prev_state or str(default_prev_state or "").upper(),
            candidate_states=record.candidate_states or default_candidates,
            metadata=record.normalized_metadata(),
        )


def write_robot_observations_jsonl(observations: Iterable[RobotObservation], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for observation in observations:
            handle.write(json.dumps(observation.to_dict(), sort_keys=True) + "\n")


def _counts(detections: Iterable[Detection]) -> Dict[str, int]:
    counter = Counter(_slug(detection.name) for detection in detections if _slug(detection.name))
    return {str(key): int(value) for key, value in sorted(counter.items()) if value > 0}


def _scene_relation_facts(scene_graph: SceneGraphFrame, min_score: float = 0.05) -> List[List[str]]:
    facts: List[List[str]] = []
    for relation in scene_graph.relations:
        if float(relation.score) < min_score:
            continue
        facts.append([
            _slug(relation.subject_name),
            _slug(relation.predicate),
            _slug(relation.object_name),
        ])
    return facts


def _extra_bbox_relation_facts(
    detections: Sequence[Detection],
    near_margin_px: float = 36.0,
    inside_overlap_ratio: float = 0.55,
    alignment_ratio: float = 0.18,
) -> List[List[str]]:
    """Infer view-agnostic relation candidates not produced by the generic scene graph."""

    facts: List[List[str]] = []
    for i, left in enumerate(detections):
        left_name = _slug(left.name)
        left_box = tuple(float(v) for v in left.xyxy)
        left_area = max(1.0, _box_area(left_box))
        left_center = _center(left_box)
        left_w = max(1.0, left_box[2] - left_box[0])
        left_h = max(1.0, left_box[3] - left_box[1])
        for j, right in enumerate(detections):
            if i == j:
                continue
            right_name = _slug(right.name)
            right_box = tuple(float(v) for v in right.xyxy)
            right_area = max(1.0, _box_area(right_box))
            right_center = _center(right_box)
            right_w = max(1.0, right_box[2] - right_box[0])
            right_h = max(1.0, right_box[3] - right_box[1])
            inter = _intersection(left_box, right_box)
            if inter / min(left_area, right_area) >= inside_overlap_ratio and left_area <= right_area:
                facts.append([left_name, "inside", right_name])
            if _edge_gap(left_box, right_box) <= near_margin_px:
                facts.append([left_name, "near", right_name])
            dx = abs(left_center[0] - right_center[0])
            dy = abs(left_center[1] - right_center[1])
            if dx <= alignment_ratio * max(left_w, right_w) or dy <= alignment_ratio * max(left_h, right_h):
                facts.append([left_name, "aligned_with", right_name])
    return facts


def _dedupe_facts(facts: Iterable[Sequence[str]]) -> List[List[str]]:
    seen = set()
    out: List[List[str]] = []
    for fact in facts:
        if len(fact) < 3:
            continue
        key = (_slug(fact[0]), _slug(fact[1]), _slug(fact[2]))
        if not all(key) or key in seen:
            continue
        seen.add(key)
        out.append([key[0], key[1], key[2]])
    return out


def _relation_counts(facts: Iterable[Sequence[str]]) -> Dict[str, int]:
    counter = Counter(_slug(fact[1]) for fact in facts if len(fact) >= 3)
    return {str(key): int(value) for key, value in sorted(counter.items()) if key and value > 0}


def _contact_evidence_keys(interaction: object) -> List[str]:
    keys = set()
    for item in getattr(interaction, "contact_facts", []) or []:
        if len(item) >= 3:
            keys.add(f"contact:{_slug(item[0])}:{_slug(item[1])}:{_slug(item[2])}")
    for name, count in dict(getattr(interaction, "contact_counts", {}) or {}).items():
        if int(count) > 0:
            keys.add(f"interaction:active_object:{_slug(name)}")
    active_object = str(getattr(interaction, "active_object", "") or "")
    if active_object:
        keys.add(f"interaction:active_object:{_slug(active_object)}")
    contact_phase = str(getattr(interaction, "contact_phase", "") or "")
    if contact_phase and contact_phase != "none":
        keys.add(f"interaction:contact_phase:{_slug(contact_phase)}")
    interaction_target = str(getattr(interaction, "interaction_target", "") or "")
    if interaction_target:
        keys.add(f"interaction_target:{_slug(interaction_target)}")
    transition_likelihood = float(getattr(interaction, "transition_likelihood", 0.0) or 0.0)
    if transition_likelihood >= 0.5:
        keys.add("transition:contact_driven_change")
    return sorted(keys)


def _track_evidence_keys(track_evidence: object) -> List[str]:
    keys = set()
    for item in getattr(track_evidence, "evidence_keys", []) or []:
        if str(item).strip():
            keys.add(str(item).strip())
    for name, count in dict(getattr(track_evidence, "stable_track_counts", {}) or {}).items():
        if int(count) > 0:
            keys.add(f"track:stable_object:{_slug(name)}")
    return sorted(keys)


class RobotObservationBuilder:
    """Run robot-frame perception and emit INSPECT ``RobotObservation`` records."""

    def __init__(
        self,
        config: AppConfig,
        yolo_weights: str,
        device: str = "cpu",
        state_specs: Optional[Dict[str, StateSpec]] = None,
    ) -> None:
        self.config = config
        self.state_specs = dict(state_specs or {})
        self.detector_checkpoint = _checkpoint_identity(yolo_weights)
        self.detector = YOLODetectorComponent(
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
            track_with_model=str(config.temporal.backend).strip().lower()
            in {"ultralytics_botsort", "ultralytics_bytetrack", "botsort", "bytetrack"},
            tracker_config=str(config.temporal.tracker_config or "botsort.yaml"),
            track_class_smoothing_alpha=config.detection.track_class_smoothing_alpha,
            track_class_switch_margin=config.detection.track_class_switch_margin,
            identity_commit_conf=config.detection.identity_commit_conf,
            identity_commit_track_margin=config.detection.identity_commit_track_margin,
        )
        geometry_backend = str(config.geometry.backend).strip().lower()
        if geometry_backend in {"moge", "moge2", "mo-ge-2"}:
            self.geometry_provider = MoGeGeometryProvider(
                model_name=config.geometry.moge_model,
                device=device,
                use_fp16=config.geometry.use_fp16,
                resolution_level=config.geometry.resolution_level,
                apply_mask=config.geometry.apply_mask,
            )
        else:
            self.geometry_provider = GradientGeometryProvider(device=device)
        temporal_backend = str(config.temporal.backend).strip().lower()
        if temporal_backend in {"ultralytics_botsort", "ultralytics_bytetrack", "botsort", "bytetrack"}:
            self.temporal_fusion = IdentityTemporalFusion()
        elif temporal_backend == "window_iou":
            self.temporal_fusion = WindowIoUFusion(window=config.temporal.window, iou_thr=config.temporal.iou_thr)
        else:
            self.temporal_fusion = ByteTrackLiteFusion(
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
        self.track_evidence_builder = TrackEvidenceAggregator(
            enabled=config.track_evidence.enabled,
            stable_hits=config.track_evidence.stable_hits,
            stable_confidence=config.track_evidence.stable_confidence,
            motion_px_threshold=config.track_evidence.motion_px_threshold,
            max_tracks=config.track_evidence.max_tracks,
        )
        self.scene_evidence_tracker = ProceduralSceneEvidenceTracker(
            enabled=config.scene_evidence.enabled,
            relation_change_memory=config.scene_evidence.relation_change_memory,
            max_objects=config.scene_evidence.max_objects,
            max_relations=config.scene_evidence.max_relations,
            max_changes=config.scene_evidence.max_changes,
        )
        self.context_selector = DepthContextSelector(tau_p=config.depth_context.tau_p, tau_d=config.depth_context.tau_d)
        self.scene_graph_builder = GeometryAwareSceneGraphBuilder(
            enabled=config.scene_graph.enabled,
            depth_margin=config.scene_graph.depth_margin,
            contact_pixel_gap=config.scene_graph.contact_pixel_gap,
            contact_depth_gap=config.scene_graph.contact_depth_gap,
            support_vertical_gap=config.scene_graph.support_vertical_gap,
            support_overlap_ratio=config.scene_graph.support_overlap_ratio,
            max_relations=config.scene_graph.max_relations,
        )
        self.contact_estimator = HandObjectContactEstimator(
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
            self.segmentation_backend = build_segmentation_backend(
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
            self.segmentation_backend = NullSegmentationBackend()

    def reset_temporal_state(self) -> None:
        """Reset stateful perception at episode or discontinuous-view boundaries."""

        for module in (
            self.detector,
            self.temporal_fusion,
            self.track_evidence_builder,
            self.scene_evidence_tracker,
            self.contact_estimator,
        ):
            reset = getattr(module, "reset", None)
            if callable(reset):
                reset()

    def build_record(self, record: RobotFrameRecord) -> RobotObservation:
        frame = cv2.imread(str(record.image_path))
        if frame is None:
            raise FileNotFoundError(f"Could not read robot frame: {record.image_path}")
        raw_detections = self.detector.detect(frame, conf=self.config.detection.conf, tta=self.config.detection.tta)
        tracked_detections = self.temporal_fusion.update(raw_detections)
        detections = tracked_detections or raw_detections
        track_evidence = self.track_evidence_builder.build(tracked_detections)
        geometry = self.geometry_provider.infer(frame)
        relevant, nearest_index = self.context_selector.select(detections, geometry)
        scene_graph = self.scene_graph_builder.build(
            detections=detections,
            geometry=geometry,
            relevant_detections=relevant,
            nearest_index=nearest_index,
        )
        prompts = list(dict.fromkeys([*self.config.segmentation.prompts, *(_slug(detection.name) for detection in detections)]))
        segmentation_masks = self.segmentation_backend.segment(frame, prompts)
        interaction = self.contact_estimator.estimate(
            detections=detections,
            geometry=geometry,
            scene_graph=scene_graph,
            relevant_detections=relevant,
            segmentation_masks=segmentation_masks,
        )
        scene_evidence = self.scene_evidence_tracker.build(
            detections=detections,
            scene_graph=scene_graph,
            track_evidence=track_evidence,
            interaction_evidence=interaction,
        )
        relation_facts = _dedupe_facts(
            [
                *_scene_relation_facts(scene_graph),
                *_extra_bbox_relation_facts(
                    detections,
                    near_margin_px=self.config.interaction.near_margin_px,
                ),
            ]
        )
        evidence_keys = sorted(
            set(_contact_evidence_keys(interaction))
            | set(_track_evidence_keys(track_evidence))
            | set(scene_evidence.evidence_keys)
        )
        metadata = record.normalized_metadata()
        metadata.update(
            {
                "rgb_path": str(record.image_path),
                "frame_shape": [int(frame.shape[0]), int(frame.shape[1])],
                "frame_height": int(frame.shape[0]),
                "frame_width": int(frame.shape[1]),
                "frame_index": record.frame_index,
                "episode_id": record.episode_id,
                "reset_tracker": bool(record.reset_tracker),
                "observation_source": "robot_image_auto",
                "detector": "ultralytics_yolo",
                "detector_checkpoint": dict(self.detector_checkpoint),
                "tracker": str(self.config.temporal.backend),
                "segmentation_backend": getattr(self.segmentation_backend, "backend_name", "unknown"),
                "geometry_provider": str(getattr(geometry, "extras", {}).get("provider", "")),
                "num_raw_detections": len(raw_detections),
                "num_detections": len(detections),
                "num_tracks": len(track_evidence.objects),
                "num_relations": len(relation_facts),
                "num_contact_facts": len(getattr(interaction, "contact_facts", []) or []),
                "detections": [_serialize_detection(detection) for detection in detections],
                "raw_detections": [_serialize_detection(detection) for detection in raw_detections],

                "track_evidence": {
                    "track_counts": dict(track_evidence.track_counts),
                    "stable_track_counts": dict(track_evidence.stable_track_counts),
                    "track_evidence_keys": list(track_evidence.evidence_keys),
                    "track_objects": [dict(item) for item in track_evidence.objects],
                },
                "scene_evidence": {
                    "evidence_keys": list(scene_evidence.evidence_keys),
                    "visible_objects": list(scene_evidence.visible_objects),
                    "stable_objects": list(scene_evidence.stable_objects),
                    "active_objects": list(scene_evidence.active_objects),
                    "moving_objects": list(scene_evidence.moving_objects),
                    "relation_keys": list(scene_evidence.relation_keys),
                    "relation_change_keys": list(scene_evidence.relation_change_keys),
                    "transition_keys": list(scene_evidence.transition_keys),
                    "objects": [dict(item) for item in scene_evidence.objects],
                    "relations": [dict(item) for item in scene_evidence.relations],
                    "extras": dict(scene_evidence.extras),
                },
                "contact": {
                    "active_object": str(getattr(interaction, "active_object", "") or ""),
                    "contact_phase": str(getattr(interaction, "contact_phase", "") or ""),
                    "transition_likelihood": float(getattr(interaction, "transition_likelihood", 0.0) or 0.0),
                    "interaction_target": str(getattr(interaction, "interaction_target", "") or ""),
                },
            }
        )
        return RobotObservation(
            observation_id=record.observation_id,
            frame_index=record.frame_index,
            prev_state=record.prev_state or None,
            view_id=record.view_id,
            visible_counts=_counts(detections),
            relevant_counts=_counts(relevant),
            relation_counts=_relation_counts(relation_facts),
            relation_facts=relation_facts,
            evidence_keys=evidence_keys,
            candidate_states=[str(item).upper() for item in record.candidate_states],
            metadata=metadata,
        )

    def build_many(self, records: Iterable[RobotFrameRecord]) -> List[RobotObservation]:
        observations: List[RobotObservation] = []
        previous_episode: Optional[str] = None
        previous_view: Optional[str] = None
        for record in records:
            episode_changed = previous_episode is not None and record.episode_id != previous_episode
            view_changed = previous_view is not None and record.view_id != previous_view
            if record.reset_tracker or episode_changed or view_changed:
                self.reset_temporal_state()
            observations.append(self.build_record(record))
            previous_episode = record.episode_id
            previous_view = record.view_id
        return observations


def candidate_states_from_specs(path: Optional[Path]) -> List[str]:
    if path is None:
        return []
    specs = load_state_specs(Path(path))
    return sorted({spec.state_id.upper() for spec in specs.values() if spec.state_id})


def build_robot_observations_from_frames(
    frames_path: Path,
    output_path: Path,
    config: AppConfig,
    yolo_weights: str,
    device: str = "cpu",
    manifest_path: Optional[Path] = None,
    candidate_states: Sequence[str] = (),
    prev_state: str = "",
    view_id: str = "",
    state_specs_path: Optional[Path] = None,
    max_frames: Optional[int] = None,
) -> List[RobotObservation]:
    """Build and write robot observations from RGB frames."""

    specs = load_state_specs(Path(state_specs_path)) if state_specs_path is not None else {}
    default_candidates = list(candidate_states) or sorted({spec.state_id.upper() for spec in specs.values() if spec.state_id})
    records = list(
        iter_robot_frame_records(
            frames_path=Path(frames_path),
            manifest_path=Path(manifest_path) if manifest_path is not None else None,
            default_view_id=view_id,
            default_prev_state=prev_state,
            default_candidate_states=default_candidates,
            max_frames=max_frames,
        )
    )
    builder = RobotObservationBuilder(
        config=config,
        yolo_weights=str(yolo_weights),
        device=device,
        state_specs=specs,
    )
    observations = builder.build_many(records)
    write_robot_observations_jsonl(observations, Path(output_path))
    return observations
