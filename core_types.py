"""Shared dataclasses used across the INSPECT runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single detection in XYXY image coordinates."""

    name: str
    xyxy: Tuple[float, float, float, float]
    confidence: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepPrediction:
    """Dense step scores produced by one expert."""

    step_id: str
    confidence: float
    scores: Dict[str, float]
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeometryFrame:
    """Dense geometry predicted for a single RGB frame."""

    depth: np.ndarray
    points: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None
    intrinsics: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneGraphRelation:
    """A directed scene-graph relation between two detected objects."""

    subject_index: int
    object_index: int
    subject_name: str
    predicate: str
    object_name: str
    score: float
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneGraphFrame:
    """Scene-graph relations and aggregate relation statistics for one frame."""

    relations: List[SceneGraphRelation]
    stats: Dict[str, float] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FusionResult:
    """Final fused step decision with diagnostic metadata."""

    step_id: str
    confidence: float
    scores: Dict[str, float]
    runner_up: Optional[str]
    gates: Dict[str, float] = field(default_factory=dict)
    contributions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeedbackEvent:
    """A human supervision event."""

    label: str
    strength: float = 1.0
    accepted: bool = False
    source: str = "human"
    note: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalInput:
    """Structured visual query passed to retrieval experts."""

    frame_bgr: np.ndarray
    focus_detection: Optional[Detection] = None


@dataclass(frozen=True)
class RuntimeAction:
    """A keyboard or voice action consumed by the live assistant."""

    action: str
    source: str = "runtime"
    label: Optional[str] = None
    text: str = ""


@dataclass(frozen=True)
class EvidenceToken:
    """Compact structured frame evidence used by streaming temporal step experts."""

    frame_index: int
    prev_step: Optional[str]
    visible_counts: Dict[str, int]
    relevant_counts: Dict[str, int]
    relation_counts: Dict[str, int]
    relation_facts: List[Tuple[str, str, str]]
    state_scores: Dict[str, float]
    retrieval_scores: Dict[str, float]
    memory_scores: Dict[str, float]
    state_confidence: float
    retrieval_confidence: float
    memory_confidence: float
    memory_active: bool = False
    memory_reason: str = ""
    review_action: str = ""
    review_reason: str = ""
    has_visual_evidence: bool = True
    hand_object_contacts: List[Dict[str, Any]] = field(default_factory=list)
    contact_counts: Dict[str, int] = field(default_factory=dict)
    contact_facts: List[Tuple[str, str, str]] = field(default_factory=list)
    active_object: str = ""
    interaction_target: str = ""
    contact_phase: str = ""
    transition_likelihood: float = 0.0
    track_counts: Dict[str, int] = field(default_factory=dict)
    stable_track_counts: Dict[str, int] = field(default_factory=dict)
    track_ids: Dict[str, List[int]] = field(default_factory=dict)
    track_confidences: Dict[str, float] = field(default_factory=dict)
    track_hits: Dict[str, int] = field(default_factory=dict)
    track_ages: Dict[str, int] = field(default_factory=dict)
    track_motion: Dict[str, float] = field(default_factory=dict)
    track_evidence_keys: List[str] = field(default_factory=list)
    track_objects: List[Dict[str, Any]] = field(default_factory=list)
    scene_evidence_keys: List[str] = field(default_factory=list)
    scene_visible_objects: List[str] = field(default_factory=list)
    scene_stable_objects: List[str] = field(default_factory=list)
    scene_active_objects: List[str] = field(default_factory=list)
    scene_moving_objects: List[str] = field(default_factory=list)
    scene_relation_keys: List[str] = field(default_factory=list)
    scene_relation_change_keys: List[str] = field(default_factory=list)
    scene_transition_keys: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class HandObjectContact:
    """One estimated hand/tool-object interaction cue."""

    hand_name: str
    object_name: str
    contact_score: float
    distance_px: float
    overlap_ratio: float
    depth_gap: float
    phase: str = "none"
    hand_index: int = -1
    object_index: int = -1
    source: str = "bbox_geometry"
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InteractionEvidence:
    """Frame-level hand/tool-object interaction evidence."""

    contacts: List[HandObjectContact]
    active_object: str = ""
    interaction_target: str = ""
    contact_phase: str = "none"
    transition_likelihood: float = 0.0
    contact_counts: Dict[str, int] = field(default_factory=dict)
    contact_facts: List[Tuple[str, str, str]] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrackEvidenceFrame:
    """Track-level object hypotheses derived from temporally fused detections."""

    objects: List[Dict[str, Any]]
    track_counts: Dict[str, int] = field(default_factory=dict)
    stable_track_counts: Dict[str, int] = field(default_factory=dict)
    track_ids: Dict[str, List[int]] = field(default_factory=dict)
    track_confidences: Dict[str, float] = field(default_factory=dict)
    track_hits: Dict[str, int] = field(default_factory=dict)
    track_ages: Dict[str, int] = field(default_factory=dict)
    track_motion: Dict[str, float] = field(default_factory=dict)
    evidence_keys: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneEvidenceFrame:
    """Scene-level procedural evidence independent of a specific workspace layout."""

    evidence_keys: List[str] = field(default_factory=list)
    visible_objects: List[str] = field(default_factory=list)
    stable_objects: List[str] = field(default_factory=list)
    active_objects: List[str] = field(default_factory=list)
    moving_objects: List[str] = field(default_factory=list)
    relation_keys: List[str] = field(default_factory=list)
    relation_change_keys: List[str] = field(default_factory=list)
    transition_keys: List[str] = field(default_factory=list)
    objects: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SegmentationMask:
    """Promptable segmentation result used by interaction/contact estimators."""

    name: str
    xyxy: Tuple[float, float, float, float]
    score: float
    mask: Any = None
    object_id: Optional[int] = None
    source: str = "segmentation"
    extras: Dict[str, Any] = field(default_factory=dict)
