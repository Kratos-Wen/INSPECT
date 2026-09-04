"""Dataclasses shared by the episodic memory subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..core_types import StepPrediction

Signature = Tuple[Tuple[str, int], ...]


@dataclass(frozen=True)
class MemoryObservation:
    """Structured evidence snapshot used for recall and capture."""

    frame_index: int
    prev_step: Optional[str]
    signature: Signature
    relevant_signature: Signature
    relation_signature: Signature
    geometry_stats: Dict[str, float]
    scene_graph_stats: Dict[str, float]
    expert_steps: Dict[str, str]
    expert_confidences: Dict[str, float]
    expert_scores: Dict[str, Dict[str, float]]
    ensemble_step: str
    ensemble_confidence: float
    ensemble_margin: float
    expert_disagreement: float
    ensemble_scores: Dict[str, float]
    num_detections: int
    num_relevant: int
    has_visual_evidence: bool
    visual_embedding: List[float]


@dataclass(frozen=True)
class MemoryRecord:
    """Persisted episodic memory record."""

    record_id: str
    run_id: str
    frame_index: int
    timestamp: float
    step_id: str
    prev_step: Optional[str]
    source: str
    trust: float
    accepted: bool
    note: str
    signature: Signature
    relevant_signature: Signature
    relation_signature: Signature
    geometry_stats: Dict[str, float]
    scene_graph_stats: Dict[str, float]
    expert_steps: Dict[str, str]
    expert_confidences: Dict[str, float]
    expert_scores: Dict[str, Dict[str, float]]
    num_detections: int
    num_relevant: int
    has_visual_evidence: bool
    visual_embedding: List[float]
    vector: List[float]
    tokens: List[str]


@dataclass(frozen=True)
class MemoryMatch:
    """One retrieved memory together with its matching diagnostics."""

    record: MemoryRecord
    store_name: str
    total_score: float
    vector_score: float
    token_score: float
    prev_step_bonus: float
    recency_weight: float


@dataclass(frozen=True)
class MemoryRecallResult:
    """Recall output returned by the memory manager."""

    prediction: StepPrediction
    matches: List[MemoryMatch] = field(default_factory=list)
    recalled: bool = False
    reason: str = ""
