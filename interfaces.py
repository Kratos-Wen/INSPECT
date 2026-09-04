"""Protocol interfaces for modular pipeline components."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np

from .core_types import Detection, EvidenceToken, FeedbackEvent, FusionResult, GeometryFrame, SceneGraphFrame, StepPrediction


class Detector(Protocol):
    """Frame detector protocol."""

    def detect(self, frame_bgr: np.ndarray, conf: float, tta: bool) -> List[Detection]:
        """Run object detection on a single BGR frame."""


class GeometryProvider(Protocol):
    """Geometry provider protocol."""

    def infer(self, frame_bgr: np.ndarray) -> GeometryFrame:
        """Return geometry for one BGR frame."""


class TemporalFusion(Protocol):
    """Temporal fusion protocol."""

    def update(self, detections: List[Detection]) -> List[Detection]:
        """Update temporal state and return fused detections."""


class ContextSelector(Protocol):
    """Geometry-guided context selection protocol."""

    def select(
        self,
        detections: List[Detection],
        geometry: GeometryFrame,
    ) -> Tuple[List[Detection], Optional[int]]:
        """Select relevant detections and return the nearest index."""


class SceneGraphBuilder(Protocol):
    """Scene-graph builder protocol."""

    def build(
        self,
        detections: List[Detection],
        geometry: GeometryFrame,
        relevant_detections: List[Detection],
        nearest_index: Optional[int],
    ) -> SceneGraphFrame:
        """Build a scene graph for the current frame."""


class StepExpert(Protocol):
    """Step-expert protocol."""

    def predict(self, payload: object) -> StepPrediction:
        """Predict dense step scores from the given payload."""


class StepCalibrator(Protocol):
    """Step-score calibration protocol."""

    def calibrate(self, prediction: StepPrediction) -> StepPrediction:
        """Transform an expert prediction before fusion."""


class StepFusion(Protocol):
    """Fusion head protocol."""

    def fuse(
        self,
        expert_predictions: Dict[str, StepPrediction],
        prev_step: Optional[str] = None,
        token: Optional[EvidenceToken] = None,
    ) -> FusionResult:
        """Fuse expert outputs into one final decision."""

    def apply_feedback(
        self,
        feedback: FeedbackEvent,
        expert_predictions: Dict[str, StepPrediction],
        fusion_result: FusionResult,
    ) -> None:
        """Update fusion parameters online from human feedback."""

    def save(self, path: Path) -> None:
        """Persist the fusion state."""


class StabilityPolicy(Protocol):
    """Stability policy protocol."""

    def update(
        self,
        detections: List[Detection],
        fused_step: str,
    ) -> Tuple[bool, Tuple[Tuple[str, int], ...]]:
        """Return whether the current state is stable and the signature used."""


class FeedbackProvider(Protocol):
    """Interactive or programmatic feedback provider."""

    def request(
        self,
        fusion_result: FusionResult,
        expert_predictions: Dict[str, StepPrediction],
        evidence_token: Optional[EvidenceToken] = None,
        review_decision: Optional[object] = None,
    ) -> Optional[FeedbackEvent]:
        """Request optional feedback for the current stable decision."""


class RunLogger(Protocol):
    """Run logger protocol."""

    @property
    def run_dir(self) -> Path:
        """Return the active run directory."""

    def log_iteration(self, payload: Dict[str, object]) -> None:
        """Write one iteration record."""

    def log_feedback(self, payload: Dict[str, object]) -> None:
        """Write one feedback record."""

    def close(self) -> None:
        """Close any open file handles."""
