"""Default component implementations for the modular step pipeline."""

from .calibration import IdentityStepCalibrator
from .context import DepthContextSelector
from .contact import HandObjectContactEstimator
from .detection import YOLODetectorComponent
from .evidence_encoder import StructuredEvidenceEncoder
from .feedback import ConsoleFeedbackProvider, TimelineFeedbackProvider
from .geometry import GradientGeometryProvider, MoGeGeometryProvider
from .kb import KnowledgeBase
from .keypoint_roles import CausalKeypointRoleBridge
from .logging import JsonlCsvLogger
from .online_fusion import AdaptiveExpertFusion, LinearMarginFusion
from .scene_graph import GeometryAwareSceneGraphBuilder
from .segmentation import (
    NullSegmentationBackend,
    SAM3OfficialImageSegmentationBackend,
    SAM3VideoSegmentationBackend,
    build_segmentation_backend,
)
from .retrieval import DummyRetrievalExpert, GalleryRetrievalExpert
from .rules import RuleBasedStepExpert
from .scene_evidence import ProceduralSceneEvidenceTracker
from .step_graph import CompiledStepGraphPrior
from .stability import DualStabilityTracker
from .temporal import ByteTrackLiteFusion, IdentityTemporalFusion, WindowIoUFusion
from .temporal_step import CausalTemporalStepExpert, StreamingGRUTemporalStepExpert, TemporalEvidenceVectorizer
from .track_evidence import TrackEvidenceAggregator
from .timeline_store import EvidenceTimelineStore
from .tts import AssistantSpeechService
from .ui import OpenCVRuntimeUI
from .visual_embedding import SharedVisualEncoder
from .voice import VoiceCommandService

__all__ = [
    "ConsoleFeedbackProvider",
    "TimelineFeedbackProvider",
    "AdaptiveExpertFusion",
    "AssistantSpeechService",
    "ByteTrackLiteFusion",
    "CausalTemporalStepExpert",
    "CompiledStepGraphPrior",
    "DepthContextSelector",
    "DummyRetrievalExpert",
    "DualStabilityTracker",
    "EvidenceTimelineStore",
    "GalleryRetrievalExpert",
    "GeometryAwareSceneGraphBuilder",
    "GradientGeometryProvider",
    "HandObjectContactEstimator",
    "IdentityStepCalibrator",
    "IdentityTemporalFusion",
    "JsonlCsvLogger",
    "KnowledgeBase",
    "CausalKeypointRoleBridge",
    "LinearMarginFusion",
    "MoGeGeometryProvider",
    "NullSegmentationBackend",
    "ProceduralSceneEvidenceTracker",
    "RuleBasedStepExpert",
    "SAM3OfficialImageSegmentationBackend",
    "SAM3VideoSegmentationBackend",
    "SharedVisualEncoder",
    "StreamingGRUTemporalStepExpert",
    "StructuredEvidenceEncoder",
    "TemporalEvidenceVectorizer",
    "TrackEvidenceAggregator",
    "OpenCVRuntimeUI",
    "VoiceCommandService",
    "WindowIoUFusion",
    "YOLODetectorComponent",
    "build_segmentation_backend",
]
