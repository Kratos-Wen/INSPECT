"""INSPECT: assistance-derived supervision for robot state verification."""

from .active_observation import choose_view_for_missing_evidence, choose_views_for_results, load_view_candidates
from .evidence_graph import induce_evidence_graph, load_evidence_graph, save_evidence_graph, summarize_evidence_graph
from .learned_verifier import CalibratedEvidenceVerifier, EvidenceFeatureExtractor, train_calibrated_verifier
from .metrics import evaluate_verification_results, summarize_traces
from .robot_adapter import (
    RecordingRobotAdapter,
    RobotControlAdapter,
    RobotProceduralDecisionLoop,
    robot_observation_from_evidence_token,
    robot_observation_from_payload,
)
from .state_spec import decompose_evidence, load_state_specs, state_spec_for
from .trace_export import export_verified_traces, load_trace_jsonl, write_trace_jsonl
from .types import (
    ActiveObservationDecision,
    EvidenceEdge,
    EvidenceNode,
    ProceduralEvidenceGraph,
    RobotObservation,
    StateSpec,
    VerificationResult,
    VerifiedTraceEvent,
    ViewCandidate,
)
from .verifier import RobotProceduralStateVerifier, load_robot_observations, observation_evidence_keys, verify_observations
from .workflow import build_inspect_artifacts

SYSTEM_NAME = "INSPECT"
SYSTEM_FULL_NAME = "Interactive Supervision for Procedural Evidence and Cross-view Task Verification"

__all__ = [
    "SYSTEM_FULL_NAME",
    "SYSTEM_NAME",
    "ActiveObservationDecision",
    "CalibratedEvidenceVerifier",
    "EvidenceEdge",
    "EvidenceFeatureExtractor",
    "EvidenceNode",
    "ProceduralEvidenceGraph",
    "RecordingRobotAdapter",
    "RobotControlAdapter",
    "RobotObservation",
    "RobotProceduralDecisionLoop",
    "RobotProceduralStateVerifier",
    "StateSpec",
    "VerificationResult",
    "VerifiedTraceEvent",
    "ViewCandidate",
    "build_inspect_artifacts",
    "choose_view_for_missing_evidence",
    "choose_views_for_results",
    "decompose_evidence",
    "evaluate_verification_results",
    "export_verified_traces",
    "induce_evidence_graph",
    "load_evidence_graph",
    "load_robot_observations",
    "load_state_specs",
    "load_trace_jsonl",
    "load_view_candidates",
    "observation_evidence_keys",
    "robot_observation_from_evidence_token",
    "robot_observation_from_payload",
    "save_evidence_graph",
    "summarize_evidence_graph",
    "summarize_traces",
    "state_spec_for",
    "train_calibrated_verifier",
    "verify_observations",
    "write_trace_jsonl",
]
