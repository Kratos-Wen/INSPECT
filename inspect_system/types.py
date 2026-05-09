"""Typed artifacts for the INSPECT robot-verification layer.

INSPECT stands for Interactive Supervision for Procedural Evidence and
Cross-view Task Verification.  The module intentionally sits above the
existing MICA step pipeline: it consumes run logs and turns assistance traces
into robot-facing verification artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class StateSpec:
    """Explicit procedural-state specification used to decompose step labels."""

    state_id: str
    nominal_step: str = ""
    description: str = ""
    preconditions: List[str] = field(default_factory=list)
    interaction_evidence: List[str] = field(default_factory=list)
    transition_evidence: List[str] = field(default_factory=list)
    postcondition_evidence: List[str] = field(default_factory=list)
    negative_evidence: List[str] = field(default_factory=list)
    admissibility_evidence: List[str] = field(default_factory=list)
    inspection_targets: List[str] = field(default_factory=list)
    next_states: List[str] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: JsonDict) -> "StateSpec":
        state_id = str(payload.get("state_id") or payload.get("state") or payload.get("id") or "").upper()
        nominal_step = str(payload.get("nominal_step") or payload.get("step") or state_id).upper()
        return cls(
            state_id=state_id,
            nominal_step=nominal_step,
            description=str(payload.get("description", "")),
            preconditions=[str(item) for item in payload.get("preconditions", [])],
            interaction_evidence=[str(item) for item in payload.get("interaction_evidence", [])],
            transition_evidence=[str(item) for item in payload.get("transition_evidence", [])],
            postcondition_evidence=[str(item) for item in payload.get("postcondition_evidence", [])],
            negative_evidence=[str(item) for item in payload.get("negative_evidence", [])],
            admissibility_evidence=[str(item) for item in payload.get("admissibility_evidence", [])],
            inspection_targets=[str(item) for item in payload.get("inspection_targets", [])],
            next_states=[str(item).upper() for item in payload.get("next_states", [])],
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class VerifiedTraceEvent:
    """One assistance-derived supervision event for a procedural state."""

    run_id: str
    frame_index: int
    iteration: int
    predicted_state: str
    verified_state: str
    candidate_state: str = ""
    verification_source: str = ""
    verification_level: str = "L0"
    trust_weight: float = 0.0
    verified: bool = False
    accepted: bool = False
    stable: bool = False
    confidence: float = 0.0
    uncertainty: float = 1.0
    prev_state: Optional[str] = None
    review_action: str = ""
    review_reason: str = ""
    review_triggers: List[str] = field(default_factory=list)
    observed_evidence: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    failure_cues: List[str] = field(default_factory=list)
    precondition_evidence: List[str] = field(default_factory=list)
    interaction_evidence: List[str] = field(default_factory=list)
    transition_evidence: List[str] = field(default_factory=list)
    postcondition_evidence: List[str] = field(default_factory=list)
    negative_evidence: List[str] = field(default_factory=list)
    admissibility_evidence: List[str] = field(default_factory=list)
    next_step_admissible: Optional[bool] = None
    visible_counts: Dict[str, int] = field(default_factory=dict)
    relevant_counts: Dict[str, int] = field(default_factory=dict)
    relation_counts: Dict[str, int] = field(default_factory=dict)
    relation_facts: List[List[str]] = field(default_factory=list)
    expert_states: Dict[str, str] = field(default_factory=dict)
    expert_confidences: Dict[str, float] = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: JsonDict) -> "VerifiedTraceEvent":
        return cls(
            run_id=str(payload.get("run_id", "")),
            frame_index=int(payload.get("frame_index", 0)),
            iteration=int(payload.get("iteration", payload.get("iter", 0))),
            predicted_state=str(payload.get("predicted_state", "")).upper(),
            verified_state=str(payload.get("verified_state", "")).upper(),
            candidate_state=str(payload.get("candidate_state", "")).upper(),
            verification_source=str(payload.get("verification_source", "")),
            verification_level=str(payload.get("verification_level", "L0")),
            trust_weight=float(payload.get("trust_weight", 0.0)),
            verified=bool(payload.get("verified", False)),
            accepted=bool(payload.get("accepted", False)),
            stable=bool(payload.get("stable", False)),
            confidence=float(payload.get("confidence", 0.0)),
            uncertainty=float(payload.get("uncertainty", 1.0)),
            prev_state=(str(payload.get("prev_state")).upper() if payload.get("prev_state") else None),
            review_action=str(payload.get("review_action", "")),
            review_reason=str(payload.get("review_reason", "")),
            review_triggers=[str(item) for item in payload.get("review_triggers", [])],
            observed_evidence=[str(item) for item in payload.get("observed_evidence", [])],
            missing_evidence=[str(item) for item in payload.get("missing_evidence", [])],
            failure_cues=[str(item) for item in payload.get("failure_cues", [])],
            precondition_evidence=[str(item) for item in payload.get("precondition_evidence", [])],
            interaction_evidence=[str(item) for item in payload.get("interaction_evidence", [])],
            transition_evidence=[str(item) for item in payload.get("transition_evidence", [])],
            postcondition_evidence=[str(item) for item in payload.get("postcondition_evidence", [])],
            negative_evidence=[str(item) for item in payload.get("negative_evidence", [])],
            admissibility_evidence=[str(item) for item in payload.get("admissibility_evidence", [])],
            next_step_admissible=(
                bool(payload.get("next_step_admissible"))
                if payload.get("next_step_admissible") is not None
                else None
            ),
            visible_counts={str(key): int(value) for key, value in dict(payload.get("visible_counts", {})).items()},
            relevant_counts={str(key): int(value) for key, value in dict(payload.get("relevant_counts", {})).items()},
            relation_counts={str(key): int(value) for key, value in dict(payload.get("relation_counts", {})).items()},
            relation_facts=[list(item) for item in payload.get("relation_facts", [])],
            expert_states={str(key): str(value).upper() for key, value in dict(payload.get("expert_states", {})).items()},
            expert_confidences={str(key): float(value) for key, value in dict(payload.get("expert_confidences", {})).items()},
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class EvidenceNode:
    """A state, object/relation evidence, or failure cue in the evidence graph."""

    node_id: str
    kind: str
    label: str
    support: float = 0.0
    weight: float = 0.0
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: JsonDict) -> "EvidenceNode":
        return cls(
            node_id=str(payload.get("node_id", "")),
            kind=str(payload.get("kind", "")),
            label=str(payload.get("label", "")),
            support=float(payload.get("support", 0.0)),
            weight=float(payload.get("weight", 0.0)),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class EvidenceEdge:
    """A directed relation in the procedural evidence graph."""

    source: str
    predicate: str
    target: str
    weight: float = 0.0
    count: int = 0
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: JsonDict) -> "EvidenceEdge":
        return cls(
            source=str(payload.get("source", "")),
            predicate=str(payload.get("predicate", "")),
            target=str(payload.get("target", "")),
            weight=float(payload.get("weight", 0.0)),
            count=int(payload.get("count", 0)),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class ProceduralEvidenceGraph:
    """View-agnostic evidence graph induced from online-corrected traces."""

    graph_id: str
    state_ids: List[str]
    nodes: Dict[str, EvidenceNode]
    edges: List[EvidenceEdge]
    state_priors: Dict[str, float] = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "graph_id": self.graph_id,
            "state_ids": list(self.state_ids),
            "nodes": {key: value.to_dict() for key, value in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "state_priors": dict(self.state_priors),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: JsonDict) -> "ProceduralEvidenceGraph":
        return cls(
            graph_id=str(payload.get("graph_id", "")),
            state_ids=[str(item).upper() for item in payload.get("state_ids", [])],
            nodes={
                str(key): EvidenceNode.from_dict(dict(value))
                for key, value in dict(payload.get("nodes", {})).items()
            },
            edges=[EvidenceEdge.from_dict(dict(item)) for item in payload.get("edges", [])],
            state_priors={str(key).upper(): float(value) for key, value in dict(payload.get("state_priors", {})).items()},
            metadata=dict(payload.get("metadata", {})),
        )

    def edges_from(self, node_id: str, predicate: Optional[str] = None) -> List[EvidenceEdge]:
        return [
            edge
            for edge in self.edges
            if edge.source == node_id and (predicate is None or edge.predicate == predicate)
        ]

    def verified_by_edges(self, state_id: str) -> List[EvidenceEdge]:
        return self.edges_from(f"state:{state_id.upper()}", predicate="verified_by")


@dataclass(frozen=True)
class RobotObservation:
    """Robot-view observation represented with the same evidence vocabulary."""

    observation_id: str
    frame_index: Optional[int] = None
    prev_state: Optional[str] = None
    view_id: str = ""
    visible_counts: Dict[str, int] = field(default_factory=dict)
    relevant_counts: Dict[str, int] = field(default_factory=dict)
    relation_counts: Dict[str, int] = field(default_factory=dict)
    relation_facts: List[List[str]] = field(default_factory=list)
    evidence_keys: List[str] = field(default_factory=list)
    candidate_states: List[str] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: JsonDict) -> "RobotObservation":
        frame_index = payload.get("frame_index")
        observation_id = payload.get("observation_id") or payload.get("id") or frame_index or ""
        metadata = dict(payload.get("metadata", {}))
        for key in (
            "ground_truth_state",
            "verified_state",
            "label",
            "state",
            "current_state",
            "next_step_admissible",
            "next_step_allowed",
            "valid_next_step",
            "anomaly",
            "is_anomaly",
            "failure",
            "invalid_state",
        ):
            if key in payload and key not in metadata:
                metadata[key] = payload[key]
        return cls(
            observation_id=str(observation_id),
            frame_index=(int(frame_index) if frame_index is not None else None),
            prev_state=(str(payload.get("prev_state") or payload.get("prev_step")).upper() if (payload.get("prev_state") or payload.get("prev_step")) else None),
            view_id=str(payload.get("view_id", "")),
            visible_counts={str(key): int(value) for key, value in dict(payload.get("visible_counts", {})).items()},
            relevant_counts={str(key): int(value) for key, value in dict(payload.get("relevant_counts", {})).items()},
            relation_counts={str(key): int(value) for key, value in dict(payload.get("relation_counts", {})).items()},
            relation_facts=[list(item) for item in payload.get("relation_facts", [])],
            evidence_keys=[str(item) for item in (payload.get("evidence_keys") or payload.get("observed_evidence") or [])],
            candidate_states=[str(item).upper() for item in payload.get("candidate_states", [])],
            metadata=metadata,
        )


@dataclass(frozen=True)
class VerificationResult:
    """Robot procedural state verification output."""

    observation_id: str
    predicted_state: str
    verified: bool
    confidence: float
    evidence_coverage: float
    observed_evidence: List[str]
    missing_evidence: List[str]
    contradicted_evidence: List[str]
    next_step_admissible: bool
    anomaly: bool
    recommended_action: str
    state_scores: Dict[str, float] = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class ViewCandidate:
    """A finite robot viewpoint candidate for active observation."""

    view_id: str
    visible_evidence: List[str]
    motion_cost: float = 0.0
    occlusion_risk: float = 0.0
    prior: float = 0.0
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: JsonDict) -> "ViewCandidate":
        return cls(
            view_id=str(payload.get("view_id") or payload.get("id") or ""),
            visible_evidence=[str(item) for item in payload.get("visible_evidence", [])],
            motion_cost=float(payload.get("motion_cost", 0.0)),
            occlusion_risk=float(payload.get("occlusion_risk", 0.0)),
            prior=float(payload.get("prior", 0.0)),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class ActiveObservationDecision:
    """Selected view and utility breakdown for evidence-seeking observation."""

    observation_id: str
    selected_view: str
    utility: float
    expected_observed_evidence: List[str]
    unresolved_evidence: List[str]
    candidate_scores: Dict[str, float]
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)
