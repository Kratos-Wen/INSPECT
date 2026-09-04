"""Fixed-view lattice policy for robot claim-disambiguating inspection.

The policy selects from robot-executable calibrated view IDs (for example
V0..V5).  It does not use human-labeled robot views for training.  Human
Assistant traces only provide claim -> evidence-view requirements; the robot
grounds those requirements into its own view candidates through metadata such
as visible evidence and evidence-view type scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

from .assistant_evidence import EvidenceRequirementProfile
from .evidence_view import EvidenceGap, EvidenceViewRequirement, EvidenceViewType, evidence_view_requirements_from_gaps
from .types import ActiveObservationDecision, ProceduralEvidenceGraph, RobotObservation, VerificationResult, ViewCandidate


def _slug(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text).strip("_")


def _state_required_evidence(graph: ProceduralEvidenceGraph, state_id: str) -> Set[str]:
    keys: Set[str] = set()
    for edge in graph.verified_by_edges(state_id):
        if edge.target.startswith("evidence:"):
            keys.add(edge.target.split(":", 1)[1])
    return keys


def _gap_from_key(key: str) -> EvidenceGap:
    low = str(key).strip().lower()
    parts = low.split(":")
    evidence_type = "generic"
    target = None
    slot = None
    relation = None
    if low.startswith("relation:") and len(parts) >= 4:
        target = _slug(parts[1])
        relation = _slug(parts[2])
        slot = _slug(parts[3])
        if relation in {"inside", "inserted", "seated", "in"}:
            evidence_type = "insertion_relation"
        elif relation in {"aligned", "aligned_with", "flush"}:
            evidence_type = "object_slot_alignment"
        elif relation in {"contact", "touching", "near"}:
            evidence_type = "contact_relation"
        else:
            evidence_type = "object_slot_alignment"
    elif low.startswith(("object:", "focus_object:", "track:stable_object:")):
        target = _slug(parts[-1])
        evidence_type = "object_presence"
        if "gear" in target:
            evidence_type = "object_identity"
    elif "identity" in low or "hard_pair" in low or "teeth" in low or "tooth" in low:
        evidence_type = "object_identity"
    elif "gap" in low or "seated" in low:
        evidence_type = "contact_relation"
    elif "align" in low:
        evidence_type = "object_slot_alignment"
    elif "inside" in low or "insert" in low or "housing" in low or "contain" in low:
        evidence_type = "insertion_relation"
    elif "occl" in low or "hand" in low:
        evidence_type = "hand_occlusion"
    elif "uncertain" in low or "confidence" in low or "margin" in low:
        evidence_type = "detector_uncertainty"
    return EvidenceGap(
        key=str(key),
        evidence_type=evidence_type,
        importance=1.0,
        target_object=target,
        slot_object=slot,
        relation=relation,
    )


def _metadata_list(metadata: Mapping[str, object], key: str) -> List[str]:
    value = metadata.get(key)
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _metadata_scores(metadata: Mapping[str, object], key: str) -> Dict[str, float]:
    value = metadata.get(key)
    if not isinstance(value, dict):
        return {}
    scores: Dict[str, float] = {}
    for raw_key, raw_value in value.items():
        try:
            scores[_slug(raw_key)] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return scores


class FixedViewLatticePolicy:
    """Select one calibrated robot view ID from a finite lattice."""

    def __init__(
        self,
        candidates: Sequence[ViewCandidate],
        *,
        requirement_profile: Optional[EvidenceRequirementProfile] = None,
        motion_weight: float = 0.20,
        occlusion_weight: float = 0.30,
        missing_evidence_weight: float = 1.00,
        evidence_view_weight: float = 1.20,
        state_context_weight: float = 0.30,
    ) -> None:
        self.candidates = list(candidates)
        self.requirement_profile = requirement_profile
        self.motion_weight = float(motion_weight)
        self.occlusion_weight = float(occlusion_weight)
        self.missing_evidence_weight = float(missing_evidence_weight)
        self.evidence_view_weight = float(evidence_view_weight)
        self.state_context_weight = float(state_context_weight)

    def select_view(
        self,
        observation: RobotObservation,
        verification: VerificationResult,
        graph: Optional[ProceduralEvidenceGraph] = None,
    ) -> ActiveObservationDecision:
        missing = set(str(item) for item in verification.missing_evidence)
        required = _state_required_evidence(graph, verification.predicted_state) if graph is not None else set()
        target_evidence = missing or (required - set(verification.observed_evidence))
        claim_type = self._claim_type(observation, verification)
        requirements = self._requirements(target_evidence, observation, verification)

        best: Optional[ViewCandidate] = None
        best_score = float("-inf")
        best_expected: Set[str] = set()
        scores: Dict[str, float] = {}
        breakdown: Dict[str, Dict[str, float]] = {}
        for candidate in self.candidates:
            utility, expected, parts = self._score_candidate(
                candidate,
                target_evidence=target_evidence,
                required=required,
                requirements=requirements,
                claim_type=claim_type,
            )
            scores[candidate.view_id] = float(utility)
            breakdown[candidate.view_id] = parts
            if utility > best_score:
                best_score = utility
                best = candidate
                best_expected = expected

        if best is None:
            return ActiveObservationDecision(
                observation_id=verification.observation_id,
                selected_view="",
                utility=0.0,
                expected_observed_evidence=[],
                unresolved_evidence=sorted(target_evidence),
                candidate_scores={},
                metadata={"policy": "fixed_view_lattice_v1", "reason": "no_view_candidates"},
            )

        return ActiveObservationDecision(
            observation_id=verification.observation_id,
            selected_view=best.view_id,
            utility=float(best_score),
            expected_observed_evidence=sorted(best_expected),
            unresolved_evidence=sorted(set(target_evidence) - best_expected),
            candidate_scores=scores,
            metadata={
                "policy": "fixed_view_lattice_v1",
                "claim_type": claim_type,
                "predicted_state": verification.predicted_state,
                "current_view": observation.view_id,
                "target_evidence": sorted(target_evidence),
                "evidence_view_requirements": [item.to_dict() for item in requirements],
                "candidate_score_breakdown": breakdown,
                "uses_human_robot_view_labels": False,
                "training_signal": "assistant_claim_to_evidence_requirements",
            },
        )

    def _claim_type(self, observation: RobotObservation, verification: VerificationResult) -> str:
        metadata = {**observation.metadata, **verification.metadata}
        for key in ("claim_type", "ambiguity_type", "claim", "state_claim"):
            if metadata.get(key):
                return _slug(metadata.get(key))
        return _slug(verification.predicted_state)

    def _requirements(
        self,
        target_evidence: Iterable[str],
        observation: RobotObservation,
        verification: VerificationResult,
    ) -> List[EvidenceViewRequirement]:
        gaps = [_gap_from_key(key) for key in target_evidence]
        metadata = {**observation.metadata, **verification.metadata}
        features = {
            "verification_confidence": float(verification.confidence),
            "evidence_coverage": float(verification.evidence_coverage),
            "target_confidence": float(metadata.get("target_confidence", metadata.get("detector_confidence", 1.0)) or 1.0),
        }
        return evidence_view_requirements_from_gaps(gaps, features)

    def _score_candidate(
        self,
        candidate: ViewCandidate,
        *,
        target_evidence: Set[str],
        required: Set[str],
        requirements: Sequence[EvidenceViewRequirement],
        claim_type: str,
    ) -> tuple[float, Set[str], Dict[str, float]]:
        visible = set(candidate.visible_evidence)
        metadata = dict(candidate.metadata or {})
        metadata_visible = set(_metadata_list(metadata, "visible_evidence"))
        visible |= metadata_visible
        expected = visible & (target_evidence | required)

        missing_hits = len(visible & target_evidence) / max(1, len(target_evidence))
        required_hits = len(visible & required) / max(1, len(required)) if required else 0.0
        view_score = self._evidence_view_score(candidate, requirements, claim_type)
        prior = float(candidate.prior)
        claim_prior = _metadata_scores(metadata, "claim_utilities").get(_slug(claim_type), 0.0)
        utility = (
            self.missing_evidence_weight * missing_hits
            + self.state_context_weight * required_hits
            + self.evidence_view_weight * view_score
            + prior
            + claim_prior
            - self.motion_weight * float(candidate.motion_cost)
            - self.occlusion_weight * float(candidate.occlusion_risk)
        )
        return float(utility), expected, {
            "missing_hits": float(missing_hits),
            "required_hits": float(required_hits),
            "evidence_view_score": float(view_score),
            "prior": float(prior),
            "claim_prior": float(claim_prior),
            "motion_cost": float(candidate.motion_cost),
            "occlusion_risk": float(candidate.occlusion_risk),
        }

    def _evidence_view_score(
        self,
        candidate: ViewCandidate,
        requirements: Sequence[EvidenceViewRequirement],
        claim_type: str,
    ) -> float:
        if not requirements:
            return 0.0
        metadata = dict(candidate.metadata or {})
        view_types = {_slug(item) for item in _metadata_list(metadata, "evidence_view_types")}
        view_scores = _metadata_scores(metadata, "evidence_view_scores")
        utility = 0.0
        denom = 0.0
        for requirement in requirements:
            view = requirement.view_type.value
            view_key = _slug(view)
            base = view_scores.get(view_key, 1.0 if view_key in view_types else 0.0)
            if base <= 0.0:
                continue
            profile_weight = self.requirement_profile.weight(claim_type, requirement.view_type) if self.requirement_profile else 1.0
            weight = float(requirement.importance) * float(profile_weight)
            utility += weight * base
            denom += weight
        return float(utility / max(1e-6, denom))


def write_lattice_decisions_jsonl(results: Iterable[ActiveObservationDecision], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
