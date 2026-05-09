"""Robot-view procedural state verification for INSPECT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

from .types import ProceduralEvidenceGraph, RobotObservation, VerificationResult


def _slug(text: object) -> str:
    value = str(text or "").strip().lower().replace("-", "_").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value).strip("_")


def _evidence_key_from_node(node_id: str) -> str:
    return node_id.split(":", 1)[1] if node_id.startswith("evidence:") else node_id


def observation_evidence_keys(observation: RobotObservation) -> List[str]:
    """Build INSPECT evidence keys from a robot observation."""

    evidence: Set[str] = set(observation.evidence_keys)
    for name, count in observation.visible_counts.items():
        if int(count) > 0:
            evidence.add(f"object:{_slug(name)}")
    for name, count in observation.relevant_counts.items():
        if int(count) > 0:
            evidence.add(f"focus_object:{_slug(name)}")
            evidence.add(f"object:{_slug(name)}")
    for key, count in observation.relation_counts.items():
        if int(count) > 0:
            evidence.add(f"relation_type:{_slug(key)}")
    for item in observation.relation_facts:
        if len(item) < 3:
            continue
        subject, predicate, obj = item[:3]
        pred = _slug(predicate)
        evidence.add(f"relation:{_slug(subject)}:{pred}:{_slug(obj)}")
        evidence.add(f"relation_type:{pred}")
    if evidence:
        evidence.add("visual:evidence_present")
    return sorted(evidence)


def load_robot_observations(path: Path) -> List[RobotObservation]:
    """Load robot observations from JSONL or a JSON list."""

    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return [RobotObservation.from_dict(dict(item)) for item in json.loads(text)]
    observations: List[RobotObservation] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            observations.append(RobotObservation.from_dict(json.loads(line)))
    return observations


def write_verification_jsonl(results: Iterable[VerificationResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")


class RobotProceduralStateVerifier:
    """Small graph-grounded verifier for robot-view procedural states."""

    def __init__(
        self,
        graph: ProceduralEvidenceGraph,
        verification_threshold: float = 0.55,
        anomaly_threshold: float = 0.30,
        transition_bonus: float = 0.10,
    ) -> None:
        self.graph = graph
        self.verification_threshold = float(verification_threshold)
        self.anomaly_threshold = float(anomaly_threshold)
        self.transition_bonus = float(transition_bonus)

    def verify(self, observation: RobotObservation) -> VerificationResult:
        observed = set(observation_evidence_keys(observation))
        candidate_states = observation.candidate_states or self.graph.state_ids
        state_scores: Dict[str, float] = {}
        state_missing: Dict[str, List[str]] = {}
        state_coverage: Dict[str, float] = {}
        state_contradictions: Dict[str, List[str]] = {}

        for state_id in candidate_states:
            state_id = state_id.upper()
            required_edges = self.graph.verified_by_edges(state_id)
            total_weight = sum(max(0.0, edge.weight) for edge in required_edges)
            observed_weight = 0.0
            missing: List[str] = []
            for edge in required_edges:
                key = _evidence_key_from_node(edge.target)
                if key in observed:
                    observed_weight += max(0.0, edge.weight)
                else:
                    missing.append(key)
            coverage = observed_weight / total_weight if total_weight > 0 else 0.0
            score = coverage
            prior = float(self.graph.state_priors.get(state_id, 0.0))
            score += min(0.05, prior * 0.05)
            if observation.prev_state:
                for edge in self.graph.edges_from(f"state:{observation.prev_state.upper()}", predicate="enables"):
                    if edge.target == f"state:{state_id}":
                        score += self.transition_bonus
                        break
            contradictions = self._observed_contradictions(state_id, observed)
            if contradictions:
                score -= min(0.35, 0.12 * len(contradictions))
            state_scores[state_id] = max(0.0, min(1.0, score))
            state_missing[state_id] = sorted(missing, key=lambda key: key)
            state_coverage[state_id] = coverage
            state_contradictions[state_id] = contradictions

        if not state_scores:
            return VerificationResult(
                observation_id=observation.observation_id,
                predicted_state="",
                verified=False,
                confidence=0.0,
                evidence_coverage=0.0,
                observed_evidence=sorted(observed),
                missing_evidence=[],
                contradicted_evidence=[],
                next_step_admissible=False,
                anomaly=True,
                recommended_action="ask_human",
                state_scores={},
                metadata={"reason": "no_candidate_states"},
            )

        ranked = sorted(state_scores.items(), key=lambda item: item[1], reverse=True)
        predicted_state, top_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence = max(0.0, min(1.0, top_score + max(0.0, top_score - runner_up) * 0.25))
        missing = state_missing[predicted_state]
        contradictions = state_contradictions[predicted_state]
        coverage = state_coverage[predicted_state]
        verified = bool(confidence >= self.verification_threshold and coverage >= self.verification_threshold and not contradictions)
        anomaly = bool(contradictions or confidence < self.anomaly_threshold)
        if verified:
            action = "continue"
        elif contradictions:
            action = "pause"
        elif missing:
            action = "observe"
        else:
            action = "ask_human"
        return VerificationResult(
            observation_id=observation.observation_id,
            predicted_state=predicted_state,
            verified=verified,
            confidence=confidence,
            evidence_coverage=coverage,
            observed_evidence=sorted(observed),
            missing_evidence=missing[:12],
            contradicted_evidence=contradictions,
            next_step_admissible=bool(verified and not anomaly),
            anomaly=anomaly,
            recommended_action=action,
            state_scores={key: float(value) for key, value in ranked},
            metadata={
                "view_id": observation.view_id,
                "runner_up_score": float(runner_up),
                "verification_threshold": self.verification_threshold,
                "anomaly_threshold": self.anomaly_threshold,
            },
        )

    def _observed_contradictions(self, state_id: str, observed: Set[str]) -> List[str]:
        contradictions: List[str] = []
        for edge in self.graph.edges_from(f"state:{state_id.upper()}", predicate="contradicted_by"):
            key = edge.target.split(":", 1)[1] if ":" in edge.target else edge.target
            if key in observed or f"failure:{key}" in observed:
                contradictions.append(key)
        return sorted(contradictions)


def verify_observations(
    graph: ProceduralEvidenceGraph,
    observations: Sequence[RobotObservation],
    verification_threshold: float = 0.55,
    anomaly_threshold: float = 0.30,
) -> List[VerificationResult]:
    verifier = RobotProceduralStateVerifier(
        graph=graph,
        verification_threshold=verification_threshold,
        anomaly_threshold=anomaly_threshold,
    )
    return [verifier.verify(observation) for observation in observations]
