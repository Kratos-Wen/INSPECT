"""Learned calibrated robot-state verifier for INSPECT."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .types import ProceduralEvidenceGraph, RobotObservation, VerificationResult
from .verifier import observation_evidence_keys


def _evidence_key(node_id: str) -> str:
    return node_id.split(":", 1)[1] if node_id.startswith("evidence:") else node_id


def _truth_state(observation: RobotObservation) -> str:
    for key in ("ground_truth_state", "verified_state", "label", "state", "current_state"):
        value = observation.metadata.get(key)
        if value:
            return str(value).upper()
    return ""


def _truth_bool(observation: RobotObservation, keys: Sequence[str]) -> Optional[bool]:
    for key in keys:
        if key not in observation.metadata:
            continue
        value = observation.metadata[key]
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "valid", "allowed", "verified"}:
            return True
        if text in {"0", "false", "no", "invalid", "blocked", "unverified"}:
            return False
    return None


@dataclass(frozen=True)
class LearnedVerifierTrainingReport:
    num_examples: int
    num_positive: int
    num_negative: int
    loss: float
    accuracy: float
    feature_names: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class EvidenceFeatureExtractor:
    """Convert graph-grounded robot observations into verifier features."""

    FEATURE_NAMES = [
        "requires_coverage",
        "interaction_coverage",
        "transition_coverage",
        "postcondition_coverage",
        "admissibility_coverage",
        "suggested_prior_coverage",
        "contradiction_count",
        "missing_required_count",
        "missing_postcondition_count",
        "observed_log_count",
        "contact_present",
        "release_phase_present",
        "transition_edge_from_prev",
    ]

    def __init__(self, graph: ProceduralEvidenceGraph) -> None:
        self.graph = graph

    def vectorize(self, observation: RobotObservation, state_id: str) -> np.ndarray:
        observed = set(observation_evidence_keys(observation))
        state_node = f"state:{state_id.upper()}"
        values = [
            self._coverage(state_node, "requires", observed)[0],
            self._coverage(state_node, "supported_by_interaction", observed)[0],
            self._coverage(state_node, "supported_by_transition", observed)[0],
            self._coverage(state_node, "verified_by", observed)[0],
            self._coverage(state_node, "admissible_by", observed)[0],
            self._coverage(state_node, "suggested_by", observed)[0],
            float(self._contradictions(state_node, observed)),
            float(self._coverage(state_node, "requires", observed)[1]),
            float(self._coverage(state_node, "verified_by", observed)[1]),
            float(math.log1p(len(observed))),
            1.0 if any(key.startswith("contact:") or key.startswith("interaction:") for key in observed) else 0.0,
            1.0 if "interaction:contact_phase:release" in observed else 0.0,
            self._transition_edge(observation.prev_state, state_id),
        ]
        return np.asarray(values, dtype=np.float32)

    def missing_evidence(self, observation: RobotObservation, state_id: str) -> List[str]:
        observed = set(observation_evidence_keys(observation))
        missing: List[str] = []
        for predicate in ("requires", "verified_by", "admissible_by"):
            for edge in self.graph.edges_from(f"state:{state_id.upper()}", predicate=predicate):
                key = _evidence_key(edge.target)
                if key not in observed:
                    missing.append(key)
        return sorted(set(missing))

    def contradicted_evidence(self, observation: RobotObservation, state_id: str) -> List[str]:
        observed = set(observation_evidence_keys(observation))
        hits: List[str] = []
        for edge in self.graph.edges_from(f"state:{state_id.upper()}", predicate="contradicted_by"):
            key = edge.target.split(":", 1)[1] if ":" in edge.target else edge.target
            if key in observed or f"failure:{key}" in observed:
                hits.append(key)
        return sorted(set(hits))

    def _coverage(self, state_node: str, predicate: str, observed: set[str]) -> Tuple[float, int]:
        edges = self.graph.edges_from(state_node, predicate=predicate)
        if not edges:
            return 0.0, 0
        total = sum(max(0.0, edge.weight) for edge in edges)
        if total <= 1e-9:
            total = float(len(edges))
        observed_weight = 0.0
        missing = 0
        for edge in edges:
            key = _evidence_key(edge.target)
            weight = max(0.0, edge.weight) or 1.0
            if key in observed:
                observed_weight += weight
            else:
                missing += 1
        return float(observed_weight / total), int(missing)

    def _contradictions(self, state_node: str, observed: set[str]) -> int:
        count = 0
        for edge in self.graph.edges_from(state_node, predicate="contradicted_by"):
            key = edge.target.split(":", 1)[1] if ":" in edge.target else edge.target
            if key in observed or f"failure:{key}" in observed:
                count += 1
        return count

    def _transition_edge(self, prev_state: Optional[str], state_id: str) -> float:
        if not prev_state:
            return 0.0
        for edge in self.graph.edges_from(f"state:{prev_state.upper()}", predicate="enables"):
            if edge.target == f"state:{state_id.upper()}":
                return 1.0
        return 0.0


@dataclass
class CalibratedEvidenceVerifier:
    """Small logistic verifier over graph-grounded evidence features."""

    graph: ProceduralEvidenceGraph
    weights: List[float]
    bias: float
    threshold: float = 0.55
    temperature: float = 1.0

    @classmethod
    def new(cls, graph: ProceduralEvidenceGraph, threshold: float = 0.55) -> "CalibratedEvidenceVerifier":
        return cls(
            graph=graph,
            weights=[0.0 for _ in EvidenceFeatureExtractor.FEATURE_NAMES],
            bias=0.0,
            threshold=float(threshold),
            temperature=1.0,
        )

    def fit(
        self,
        observations: Sequence[RobotObservation],
        epochs: int = 600,
        lr: float = 0.08,
        l2: float = 0.001,
    ) -> LearnedVerifierTrainingReport:
        extractor = EvidenceFeatureExtractor(self.graph)
        examples: List[np.ndarray] = []
        labels: List[float] = []
        for observation in observations:
            truth_state = _truth_state(observation)
            truth_verified = _truth_bool(observation, ("postcondition_verified", "verified", "next_step_admissible", "valid_next_step"))
            candidates = observation.candidate_states or self.graph.state_ids
            for candidate in candidates:
                candidate = candidate.upper()
                if not candidate:
                    continue
                if truth_state:
                    label = 1.0 if candidate == truth_state and truth_verified is not False else 0.0
                elif truth_verified is not None and len(candidates) == 1:
                    label = 1.0 if truth_verified else 0.0
                else:
                    continue
                examples.append(extractor.vectorize(observation, candidate))
                labels.append(label)
        if not examples:
            return LearnedVerifierTrainingReport(
                num_examples=0,
                num_positive=0,
                num_negative=0,
                loss=0.0,
                accuracy=0.0,
                feature_names=list(EvidenceFeatureExtractor.FEATURE_NAMES),
            )

        x = np.vstack(examples).astype(np.float32)
        y = np.asarray(labels, dtype=np.float32)
        w = np.asarray(self.weights, dtype=np.float32)
        b = float(self.bias)
        for _ in range(max(1, int(epochs))):
            logits = x @ w + b
            pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
            error = pred - y
            grad_w = (x.T @ error) / len(y) + float(l2) * w
            grad_b = float(np.mean(error))
            w -= float(lr) * grad_w
            b -= float(lr) * grad_b

        logits = x @ w + b
        pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        loss = float(-np.mean(y * np.log(pred + 1e-9) + (1.0 - y) * np.log(1.0 - pred + 1e-9)))
        acc = float(np.mean((pred >= self.threshold) == (y >= 0.5)))
        self.weights = [float(item) for item in w.tolist()]
        self.bias = float(b)
        self.temperature = self._fit_temperature(logits, y)
        return LearnedVerifierTrainingReport(
            num_examples=len(labels),
            num_positive=int(np.sum(y >= 0.5)),
            num_negative=int(np.sum(y < 0.5)),
            loss=loss,
            accuracy=acc,
            feature_names=list(EvidenceFeatureExtractor.FEATURE_NAMES),
        )

    def verify(self, observation: RobotObservation) -> VerificationResult:
        extractor = EvidenceFeatureExtractor(self.graph)
        observed = observation_evidence_keys(observation)
        candidates = observation.candidate_states or self.graph.state_ids
        scores: Dict[str, float] = {}
        for candidate in candidates:
            candidate = candidate.upper()
            vector = extractor.vectorize(observation, candidate)
            score = self._probability(vector)
            scores[candidate] = score
        if not scores:
            return VerificationResult(
                observation_id=observation.observation_id,
                predicted_state="",
                verified=False,
                confidence=0.0,
                evidence_coverage=0.0,
                observed_evidence=observed,
                missing_evidence=[],
                contradicted_evidence=[],
                next_step_admissible=False,
                anomaly=True,
                recommended_action="ask_human",
                state_scores={},
            )
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        state_id, confidence = ranked[0]
        missing = extractor.missing_evidence(observation, state_id)
        contradicted = extractor.contradicted_evidence(observation, state_id)
        verified = bool(confidence >= self.threshold and not contradicted)
        if verified:
            action = "continue"
        elif contradicted:
            action = "pause"
        elif missing:
            action = "observe"
        else:
            action = "ask_human"
        return VerificationResult(
            observation_id=observation.observation_id,
            predicted_state=state_id,
            verified=verified,
            confidence=float(confidence),
            evidence_coverage=float(confidence),
            observed_evidence=observed,
            missing_evidence=missing[:12],
            contradicted_evidence=contradicted,
            next_step_admissible=bool(verified),
            anomaly=bool(contradicted or confidence < 0.25),
            recommended_action=action,
            state_scores={key: float(value) for key, value in ranked},
            metadata={"model": "calibrated_logistic_evidence_verifier"},
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "graph": self.graph.to_dict(),
            "weights": list(self.weights),
            "bias": float(self.bias),
            "threshold": float(self.threshold),
            "temperature": float(self.temperature),
            "feature_names": list(EvidenceFeatureExtractor.FEATURE_NAMES),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CalibratedEvidenceVerifier":
        from .types import ProceduralEvidenceGraph

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            graph=ProceduralEvidenceGraph.from_dict(dict(payload.get("graph", {}))),
            weights=[float(item) for item in payload.get("weights", [])],
            bias=float(payload.get("bias", 0.0)),
            threshold=float(payload.get("threshold", 0.55)),
            temperature=float(payload.get("temperature", 1.0)),
        )

    def _probability(self, vector: np.ndarray) -> float:
        w = np.asarray(self.weights, dtype=np.float32)
        if w.size != vector.size:
            w = np.resize(w, vector.size).astype(np.float32)
        logit = float(vector @ w + float(self.bias))
        temp = max(1e-3, float(self.temperature))
        return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit / temp)))))

    @staticmethod
    def _fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
        best_temp = 1.0
        best_loss = float("inf")
        for temp in np.linspace(0.5, 3.0, 26):
            pred = 1.0 / (1.0 + np.exp(-np.clip(logits / temp, -30.0, 30.0)))
            loss = float(-np.mean(labels * np.log(pred + 1e-9) + (1.0 - labels) * np.log(1.0 - pred + 1e-9)))
            if loss < best_loss:
                best_loss = loss
                best_temp = float(temp)
        return best_temp


def train_calibrated_verifier(
    graph: ProceduralEvidenceGraph,
    observations: Sequence[RobotObservation],
    model_path: Path,
    threshold: float = 0.55,
    epochs: int = 600,
    lr: float = 0.08,
    l2: float = 0.001,
) -> LearnedVerifierTrainingReport:
    model = CalibratedEvidenceVerifier.new(graph, threshold=threshold)
    report = model.fit(observations, epochs=epochs, lr=lr, l2=l2)
    model.save(Path(model_path))
    return report
