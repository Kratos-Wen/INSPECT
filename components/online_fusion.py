"""Online adaptive fusion with a margin-based linear head."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..types import EvidenceToken, FeedbackEvent, FusionResult, StepPrediction


class AdaptiveExpertFusion:
    """Fuse multiple dense step experts and adapt online with margin-based updates."""

    CONTEXT_FEATURE_NAMES = (
        "has_visual_evidence",
        "memory_active",
        "review_hold",
        "review_request_human",
        "visible_density",
        "relevant_density",
        "contact_density",
        "support_density",
        "directional_density",
        "overlap_density",
        "bootstrap_margin",
        "expert_disagreement",
        "transition_active",
    )

    def __init__(
        self,
        steps: List[str],
        expert_names: Optional[List[str]] = None,
        state_gate: float = 0.5,
        temporal_gate: float = 0.45,
        retrieval_gate: float = 0.5,
        memory_gate: float = 0.2,
        leak_state: float = 0.05,
        leak_temporal: float = 0.03,
        leak_retrieval: float = 0.05,
        leak_memory: float = 0.02,
        clamp_lo: float = 0.05,
        clamp_hi: float = 0.95,
        floor_per_class: float = 0.02,
        bias_cap: float = 0.5,
        eta: float = 0.10,
        gate_eta: float = 0.05,
        margin: float = 0.20,
        positive_margin: float = 0.05,
        positive_scale: float = 0.35,
        hit_gamma: float = 2.0,
        error_gamma: float = 2.0,
        freeze_confidence: float = 0.90,
        exposure_rho: float = 0.5,
        balance_window: int = 50,
        balance_tau: float = 0.6,
        lambda_transition: float = 2.0,
        context_gate_enabled: bool = True,
        context_gate_scale: float = 0.35,
        context_gate_eta: float = 0.02,
        context_gate_path: Optional[Path] = None,
        transitions: Optional[Dict[str, List[str]]] = None,
        state_path: Optional[Path] = None,
    ) -> None:
        self.steps = [str(step).strip().upper() for step in steps]
        self.step_to_index = {step_id: index for index, step_id in enumerate(self.steps)}
        self.num_steps = len(self.steps)
        self.expert_names = [str(name).strip().lower() for name in (expert_names or ["state", "retrieval", "memory"])]
        self.expert_names = [name for name in self.expert_names if name]
        if not self.expert_names:
            raise ValueError("AdaptiveExpertFusion requires at least one expert name.")
        self.expert_to_index = {name: index for index, name in enumerate(self.expert_names)}
        self.num_experts = len(self.expert_names)

        self.leak_by_expert = {
            "state": float(leak_state),
            "temporal": float(leak_temporal),
            "retrieval": float(leak_retrieval),
            "memory": float(leak_memory),
        }
        for expert_name in self.expert_names:
            self.leak_by_expert.setdefault(expert_name, float(leak_memory))
        self.clamp_lo = float(clamp_lo)
        self.clamp_hi = float(clamp_hi)
        self.floor_per_class = float(floor_per_class)
        self.bias_cap = float(bias_cap)
        self.eta = float(eta)
        self.gate_eta = float(gate_eta)
        self.margin = float(margin)
        self.positive_margin = float(positive_margin)
        self.positive_scale = float(positive_scale)
        self.hit_gamma = float(hit_gamma)
        self.error_gamma = float(error_gamma)
        self.freeze_confidence = float(freeze_confidence)
        self.exposure_rho = float(exposure_rho)
        self.balance_window = int(balance_window)
        self.balance_tau = float(balance_tau)
        self.lambda_transition = float(lambda_transition)
        self.context_gate_enabled = bool(context_gate_enabled)
        self.context_gate_scale = float(context_gate_scale)
        self.context_gate_eta = float(context_gate_eta)
        self.context_gate_path = Path(context_gate_path) if context_gate_path else None
        self.transitions = {
            str(key).strip().upper(): [str(value).strip().upper() for value in values]
            for key, values in (transitions or {}).items()
        }
        self.state_path = Path(state_path) if state_path else None

        initial_gate_map = {
            "state": float(state_gate),
            "temporal": float(temporal_gate),
            "retrieval": float(retrieval_gate),
            "memory": float(memory_gate),
        }
        initial_gates = np.array(
            [float(initial_gate_map.get(name, memory_gate)) for name in self.expert_names],
            dtype=np.float32,
        )
        initial_gates = np.maximum(initial_gates, 1e-6)
        self.gates = initial_gates / initial_gates.sum()
        self.context_gate_weights = np.zeros((self.num_experts, len(self.CONTEXT_FEATURE_NAMES)), dtype=np.float32)
        self.context_gate_bias = np.zeros((self.num_experts,), dtype=np.float32)
        self.weights = np.full((self.num_steps, self.num_experts), 1.0 / self.num_steps, dtype=np.float32)
        self.bias = np.zeros((self.num_steps,), dtype=np.float32)
        self.feedback_counts = np.zeros((self.num_steps,), dtype=np.int64)
        self.recent_feedback: List[str] = []
        self.recent_predictions: List[str] = []
        self.context_gate_loaded = False

        self._initialize_context_gate_weights()
        if self.context_gate_path and self.context_gate_path.exists():
            self._load_context_gate(self.context_gate_path)

        if self.state_path and self.state_path.exists():
            self._load(self.state_path)

    def fuse(
        self,
        expert_predictions: Dict[str, StepPrediction],
        prev_step: Optional[str] = None,
        token: Optional[EvidenceToken] = None,
    ) -> FusionResult:
        """Fuse expert score vectors into one final step prediction."""

        dense_scores = {
            expert_name: self._dense_scores(expert_predictions.get(expert_name), expert_name)
            for expert_name in self.expert_names
        }
        context_vector = self._context_vector(token=token, expert_predictions=expert_predictions, prev_step=prev_step)
        effective_gates = self._effective_gates(context_vector)

        fused_scores: Dict[str, float] = {}
        contributions: Dict[str, Dict[str, float]] = {}
        for index, step_id in enumerate(self.steps):
            transition_term = self._transition_term(step_id, prev_step)
            total = float(self.bias[index] + transition_term)
            contributions[step_id] = {
                "bias": float(self.bias[index]),
                "transition": transition_term,
            }
            for expert_index, expert_name in enumerate(self.expert_names):
                expert_term = float(
                    effective_gates[expert_index] * self.weights[index, expert_index] * dense_scores[expert_name][index]
                )
                total += expert_term
                contributions[step_id][expert_name] = expert_term
            fused_scores[step_id] = total

        ordered = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        top_step, top_score = ordered[0]
        runner_up = ordered[1][0] if len(ordered) > 1 else None
        probabilities = self._softmax([score for _, score in ordered])
        confidence = float(probabilities[0]) if probabilities else 0.0

        self.recent_predictions.append(top_step)
        if len(self.recent_predictions) > self.balance_window:
            self.recent_predictions.pop(0)

        return FusionResult(
            step_id=top_step,
            confidence=confidence,
            scores=fused_scores,
            runner_up=runner_up,
            gates={expert_name: float(effective_gates[index]) for expert_name, index in self.expert_to_index.items()},
            contributions=contributions,
            extras={
                "expert_scores": {
                    expert_name: {
                        step_id: float(dense_scores[expert_name][index]) for index, step_id in enumerate(self.steps)
                    }
                    for expert_name in self.expert_names
                },
                "top_score": top_score,
                "base_gates": {expert_name: float(self.gates[index]) for expert_name, index in self.expert_to_index.items()},
                "context_gate_features": {
                    name: float(context_vector[index]) for index, name in enumerate(self.CONTEXT_FEATURE_NAMES)
                },
                "context_gate_loaded": bool(self.context_gate_loaded),
            },
        )

    def apply_feedback(
        self,
        feedback: FeedbackEvent,
        expert_predictions: Dict[str, StepPrediction],
        fusion_result: FusionResult,
    ) -> None:
        """Apply one online update from a human supervision event."""

        label = str(feedback.label).strip().upper()
        if label not in self.step_to_index:
            return

        runner_up = fusion_result.runner_up
        if runner_up is None:
            return

        dense_scores = {
            expert_name: self._dense_scores(expert_predictions.get(expert_name), expert_name)
            for expert_name in self.expert_names
        }
        target_index = self.step_to_index[label]
        competitor_index = self.step_to_index[runner_up]

        current_margin = float(fusion_result.scores[label] - fusion_result.scores[runner_up])
        desired_margin = self.positive_margin if feedback.accepted else self.margin
        loss = max(0.0, desired_margin - current_margin)

        impact = self._feedback_impact(feedback, fusion_result.confidence)
        if impact <= 0.0:
            return
        if feedback.accepted and loss <= 0.0:
            return

        eta_eff = self._effective_eta(label, feedback.strength)
        if feedback.accepted:
            eta_eff *= self.positive_scale

        target_features = np.array(
            [dense_scores[expert_name][target_index] for expert_name in self.expert_names],
            dtype=np.float32,
        )
        competitor_features = np.array(
            [dense_scores[expert_name][competitor_index] for expert_name in self.expert_names],
            dtype=np.float32,
        )
        feature_norm = float(
            np.dot(target_features, target_features) + np.dot(competitor_features, competitor_features) + 1.0
        )
        tau = eta_eff * impact * max(loss, 1e-6) / feature_norm
        if tau <= 0.0:
            return

        self.weights[target_index, :] += tau * target_features
        self.weights[competitor_index, :] -= tau * competitor_features
        self.bias[target_index] += tau
        self.bias[competitor_index] -= tau

        self._update_gates(
            feedback=feedback,
            label=label,
            expert_predictions=expert_predictions,
            fusion_result=fusion_result,
            target_features=target_features,
            competitor_features=competitor_features,
            impact=impact,
        )

        self.feedback_counts[target_index] += 1
        self.recent_feedback.append(label)
        if len(self.recent_feedback) > self.balance_window:
            self.recent_feedback.pop(0)
        self._normalize()

        if self.state_path is not None:
            self.save(self.state_path)

    def save(self, path: Path) -> None:
        """Persist the fusion state to disk."""

        payload = {
            "steps": self.steps,
            "expert_names": self.expert_names,
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "gates": self.gates.tolist(),
            "context_gate_weights": self.context_gate_weights.tolist(),
            "context_gate_bias": self.context_gate_bias.tolist(),
            "feedback_counts": self.feedback_counts.tolist(),
            "recent_feedback": list(self.recent_feedback),
            "recent_predictions": list(self.recent_predictions),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("steps") != self.steps:
            return
        if payload.get("expert_names") != self.expert_names:
            return
        weights = payload.get("weights", self.weights.tolist())
        gates = payload.get("gates", self.gates.tolist())
        if len(weights) != self.num_steps:
            return
        if any(len(row) != self.num_experts for row in weights):
            return
        if len(gates) != self.num_experts:
            return
        self.weights = np.array(weights, dtype=np.float32)
        self.bias = np.array(payload.get("bias", self.bias.tolist()), dtype=np.float32)
        self.gates = np.array(gates, dtype=np.float32)
        context_gate_weights = payload.get("context_gate_weights")
        context_gate_bias = payload.get("context_gate_bias")
        if (
            isinstance(context_gate_weights, list)
            and len(context_gate_weights) == self.num_experts
            and all(isinstance(row, list) and len(row) == len(self.CONTEXT_FEATURE_NAMES) for row in context_gate_weights)
        ):
            self.context_gate_weights = np.array(context_gate_weights, dtype=np.float32)
        if isinstance(context_gate_bias, list) and len(context_gate_bias) == self.num_experts:
            self.context_gate_bias = np.array(context_gate_bias, dtype=np.float32)
        self.feedback_counts = np.array(payload.get("feedback_counts", self.feedback_counts.tolist()), dtype=np.int64)
        self.recent_feedback = [str(item).strip().upper() for item in payload.get("recent_feedback", [])]
        self.recent_predictions = [str(item).strip().upper() for item in payload.get("recent_predictions", [])]
        self._normalize()

    def _dense_scores(self, prediction: Optional[StepPrediction], expert_name: str) -> np.ndarray:
        vector = np.zeros((self.num_steps,), dtype=np.float32)
        scores = prediction.scores if prediction is not None else {}
        leak = self.leak_by_expert.get(expert_name, 0.0)
        for index, step_id in enumerate(self.steps):
            vector[index] = float(scores.get(step_id, leak))
        return np.clip(vector, 0.0, 1.0)

    def _transition_term(self, step_id: str, prev_step: Optional[str]) -> float:
        if not prev_step or not self.transitions:
            return 0.0
        allowed = set(self.transitions.get(str(prev_step).strip().upper(), []))
        if step_id in allowed:
            return 0.0
        return -self.lambda_transition

    def _feedback_impact(self, feedback: FeedbackEvent, confidence: float) -> float:
        confidence = float(np.clip(confidence, 0.0, 1.0))
        if feedback.accepted:
            if confidence >= self.freeze_confidence:
                return 0.0
            return math.pow(max(1e-6, 1.0 - confidence), self.hit_gamma)
        return math.pow(max(1e-3, confidence), self.error_gamma)

    def _effective_eta(self, label: str, strength: float) -> float:
        label_index = self.step_to_index[label]
        exposure = 1.0 / math.pow(max(1, int(self.feedback_counts[label_index]) + 1), self.exposure_rho)
        feedback_fraction = 0.0
        if self.recent_feedback:
            feedback_fraction = sum(1 for item in self.recent_feedback if item == label) / float(len(self.recent_feedback))
        damping = 0.5 if feedback_fraction > self.balance_tau else 1.0
        return float(self.eta * max(0.0, strength) * exposure * damping)

    def _update_gates(
        self,
        feedback: FeedbackEvent,
        label: str,
        expert_predictions: Dict[str, StepPrediction],
        fusion_result: FusionResult,
        target_features: np.ndarray,
        competitor_features: np.ndarray,
        impact: float,
    ) -> None:
        expert_tops = {
            expert_name: (expert_predictions.get(expert_name).step_id.strip().upper() if expert_predictions.get(expert_name) else "")
            for expert_name in self.expert_names
        }
        support = target_features / max(1e-6, float(target_features.sum()))
        opposition = competitor_features / max(1e-6, float(competitor_features.sum()))

        deltas = np.zeros((self.num_experts,), dtype=np.float32)
        if feedback.accepted:
            deltas += self.gate_eta * impact * support
        else:
            supporters = [index for index, expert_name in enumerate(self.expert_names) if expert_tops.get(expert_name) == label]
            if 0 < len(supporters) < self.num_experts:
                supporter_gain = self.gate_eta * impact / max(1, len(supporters))
                non_supporters = max(1, self.num_experts - len(supporters))
                for index in range(self.num_experts):
                    if index in supporters:
                        deltas[index] += supporter_gain
                    else:
                        deltas[index] -= self.gate_eta * impact / non_supporters
            else:
                deltas += self.gate_eta * impact * (support - opposition)

        if fusion_result.step_id != label:
            wrong_contrib = fusion_result.contributions.get(fusion_result.step_id, {})
            total_wrong = sum(abs(float(wrong_contrib.get(expert_name, 0.0))) for expert_name in self.expert_names) + 1e-6
            for expert_index, expert_name in enumerate(self.expert_names):
                deltas[expert_index] -= (
                    self.gate_eta
                    * impact
                    * abs(float(wrong_contrib.get(expert_name, 0.0)))
                    / total_wrong
                )

        self.gates += deltas
        self._update_context_weights_from_feedback(
            feedback=feedback,
            expert_predictions=expert_predictions,
            fusion_result=fusion_result,
            label=label,
            impact=impact,
        )

    def _initialize_context_gate_weights(self) -> None:
        feature_index = {name: index for index, name in enumerate(self.CONTEXT_FEATURE_NAMES)}
        for expert_name, expert_index in self.expert_to_index.items():
            if expert_name == "state":
                self.context_gate_weights[expert_index, feature_index["has_visual_evidence"]] = 0.40
                self.context_gate_weights[expert_index, feature_index["relevant_density"]] = 0.32
                self.context_gate_weights[expert_index, feature_index["bootstrap_margin"]] = 0.18
            elif expert_name == "temporal":
                self.context_gate_weights[expert_index, feature_index["contact_density"]] = 0.22
                self.context_gate_weights[expert_index, feature_index["support_density"]] = 0.28
                self.context_gate_weights[expert_index, feature_index["directional_density"]] = 0.18
                self.context_gate_weights[expert_index, feature_index["expert_disagreement"]] = 0.22
                self.context_gate_weights[expert_index, feature_index["transition_active"]] = 0.30
            elif expert_name == "retrieval":
                self.context_gate_weights[expert_index, feature_index["bootstrap_margin"]] = -0.20
                self.context_gate_weights[expert_index, feature_index["relevant_density"]] = -0.10
                self.context_gate_weights[expert_index, feature_index["expert_disagreement"]] = 0.12
            elif expert_name == "memory":
                self.context_gate_weights[expert_index, feature_index["memory_active"]] = 0.42
                self.context_gate_weights[expert_index, feature_index["review_hold"]] = 0.18
                self.context_gate_weights[expert_index, feature_index["review_request_human"]] = 0.18
                self.context_gate_weights[expert_index, feature_index["has_visual_evidence"]] = -0.14
                self.context_gate_weights[expert_index, feature_index["expert_disagreement"]] = 0.18

    def _load_context_gate(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("expert_names") != self.expert_names:
            return
        if payload.get("feature_names") != list(self.CONTEXT_FEATURE_NAMES):
            return
        base_gates = payload.get("base_gates")
        weights = payload.get("context_gate_weights")
        bias = payload.get("context_gate_bias")
        if isinstance(base_gates, list) and len(base_gates) == self.num_experts:
            self.gates = np.array(base_gates, dtype=np.float32)
        if (
            isinstance(weights, list)
            and len(weights) == self.num_experts
            and all(isinstance(row, list) and len(row) == len(self.CONTEXT_FEATURE_NAMES) for row in weights)
        ):
            self.context_gate_weights = np.array(weights, dtype=np.float32)
        if isinstance(bias, list) and len(bias) == self.num_experts:
            self.context_gate_bias = np.array(bias, dtype=np.float32)
        self.context_gate_loaded = True
        self._normalize()

    def _context_vector(
        self,
        *,
        token: Optional[EvidenceToken],
        expert_predictions: Dict[str, StepPrediction],
        prev_step: Optional[str],
    ) -> np.ndarray:
        if token is None:
            return np.zeros((len(self.CONTEXT_FEATURE_NAMES),), dtype=np.float32)
        review_action = str(token.review_action).strip().lower()
        visible_total = float(sum(int(value) for value in token.visible_counts.values()))
        relevant_total = float(sum(int(value) for value in token.relevant_counts.values()))
        contact_total = float(token.relation_counts.get("contacting", 0) + token.relation_counts.get("supported_by", 0))
        support_total = float(token.relation_counts.get("supporting", 0) + token.relation_counts.get("supported_by", 0))
        directional_total = float(
            token.relation_counts.get("in_front_of", 0)
            + token.relation_counts.get("behind", 0)
            + token.relation_counts.get("left_of", 0)
            + token.relation_counts.get("right_of", 0)
            + token.relation_counts.get("above", 0)
            + token.relation_counts.get("below", 0)
        )
        overlap_total = float(token.relation_counts.get("overlapping", 0))

        bootstrap_scores = {}
        for step_id in self.steps:
            bootstrap_scores[step_id] = (
                float(token.state_scores.get(step_id, 0.0))
                + float(token.retrieval_scores.get(step_id, 0.0))
                + float(token.memory_scores.get(step_id, 0.0))
            ) / 3.0
        ordered = sorted(bootstrap_scores.values(), reverse=True)
        bootstrap_margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else float(ordered[0] if ordered else 0.0)

        top_steps = []
        for expert_name in self.expert_names:
            prediction = expert_predictions.get(expert_name)
            if prediction is None:
                continue
            top_steps.append(str(prediction.step_id).strip().upper())
        unique_top_steps = len({step_id for step_id in top_steps if step_id})
        disagreement = float(max(0, unique_top_steps - 1)) / float(max(1, len(top_steps)))
        bootstrap_top = max(bootstrap_scores, key=bootstrap_scores.get) if bootstrap_scores else ""
        transition_active = 1.0 if prev_step and bootstrap_top and bootstrap_top != str(prev_step).strip().upper() else 0.0

        return np.asarray(
            [
                1.0 if token.has_visual_evidence else 0.0,
                1.0 if token.memory_active else 0.0,
                1.0 if review_action == "hold" else 0.0,
                1.0 if review_action == "request_human" else 0.0,
                min(1.0, visible_total / 4.0),
                min(1.0, relevant_total / 4.0),
                min(1.0, contact_total / 3.0),
                min(1.0, support_total / 3.0),
                min(1.0, directional_total / 4.0),
                min(1.0, overlap_total / 2.0),
                max(0.0, min(1.0, bootstrap_margin)),
                max(0.0, min(1.0, disagreement)),
                transition_active,
            ],
            dtype=np.float32,
        )

    def _effective_gates(self, context_vector: np.ndarray) -> np.ndarray:
        if not self.context_gate_enabled:
            return self.gates.copy()
        logits = np.log(np.maximum(self.gates, 1e-6))
        logits = logits + self.context_gate_bias + self.context_gate_scale * np.matmul(self.context_gate_weights, context_vector)
        exp_logits = np.exp(logits - np.max(logits))
        denom = float(exp_logits.sum())
        if denom <= 0.0:
            return self.gates.copy()
        return exp_logits / denom

    def _update_context_weights_from_feedback(
        self,
        *,
        feedback: FeedbackEvent,
        expert_predictions: Dict[str, StepPrediction],
        fusion_result: FusionResult,
        label: str,
        impact: float,
    ) -> None:
        if not self.context_gate_enabled:
            return
        feature_payload = fusion_result.extras.get("context_gate_features", {})
        if not isinstance(feature_payload, dict) or not feature_payload:
            return
        context_vector = np.asarray(
            [float(feature_payload.get(name, 0.0)) for name in self.CONTEXT_FEATURE_NAMES],
            dtype=np.float32,
        )
        if np.allclose(context_vector, 0.0):
            return
        target_sign = np.zeros((self.num_experts,), dtype=np.float32)
        for expert_name, expert_index in self.expert_to_index.items():
            prediction = expert_predictions.get(expert_name)
            pred_step = str(prediction.step_id).strip().upper() if prediction is not None else ""
            if feedback.accepted:
                if pred_step == fusion_result.step_id:
                    target_sign[expert_index] += 1.0
            else:
                if pred_step == label:
                    target_sign[expert_index] += 1.0
                if pred_step == fusion_result.step_id and fusion_result.step_id != label:
                    target_sign[expert_index] -= 1.0
        self.context_gate_weights += self.context_gate_eta * impact * np.outer(target_sign, context_vector)
        self.context_gate_bias += self.context_gate_eta * impact * target_sign

    def _normalize(self) -> None:
        self.weights = np.clip(self.weights, self.clamp_lo, self.clamp_hi)
        for expert_index in range(self.num_experts):
            column = np.maximum(self.weights[:, expert_index], self.floor_per_class)
            column_sum = float(column.sum())
            if column_sum <= 0.0:
                column = np.full((self.num_steps,), 1.0 / self.num_steps, dtype=np.float32)
            else:
                column = column / column_sum
            self.weights[:, expert_index] = column

        self.bias = np.clip(self.bias, -self.bias_cap, self.bias_cap)
        self.context_gate_weights = np.clip(self.context_gate_weights, -1.5, 1.5)
        self.context_gate_bias = np.clip(self.context_gate_bias, -0.75, 0.75)
        self.gates = np.clip(self.gates, self.clamp_lo, self.clamp_hi)
        gate_sum = float(self.gates.sum())
        if gate_sum <= 0.0:
            self.gates = np.full((self.num_experts,), 1.0 / float(self.num_experts), dtype=np.float32)
        else:
            self.gates = self.gates / gate_sum

    @staticmethod
    def _softmax(values: List[float]) -> List[float]:
        if not values:
            return []
        array = np.array(values, dtype=np.float32)
        array = array - array.max()
        probs = np.exp(array)
        probs = probs / max(1e-6, float(probs.sum()))
        return [float(item) for item in probs]


class LinearMarginFusion(AdaptiveExpertFusion):
    """Backward-compatible name for the default adaptive fusion head."""

    pass
