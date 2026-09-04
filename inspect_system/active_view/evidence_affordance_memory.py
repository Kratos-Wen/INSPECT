"""Continual outcome memory for evidence-seeking observation changes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from .ontology import normalize_key


def _safe_key(value: object, fallback: str = "generic") -> str:
    return normalize_key(value) or fallback


@dataclass(frozen=True)
class EvidenceAffordanceContext:
    """Semantic context in which an observation change was attempted."""

    action: str
    role: str
    counterfactual: str
    claim: str = "generic"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EvidenceAffordanceContext":
        return cls(
            action=_safe_key(payload.get("action"), "unknown"),
            role=_safe_key(payload.get("role")),
            counterfactual=_safe_key(payload.get("counterfactual")),
            claim=_safe_key(payload.get("claim")),
        )

    @property
    def semantic_key(self) -> Tuple[str, str]:
        return self.role, self.counterfactual

    @property
    def exact_key(self) -> Tuple[str, str, str]:
        return self.action, self.role, self.counterfactual


@dataclass
class OutcomeStatistics:
    """Fractional outcome counts and signed counterfactual-gain moments."""

    positive_weight: float = 0.0
    negative_weight: float = 0.0
    gain_sum: float = 0.0
    gain_square_sum: float = 0.0
    update_count: int = 0

    @property
    def weight(self) -> float:
        return self.positive_weight + self.negative_weight

    @property
    def mean_gain(self) -> float:
        return self.gain_sum / self.weight if self.weight > 0.0 else 0.0

    def update(self, gain: float, weight: float = 1.0) -> None:
        value = float(gain)
        mass = max(0.0, float(weight))
        if not math.isfinite(value) or not math.isfinite(mass) or mass == 0.0:
            return
        if value > 0.0:
            self.positive_weight += mass
        else:
            self.negative_weight += mass
        self.gain_sum += mass * value
        self.gain_square_sum += mass * value * value
        self.update_count += 1

    def probability(self, prior_probability: float, prior_strength: float) -> float:
        strength = max(0.0, float(prior_strength))
        denominator = self.weight + strength
        if denominator <= 0.0:
            return float(prior_probability)
        return (self.positive_weight + strength * prior_probability) / denominator

    def expected_gain(self, prior_gain: float, prior_strength: float) -> float:
        strength = max(0.0, float(prior_strength))
        denominator = self.weight + strength
        if denominator <= 0.0:
            return float(prior_gain)
        return (self.gain_sum + strength * prior_gain) / denominator

    def to_dict(self) -> Dict[str, Any]:
        return {
            "positive_weight": self.positive_weight,
            "negative_weight": self.negative_weight,
            "gain_sum": self.gain_sum,
            "gain_square_sum": self.gain_square_sum,
            "update_count": self.update_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OutcomeStatistics":
        return cls(
            positive_weight=float(payload.get("positive_weight", 0.0)),
            negative_weight=float(payload.get("negative_weight", 0.0)),
            gain_sum=float(payload.get("gain_sum", 0.0)),
            gain_square_sum=float(payload.get("gain_square_sum", 0.0)),
            update_count=int(payload.get("update_count", 0)),
        )


@dataclass(frozen=True)
class EvidenceAffordancePrediction:
    helpful_probability: float
    expected_gain: float
    uncertainty: float
    effective_support: float
    scope: str


@dataclass(frozen=True)
class EvidenceAffordanceConfig:
    global_prior_strength: float = 4.0
    action_prior_strength: float = 6.0
    semantic_prior_strength: float = 6.0
    exact_prior_strength: float = 3.0
    session_prior_strength: float = 2.0
    max_exact_atoms: int = 256


@dataclass
class EvidenceAffordanceMemory:
    """Hierarchical evidence atoms that update after verified daily use."""

    config: EvidenceAffordanceConfig = field(default_factory=EvidenceAffordanceConfig)
    global_outcome: OutcomeStatistics = field(default_factory=OutcomeStatistics)
    action_atoms: Dict[str, OutcomeStatistics] = field(default_factory=dict)
    semantic_atoms: Dict[Tuple[str, str], OutcomeStatistics] = field(default_factory=dict)
    exact_atoms: Dict[Tuple[str, str, str], OutcomeStatistics] = field(default_factory=dict)
    session_atoms: Dict[str, OutcomeStatistics] = field(default_factory=dict)
    session_exact_atoms: Dict[Tuple[str, str, str, str], OutcomeStatistics] = field(
        default_factory=dict
    )

    @staticmethod
    def _atom(
        table: Dict[Any, OutcomeStatistics], key: Any
    ) -> OutcomeStatistics:
        if key not in table:
            table[key] = OutcomeStatistics()
        return table[key]

    def _prune_exact_atoms(self) -> None:
        limit = max(1, int(self.config.max_exact_atoms))
        if len(self.exact_atoms) <= limit:
            return
        victim = min(
            self.exact_atoms,
            key=lambda key: (
                self.exact_atoms[key].weight,
                self.exact_atoms[key].update_count,
                key,
            ),
        )
        del self.exact_atoms[victim]

    def update(
        self,
        context: EvidenceAffordanceContext,
        signed_gain: float,
        weight: float = 1.0,
        session_id: str | None = None,
        update_shared: bool = True,
    ) -> None:
        if update_shared:
            self.global_outcome.update(signed_gain, weight)
            self._atom(self.action_atoms, context.action).update(signed_gain, weight)
            self._atom(self.semantic_atoms, context.semantic_key).update(signed_gain, weight)
            self._atom(self.exact_atoms, context.exact_key).update(signed_gain, weight)
            self._prune_exact_atoms()
        if session_id:
            session = _safe_key(session_id, "session")
            self._atom(self.session_atoms, session).update(signed_gain, weight)
            key = (session, *context.exact_key)
            self._atom(self.session_exact_atoms, key).update(signed_gain, weight)

    def fit(
        self,
        examples: Iterable[Tuple[EvidenceAffordanceContext, float, float]],
    ) -> "EvidenceAffordanceMemory":
        for context, signed_gain, weight in examples:
            self.update(context, signed_gain, weight)
        return self

    def _base_prediction(
        self, context: EvidenceAffordanceContext
    ) -> Tuple[float, float, float, str]:
        cfg = self.config
        global_probability = self.global_outcome.probability(
            0.5, cfg.global_prior_strength
        )
        global_gain = self.global_outcome.expected_gain(0.0, cfg.global_prior_strength)

        action = self.action_atoms.get(context.action, OutcomeStatistics())
        action_probability = action.probability(
            global_probability, cfg.action_prior_strength
        )
        action_gain = action.expected_gain(global_gain, cfg.action_prior_strength)

        semantic = self.semantic_atoms.get(context.semantic_key, OutcomeStatistics())
        semantic_probability = semantic.probability(
            global_probability, cfg.semantic_prior_strength
        )
        semantic_gain = semantic.expected_gain(global_gain, cfg.semantic_prior_strength)

        action_reliability = action.weight / (action.weight + cfg.action_prior_strength)
        semantic_reliability = semantic.weight / (
            semantic.weight + cfg.semantic_prior_strength
        )
        total_reliability = action_reliability + semantic_reliability
        if total_reliability > 0.0:
            parent_probability = (
                action_reliability * action_probability
                + semantic_reliability * semantic_probability
            ) / total_reliability
            parent_gain = (
                action_reliability * action_gain
                + semantic_reliability * semantic_gain
            ) / total_reliability
            scope = "action+semantic"
        else:
            parent_probability = global_probability
            parent_gain = global_gain
            scope = "global"

        exact = self.exact_atoms.get(context.exact_key, OutcomeStatistics())
        probability = exact.probability(parent_probability, cfg.exact_prior_strength)
        gain = exact.expected_gain(parent_gain, cfg.exact_prior_strength)
        if exact.weight > 0.0:
            scope = "exact"
        support = action.weight + semantic.weight + exact.weight
        return probability, gain, support, scope

    def predict(
        self,
        context: EvidenceAffordanceContext,
        session_id: str | None = None,
    ) -> EvidenceAffordancePrediction:
        probability, gain, support, scope = self._base_prediction(context)
        if session_id:
            session = _safe_key(session_id, "session")
            session_global = self.session_atoms.get(session, OutcomeStatistics())
            session_exact = self.session_exact_atoms.get(
                (session, *context.exact_key), OutcomeStatistics()
            )
            local_weight = session_global.weight + session_exact.weight
            if local_weight > 0.0:
                local_probability = session_global.probability(
                    probability, self.config.session_prior_strength
                )
                local_gain = session_global.expected_gain(
                    gain, self.config.session_prior_strength
                )
                local_probability = session_exact.probability(
                    local_probability, self.config.session_prior_strength
                )
                local_gain = session_exact.expected_gain(
                    local_gain, self.config.session_prior_strength
                )
                reliability = local_weight / (
                    local_weight + self.config.session_prior_strength
                )
                probability = (1.0 - reliability) * probability + reliability * local_probability
                gain = (1.0 - reliability) * gain + reliability * local_gain
                support += local_weight
                scope = "session+" + scope

        probability = min(1.0 - 1e-6, max(1e-6, float(probability)))
        uncertainty = 4.0 * probability * (1.0 - probability) / math.sqrt(1.0 + support)
        return EvidenceAffordancePrediction(
            helpful_probability=probability,
            expected_gain=float(gain),
            uncertainty=float(uncertainty),
            effective_support=float(support),
            scope=scope,
        )

    def to_dict(self) -> Dict[str, Any]:
        def encode_key(key: Sequence[str]) -> str:
            return "\u241f".join(key)

        return {
            "model_type": "evidence_affordance_memory",
            "config": self.config.__dict__,
            "global_outcome": self.global_outcome.to_dict(),
            "action_atoms": {
                key: value.to_dict() for key, value in self.action_atoms.items()
            },
            "semantic_atoms": {
                encode_key(key): value.to_dict()
                for key, value in self.semantic_atoms.items()
            },
            "exact_atoms": {
                encode_key(key): value.to_dict() for key, value in self.exact_atoms.items()
            },
            "session_atoms": {
                key: value.to_dict() for key, value in self.session_atoms.items()
            },
            "session_exact_atoms": {
                encode_key(key): value.to_dict()
                for key, value in self.session_exact_atoms.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceAffordanceMemory":
        if payload.get("model_type") != "evidence_affordance_memory":
            raise ValueError("Evidence affordance memory schema mismatch")

        def decode_key(value: str) -> Tuple[str, ...]:
            return tuple(value.split("\u241f"))

        model = cls(
            config=EvidenceAffordanceConfig(**dict(payload.get("config", {}))),
            global_outcome=OutcomeStatistics.from_dict(
                payload.get("global_outcome", {})
            ),
        )
        model.action_atoms = {
            str(key): OutcomeStatistics.from_dict(value)
            for key, value in payload.get("action_atoms", {}).items()
        }
        model.semantic_atoms = {
            decode_key(key): OutcomeStatistics.from_dict(value)
            for key, value in payload.get("semantic_atoms", {}).items()
        }
        model.exact_atoms = {
            decode_key(key): OutcomeStatistics.from_dict(value)
            for key, value in payload.get("exact_atoms", {}).items()
        }
        model.session_atoms = {
            str(key): OutcomeStatistics.from_dict(value)
            for key, value in payload.get("session_atoms", {}).items()
        }
        model.session_exact_atoms = {
            decode_key(key): OutcomeStatistics.from_dict(value)
            for key, value in payload.get("session_exact_atoms", {}).items()
        }
        return model
