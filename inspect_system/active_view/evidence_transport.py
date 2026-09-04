"""Hierarchical, symmetry-aware evidence transport learned from assistant use."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping

from .ontology import normalize_claim, normalize_key
from .reveal_model import PriorTableRevealModel, TrainingExample
from .trace_event_miner import MinedEvent
from .view_lattice import ORBIT_ACTIONS


ACTION_FAMILIES = (
    "horizontal",
    "up",
    "down",
    "upper_diagonal",
    "lower_diagonal",
)

FAMILY_MEMBERS: Mapping[str, tuple[str, ...]] = {
    "horizontal": ("orbit_left", "orbit_right"),
    "up": ("orbit_up",),
    "down": ("orbit_down",),
    "upper_diagonal": ("orbit_left_up", "orbit_right_up"),
    "lower_diagonal": ("orbit_left_down", "orbit_right_down"),
}

ACTION_TO_FAMILY = {
    action: family
    for family, actions in FAMILY_MEMBERS.items()
    for action in actions
}


def action_family(action: str) -> str:
    return ACTION_TO_FAMILY.get(normalize_key(action), "")


def _normalized(values: Mapping[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(values.get(key, 0.0))) for key in ACTION_FAMILIES)
    if total <= 1e-12:
        return {key: 1.0 / len(ACTION_FAMILIES) for key in ACTION_FAMILIES}
    return {
        key: max(0.0, float(values.get(key, 0.0))) / total
        for key in ACTION_FAMILIES
    }


def _posterior(
    counts: Mapping[str, float],
    prior: Mapping[str, float],
    strength: float,
) -> Dict[str, float]:
    concentration = max(1e-9, float(strength))
    total = sum(max(0.0, float(counts.get(key, 0.0))) for key in ACTION_FAMILIES)
    return _normalized(
        {
            key: max(0.0, float(counts.get(key, 0.0)))
            + concentration * max(0.0, float(prior.get(key, 0.0)))
            for key in ACTION_FAMILIES
        }
        if total + concentration > 0.0
        else prior
    )


def _event_weight(event: MinedEvent, gain_power: float) -> float:
    base = (
        max(0.0, float(event.transfer_weight))
        * max(0.0, float(event.label_confidence))
        * max(0.0, float(event.evidence_stability))
        * max(0.0, float(event.evidence_importance))
    )
    metadata = event.metadata or {}
    gain = metadata.get("score_gain")
    if gain in (None, ""):
        before = metadata.get("before_score")
        after = metadata.get("after_score")
        if before not in (None, "") and after not in (None, ""):
            gain = float(after) - float(before)
    gain_value = max(0.05, min(1.0, float(gain if gain not in (None, "") else 1.0)))
    return base * gain_value ** max(0.0, float(gain_power))


@dataclass
class HierarchicalEvidenceTransportModel(PriorTableRevealModel):
    """Empirical-Bayes transport posterior over reflection-equivalent actions."""

    base_concentration: float = 1.0
    hierarchy_strength: float = 1.0
    risk_beta: float = 0.0
    gain_power: float = 1.0
    confidence_scale: float = 0.0
    global_family_counts: Dict[str, float] = field(default_factory=dict)
    claim_family_counts: Dict[str, Dict[str, float]] = field(default_factory=dict)
    role_family_counts: Dict[str, Dict[str, float]] = field(default_factory=dict)
    claim_role_family_counts: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @staticmethod
    def _claim_role_key(claim_id: str, evidence_role: str) -> str:
        return f"{normalize_claim(claim_id)}|{normalize_key(evidence_role)}"

    @staticmethod
    def _add(
        table: Dict[str, Dict[str, float]],
        key: str,
        family: str,
        weight: float,
    ) -> None:
        bucket = table.setdefault(key, {})
        bucket[family] = float(bucket.get(family, 0.0)) + float(weight)

    def _reset(self) -> None:
        self.counts = {}
        self.global_family_counts = {}
        self.claim_family_counts = {}
        self.role_family_counts = {}
        self.claim_role_family_counts = {}

    def _observe(
        self,
        claim_id: str,
        evidence_role: str,
        relative_action: str,
        weight: float,
    ) -> bool:
        family = action_family(relative_action)
        if not family or weight <= 0.0:
            return False
        claim = normalize_claim(claim_id)
        role = normalize_key(evidence_role)
        self.global_family_counts[family] = (
            float(self.global_family_counts.get(family, 0.0)) + float(weight)
        )
        self._add(self.claim_family_counts, claim, family, weight)
        self._add(self.role_family_counts, role, family, weight)
        self._add(
            self.claim_role_family_counts,
            self._claim_role_key(claim, role),
            family,
            weight,
        )
        return True

    def fit_events(self, events: Iterable[MinedEvent]) -> "HierarchicalEvidenceTransportModel":
        self._reset()
        seen = 0
        videos = set()
        for event in events:
            if not event.transferable or not event.label:
                continue
            weight = _event_weight(event, self.gain_power)
            if self._observe(
                event.claim_id,
                event.evidence_role,
                event.relative_action,
                weight,
            ):
                seen += 1
                if event.video:
                    videos.add(event.video)
        self.metadata.update(
            {
                "model_type": "hierarchical_equivariant_evidence_transport",
                "num_training_events": seen,
                "num_training_videos": len(videos),
                "action_families": list(ACTION_FAMILIES),
                "symmetry_group": "horizontal_reflection",
                "uses_robot_view_training": False,
            }
        )
        return self

    def fit(self, examples: Iterable[TrainingExample]) -> "HierarchicalEvidenceTransportModel":
        self._reset()
        seen = 0
        for example in examples:
            if not int(example.label):
                continue
            if self._observe(
                example.claim_id,
                example.evidence_role,
                example.relative_action,
                max(0.0, float(example.weight)),
            ):
                seen += 1
        self.metadata.update(
            {
                "model_type": "hierarchical_equivariant_evidence_transport",
                "num_training_events": seen,
                "action_families": list(ACTION_FAMILIES),
                "symmetry_group": "horizontal_reflection",
                "uses_robot_view_training": False,
            }
        )
        return self

    @staticmethod
    def _mass(counts: Mapping[str, float]) -> float:
        return sum(max(0.0, float(value)) for value in counts.values())

    def family_distribution(self, claim_id: str, evidence_role: str) -> Dict[str, float]:
        uniform = {key: 1.0 / len(ACTION_FAMILIES) for key in ACTION_FAMILIES}
        global_p = _posterior(
            self.global_family_counts,
            uniform,
            self.base_concentration,
        )
        claim = normalize_claim(claim_id)
        role = normalize_key(evidence_role)
        claim_counts = self.claim_family_counts.get(claim, {})
        role_counts = self.role_family_counts.get(role, {})
        exact_counts = self.claim_role_family_counts.get(
            self._claim_role_key(claim, role),
            {},
        )
        claim_p = _posterior(
            claim_counts,
            global_p,
            self.hierarchy_strength,
        )
        role_p = _posterior(
            role_counts,
            global_p,
            self.hierarchy_strength,
        )
        claim_reliability = self._mass(claim_counts) / (
            self._mass(claim_counts) + self.hierarchy_strength
        )
        role_reliability = self._mass(role_counts) / (
            self._mass(role_counts) + self.hierarchy_strength
        )
        parent = _normalized(
            {
                family: global_p[family]
                + claim_reliability * claim_p[family]
                + role_reliability * role_p[family]
                for family in ACTION_FAMILIES
            }
        )
        posterior = _posterior(
            exact_counts,
            parent,
            self.hierarchy_strength,
        )
        if self.risk_beta <= 0.0:
            return posterior
        concentration = self._mass(exact_counts) + self.hierarchy_strength
        lower = {}
        for family, probability in posterior.items():
            variance = probability * (1.0 - probability) / max(1.0, concentration + 1.0)
            lower[family] = max(
                1e-6,
                probability - self.risk_beta * math.sqrt(max(0.0, variance)),
            )
        return _normalized(lower)

    def transport_confidence(
        self,
        claim_id: str,
        evidence_role: str,
        context: Mapping[str, Any] | None = None,
    ) -> float:
        if self.confidence_scale <= 0.0:
            return 1.0
        claim = normalize_claim(claim_id)
        role = normalize_key(evidence_role)
        exact = self._mass(
            self.claim_role_family_counts.get(self._claim_role_key(claim, role), {})
        )
        claim_mass = self._mass(self.claim_family_counts.get(claim, {}))
        role_mass = self._mass(self.role_family_counts.get(role, {}))
        support = exact + 0.25 * claim_mass + 0.25 * role_mass
        return 1.0 - math.exp(-support / max(1e-9, self.confidence_scale))

    def probability(
        self,
        claim_id: str,
        evidence_role: str,
        action: str,
        context: Mapping[str, Any] | None = None,
    ) -> float:
        normalized_action = normalize_key(action)
        family = action_family(normalized_action)
        if not family:
            return 0.0
        members = FAMILY_MEMBERS[family]
        return self.family_distribution(claim_id, evidence_role)[family] / len(members)

    def claim_conditioned_copy(self) -> "HierarchicalEvidenceTransportModel":
        return HierarchicalEvidenceTransportModel(
            alpha=self.alpha,
            default_probability=self.default_probability,
            metadata={
                **self.metadata,
                "model_type": "hierarchical_equivariant_evidence_transport",
                "ablation": "claim_conditioned_prior",
            },
            base_concentration=self.base_concentration,
            hierarchy_strength=self.hierarchy_strength,
            risk_beta=self.risk_beta,
            gain_power=self.gain_power,
            confidence_scale=self.confidence_scale,
            global_family_counts=dict(self.global_family_counts),
            claim_family_counts={
                key: dict(value) for key, value in self.claim_family_counts.items()
            },
            role_family_counts={},
            claim_role_family_counts={},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "default_probability": self.default_probability,
            "counts": {},
            "metadata": self.metadata,
            "base_concentration": self.base_concentration,
            "hierarchy_strength": self.hierarchy_strength,
            "risk_beta": self.risk_beta,
            "gain_power": self.gain_power,
            "confidence_scale": self.confidence_scale,
            "global_family_counts": self.global_family_counts,
            "claim_family_counts": self.claim_family_counts,
            "role_family_counts": self.role_family_counts,
            "claim_role_family_counts": self.claim_role_family_counts,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "HierarchicalEvidenceTransportModel":
        def nested(name: str) -> Dict[str, Dict[str, float]]:
            return {
                str(key): {str(k): float(v) for k, v in dict(value).items()}
                for key, value in dict(payload.get(name, {}) or {}).items()
            }

        return cls(
            alpha=float(payload.get("alpha", 1.0)),
            default_probability=float(payload.get("default_probability", 0.20)),
            metadata=dict(payload.get("metadata", {}) or {}),
            base_concentration=float(payload.get("base_concentration", 1.0)),
            hierarchy_strength=float(payload.get("hierarchy_strength", 1.0)),
            risk_beta=float(payload.get("risk_beta", 0.0)),
            gain_power=float(payload.get("gain_power", 1.0)),
            confidence_scale=float(payload.get("confidence_scale", 0.0)),
            global_family_counts={
                str(key): float(value)
                for key, value in dict(payload.get("global_family_counts", {}) or {}).items()
            },
            claim_family_counts=nested("claim_family_counts"),
            role_family_counts=nested("role_family_counts"),
            claim_role_family_counts=nested("claim_role_family_counts"),
        )
