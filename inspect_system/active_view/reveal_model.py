"""Prior-table reveal model trained from human assistant relative-view events."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .ontology import normalize_claim, normalize_key
from .trace_event_miner import MinedEvent
from .view_lattice import ORBIT_ACTIONS, RelativeAction


@dataclass
class TrainingExample:
    claim_id: str
    evidence_role: str
    relative_action: str
    label: int
    weight: float = 1.0
    source_event_id: str = ""
    utility_before: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def examples_from_mined_events(
    events: Iterable[MinedEvent],
    *,
    negative_other_actions: bool = True,
    negative_weight: float = 0.25,
) -> List[TrainingExample]:
    examples: List[TrainingExample] = []
    for event in events:
        if not event.transferable:
            continue
        if event.relative_action not in ORBIT_ACTIONS:
            continue
        weight = (
            max(0.0, float(event.transfer_weight))
            * max(0.0, float(event.label_confidence))
            * max(0.0, float(event.evidence_stability))
            * max(0.0, float(event.evidence_importance))
        )
        if weight <= 0.0:
            continue
        examples.append(
            TrainingExample(
                claim_id=event.claim_id,
                evidence_role=event.evidence_role,
                relative_action=event.relative_action,
                label=int(event.label),
                weight=weight,
                source_event_id=event.event_id,
                utility_before=(
                    float(event.metadata["utility_before"])
                    if event.metadata.get("utility_before") not in (None, "")
                    else None
                ),
            )
        )
        if negative_other_actions and event.label:
            for action in ORBIT_ACTIONS:
                if action == event.relative_action:
                    continue
                examples.append(
                    TrainingExample(
                        claim_id=event.claim_id,
                        evidence_role=event.evidence_role,
                        relative_action=action,
                        label=0,
                        weight=negative_weight * weight,
                        source_event_id=event.event_id,
                        utility_before=(
                            float(event.metadata["utility_before"])
                            if event.metadata.get("utility_before") not in (None, "")
                            else None
                        ),
                    )
                )
    return examples


@dataclass
class PriorTableRevealModel:
    alpha: float = 1.0
    default_probability: float = 0.20
    counts: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _key(claim_id: str, evidence_role: str, action: str) -> str:
        return "|".join([normalize_claim(claim_id), normalize_key(evidence_role), normalize_key(action)])

    @staticmethod
    def _fallback_keys(claim_id: str, evidence_role: str, action: str) -> List[str]:
        claim = normalize_claim(claim_id)
        role = normalize_key(evidence_role)
        act = normalize_key(action)
        return [
            PriorTableRevealModel._key(claim, role, act),
            PriorTableRevealModel._key(claim, "__any__", act),
            PriorTableRevealModel._key("__global__", role, act),
            PriorTableRevealModel._key("__global__", "__any__", act),
        ]

    def fit(self, examples: Iterable[TrainingExample]) -> "PriorTableRevealModel":
        self.counts = {}
        total_seen = 0
        for example in examples:
            action = normalize_key(example.relative_action)
            if action not in ORBIT_ACTIONS:
                continue
            label = 1.0 if int(example.label) else 0.0
            weight = max(0.0, float(example.weight))
            if weight <= 0.0:
                continue
            total_seen += 1
            keys = [
                self._key(example.claim_id, example.evidence_role, action),
                self._key(example.claim_id, "__any__", action),
                self._key("__global__", example.evidence_role, action),
                self._key("__global__", "__any__", action),
            ]
            for key in keys:
                bucket = self.counts.setdefault(key, {"pos": 0.0, "total": 0.0})
                bucket["pos"] += label * weight
                bucket["total"] += weight
        self.metadata.update(
            {
                "model_type": "prior_table_relative_reveal",
                "num_training_examples": total_seen,
                "actions": list(ORBIT_ACTIONS),
                "uses_robot_view_training": False,
            }
        )
        return self

    def probability(self, claim_id: str, evidence_role: str, action: str, context: Mapping[str, Any] | None = None) -> float:
        action = normalize_key(action)
        for key in self._fallback_keys(claim_id, evidence_role, action):
            bucket = self.counts.get(key)
            if bucket and bucket.get("total", 0.0) > 0:
                pos = float(bucket.get("pos", 0.0))
                total = float(bucket.get("total", 0.0))
                return (pos + self.alpha * self.default_probability) / (total + self.alpha)
        return self.default_probability

    def action_distribution(
        self,
        claim_id: str,
        evidence_role: str,
        context: Mapping[str, Any] | None = None,
    ) -> Dict[str, float]:
        raw = {
            action: self.probability(claim_id, evidence_role, action, context=context)
            for action in ORBIT_ACTIONS
        }
        total = sum(max(0.0, value) for value in raw.values())
        if total <= 1e-9:
            return {action: 1.0 / len(ORBIT_ACTIONS) for action in ORBIT_ACTIONS}
        return {action: value / total for action, value in raw.items()}

    def counterfactual_relevance(
        self,
        claim_id: str,
        evidence_role: str,
        context: Mapping[str, Any] | None = None,
    ) -> float:
        """Return the KB compatibility H(c, q) for the active counterfactual."""

        # Imported lazily because the learned counterfactual model extends this
        # base class. The base implementation supplies H(c, q) even when pi is
        # represented by a simpler hierarchical transport posterior.
        from .counterfactual_transport import (
            counterfactual_mixture,
            counterfactual_role_gate,
            infer_counterfactual_family,
        )

        mixture = counterfactual_mixture(
            context,
            infer_counterfactual_family(claim_id, evidence_role, context),
        )
        return sum(
            weight * counterfactual_role_gate(family, evidence_role)
            for family, weight in mixture.items()
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "default_probability": self.default_probability,
            "counts": self.counts,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PriorTableRevealModel":
        return cls(
            alpha=float(payload.get("alpha", 1.0)),
            default_probability=float(payload.get("default_probability", 0.20)),
            counts={str(key): {"pos": float(value.get("pos", 0.0)), "total": float(value.get("total", 0.0))} for key, value in dict(payload.get("counts", {})).items()},
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PriorTableRevealModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model_type = str((payload.get("metadata") or {}).get("model_type", payload.get("model_type", "")))
        if model_type == "feature_conditioned_reveal_table":
            return FeatureConditionedRevealModel.from_dict(payload)
        if model_type == "hierarchical_equivariant_evidence_transport":
            from .evidence_transport import HierarchicalEvidenceTransportModel

            return HierarchicalEvidenceTransportModel.from_dict(payload)
        if model_type == "counterfactual_evidence_transport":
            from .counterfactual_transport import CounterfactualEvidenceTransportModel

            return CounterfactualEvidenceTransportModel.from_dict(payload)
        if model_type == "affordance_calibrated_reveal":
            from .affordance_calibrated_reveal import AffordanceCalibratedRevealModel

            return AffordanceCalibratedRevealModel.from_dict(payload)
        if model_type == "object_centric_calibrated_reveal":
            from .object_centric_reveal import ObjectCentricCalibratedRevealModel

            return ObjectCentricCalibratedRevealModel.from_dict(payload)
        if model_type == "camera_rotation_gain_reveal":
            from .camera_rotation_reveal import CameraRotationGainRevealModel

            return CameraRotationGainRevealModel.from_dict(payload)
        if model_type == "camera_motion_gain_reveal":
            from .camera_motion_reveal import CameraMotionGainRevealModel

            return CameraMotionGainRevealModel.from_dict(payload)
        return cls.from_dict(payload)


def _utility_bin(value: Any) -> str:
    try:
        score = float(value)
    except Exception:
        return "__unknown__"
    if score < 0.35:
        return "low"
    if score < 0.70:
        return "mid"
    return "high"


@dataclass
class FeatureConditionedRevealModel(PriorTableRevealModel):
    feature_counts: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @staticmethod
    def _feature_key(claim_id: str, evidence_role: str, action: str, utility_bin: str) -> str:
        return "|".join([
            normalize_claim(claim_id),
            normalize_key(evidence_role),
            normalize_key(action),
            normalize_key(utility_bin),
        ])

    @staticmethod
    def _feature_fallback_keys(claim_id: str, evidence_role: str, action: str, utility_bin: str) -> List[str]:
        claim = normalize_claim(claim_id)
        role = normalize_key(evidence_role)
        act = normalize_key(action)
        b = normalize_key(utility_bin)
        return [
            FeatureConditionedRevealModel._feature_key(claim, role, act, b),
            FeatureConditionedRevealModel._feature_key(claim, "__any__", act, b),
            FeatureConditionedRevealModel._feature_key("__global__", role, act, b),
            FeatureConditionedRevealModel._feature_key("__global__", "__any__", act, b),
        ]

    def fit(self, examples: Iterable[TrainingExample]) -> "FeatureConditionedRevealModel":
        self.counts = {}
        self.feature_counts = {}
        total_seen = 0
        feature_seen = 0
        for example in examples:
            action = normalize_key(example.relative_action)
            if action not in ORBIT_ACTIONS:
                continue
            label = 1.0 if int(example.label) else 0.0
            weight = max(0.0, float(example.weight))
            if weight <= 0.0:
                continue
            total_seen += 1
            base_keys = [
                self._key(example.claim_id, example.evidence_role, action),
                self._key(example.claim_id, "__any__", action),
                self._key("__global__", example.evidence_role, action),
                self._key("__global__", "__any__", action),
            ]
            for key in base_keys:
                bucket = self.counts.setdefault(key, {"pos": 0.0, "total": 0.0})
                bucket["pos"] += label * weight
                bucket["total"] += weight
            if example.utility_before is not None:
                feature_seen += 1
                bin_name = _utility_bin(example.utility_before)
                feature_keys = self._feature_fallback_keys(example.claim_id, example.evidence_role, action, bin_name)
                for key in feature_keys:
                    bucket = self.feature_counts.setdefault(key, {"pos": 0.0, "total": 0.0})
                    bucket["pos"] += label * weight
                    bucket["total"] += weight
        self.metadata.update(
            {
                "model_type": "feature_conditioned_reveal_table",
                "num_training_examples": total_seen,
                "num_feature_examples": feature_seen,
                "actions": list(ORBIT_ACTIONS),
                "uses_robot_view_training": False,
                "feature": "utility_before_bin",
            }
        )
        return self

    def probability(self, claim_id: str, evidence_role: str, action: str, context: Mapping[str, Any] | None = None) -> float:
        action = normalize_key(action)
        context = context or {}
        value = context.get("utility_before", context.get("item_score", context.get("observed_score", None)))
        bin_name = _utility_bin(value)
        if bin_name != "__unknown__":
            for key in self._feature_fallback_keys(claim_id, evidence_role, action, bin_name):
                bucket = self.feature_counts.get(key)
                if bucket and bucket.get("total", 0.0) > 0:
                    pos = float(bucket.get("pos", 0.0))
                    total = float(bucket.get("total", 0.0))
                    return (pos + self.alpha * self.default_probability) / (total + self.alpha)
        return super().probability(claim_id, evidence_role, action, context=context)

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        payload["feature_counts"] = self.feature_counts
        payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureConditionedRevealModel":
        return cls(
            alpha=float(payload.get("alpha", 1.0)),
            default_probability=float(payload.get("default_probability", 0.20)),
            counts={str(key): {"pos": float(value.get("pos", 0.0)), "total": float(value.get("total", 0.0))} for key, value in dict(payload.get("counts", {})).items()},
            metadata=dict(payload.get("metadata", {}) or {}),
            feature_counts={str(key): {"pos": float(value.get("pos", 0.0)), "total": float(value.get("total", 0.0))} for key, value in dict(payload.get("feature_counts", {})).items()},
        )
