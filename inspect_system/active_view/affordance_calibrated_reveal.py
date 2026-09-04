"""Fallback-safe calibration of a reveal policy with evidence outcome atoms."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from .counterfactual_transport import (
    counterfactual_mixture,
    infer_counterfactual_family,
)
from .evidence_affordance_memory import (
    EvidenceAffordanceContext,
    EvidenceAffordanceMemory,
)
from .ontology import normalize_key
from .reveal_model import PriorTableRevealModel
from .view_lattice import ORBIT_ACTIONS


class AffordanceCalibratedRevealModel(PriorTableRevealModel):
    """Reweight a frozen reveal policy by assistant-observed action reliability."""

    def __init__(
        self,
        base_model: PriorTableRevealModel,
        memory: EvidenceAffordanceMemory,
        calibration_strength: float = 0.5,
        support_strength: float = 4.0,
    ) -> None:
        super().__init__(
            alpha=float(base_model.alpha),
            default_probability=float(base_model.default_probability),
            counts={key: dict(value) for key, value in base_model.counts.items()},
            metadata={
                **dict(base_model.metadata),
                "model_type": "affordance_calibrated_reveal",
                "base_model_type": str(
                    base_model.metadata.get("model_type", "prior_table_relative_reveal")
                ),
                "uses_robot_view_training": False,
            },
        )
        self.base_model = base_model
        self.memory = memory
        self.calibration_strength = max(0.0, min(1.0, float(calibration_strength)))
        self.support_strength = max(1e-6, float(support_strength))

    def _reliability(
        self,
        claim_id: str,
        evidence_role: str,
        action: str,
        context: Mapping[str, Any] | None,
    ) -> tuple[float, float]:
        mixture = counterfactual_mixture(
            context,
            infer_counterfactual_family(claim_id, evidence_role, context),
        )
        helpful_probability = 0.0
        support = 0.0
        for family, mixture_weight in mixture.items():
            prediction = self.memory.predict(
                EvidenceAffordanceContext(
                    action=normalize_key(action),
                    role=normalize_key(evidence_role),
                    counterfactual=family,
                    claim=claim_id,
                )
            )
            helpful_probability += mixture_weight * prediction.helpful_probability
            support += mixture_weight * prediction.effective_support
        return helpful_probability, support

    def action_distribution(
        self,
        claim_id: str,
        evidence_role: str,
        context: Mapping[str, Any] | None = None,
    ) -> Dict[str, float]:
        base = self.base_model.action_distribution(
            claim_id, evidence_role, context=context
        )
        calibrated: Dict[str, float] = {}
        for action in ORBIT_ACTIONS:
            probability, support = self._reliability(
                claim_id, evidence_role, action, context
            )
            support_reliability = support / (support + self.support_strength)
            factor = 1.0 + self.calibration_strength * support_reliability * (
                2.0 * probability - 1.0
            )
            calibrated[action] = max(0.0, float(base[action])) * max(1e-6, factor)
        total = sum(calibrated.values())
        if total <= 1e-12:
            return dict(base)
        return {action: value / total for action, value in calibrated.items()}

    def probability(
        self,
        claim_id: str,
        evidence_role: str,
        action: str,
        context: Mapping[str, Any] | None = None,
    ) -> float:
        return self.action_distribution(
            claim_id, evidence_role, context=context
        ).get(normalize_key(action), 0.0)

    def counterfactual_relevance(
        self,
        claim_id: str,
        evidence_role: str,
        context: Mapping[str, Any] | None = None,
    ) -> float:
        function = getattr(self.base_model, "counterfactual_relevance", None)
        return (
            float(function(claim_id, evidence_role, context=context))
            if callable(function)
            else 1.0
        )

    def transport_confidence(
        self,
        claim_id: str,
        evidence_role: str,
        context: Mapping[str, Any] | None = None,
    ) -> float:
        function = getattr(self.base_model, "transport_confidence", None)
        return (
            float(function(claim_id, evidence_role, context=context))
            if callable(function)
            else 1.0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": "affordance_calibrated_reveal",
            "base_model": self.base_model.to_dict(),
            "memory": self.memory.to_dict(),
            "calibration_strength": self.calibration_strength,
            "support_strength": self.support_strength,
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def _load_base(payload: Mapping[str, Any]) -> PriorTableRevealModel:
        model_type = str((payload.get("metadata") or {}).get("model_type", ""))
        if model_type == "counterfactual_evidence_transport":
            from .counterfactual_transport import CounterfactualEvidenceTransportModel

            return CounterfactualEvidenceTransportModel.from_dict(payload)
        if model_type == "hierarchical_equivariant_evidence_transport":
            from .evidence_transport import HierarchicalEvidenceTransportModel

            return HierarchicalEvidenceTransportModel.from_dict(payload)
        return PriorTableRevealModel.from_dict(payload)

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "AffordanceCalibratedRevealModel":
        if payload.get("model_type") != "affordance_calibrated_reveal":
            raise ValueError("Affordance-calibrated reveal schema mismatch")
        return cls(
            base_model=cls._load_base(dict(payload.get("base_model", {}))),
            memory=EvidenceAffordanceMemory.from_dict(
                dict(payload.get("memory", {}))
            ),
            calibration_strength=float(payload.get("calibration_strength", 0.5)),
            support_strength=float(payload.get("support_strength", 4.0)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "AffordanceCalibratedRevealModel":
        import json

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
