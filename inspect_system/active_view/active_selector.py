"""Runtime selector for INSPECT-Active view transfer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping

from .evidence_state import EvidenceState, merge_missing_names
from .reveal_model import PriorTableRevealModel
from .six_view_projector import rank_candidate_views
from .view_lattice import SIX_VIEWS, ViewNode


@dataclass
class SelectionResult:
    action: str
    current_view: str
    selected_view: str = ""
    score: float = 0.0
    reason: str = ""
    missing_evidence: List[str] = field(default_factory=list)
    ranked_views: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActiveSelector:
    model: PriorTableRevealModel
    views: Mapping[str, ViewNode] = field(default_factory=lambda: dict(SIX_VIEWS))
    lambda_cost: float = 0.15
    lambda_risk: float = 0.0
    lambda_revisit: float = 0.0
    tau_view: float = 0.02
    certainty_threshold: float = 0.85
    tau_decide: float = 0.85
    tau_missing: float = 0.25
    tau_margin: float = 0.05
    tau_conflict: float = 0.50
    view_risk: Mapping[str, float] = field(default_factory=dict)

    @staticmethod
    def _missing_weight(evidence_state: EvidenceState) -> float:
        total = 0.0
        weight = 0.0
        for item in evidence_state.items:
            importance = max(0.0, float(item.importance))
            total += importance
            if item.threshold <= 1e-9:
                missing = 0.0
            else:
                missing = max(0.0, min(1.0, 1.0 - float(item.score) / float(item.threshold)))
            weight += importance * missing
        return weight / max(1e-9, total)

    def _stay_reason(self, evidence_state: EvidenceState) -> str:
        if evidence_state.current_utility_proxy is not None:
            return "current_evidence_sufficient" if float(evidence_state.current_utility_proxy) >= 2.0 else ""
        missing_weight = self._missing_weight(evidence_state)
        if (
            evidence_state.claim_score >= self.tau_decide
            and evidence_state.contradiction_score <= self.tau_conflict
            and evidence_state.margin >= self.tau_margin
            and missing_weight <= self.tau_missing
        ):
            return "supported_with_sufficient_margin"
        if evidence_state.contradiction_score >= self.tau_decide and missing_weight <= self.tau_missing:
            return "contradicted_with_sufficient_margin"
        return ""

    def select(
        self,
        *,
        current_view: str,
        evidence_state: EvidenceState,
        visited_views: List[str] | None = None,
        candidate_context: Mapping[str, Any] | None = None,
    ) -> SelectionResult:
        stay_reason = self._stay_reason(evidence_state)
        if stay_reason:
            return SelectionResult(
                action="stay",
                current_view=current_view,
                selected_view=current_view,
                score=0.0,
                reason=stay_reason,
                missing_evidence=[],
            )
        missing = evidence_state.missing()
        if not missing:
            return SelectionResult(
                action="defer",
                current_view=current_view,
                reason="no_structured_missing_evidence",
            )
        counterfactual_family = evidence_state.counterfactual_family
        if not counterfactual_family and evidence_state.counterfactual_scores:
            positive_scores = {
                key: float(value)
                for key, value in evidence_state.counterfactual_scores.items()
                if float(value) > 1e-12
            }
            if positive_scores:
                counterfactual_family = max(
                    positive_scores,
                    key=lambda key: (positive_scores[key], key),
                )
        ranked = rank_candidate_views(
            model=self.model,
            claim_id=evidence_state.claim_id,
            missing_evidence=missing,
            current_view=current_view,
            views=self.views,
            lambda_cost=self.lambda_cost,
            lambda_risk=self.lambda_risk,
            lambda_revisit=self.lambda_revisit,
            view_risk=self.view_risk,
            visited_views=visited_views,
            counterfactual_context={
                **dict(candidate_context or {}),
                "counterfactual_family": counterfactual_family,
                "counterfactual_scores": dict(evidence_state.counterfactual_scores),
                "counterfactual_margin": float(evidence_state.margin),
            },
        )
        visited = set(visited_views or [])
        ranked = [item for item in ranked if item.view_id not in visited]
        ranked_dicts = [
            {
                "view_id": item.view_id,
                "score": item.score,
                "evidence_gain": item.evidence_gain,
                "cost": item.cost,
                "risk_penalty": item.risk_penalty,
                "revisit_penalty": item.revisit_penalty,
                "action_contributions": item.action_contributions,
            }
            for item in ranked
        ]
        if not ranked or ranked[0].score < self.tau_view:
            return SelectionResult(
                action="defer",
                current_view=current_view,
                score=ranked[0].score if ranked else 0.0,
                reason="no_informative_reachable_view",
                missing_evidence=merge_missing_names(missing),
                ranked_views=ranked_dicts,
            )
        return SelectionResult(
            action="move",
            current_view=current_view,
            selected_view=ranked[0].view_id,
            score=ranked[0].score,
            reason="projected_relative_action",
            missing_evidence=merge_missing_names(missing),
            ranked_views=ranked_dicts,
        )
