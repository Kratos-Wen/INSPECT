"""Monotone claim-evidence transport for fixed-lattice inspection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from .active_selector import SelectionResult
from .decidability_transport import StructuredDecidabilitySelector, claim_decidability
from .evidence_state import EvidenceState, merge_missing_names
from .six_view_projector import rank_candidate_views
from .view_lattice import SIX_VIEWS, ViewNode


@dataclass
class MonotoneClaimEvidenceTransportSelector:
    """Maximize assistant-derived reveal under an evidence-retention constraint.

    A move is admissible only when every active claim-evidence clause is
    projected to be non-decreasing and at least one clause strictly improves.
    Motion cost remains in the reveal objective, but cannot turn a genuine
    evidence improvement into a verifier-side rejection.
    """

    model: Any
    views: Mapping[str, ViewNode] = field(default_factory=lambda: dict(SIX_VIEWS))
    lambda_cost: float = 0.05
    tau_view: float = 0.02
    geometry_loss_cap: float = 1.0
    numerical_tolerance: float = 1e-9

    def select(
        self,
        *,
        current_view: str,
        evidence_state: EvidenceState,
        visited_views: List[str] | None = None,
        candidate_context: Mapping[str, Any] | None = None,
    ) -> SelectionResult:
        missing = evidence_state.missing()
        if not missing:
            return SelectionResult(
                action="defer",
                current_view=current_view,
                selected_view=current_view,
                reason="no_structured_missing_evidence",
            )
        context = {
            **dict(candidate_context or {}),
            "counterfactual_family": evidence_state.counterfactual_family,
            "counterfactual_scores": dict(evidence_state.counterfactual_scores),
            "counterfactual_margin": float(evidence_state.margin),
        }
        if (
            "preservation_role_factors" not in context
            and "candidate_role_factors" in context
        ):
            context["preservation_role_factors"] = dict(
                context.get("candidate_role_factors") or {}
            )
        proposal_context = {**context, "candidate_role_factors": {}}
        proposals = rank_candidate_views(
            model=self.model,
            claim_id=evidence_state.claim_id,
            missing_evidence=missing,
            current_view=current_view,
            views=self.views,
            lambda_cost=self.lambda_cost,
            visited_views=visited_views,
            counterfactual_context=proposal_context,
        )
        visited = set(visited_views or [])
        proposals = [proposal for proposal in proposals if proposal.view_id not in visited]

        projector = StructuredDecidabilitySelector(
            model=self.model,
            views=self.views,
            lambda_cost=self.lambda_cost,
            tau_view=self.tau_view,
            geometry_loss_cap=self.geometry_loss_cap,
        )
        current_values = {
            (item.evidence_role or item.name): min(
                1.0,
                max(0.0, float(item.score) / max(1e-9, float(item.threshold))),
            )
            for item in evidence_state.items
        }
        current_decidability, current_clauses = claim_decidability(
            evidence_state, current_values
        )
        projected = projector._candidate_role_values(
            evidence_state, current_view, context
        )

        ranked: List[Dict[str, Any]] = []
        for proposal in proposals:
            role_values = projected.get(proposal.view_id, {})
            decidability, clauses = claim_decidability(evidence_state, role_values)
            clause_deltas = {
                name: float(clauses.get(name, 0.0))
                - float(current_clauses.get(name, 0.0))
                for name in current_clauses
            }
            non_decreasing = all(
                value >= -self.numerical_tolerance
                for value in clause_deltas.values()
            )
            strict_improvement = any(
                value > self.numerical_tolerance
                for value in clause_deltas.values()
            )
            move_verified = bool(
                proposal.score >= self.tau_view
                and non_decreasing
                and strict_improvement
            )
            ranked.append(
                {
                    "view_id": proposal.view_id,
                    "score": float(proposal.score),
                    "evidence_gain": float(proposal.evidence_gain),
                    "cost": float(proposal.cost),
                    "current_decidability": current_decidability,
                    "projected_decidability": decidability,
                    "decidability_gain": decidability - current_decidability,
                    "current_clauses": current_clauses,
                    "projected_clauses": clauses,
                    "clause_deltas": clause_deltas,
                    "evidence_retained": non_decreasing,
                    "evidence_improved": strict_improvement,
                    "move_verified": move_verified,
                    "projected_role_values": role_values,
                    "action_contributions": dict(proposal.action_contributions),
                }
            )

        selected = next((item for item in ranked if item["move_verified"]), None)
        if selected is None:
            return SelectionResult(
                action="defer",
                current_view=current_view,
                selected_view=current_view,
                score=float(ranked[0]["score"]) if ranked else 0.0,
                reason="no_monotone_evidence_improvement",
                missing_evidence=merge_missing_names(missing),
                ranked_views=ranked,
            )
        return SelectionResult(
            action="move",
            current_view=current_view,
            selected_view=str(selected["view_id"]),
            score=float(selected["score"]),
            reason="monotone_claim_evidence_transport",
            missing_evidence=merge_missing_names(missing),
            ranked_views=ranked,
        )
