"""Claim-structured evidence transport for fixed-lattice inspection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from .active_selector import SelectionResult
from .decidability_transport import (
    StructuredDecidabilitySelector,
    claim_decidability,
)
from .evidence_state import EvidenceState, merge_missing_names
from .six_view_projector import rank_candidate_views
from .view_lattice import SIX_VIEWS, ViewNode, transition_cost


@dataclass
class ClaimStructuredEvidenceTransportSelector:
    """Let evidence transport propose and claim clauses verify a view move.

    The proposal stage preserves the relative reveal behavior learned from
    assistant use. The verification stage compares every proposal with STAY
    under the active claim's evidence clauses and rejects moves that do not
    have positive conservative decidability gain.
    """

    model: Any
    views: Mapping[str, ViewNode] = field(default_factory=lambda: dict(SIX_VIEWS))
    lambda_cost: float = 0.05
    tau_view: float = 0.02
    geometry_loss_cap: float = 1.0
    use_projected_role_factors_for_proposal: bool = False
    use_counterfactual_relevance_for_proposal: bool = True

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
        # Candidate geometry is never an input to the transport proposal. A
        # current-view geometry estimate may only test whether a proposed move
        # would preserve evidence that is already visible.
        if (
            "preservation_role_factors" not in context
            and "candidate_role_factors" in context
        ):
            context["preservation_role_factors"] = dict(
                context.get("candidate_role_factors") or {}
            )
        proposal_context = dict(context)
        if not self.use_counterfactual_relevance_for_proposal:
            for key in (
                "counterfactual_family",
                "counterfactual_scores",
                "counterfactual_margin",
            ):
                proposal_context.pop(key, None)
        if not self.use_projected_role_factors_for_proposal:
            proposal_context["candidate_role_factors"] = {}
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

        verifier = StructuredDecidabilitySelector(
            model=self.model,
            views=self.views,
            lambda_cost=self.lambda_cost,
            tau_view=self.tau_view,
            geometry_loss_cap=self.geometry_loss_cap,
        )
        current_values = {
            (item.evidence_role or item.name): min(
                1.0, max(0.0, float(item.score) / max(1e-9, float(item.threshold)))
            )
            for item in evidence_state.items
        }
        current_decidability, current_clauses = claim_decidability(
            evidence_state, current_values
        )
        projected = verifier._candidate_role_values(
            evidence_state, current_view, context
        )
        proposal_by_view = {proposal.view_id: proposal for proposal in proposals}
        ranked: List[Dict[str, Any]] = []
        for proposal in proposals:
            role_values = projected.get(proposal.view_id, {})
            decidability, clauses = claim_decidability(evidence_state, role_values)
            cost = transition_cost(current_view, proposal.view_id, self.views)
            gate_gain = decidability - current_decidability
            gate_score = gate_gain - self.lambda_cost * cost
            ranked.append(
                {
                    "view_id": proposal.view_id,
                    "score": float(proposal.score),
                    "evidence_gain": float(proposal.evidence_gain),
                    "cost": float(proposal.cost),
                    "transport_score": float(proposal.score),
                    "current_decidability": current_decidability,
                    "projected_decidability": decidability,
                    "decidability_gain": gate_gain,
                    "decidability_gate_score": gate_score,
                    "current_clauses": current_clauses,
                    "projected_clauses": clauses,
                    "current_bottleneck": min(current_clauses.values(), default=0.0),
                    "projected_bottleneck": min(clauses.values(), default=0.0),
                    "projected_role_values": role_values,
                    "move_verified": bool(
                        proposal.score >= self.tau_view
                        and gate_score > self.tau_view
                    ),
                    "action_contributions": dict(proposal.action_contributions),
                }
            )
        selected = next(
            (item for item in ranked if bool(item["move_verified"])), None
        )
        if selected is None:
            best_score = (
                max(
                    (
                        float(item["decidability_gate_score"])
                        for item in ranked
                    ),
                    default=0.0,
                )
            )
            return SelectionResult(
                action="defer",
                current_view=current_view,
                selected_view=current_view,
                score=best_score,
                reason="transport_proposal_not_decidability_improving",
                missing_evidence=merge_missing_names(missing),
                ranked_views=ranked,
            )
        selected_view = str(selected["view_id"])
        if selected_view not in proposal_by_view:
            raise RuntimeError("Verified view was not produced by transport.")
        return SelectionResult(
            action="move",
            current_view=current_view,
            selected_view=selected_view,
            score=float(selected["decidability_gate_score"]),
            reason="decidability_verified_transport",
            missing_evidence=merge_missing_names(missing),
            ranked_views=ranked,
        )
# Backward-compatible name for frozen evaluation scripts.
DecidabilityGatedTransportSelector = ClaimStructuredEvidenceTransportSelector
