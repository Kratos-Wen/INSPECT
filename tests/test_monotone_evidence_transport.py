from inspect_system.active_view.evidence_state import EvidenceItem, EvidenceState
from inspect_system.active_view.monotone_evidence_transport import (
    MonotoneClaimEvidenceTransportSelector,
)
from inspect_system.active_view.view_lattice import ORBIT_ACTIONS


class RevealModel:
    metadata = {"uses_robot_view_training": False}

    def __init__(self, action: str | None) -> None:
        self.action = action

    def action_distribution(self, claim_id, evidence_role, context=None):
        del claim_id, evidence_role, context
        if self.action is None:
            return {action: 1.0 / len(ORBIT_ACTIONS) for action in ORBIT_ACTIONS}
        return {action: float(action == self.action) for action in ORBIT_ACTIONS}

    def transport_confidence(self, claim_id, evidence_role, context=None):
        del claim_id, evidence_role, context
        return 1.0

    def counterfactual_relevance(self, claim_id, evidence_role, context=None):
        del claim_id, evidence_role, context
        return 1.0

    def candidate_affordance_factor(self, *args, **kwargs):
        del args, kwargs
        return 1.0


def state() -> EvidenceState:
    return EvidenceState(
        claim_id="step2_small_gear_inserted",
        items=[
            EvidenceItem(
                "identity_disambiguation_view",
                score=0.0,
                threshold=1.0,
                importance=1.0,
                evidence_role="identity_disambiguation_view",
            )
        ],
        counterfactual_family="identity",
    )


def test_directional_transport_improves_without_discarding_evidence() -> None:
    result = MonotoneClaimEvidenceTransportSelector(
        model=RevealModel("orbit_right"), lambda_cost=0.0, tau_view=0.0
    ).select(current_view="V1", evidence_state=state())
    assert result.action == "move"
    assert result.selected_view == "V2"
    assert result.ranked_views[0]["evidence_retained"]
    assert result.ranked_views[0]["evidence_improved"]


def test_uniform_transport_without_clause_gain_defers() -> None:
    result = MonotoneClaimEvidenceTransportSelector(
        model=RevealModel(None), lambda_cost=0.0, tau_view=0.0
    ).select(current_view="V1", evidence_state=state())
    assert result.action == "defer"
    assert result.selected_view == "V1"
