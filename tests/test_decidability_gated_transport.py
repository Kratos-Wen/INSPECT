from inspect_system.active_view.decidability_gated_transport import (
    ClaimStructuredEvidenceTransportSelector,
)
from inspect_system.active_view.evidence_state import EvidenceItem, EvidenceState
from inspect_system.active_view.view_lattice import ORBIT_ACTIONS
from inspect_system.learned_view_lattice_policy import LearnedViewLatticePolicy


class RevealModel:
    alpha = 2.0
    metadata = {"uses_robot_view_training": False}

    def __init__(self, action: str | None) -> None:
        self.action = action

    def action_distribution(self, claim_id, evidence_role, context=None):
        del claim_id, evidence_role, context
        if self.action is None:
            return {action: 1.0 / len(ORBIT_ACTIONS) for action in ORBIT_ACTIONS}
        return {action: float(action == self.action) for action in ORBIT_ACTIONS}

    def probability(self, claim_id, evidence_role, action, context=None):
        return self.action_distribution(claim_id, evidence_role, context)[action]

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


def test_directional_transport_is_verified_before_move() -> None:
    result = ClaimStructuredEvidenceTransportSelector(
        model=RevealModel("orbit_right"), lambda_cost=0.0, tau_view=0.0
    ).select(current_view="V1", evidence_state=state())
    assert result.action == "move"
    assert result.selected_view == "V2"
    assert result.reason == "decidability_verified_transport"


def test_uninformative_transport_does_not_force_motion() -> None:
    result = ClaimStructuredEvidenceTransportSelector(
        model=RevealModel(None), lambda_cost=0.0, tau_view=0.0
    ).select(current_view="V1", evidence_state=state())
    assert result.action == "defer"
    assert result.selected_view == "V1"


def test_unified_claim_ledger_does_not_fragment_by_counterfactual_family() -> None:
    policy = LearnedViewLatticePolicy(
        selector=None,
        requirement_counts={},
        requirement_calibration=None,
        unified_claim_evidence_ledger=True,
    )
    first = state()
    first.counterfactual_family = "identity"
    second = state()
    second.counterfactual_family = "spatial_relation"
    assert policy._evidence_ledger_key(first) == policy._evidence_ledger_key(second)


def _multi_clause_state(identity: float, relation: float) -> EvidenceState:
    return EvidenceState(
        claim_id="step2_small_gear_inserted",
        items=[
            EvidenceItem(
                "identity_disambiguation_view",
                score=identity,
                threshold=1.0,
                importance=1.0,
                evidence_role="identity_disambiguation_view",
            ),
            EvidenceItem(
                "insertion_verification_view",
                score=relation,
                threshold=1.0,
                importance=1.0,
                evidence_role="insertion_verification_view",
            ),
        ],
        counterfactual_family="",
    )


def test_evidence_dominance_checkpoint_does_not_trade_away_a_clause() -> None:
    policy = LearnedViewLatticePolicy(
        selector=None,
        requirement_counts={},
        requirement_calibration=None,
        unified_claim_evidence_ledger=True,
        pareto_evidence_checkpoint=True,
    )
    policy._remember_evidence(
        session_id="session",
        view_id="V0",
        state=_multi_clause_state(0.8, 0.8),
    )
    policy._remember_evidence(
        session_id="session",
        view_id="V1",
        state=_multi_clause_state(1.0, 0.7),
    )
    assert policy.best_evidence_view(
        "session", "step2_small_gear_inserted", "identity"
    ) == "V0"

    policy._remember_evidence(
        session_id="session",
        view_id="V2",
        state=_multi_clause_state(0.9, 0.9),
    )
    assert policy.best_evidence_view(
        "session", "step2_small_gear_inserted", "identity"
    ) == "V2"
