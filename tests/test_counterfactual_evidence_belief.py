from inspect_system.active_view.counterfactual_evidence_belief import (
    CounterfactualEvidenceBelief,
    CounterfactualEvidenceBeliefConfig,
)
from inspect_system.types import VerificationResult


def verification(
    observation_id: str,
    *,
    support: float,
    contradiction: float,
    identity: bool,
    relation: bool,
    contradiction_available: bool = False,
    role_scores: dict[str, float] | None = None,
) -> VerificationResult:
    return VerificationResult(
        observation_id=observation_id,
        predicted_state="gear_inserted",
        verified=False,
        confidence=max(support, contradiction),
        evidence_coverage=0.5,
        observed_evidence=[],
        missing_evidence=[],
        contradicted_evidence=[],
        next_step_admissible=False,
        anomaly=False,
        recommended_action="observe",
        metadata={
            "claim_id": "gear_inserted",
            "decision": "insufficient",
            "support_score": support,
            "contradiction_score": contradiction,
            "identity_available": identity,
            "relation_available": relation,
            "contradiction_evidence_available": contradiction_available,
            "role_scores": role_scores or {},
        },
    )


def test_complementary_views_can_resolve_a_claim():
    belief = CounterfactualEvidenceBelief(
        claim_id="gear_inserted",
        config=CounterfactualEvidenceBeliefConfig(conflict_threshold=0.25),
    )
    first = belief.observe(
        verification(
            "view_1",
            support=0.80,
            contradiction=0.20,
            identity=True,
            relation=False,
            role_scores={"identity_disambiguation_view": 0.90},
        )
    )
    second = belief.observe(
        verification(
            "view_2",
            support=0.75,
            contradiction=0.25,
            identity=False,
            relation=True,
            role_scores={"insertion_verification_view": 0.85},
        )
    )

    assert not first.verified
    assert second.verified
    assert second.metadata["decision"] == "supported"
    assert second.metadata["identity_available"]
    assert second.metadata["relation_available"]


def test_conflicting_views_return_to_insufficient():
    belief = CounterfactualEvidenceBelief(claim_id="gear_inserted")
    belief.observe(
        verification(
            "view_1",
            support=0.90,
            contradiction=0.10,
            identity=True,
            relation=True,
        )
    )
    result = belief.observe(
        verification(
            "view_2",
            support=0.10,
            contradiction=0.90,
            identity=False,
            relation=False,
            contradiction_available=True,
        )
    )

    assert not result.verified
    assert result.metadata["decision"] == "insufficient"
    assert result.state_scores["insufficient"] >= 0.20


def test_duplicate_observation_is_not_double_counted():
    belief = CounterfactualEvidenceBelief(claim_id="gear_inserted")
    item = verification(
        "view_1",
        support=0.70,
        contradiction=0.30,
        identity=True,
        relation=False,
    )
    first = belief.observe(item)
    second = belief.observe(item)

    assert first.state_scores == second.state_scores
    assert second.metadata["counterfactual_evidence_belief"]["observations"] == 1


def test_role_evidence_is_monotone_across_views():
    belief = CounterfactualEvidenceBelief(claim_id="cover_seated")
    belief.observe(
        verification(
            "view_1",
            support=0.55,
            contradiction=0.45,
            identity=True,
            relation=False,
            role_scores={"gap_visibility_view": 0.75},
        )
    )
    result = belief.observe(
        verification(
            "view_2",
            support=0.55,
            contradiction=0.45,
            identity=False,
            relation=True,
            role_scores={"gap_visibility_view": 0.30},
        )
    )

    assert result.metadata["role_scores"]["gap_visibility_view"] == 0.75
