from scripts.external_benchmarks.evaluate_impact_reveal_policy import (
    evidence_roles,
    evidence_state_from_row,
    semantic_claim_id,
)


def test_impact_metadata_maps_to_shared_semantics() -> None:
    row = {
        "claim_id": "bearing_screw_lowright.installed_correctly",
        "evidence_roles": "identity;alignment;containment;boundary_visibility",
        "decision_confidence": 0.2,
    }

    assert semantic_claim_id(row) == "component_installed_correctly"
    assert evidence_roles(row) == [
        "identity_disambiguation_view",
        "slot_relation_view",
        "insertion_verification_view",
        "gap_visibility_view",
    ]
    state = evidence_state_from_row(row, cutoff=0.4, semantics="metadata")
    assert state.claim_id == "component_installed_correctly"
    assert [item.evidence_role for item in state.items] == evidence_roles(row)
    assert sum(item.importance for item in state.items) == 1.0


def test_impact_generic_protocol_remains_reproducible() -> None:
    state = evidence_state_from_row(
        {"decision_confidence": 0.2},
        cutoff=0.4,
        semantics="generic",
    )

    assert state.claim_id == "state_validity"
    assert [item.evidence_role for item in state.items] == [
        "claim_disambiguation_view"
    ]

def test_zero_mass_counterfactual_scores_do_not_create_a_hard_family() -> None:
    state = evidence_state_from_row(
        {
            "claim_id": "lever.installed_correctly",
            "evidence_roles": "alignment",
            "decision_confidence": 0.2,
        },
        cutoff=0.4,
        semantics="metadata",
    )

    assert state.counterfactual_family == ""
    assert state.counterfactual_scores == {
        "spatial_relation": 0.0,
        "state_absence": 0.0,
    }
