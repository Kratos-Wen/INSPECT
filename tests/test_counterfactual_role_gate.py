from __future__ import annotations

from inspect_system.active_view.reveal_model import PriorTableRevealModel


def test_base_reveal_model_applies_counterfactual_role_gate() -> None:
    model = PriorTableRevealModel()

    identity = model.counterfactual_relevance(
        "gear_inserted",
        "identity_disambiguation_view",
        context={"counterfactual_family": "identity"},
    )
    insertion = model.counterfactual_relevance(
        "gear_inserted",
        "insertion_verification_view",
        context={"counterfactual_family": "identity"},
    )
    spatial = model.counterfactual_relevance(
        "gear_inserted",
        "insertion_verification_view",
        context={"counterfactual_family": "spatial_relation"},
    )

    assert identity == 1.0
    assert insertion < identity
    assert spatial == 1.0


def test_claim_disambiguation_role_separates_explicit_counterfactuals() -> None:
    model = PriorTableRevealModel()

    for family in ("identity", "spatial_relation", "seating_contact"):
        assert model.counterfactual_relevance(
            "gear_inserted",
            "claim_disambiguation_view",
            context={"counterfactual_family": family},
        ) > 0.0

def test_zero_mass_scores_fall_back_to_explicit_counterfactual() -> None:
    model = PriorTableRevealModel()
    value = model.counterfactual_relevance(
        "gear_inserted",
        "insertion_verification_view",
        context={
            "counterfactual_family": "identity",
            "counterfactual_scores": {"identity": 0.0, "spatial_relation": 0.0},
        },
    )

    assert value == 0.0


def test_counterfactual_scores_form_a_soft_role_mixture() -> None:
    model = PriorTableRevealModel()
    value = model.counterfactual_relevance(
        "gear_inserted",
        "identity_disambiguation_view",
        context={
            "counterfactual_scores": {
                "identity": 0.75,
                "spatial_relation": 0.25,
            }
        },
    )

    assert value == 0.75
