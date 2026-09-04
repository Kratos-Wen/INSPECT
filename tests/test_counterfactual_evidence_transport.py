from __future__ import annotations

import math

from inspect_system.active_view.counterfactual_transport import (
    CounterfactualEvidenceTransportModel,
    counterfactual_mixture,
    counterfactual_role_relevance,
    infer_counterfactual_family,
    resolution_decidability_gain,
)
from inspect_system.active_view.evidence_state import EvidenceState
from inspect_system.active_view.reveal_model import PriorTableRevealModel
from inspect_system.active_view.trace_event_miner import MinedEvent


def event(event_id: str, action: str, gain: float) -> MinedEvent:
    return MinedEvent(
        event_id=event_id,
        video=f"{event_id}.mp4",
        claim_id="gear_inserted",
        evidence_role="identity_disambiguation_view",
        relative_action=action,
        label=1,
        transferable=True,
        transfer_weight=1.0,
        label_confidence=1.0,
        evidence_stability=1.0,
        evidence_importance=1.0,
        metadata={
            "score_gain": 0.8,
            "counterfactual_family": "wrong_identity",
            "signed_counterfactual_margin_gain": gain,
        },
    )


def test_counterfactual_taxonomy_uses_claim_and_role() -> None:
    assert infer_counterfactual_family("gear_inserted", "identity_disambiguation_view") == "identity"
    assert infer_counterfactual_family("gear_inserted", "slot_relation_view") == "spatial_relation"
    assert infer_counterfactual_family("cover_seated", "gap_visibility_view") == "seating_contact"


def test_projected_role_overrides_generic_source_role() -> None:
    metadata = {"source_evidence_role": "claim_disambiguation_view"}
    assert (
        infer_counterfactual_family(
            "gear_inserted",
            "identity_disambiguation_view",
            metadata,
        )
        == "identity"
    )
    assert (
        infer_counterfactual_family(
            "gear_inserted",
            "slot_relation_view",
            metadata,
        )
        == "spatial_relation"
    )


def test_failed_counterfactual_separation_penalizes_action_family() -> None:
    model = CounterfactualEvidenceTransportModel(counterfactual_strength=0.5).fit_events(
        [
            event("success", "orbit_up", 0.8),
            event("failure", "orbit_left", -0.8),
        ]
    )
    context = {"counterfactual_family": "identity"}
    distribution = model.action_distribution(
        "gear_inserted",
        "identity_disambiguation_view",
        context=context,
    )
    assert math.isclose(sum(distribution.values()), 1.0)
    assert distribution["orbit_up"] > distribution["orbit_left"]


def test_counterfactual_mixture_preserves_uncertainty_mass() -> None:
    mixture = counterfactual_mixture(
        {"counterfactual_scores": {"wrong_identity": 0.3, "not_installed": 0.2}},
        "generic",
    )
    assert math.isclose(sum(mixture.values()), 1.0)
    assert math.isclose(mixture["identity"], 0.3)
    assert math.isclose(mixture["state_absence"], 0.2)
    assert math.isclose(mixture["generic"], 0.5)


def test_counterfactual_role_relevance_is_discriminative() -> None:
    assert counterfactual_role_relevance("wrong_identity", "identity_disambiguation_view") == 1.0
    assert counterfactual_role_relevance("wrong_identity", "slot_relation_view") < 1.0
    assert counterfactual_role_relevance("not_seated", "gap_visibility_view") == 1.0


def test_counterfactual_model_round_trip(tmp_path) -> None:
    model = CounterfactualEvidenceTransportModel().fit_events(
        [event("success", "orbit_right_up", 0.4)]
    )
    path = tmp_path / "counterfactual_transport.json"
    model.save(path)
    loaded = PriorTableRevealModel.load(path)
    assert isinstance(loaded, CounterfactualEvidenceTransportModel)
    assert loaded.counterfactual_success_counts == model.counterfactual_success_counts


def test_contradicted_resolution_is_positive_decidability_gain() -> None:
    item = event("contradicted", "orbit_left", -0.4)
    item.metadata = {
        "before_score": 0.2,
        "after_score": 0.8,
        "score_gain": 0.6,
        "world_outcome": "contradicted",
    }
    assert resolution_decidability_gain(item) == 0.6


def test_evidence_state_counterfactual_round_trip() -> None:
    state = EvidenceState(
        claim_id="gear_inserted",
        margin=-0.2,
        counterfactual_family="identity",
        counterfactual_scores={"identity": 0.7, "spatial_relation": 0.2},
    )
    restored = EvidenceState.from_dict(state.to_dict())
    assert restored.counterfactual_family == "identity"
    assert restored.counterfactual_scores == state.counterfactual_scores
