from inspect_system.active_view.session_ray_field import SessionRayEvidenceField
from inspect_system.active_view.view_lattice import SIX_VIEWS


def test_positive_destination_observation_updates_only_matching_session() -> None:
    field = SessionRayEvidenceField()
    assert field.update(
        session_id="assembly_01",
        claim_id="gear_inserted",
        evidence_role="identity_disambiguation_view",
        counterfactual_family="identity",
        view_id="V1",
        views=SIX_VIEWS,
        signed_gain=1.0,
        weight=1.0,
    )

    matching = field.factor(
        session_id="assembly_01",
        claim_id="gear_inserted",
        evidence_role="identity_disambiguation_view",
        counterfactual_weights={"identity": 1.0},
        candidate_view="V2",
        views=SIX_VIEWS,
    )
    unrelated = field.factor(
        session_id="assembly_02",
        claim_id="gear_inserted",
        evidence_role="identity_disambiguation_view",
        counterfactual_weights={"identity": 1.0},
        candidate_view="V2",
        views=SIX_VIEWS,
    )

    assert matching > 1.0
    assert unrelated == 1.0


def test_negative_destination_observation_suppresses_nearby_rays() -> None:
    field = SessionRayEvidenceField()
    field.update(
        session_id="assembly_01",
        claim_id="cover_seated",
        evidence_role="gap_visibility_view",
        counterfactual_family="seating_contact",
        view_id="V4",
        views=SIX_VIEWS,
        signed_gain=-1.0,
        weight=1.0,
    )

    factor = field.factor(
        session_id="assembly_01",
        claim_id="cover_seated",
        evidence_role="gap_visibility_view",
        counterfactual_weights={"seating_contact": 1.0},
        candidate_view="V5",
        views=SIX_VIEWS,
    )

    assert factor < 1.0
