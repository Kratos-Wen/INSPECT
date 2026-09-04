import pytest
import numpy as np

from inspect_system.active_view.ray_visibility import (
    COVER_CLASSES,
    GEAR_CLASSES,
    HOUSING_CLASSES,
    ROLE_SURFACE_CLASSES,
    candidate_role_factors,
    merge_role_fallback_detections,
    relation_frame_camera,
    role_conditioned_candidate_factors,
)
from inspect_system.active_view.six_view_projector import (
    relative_role_affordance_factor,
)
from inspect_system.active_view.view_lattice import ViewNode


VIEWS = {
    "side": ViewNode("side", yaw=0.0, elevation=0.0),
    "oblique": ViewNode("oblique", yaw=0.0, elevation=30.0),
    "top": ViewNode("top", yaw=0.0, elevation=90.0),
}


def test_missing_surface_normal_is_neutral() -> None:
    factors = candidate_role_factors(
        normal_camera=None,
        current_view="side",
        views=VIEWS,
    )

    assert all(values["gap_visibility_view"] == pytest.approx(1.0) for values in factors.values())


def test_roles_follow_current_local_surface_not_workspace_up() -> None:
    factors = candidate_role_factors(
        normal_camera=np.asarray([0.0, 0.0, 1.0]),
        current_view="side",
        views=VIEWS,
    )

    assert factors["side"]["identity_disambiguation_view"] > factors["oblique"]["identity_disambiguation_view"]
    assert factors["oblique"]["identity_disambiguation_view"] > factors["top"]["identity_disambiguation_view"]
    assert factors["side"]["gap_visibility_view"] < factors["oblique"]["gap_visibility_view"]
    assert factors["oblique"]["gap_visibility_view"] < factors["top"]["gap_visibility_view"]


def test_low_reliability_shrinks_geometry_toward_neutral() -> None:
    full = candidate_role_factors(
        normal_camera=np.asarray([0.0, 0.0, 1.0]),
        current_view="side",
        views=VIEWS,
        reliability=1.0,
    )
    weak = candidate_role_factors(
        normal_camera=np.asarray([0.0, 0.0, 1.0]),
        current_view="side",
        views=VIEWS,
        reliability=0.2,
    )
    neutral = candidate_role_factors(
        normal_camera=np.asarray([0.0, 0.0, 1.0]),
        current_view="side",
        views=VIEWS,
        reliability=0.0,
    )

    role = "identity_disambiguation_view"
    assert neutral["side"][role] == pytest.approx(1.0)
    assert abs(weak["side"][role] - 1.0) < abs(full["side"][role] - 1.0)


@pytest.mark.parametrize(
    "role",
    [
        "identity_disambiguation_view",
        "insertion_verification_view",
        "containment_verification_view",
        "slot_relation_view",
        "claim_disambiguation_view",
    ],
)
def test_non_grazing_roles_are_not_overridden(role: str) -> None:
    factors = candidate_role_factors(
        normal_camera=None,
        current_view="side",
        views=VIEWS,
    )

    assert all(values[role] == pytest.approx(1.0) for values in factors.values())


def test_roles_use_independent_local_surface_normals() -> None:
    factors = role_conditioned_candidate_factors(
        normals_camera={
            "identity_disambiguation_view": np.asarray([0.0, 0.0, 1.0]),
            "gap_visibility_view": None,
        },
        current_view="side",
        views=VIEWS,
    )

    assert factors["side"]["identity_disambiguation_view"] > factors["top"]["identity_disambiguation_view"]
    assert all(values["gap_visibility_view"] == pytest.approx(1.0) for values in factors.values())


def test_role_surfaces_follow_claim_topology() -> None:
    assert ROLE_SURFACE_CLASSES["identity_disambiguation_view"] == GEAR_CLASSES
    assert ROLE_SURFACE_CLASSES["slot_relation_view"] == GEAR_CLASSES | HOUSING_CLASSES
    assert ROLE_SURFACE_CLASSES["gap_visibility_view"] == COVER_CLASSES | HOUSING_CLASSES


def test_relation_frame_uses_current_predicted_part_and_housing() -> None:
    y, x = np.mgrid[-1.0:1.0:60j, -1.0:1.0:60j]
    points = np.stack([x, y, np.full_like(x, 2.0)], axis=-1)
    detections = [
        {"name": "type_8_gear", "confidence": 0.9, "xyxy": [34, 18, 56, 42]},
        {
            "name": "type_5_gearbox_housing",
            "confidence": 0.9,
            "xyxy": [4, 18, 28, 42],
        },
    ]

    frame = relation_frame_camera(
        points,
        np.ones((60, 60), dtype=bool),
        detections,
        target_classes=GEAR_CLASSES,
        reference_classes=HOUSING_CLASSES,
        reference_normal=np.asarray([0.0, 0.0, -1.0]),
    )

    assert frame is not None
    assert frame["axis_camera"][0] > 0.95
    assert frame["quality"] > 0.95


def test_relation_frame_accepts_proposal_only_role_labels() -> None:
    y, x = np.mgrid[-1.0:1.0:60j, -1.0:1.0:60j]
    points = np.stack([x, y, np.full_like(x, 2.0)], axis=-1)
    detections = [
        {"name": "big_gear", "confidence": 0.9, "xyxy": [34, 18, 56, 42]},
        {"name": "housing", "confidence": 0.9, "xyxy": [4, 18, 28, 42]},
    ]

    frame = relation_frame_camera(
        points,
        np.ones((60, 60), dtype=bool),
        detections,
        target_classes=GEAR_CLASSES,
        reference_classes=HOUSING_CLASSES,
        reference_normal=np.asarray([0.0, 0.0, -1.0]),
    )

    assert frame is not None
    assert frame["axis_camera"][0] > 0.95

def test_coarse_roles_only_fill_missing_fine_roles() -> None:
    detections = merge_role_fallback_detections(
        [{"name": "type_7_gear", "confidence": 0.8}],
        [
            {"name": "small_gear", "confidence": 0.9},
            {"name": "housing", "confidence": 0.7},
        ],
    )

    assert [item["name"] for item in detections] == ["type_7_gear", "housing"]

def test_relative_role_affordance_keeps_stay_neutral() -> None:
    factors = {
        "V0": {"slot_relation_view": 0.4},
        "V1": {"slot_relation_view": 0.8},
    }

    value = relative_role_affordance_factor(
        candidate_role_factors=factors,
        current_view="V0",
        candidate_view="V0",
        role="slot_relation_view",
    )

    assert value == pytest.approx(1.0)


def test_relative_role_affordance_rewards_relative_reveal() -> None:
    factors = {
        "V0": {"slot_relation_view": 0.4},
        "V1": {"slot_relation_view": 0.8},
    }

    value = relative_role_affordance_factor(
        candidate_role_factors=factors,
        current_view="V0",
        candidate_view="V1",
        role="slot_relation_view",
    )

    assert value == pytest.approx(2.0)
