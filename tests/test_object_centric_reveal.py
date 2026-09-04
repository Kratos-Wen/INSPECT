from __future__ import annotations

import math

from inspect_system.active_view.evidence_affordance_memory import (
    EvidenceAffordanceContext,
    EvidenceAffordanceMemory,
)
from inspect_system.active_view.affordance_calibrated_reveal import (
    AffordanceCalibratedRevealModel,
)
from inspect_system.active_view.evidence_state import EvidenceItem
from inspect_system.active_view.object_centric_evidence_memory import (
    ObjectCentricEvidenceMemory,
    ObjectCentricKernelConfig,
    ObjectCentricMotion,
)
from inspect_system.active_view.object_centric_reveal import (
    ObjectCentricCalibratedRevealModel,
    camera_motion_from_lattice,
)
from inspect_system.active_view.reveal_model import PriorTableRevealModel
from inspect_system.active_view.six_view_projector import score_candidate_view
from inspect_system.active_view.view_lattice import SIX_VIEWS


def build_model() -> ObjectCentricCalibratedRevealModel:
    memory = ObjectCentricEvidenceMemory(
        kernel_config=ObjectCentricKernelConfig(
            geometry_mode="parallax",
            azimuth_bandwidth=0.25,
            kernel_prior_strength=1.0,
        )
    )
    context = EvidenceAffordanceContext(
        action="orbit_right",
        role="slot_relation_view",
        counterfactual="spatial_relation",
        claim="gear_inserted",
    )
    for _ in range(4):
        memory.update(
            context,
            ObjectCentricMotion(
                azimuth=math.radians(45.0),
                elevation=0.0,
                reference_type="scene_transport",
            ),
            0.2,
        )
        memory.update(
            context,
            ObjectCentricMotion(
                azimuth=math.radians(90.0),
                elevation=0.0,
                reference_type="scene_transport",
            ),
            -0.2,
        )
    return ObjectCentricCalibratedRevealModel(
        PriorTableRevealModel(), memory, calibration_strength=1.0
    )


def test_known_lattice_displacement_changes_candidate_factor() -> None:
    model = build_model()
    context = {"counterfactual_family": "spatial_relation"}
    near = model.candidate_affordance_factor(
        "gear_inserted",
        "slot_relation_view",
        "V0",
        "V1",
        SIX_VIEWS,
        context=context,
    )
    far = model.candidate_affordance_factor(
        "gear_inserted",
        "slot_relation_view",
        "V0",
        "V2",
        SIX_VIEWS,
        context=context,
    )

    assert near > far


def test_lattice_camera_motion_is_inverse_pose_without_object_center() -> None:
    motion = camera_motion_from_lattice("V0", "V1", SIX_VIEWS)

    translation_norm = math.sqrt(
        motion.camera_translation_x**2
        + motion.camera_translation_y**2
        + motion.camera_translation_z**2
    )
    assert math.isclose(translation_norm, 1.0)
    assert abs(motion.camera_rotation_y) > 0.0


def test_projector_uses_candidate_factor_without_candidate_images() -> None:
    model = build_model()
    evidence = EvidenceItem(
        name="slot_relation_view",
        evidence_role="slot_relation_view",
        score=0.0,
        threshold=1.0,
        importance=1.0,
    )
    near = score_candidate_view(
        model=model,
        claim_id="gear_inserted",
        missing_evidence=[evidence],
        current_view="V0",
        candidate_view="V1",
        lambda_cost=0.0,
        counterfactual_context={"counterfactual_family": "spatial_relation"},
    )
    far = score_candidate_view(
        model=model,
        claim_id="gear_inserted",
        missing_evidence=[evidence],
        current_view="V0",
        candidate_view="V2",
        lambda_cost=0.0,
        counterfactual_context={"counterfactual_family": "spatial_relation"},
    )

    assert near.evidence_gain > far.evidence_gain


def test_object_centric_policy_roundtrip() -> None:
    model = build_model()
    restored = ObjectCentricCalibratedRevealModel.from_dict(model.to_dict())
    context = {"counterfactual_family": "spatial_relation"}
    expected = model.candidate_affordance_factor(
        "gear_inserted", "slot_relation_view", "V0", "V1", SIX_VIEWS, context
    )
    actual = restored.candidate_affordance_factor(
        "gear_inserted", "slot_relation_view", "V0", "V1", SIX_VIEWS, context
    )

    assert math.isclose(actual, expected)


def test_roundtrip_preserves_affordance_calibrated_base() -> None:
    model = build_model()
    model.base_model = AffordanceCalibratedRevealModel(
        PriorTableRevealModel(),
        EvidenceAffordanceMemory(),
    )
    restored = ObjectCentricCalibratedRevealModel.from_dict(model.to_dict())

    assert isinstance(restored.base_model, AffordanceCalibratedRevealModel)


def test_roundtrip_preserves_counterfactual_family_memories() -> None:
    model = build_model()
    identity_memory = ObjectCentricEvidenceMemory(
        kernel_config=ObjectCentricKernelConfig(
            geometry_mode="camera_pose",
            kernel_prior_strength=1.0,
        )
    )
    model.family_memories = {"identity": identity_memory}

    restored = ObjectCentricCalibratedRevealModel.from_dict(model.to_dict())

    assert set(restored.family_memories) == {"identity"}
    assert (
        restored.family_memories["identity"].kernel_config.geometry_mode
        == "camera_pose"
    )


def test_counterfactual_family_selects_its_own_observation_basis() -> None:
    model = build_model()
    identity_memory = ObjectCentricEvidenceMemory(
        kernel_config=ObjectCentricKernelConfig(
            geometry_mode="camera_pose",
            rotation_bandwidth=0.25,
            radius_bandwidth=0.35,
            kernel_prior_strength=1.0,
        )
    )
    context = EvidenceAffordanceContext(
        action="orbit_right",
        role="identity_disambiguation_view",
        counterfactual="identity",
        claim="gear_identity_safe",
    )
    identity_memory.update(
        context,
        camera_motion_from_lattice("V0", "V1", SIX_VIEWS),
        0.5,
        weight=8.0,
    )
    identity_memory.update(
        context,
        camera_motion_from_lattice("V0", "V5", SIX_VIEWS),
        -0.5,
        weight=8.0,
    )
    model.family_memories = {"identity": identity_memory}

    identity_factor = model.candidate_affordance_factor(
        "gear_identity_safe",
        "identity_disambiguation_view",
        "V0",
        "V1",
        SIX_VIEWS,
        context={"counterfactual_family": "identity"},
    )
    generic_factor = model.candidate_affordance_factor(
        "gear_identity_safe",
        "identity_disambiguation_view",
        "V0",
        "V1",
        SIX_VIEWS,
        context={"counterfactual_family": "generic"},
    )

    assert identity_factor > generic_factor


def test_geometry_can_be_restricted_to_top_coarse_action_family() -> None:
    model = build_model()
    model.geometry_scope = "top_action_family"
    model.base_model = PriorTableRevealModel(
        counts={
            "gear_inserted|identity_disambiguation_view|orbit_left_down": {
                "pos": 8.0,
                "total": 8.0,
            },
            "gear_inserted|identity_disambiguation_view|orbit_right_down": {
                "pos": 8.0,
                "total": 8.0,
            },
            "gear_inserted|identity_disambiguation_view|orbit_left": {
                "pos": 0.0,
                "total": 8.0,
            },
            "gear_inserted|identity_disambiguation_view|orbit_right": {
                "pos": 0.0,
                "total": 8.0,
            },
        }
    )

    out_of_family = model.candidate_affordance_factor(
        "gear_inserted",
        "identity_disambiguation_view",
        "V0",
        "V1",
        SIX_VIEWS,
        context={"counterfactual_family": "identity"},
    )

    assert math.isclose(out_of_family, 1.0)
