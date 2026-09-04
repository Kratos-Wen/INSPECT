from __future__ import annotations

from inspect_system.active_view.evidence_affordance_memory import (
    EvidenceAffordanceContext,
)
from inspect_system.active_view.object_centric_evidence_memory import (
    ObjectCentricEvidenceMemory,
    ObjectCentricKernelConfig,
    ObjectCentricMotion,
)


def context() -> EvidenceAffordanceContext:
    return EvidenceAffordanceContext(
        action="orbit",
        role="slot_relation",
        counterfactual="spatial_relation",
        claim="gear_inserted",
    )


def motion(azimuth: float, elevation: float = 0.0) -> ObjectCentricMotion:
    return ObjectCentricMotion(
        azimuth=azimuth,
        elevation=elevation,
        quality=1.0,
        reference_type="scene_transport",
    )


def test_axis_invariant_field_does_not_assume_shared_left_right_orientation() -> None:
    invariant = ObjectCentricEvidenceMemory(
        kernel_config=ObjectCentricKernelConfig(geometry_mode="axis_invariant")
    )
    directional = ObjectCentricEvidenceMemory(
        kernel_config=ObjectCentricKernelConfig(geometry_mode="direction")
    )

    assert invariant._distance_square(motion(0.4), motion(-0.4)) == 0.0
    assert directional._distance_square(motion(0.4), motion(-0.4)) > 0.0


def test_verified_session_outcome_updates_prediction_after_not_before() -> None:
    memory = ObjectCentricEvidenceMemory(
        kernel_config=ObjectCentricKernelConfig(
            geometry_mode="parallax",
            kernel_prior_strength=1.0,
            session_boost=12.0,
        )
    )
    memory.fit([(context(), motion(0.25), 0.2, 1.0)])
    before = memory.predict(context(), motion(0.22), session_id="held_out_day")

    memory.update(
        context(),
        motion(0.22),
        -0.3,
        session_id="held_out_day",
        update_shared=True,
    )
    after = memory.predict(context(), motion(0.22), session_id="held_out_day")

    assert after.helpful_probability < before.helpful_probability
    assert after.expected_gain < before.expected_gain


def test_nearby_verified_changes_merge_into_a_bounded_atom() -> None:
    memory = ObjectCentricEvidenceMemory(
        kernel_config=ObjectCentricKernelConfig(
            geometry_mode="parallax",
            merge_radius=1.0,
            max_atoms=2,
        )
    )
    memory.update(context(), motion(0.20), 0.1)
    memory.update(context(), motion(0.22), 0.2)

    assert len(memory.atoms) == 1
    assert memory.atoms[0].outcome.update_count == 2


def test_relation_surface_motion_preserves_signed_relation_axis() -> None:
    memory = ObjectCentricEvidenceMemory(
        kernel_config=ObjectCentricKernelConfig(
            geometry_mode="relation_surface_motion"
        )
    )
    right = ObjectCentricMotion(
        azimuth=0.0,
        elevation=0.0,
        relation_axis_translation=0.8,
        relation_frame_quality=1.0,
    )
    left = ObjectCentricMotion(
        azimuth=0.0,
        elevation=0.0,
        relation_axis_translation=-0.8,
        relation_frame_quality=1.0,
    )

    assert memory._distance_square(right, left) > 0.0
