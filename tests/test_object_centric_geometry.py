from __future__ import annotations

import math

import cv2
import numpy as np

from inspect_system.active_view.object_centric_geometry import (
    SimilarityEstimate,
    estimate_similarity_ransac,
    inverse_camera_motion,
    local_surface_camera_motion,
    relation_aligned_camera_motion,
    relative_similarity_residual,
    select_anchor_pair,
    surface_view_change,
    umeyama_similarity,
)
from inspect_system.active_view.object_centric_evidence_memory import (
    ObjectCentricEvidenceMemory,
    ObjectCentricKernelConfig,
    ObjectCentricMotion,
)
from inspect_system.active_view.object_centric_reveal import camera_motion_from_lattice
from inspect_system.active_view.view_lattice import ViewNode


def _rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    matrix, _ = cv2.Rodrigues(vector * angle)
    return matrix


def test_umeyama_recovers_similarity() -> None:
    rng = np.random.default_rng(4)
    source = rng.normal(size=(80, 3))
    expected_scale = 1.18
    expected_rotation = _rotation((0.2, 1.0, -0.1), 0.27)
    expected_translation = np.asarray([0.3, -0.2, 0.4])
    target = expected_scale * (source @ expected_rotation.T) + expected_translation

    scale, rotation, translation = umeyama_similarity(source, target)

    assert math.isclose(scale, expected_scale, rel_tol=1e-8)
    assert np.allclose(rotation, expected_rotation, atol=1e-8)
    assert np.allclose(translation, expected_translation, atol=1e-8)


def test_ransac_similarity_rejects_outliers() -> None:
    rng = np.random.default_rng(9)
    source = rng.normal(size=(120, 3))
    expected_rotation = _rotation((0.0, 1.0, 0.1), -0.19)
    target = 0.92 * (source @ expected_rotation.T) + np.asarray([-0.15, 0.08, 0.22])
    target += rng.normal(scale=0.002, size=target.shape)
    target[:25] = rng.normal(loc=4.0, scale=1.0, size=(25, 3))

    estimate = estimate_similarity_ransac(source, target, threshold_ratio=0.03)

    assert estimate.inliers.sum() >= 90
    assert np.allclose(estimate.rotation, expected_rotation, atol=0.01)
    assert math.isclose(estimate.scale, 0.92, rel_tol=0.02)
    assert estimate.normalized_residual < 0.01


def test_inverse_camera_motion_recovers_pose_without_object_frame() -> None:
    object_rotation = _rotation((0.1, 1.0, -0.2), 0.24)
    object_translation = np.asarray([0.3, -0.1, 0.4])
    estimate = SimilarityEstimate(
        scale=1.2,
        rotation=object_rotation,
        translation=object_translation,
        inliers=np.ones((12,), dtype=bool),
        normalized_residual=0.0,
    )

    rotation_vector, translation = inverse_camera_motion(
        estimate,
        reference_scale=0.5,
    )

    recovered_rotation, _ = cv2.Rodrigues(rotation_vector)
    expected_translation = -(object_rotation.T @ object_translation) / 1.2 / 0.5
    assert np.allclose(recovered_rotation, object_rotation.T, atol=1e-8)
    assert np.allclose(translation, expected_translation, atol=1e-8)


def test_relative_similarity_residual_is_zero_for_static_foreground() -> None:
    background = SimilarityEstimate(
        scale=1.1,
        rotation=_rotation((0.2, 1.0, -0.1), 0.18),
        translation=np.asarray([0.2, -0.1, 0.3]),
        inliers=np.ones((16,), dtype=bool),
        normalized_residual=0.0,
    )

    residual = relative_similarity_residual(
        background,
        background,
        reference_scale=0.5,
    )

    assert math.isclose(residual.log_scale, 0.0, abs_tol=1e-10)
    assert np.allclose(residual.rotation_vector, 0.0, atol=1e-10)
    assert np.allclose(residual.translation, 0.0, atol=1e-10)


def test_relative_similarity_residual_recovers_object_motion() -> None:
    background_scale = 0.9
    background_rotation = _rotation((0.1, 1.0, 0.2), -0.16)
    background_translation = np.asarray([-0.2, 0.15, 0.4])
    relative_scale = 1.04
    relative_rotation = _rotation((0.8, -0.1, 0.3), 0.21)
    relative_translation = np.asarray([0.06, -0.03, 0.02])
    background = SimilarityEstimate(
        scale=background_scale,
        rotation=background_rotation,
        translation=background_translation,
        inliers=np.ones((16,), dtype=bool),
        normalized_residual=0.0,
    )
    foreground = SimilarityEstimate(
        scale=background_scale * relative_scale,
        rotation=background_rotation @ relative_rotation,
        translation=(
            background_translation
            + background_scale * (background_rotation @ relative_translation)
        ),
        inliers=np.ones((16,), dtype=bool),
        normalized_residual=0.0,
    )

    residual = relative_similarity_residual(
        background,
        foreground,
        reference_scale=0.5,
    )
    recovered_rotation, _ = cv2.Rodrigues(residual.rotation_vector)

    assert math.isclose(residual.log_scale, math.log(relative_scale), abs_tol=1e-8)
    assert np.allclose(recovered_rotation, relative_rotation, atol=1e-8)
    assert np.allclose(residual.translation, relative_translation / 0.5, atol=1e-8)


def test_surface_view_change_is_center_and_axis_independent() -> None:
    center = np.asarray([0.0, 0.0, 2.0])
    normal = np.asarray([0.0, 0.0, -1.0])
    camera_motion = SimilarityEstimate(
        scale=1.0,
        rotation=np.eye(3),
        translation=np.asarray([0.8, 0.0, 0.0]),
        inliers=np.ones((12,), dtype=bool),
        normalized_residual=0.0,
    )

    before, after, delta, grazing_delta, parallax = surface_view_change(
        center,
        normal,
        camera_motion,
    )

    assert math.isclose(before, 1.0, abs_tol=1e-8)
    assert after < before
    assert delta < 0.0
    assert grazing_delta > 0.0
    assert parallax > 0.0

    coordinate_rotation = _rotation((0.2, 0.8, -0.1), 0.73)
    rotated_similarity = SimilarityEstimate(
        scale=1.0,
        rotation=coordinate_rotation @ camera_motion.rotation @ coordinate_rotation.T,
        translation=coordinate_rotation @ camera_motion.translation,
        inliers=np.ones((12,), dtype=bool),
        normalized_residual=0.0,
    )
    rotated = surface_view_change(
        coordinate_rotation @ center,
        coordinate_rotation @ normal,
        rotated_similarity,
    )
    assert np.allclose(rotated, (before, after, delta, grazing_delta, parallax))


def test_local_surface_motion_is_coordinate_rotation_invariant() -> None:
    center = np.asarray([0.2, -0.1, 2.0])
    normal = np.asarray([0.4, -0.2, -1.0])
    motion = SimilarityEstimate(
        scale=1.0,
        rotation=_rotation((0.2, 0.8, 0.1), 0.12),
        translation=np.asarray([0.5, -0.2, 0.1]),
        inliers=np.ones((12,), dtype=bool),
        normalized_residual=0.0,
    )
    expected = local_surface_camera_motion(center, normal, motion)

    coordinate_rotation = _rotation((-0.3, 0.4, 0.8), 0.67)
    transformed = SimilarityEstimate(
        scale=motion.scale,
        rotation=(
            coordinate_rotation
            @ motion.rotation
            @ coordinate_rotation.T
        ),
        translation=coordinate_rotation @ motion.translation,
        inliers=motion.inliers,
        normalized_residual=0.0,
    )
    actual = local_surface_camera_motion(
        coordinate_rotation @ center,
        coordinate_rotation @ normal,
        transformed,
    )

    assert np.allclose(actual, expected, atol=1e-8)


def test_relation_aligned_motion_uses_part_to_reference_axis() -> None:
    y, x = np.mgrid[-1.0:1.0:60j, -1.0:1.0:60j]
    points = np.stack([x, y, np.full_like(x, 2.0)], axis=-1)
    valid = np.ones((60, 60), dtype=bool)

    axis, binormal, normal, quality = relation_aligned_camera_motion(
        points,
        valid,
        target_bbox=[34, 18, 56, 42],
        reference_bbox=[4, 18, 28, 42],
        camera_translation=[1.0, 0.0, 0.0],
    )

    assert axis > 0.95
    assert abs(binormal) < 0.05
    assert abs(normal) < 0.05
    assert quality > 0.95


def test_lattice_motion_uses_current_surface_normal() -> None:
    views = {
        "front": ViewNode("front", yaw=0.0, elevation=0.0),
        "top": ViewNode("top", yaw=0.0, elevation=90.0),
    }

    motion = camera_motion_from_lattice(
        "front",
        "top",
        views,
        surface_normal_world=np.asarray([0.0, -1.0, 0.0]),
    )

    assert math.isclose(motion.surface_incidence_before, 1.0)
    assert math.isclose(motion.surface_incidence_after, 0.0, abs_tol=1e-8)
    assert math.isclose(motion.surface_incidence_delta, -1.0)
    assert math.isclose(motion.surface_view_parallax, math.pi / 2.0)
    assert abs(motion.surface_tangent_x) + abs(motion.surface_tangent_y) > 0.0


def test_lattice_motion_projects_into_relation_frame() -> None:
    views = {
        "front": ViewNode("front", yaw=0.0, elevation=0.0),
        "right": ViewNode("right", yaw=90.0, elevation=0.0),
    }
    motion = camera_motion_from_lattice(
        "front",
        "right",
        views,
        relation_frame_world={
            "axis_world": [1.0, 0.0, 0.0],
            "binormal_world": [0.0, 1.0, 0.0],
            "normal_world": [0.0, 0.0, 1.0],
            "quality": 0.8,
        },
    )

    assert abs(motion.relation_axis_translation) > 0.0
    assert motion.relation_frame_quality == 0.8


def test_anchor_selection_prefers_shared_housing_track() -> None:
    before = [
        {"name": "type_8_gear", "track_id": 2, "confidence": 0.99, "xyxy": [0, 0, 80, 80]},
        {
            "name": "type_5_gearbox_housing",
            "track_id": 7,
            "confidence": 0.75,
            "stable": True,
            "xyxy": [10, 10, 210, 180],
        },
    ]
    after = [
        {"name": "type_8_gear", "track_id": 2, "confidence": 0.95, "xyxy": [8, 4, 88, 84]},
        {
            "name": "type_5_gearbox_housing",
            "track_id": 7,
            "confidence": 0.72,
            "stable": True,
            "xyxy": [20, 12, 220, 182],
        },
    ]

    selected_before, selected_after = select_anchor_pair(before, after)

    assert selected_before["track_id"] == 7
    assert selected_after["track_id"] == 7

def test_spherical_bearing_is_scale_invariant_but_sign_sensitive() -> None:
    memory = ObjectCentricEvidenceMemory(
        kernel_config=ObjectCentricKernelConfig(
            geometry_mode="spherical_bearing",
            rotation_bandwidth=0.5,
        )
    )
    small = ObjectCentricMotion(azimuth=0.1, elevation=0.1)
    large = ObjectCentricMotion(azimuth=1.0, elevation=1.0)
    mirrored = ObjectCentricMotion(azimuth=-1.0, elevation=1.0)

    assert math.isclose(memory._distance_square(small, large), 0.0, abs_tol=1e-12)
    assert memory._distance_square(small, mirrored) > 1.0
