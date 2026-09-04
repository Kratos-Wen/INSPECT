"""Current-view ray affordances for a fixed camera lattice.

The scorer estimates a task-surface normal from the current MoGe point map
and predicted object boxes. Candidate images and robot utility labels are not
used. Known lattice rays are compared with that normal to estimate whether a
candidate is likely to expose frontal identity/slot evidence or grazing
gap/boundary evidence.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from .view_lattice import SIX_VIEWS, ViewNode


GEAR_CLASSES = {
    "small_gear",
    "big_gear",
    "type_2_gear",
    "type_3_gear",
    "type_7_gear",
    "type_8_gear",
}
COVER_CLASSES = {
    "cover",
    "type_5_gearbox_cover",
    "type_6_gearbox_cover",
}
HOUSING_CLASSES = {
    "housing",
    "type_5_gearbox_housing",
    "type_6_gearbox_housing",
}

GRAZING_ROLES = {
    "gap_visibility_view",
    "boundary_alignment_view",
    "contact_verification_view",
    "insertion_verification_view",
    "containment_verification_view",
}
FRONTAL_ROLES = {
    "identity_disambiguation_view",
}
RELATIONAL_ROLES = {
    "slot_relation_view",
}

ROLE_SURFACE_CLASSES: Mapping[str, set[str]] = {
    "identity_disambiguation_view": GEAR_CLASSES,
    "insertion_verification_view": GEAR_CLASSES | HOUSING_CLASSES,
    "containment_verification_view": GEAR_CLASSES | HOUSING_CLASSES,
    "slot_relation_view": GEAR_CLASSES | HOUSING_CLASSES,
    "gap_visibility_view": COVER_CLASSES | HOUSING_CLASSES,
    "boundary_alignment_view": COVER_CLASSES | HOUSING_CLASSES,
    "contact_verification_view": COVER_CLASSES | HOUSING_CLASSES,
}

ROLE_RELATION_CLASSES: Mapping[str, tuple[set[str], set[str]]] = {
    "identity_disambiguation_view": (GEAR_CLASSES, HOUSING_CLASSES),
    "insertion_verification_view": (GEAR_CLASSES, HOUSING_CLASSES),
    "containment_verification_view": (GEAR_CLASSES, HOUSING_CLASSES),
    "slot_relation_view": (GEAR_CLASSES, HOUSING_CLASSES),
    "gap_visibility_view": (COVER_CLASSES, HOUSING_CLASSES),
    "boundary_alignment_view": (COVER_CLASSES, HOUSING_CLASSES),
    "contact_verification_view": (COVER_CLASSES, HOUSING_CLASSES),
}


FINE_CLASS_ROLES: Mapping[str, str] = {
    "type_3_gear": "small_gear",
    "type_7_gear": "small_gear",
    "type_2_gear": "big_gear",
    "type_8_gear": "big_gear",
    "type_5_gearbox_cover": "cover",
    "type_6_gearbox_cover": "cover",
    "type_5_gearbox_housing": "housing",
    "type_6_gearbox_housing": "housing",
}


def merge_role_fallback_detections(
    fine_detections: Iterable[Mapping[str, Any]],
    role_detections: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Append coarse boxes only for roles absent from fine detections."""

    fine = list(fine_detections)
    present_roles = {
        FINE_CLASS_ROLES[name]
        for item in fine
        if (name := str(item.get("name", ""))) in FINE_CLASS_ROLES
    }
    fallback = [
        item
        for item in role_detections
        if str(item.get("name", "")) not in present_roles
    ]
    return fine + fallback

def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def camera_position(node: ViewNode) -> np.ndarray:
    yaw = math.radians(float(node.yaw))
    elevation = math.radians(float(node.elevation))
    return np.asarray(
        [
            math.sin(yaw) * math.cos(elevation),
            -math.cos(yaw) * math.cos(elevation),
            math.sin(elevation),
        ],
        dtype=np.float64,
    )


def camera_to_world_rotation(node: ViewNode) -> np.ndarray:
    position = camera_position(node)
    forward = _unit(-position)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = _unit(np.cross(forward, world_up))
    down = _unit(np.cross(forward, right))
    return np.stack([right, down, forward], axis=1)


def _box(item: Mapping[str, Any], width: int, height: int) -> tuple[int, int, int, int] | None:
    values = item.get("xyxy") or item.get("bbox")
    if not isinstance(values, Sequence) or len(values) < 4:
        return None
    x1, y1, x2, y2 = (float(values[index]) for index in range(4))
    x1 = max(0, min(width - 1, int(math.floor(x1))))
    y1 = max(0, min(height - 1, int(math.floor(y1))))
    x2 = max(x1 + 1, min(width, int(math.ceil(x2))))
    y2 = max(y1 + 1, min(height, int(math.ceil(y2))))
    return x1, y1, x2, y2


def _role_classes(claim_id: str) -> set[str]:
    claim = str(claim_id or "").lower()
    if "cover" in claim:
        return COVER_CLASSES | HOUSING_CLASSES
    return GEAR_CLASSES | HOUSING_CLASSES


def _class_center_camera(
    points: np.ndarray,
    valid_mask: np.ndarray | None,
    detections: Iterable[Mapping[str, Any]],
    allowed_classes: set[str],
) -> tuple[np.ndarray | None, int]:
    height, width = points.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    for item in detections:
        if str(item.get("name", "")) not in allowed_classes:
            continue
        if float(item.get("confidence", item.get("conf", 0.0)) or 0.0) < 0.08:
            continue
        bounds = _box(item, width, height)
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        inset_x = max(1, int(round(0.12 * (x2 - x1))))
        inset_y = max(1, int(round(0.12 * (y2 - y1))))
        mask[
            y1 + inset_y : max(y1 + inset_y + 1, y2 - inset_y),
            x1 + inset_x : max(x1 + inset_x + 1, x2 - inset_x),
        ] = True
    finite = np.all(np.isfinite(points), axis=-1) & (points[..., 2] > 1e-6)
    if valid_mask is not None:
        finite &= np.asarray(valid_mask, dtype=bool)
    selected = np.asarray(points[mask & finite], dtype=np.float64)
    if len(selected) < 32:
        return None, int(len(selected))
    center = np.median(selected, axis=0)
    distances = np.linalg.norm(selected - center, axis=1)
    selected = selected[distances <= float(np.quantile(distances, 0.80))]
    if len(selected) < 24:
        return None, int(len(selected))
    return np.median(selected, axis=0), int(len(selected))


def relation_frame_camera(
    points: np.ndarray,
    valid_mask: np.ndarray | None,
    detections: Iterable[Mapping[str, Any]],
    *,
    target_classes: set[str],
    reference_classes: set[str],
    reference_normal: np.ndarray | None,
) -> Dict[str, Any] | None:
    if reference_normal is None:
        return None
    target_center, target_support = _class_center_camera(
        points, valid_mask, detections, target_classes
    )
    reference_center, reference_support = _class_center_camera(
        points, valid_mask, detections, reference_classes
    )
    if target_center is None or reference_center is None:
        return None
    normal = _unit(np.asarray(reference_normal, dtype=np.float64))
    view_to_camera = _unit(-reference_center)
    if float(np.dot(normal, view_to_camera)) < 0.0:
        normal = -normal
    relation = target_center - reference_center
    relation_norm = float(np.linalg.norm(relation))
    if relation_norm <= 1e-8:
        return None
    tangent = relation - float(np.dot(relation, normal)) * normal
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-8:
        return None
    axis = tangent / tangent_norm
    binormal = _unit(np.cross(normal, axis))
    return {
        "axis_camera": axis,
        "binormal_camera": binormal,
        "normal_camera": normal,
        "quality": float(np.clip(tangent_norm / relation_norm, 0.0, 1.0)),
        "target_support_points": int(target_support),
        "reference_support_points": int(reference_support),
    }


def task_surface_normal_camera(
    points: np.ndarray,
    valid_mask: np.ndarray | None,
    detections: Iterable[Mapping[str, Any]],
    claim_id: str,
    *,
    normals: np.ndarray | None = None,
    allowed_classes: set[str] | None = None,
    max_points: int = 12000,
) -> tuple[np.ndarray | None, int, str, float]:
    """Estimate a normal and an observation-only reliability score."""
    height, width = points.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    allowed = set(allowed_classes) if allowed_classes is not None else _role_classes(claim_id)
    for item in detections:
        if str(item.get("name", "")) not in allowed:
            continue
        if float(item.get("confidence", item.get("conf", 0.0)) or 0.0) < 0.08:
            continue
        bounds = _box(item, width, height)
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        inset_x = max(1, int(round(0.08 * (x2 - x1))))
        inset_y = max(1, int(round(0.08 * (y2 - y1))))
        mask[y1 + inset_y : max(y1 + inset_y + 1, y2 - inset_y),
             x1 + inset_x : max(x1 + inset_x + 1, x2 - inset_x)] = True
    finite = np.all(np.isfinite(points), axis=-1) & (points[..., 2] > 1e-6)
    if valid_mask is not None:
        finite &= np.asarray(valid_mask, dtype=bool)
    mask &= finite
    selected = np.asarray(points[mask], dtype=np.float64)
    if len(selected) < 48:
        return None, int(len(selected)), "none", 0.0
    if normals is not None and np.asarray(normals).shape[:2] == points.shape[:2]:
        selected_normals = np.asarray(normals[mask], dtype=np.float64)
        normal_lengths = np.linalg.norm(selected_normals, axis=1)
        selected_normals = selected_normals[
            np.all(np.isfinite(selected_normals), axis=1) & (normal_lengths > 0.25)
        ]
        if len(selected_normals) >= 48:
            selected_normals = selected_normals / np.maximum(
                np.linalg.norm(selected_normals, axis=1, keepdims=True),
                1e-8,
            )
            orientation = selected_normals.T @ selected_normals
            values, vectors = np.linalg.eigh(orientation)
            normal = _unit(vectors[:, int(np.argmax(values))])
            if np.all(np.isfinite(normal)):
                concentration = float(np.max(values) / max(1e-8, np.sum(values)))
                coherence = float(
                    np.clip(
                        (concentration - 1.0 / 3.0) / (2.0 / 3.0),
                        0.0,
                        1.0,
                    )
                )
                support_reliability = float(
                    len(selected_normals) / (len(selected_normals) + 256.0)
                )
                return (
                    normal,
                    int(len(selected_normals)),
                    "moge_normals",
                    coherence * support_reliability,
                )
    if len(selected) > max_points:
        indices = np.linspace(0, len(selected) - 1, max_points, dtype=np.int64)
        selected = selected[indices]
    center = np.median(selected, axis=0)
    distances = np.linalg.norm(selected - center, axis=1)
    cutoff = float(np.quantile(distances, 0.85))
    selected = selected[distances <= cutoff]
    if len(selected) < 32:
        return None, int(len(selected)), "none", 0.0
    covariance = np.cov((selected - np.mean(selected, axis=0)).T)
    values, vectors = np.linalg.eigh(covariance)
    normal = _unit(vectors[:, int(np.argmin(values))])
    if not np.all(np.isfinite(normal)):
        return None, int(len(selected)), "none", 0.0
    ordered = np.sort(np.maximum(values, 0.0))
    planarity = float(
        np.clip(
            (ordered[1] - ordered[0]) / max(1e-8, ordered[2]),
            0.0,
            1.0,
        )
    )
    support_reliability = float(len(selected) / (len(selected) + 256.0))
    return (
        normal,
        int(len(selected)),
        "point_pca",
        planarity * support_reliability,
    )


def candidate_role_factors(
    *,
    normal_camera: np.ndarray | None,
    current_view: str,
    views: Mapping[str, ViewNode] | None = None,
    reliability: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    table = views or SIX_VIEWS
    reliability = float(np.clip(reliability, 0.0, 1.0))
    normal_world = (
        _unit(camera_to_world_rotation(table[current_view]) @ normal_camera)
        if current_view in table and normal_camera is not None
        else None
    )
    result: Dict[str, Dict[str, float]] = {}
    for view_id, node in table.items():
        surface_incidence = (
            abs(float(np.dot(normal_world, _unit(camera_position(node)))))
            if normal_world is not None
            else None
        )
        values: Dict[str, float] = {}
        if surface_incidence is None:
            for role in GRAZING_ROLES | FRONTAL_ROLES | RELATIONAL_ROLES:
                values[role] = 1.0
        else:
            frontal_lobe = surface_incidence**2
            grazing_lobe = (1.0 - surface_incidence) ** 2
            for role in GRAZING_ROLES:
                raw = 0.25 + 1.75 * grazing_lobe
                values[role] = 1.0 + reliability * (raw - 1.0)
            for role in FRONTAL_ROLES:
                raw = 0.25 + 1.75 * frontal_lobe
                values[role] = 1.0 + reliability * (raw - 1.0)
            for role in RELATIONAL_ROLES:
                # Slot topology benefits from an incidence change but is not
                # tied to only a frontal or only a grazing view.
                raw = 0.25 + 1.75 * max(
                    frontal_lobe,
                    grazing_lobe,
                )
                values[role] = 1.0 + reliability * (raw - 1.0)
        values["claim_disambiguation_view"] = 1.0
        values["surface_incidence"] = (
            0.5 if surface_incidence is None else surface_incidence
        )
        result[view_id] = values
    return result


def role_conditioned_candidate_factors(
    *,
    normals_camera: Mapping[str, np.ndarray | None],
    current_view: str,
    views: Mapping[str, ViewNode] | None = None,
    reliability: Mapping[str, float] | None = None,
) -> Dict[str, Dict[str, float]]:
    table = views or SIX_VIEWS
    result = {
        view_id: {
            role: 1.0
            for role in ROLE_SURFACE_CLASSES
        }
        for view_id in table
    }
    for role, normal_camera in normals_camera.items():
        if role not in ROLE_SURFACE_CLASSES:
            continue
        factors = candidate_role_factors(
            normal_camera=normal_camera,
            current_view=current_view,
            views=table,
            reliability=(1.0 if reliability is None else float(reliability.get(role, 0.0))),
        )
        for view_id in table:
            result[view_id][role] = float(factors[view_id].get(role, 1.0))
    for values in result.values():
        values["claim_disambiguation_view"] = 1.0
    return result


def estimate_ray_affordances(
    *,
    points: np.ndarray,
    normals: np.ndarray | None,
    valid_mask: np.ndarray | None,
    detections: Iterable[Mapping[str, Any]],
    claim_id: str,
    current_view: str,
    views: Mapping[str, ViewNode] | None = None,
) -> Dict[str, Any]:
    detections = list(detections)
    normal_camera, support, normal_source, normal_reliability = task_surface_normal_camera(
        points,
        valid_mask,
        detections,
        claim_id,
        normals=normals,
    )
    factors = candidate_role_factors(
        normal_camera=normal_camera,
        current_view=current_view,
        views=views,
        reliability=normal_reliability,
    )
    normal_world = None
    table = views or SIX_VIEWS
    if normal_camera is not None and current_view in table:
        normal_world = _unit(
            camera_to_world_rotation(table[current_view]) @ normal_camera
        ).tolist()
    role_normals_camera: Dict[str, np.ndarray | None] = {}
    role_normals_world: Dict[str, list[float] | None] = {}
    role_support: Dict[str, int] = {}
    role_sources: Dict[str, str] = {}
    role_reliability: Dict[str, float] = {}
    for role, classes in ROLE_SURFACE_CLASSES.items():
        role_normal, role_points, role_source, role_quality = task_surface_normal_camera(
            points,
            valid_mask,
            detections,
            claim_id,
            normals=normals,
            allowed_classes=classes,
        )
        role_normals_camera[role] = role_normal
        role_support[role] = int(role_points)
        role_sources[role] = role_source
        role_reliability[role] = float(role_quality)
        role_normals_world[role] = (
            _unit(camera_to_world_rotation(table[current_view]) @ role_normal).tolist()
            if role_normal is not None and current_view in table
            else None
        )
    housing_normal, _, _, _ = task_surface_normal_camera(
        points,
        valid_mask,
        detections,
        claim_id,
        normals=normals,
        allowed_classes=HOUSING_CLASSES,
    )
    role_relation_frames_world: Dict[str, Dict[str, Any] | None] = {}
    rotation_to_world = (
        camera_to_world_rotation(table[current_view])
        if current_view in table
        else None
    )
    for role, (target_classes, reference_classes) in ROLE_RELATION_CLASSES.items():
        frame = relation_frame_camera(
            points,
            valid_mask,
            detections,
            target_classes=target_classes,
            reference_classes=reference_classes,
            reference_normal=housing_normal,
        )
        if frame is None or rotation_to_world is None:
            role_relation_frames_world[role] = None
            continue
        role_relation_frames_world[role] = {
            "axis_world": _unit(
                rotation_to_world @ frame["axis_camera"]
            ).tolist(),
            "binormal_world": _unit(
                rotation_to_world @ frame["binormal_camera"]
            ).tolist(),
            "normal_world": _unit(
                rotation_to_world @ frame["normal_camera"]
            ).tolist(),
            "quality": float(frame["quality"]),
            "target_support_points": int(frame["target_support_points"]),
            "reference_support_points": int(frame["reference_support_points"]),
        }
    factors = role_conditioned_candidate_factors(
        normals_camera=role_normals_camera,
        current_view=current_view,
        views=table,
        reliability=role_reliability,
    )
    return {
        "surface_normal_camera": None if normal_camera is None else normal_camera.tolist(),
        "surface_normal_world": normal_world,
        "normal_support_points": support,
        "normal_source": normal_source,
        "normal_reliability": float(normal_reliability),
        "role_surface_normals_world": role_normals_world,
        "role_normal_support_points": role_support,
        "role_normal_source": role_sources,
        "role_normal_reliability": role_reliability,
        "role_relation_frames_world": role_relation_frames_world,
        "candidate_role_factors": factors,
        "candidate_images_used": False,
        "robot_utility_labels_used": False,
    }
