"""Object-centric observation changes from tracked RGB-D endpoints."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class SimilarityEstimate:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    inliers: np.ndarray
    normalized_residual: float


@dataclass(frozen=True)
class RelativeSimilarityResidual:
    log_scale: float
    rotation_vector: np.ndarray
    translation: np.ndarray

    @property
    def rotation_magnitude(self) -> float:
        return float(np.linalg.norm(self.rotation_vector))

    @property
    def translation_magnitude(self) -> float:
        return float(np.linalg.norm(self.translation))

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_scale": float(self.log_scale),
            "rotation_vector": [float(value) for value in self.rotation_vector],
            "translation": [float(value) for value in self.translation],
            "rotation_magnitude": self.rotation_magnitude,
            "translation_magnitude": self.translation_magnitude,
        }


@dataclass(frozen=True)
class ObjectCentricTransition:
    azimuth_delta: float
    elevation_delta: float
    log_radius_delta: float
    displacement_x: float
    displacement_y: float
    displacement_z: float
    rotation_x: float
    rotation_y: float
    rotation_z: float
    inlier_ratio: float
    normalized_residual: float
    tracked_points: int
    inlier_points: int
    anchor_scale: float
    camera_rotation_x: float = 0.0
    camera_rotation_y: float = 0.0
    camera_rotation_z: float = 0.0
    camera_translation_x: float = 0.0
    camera_translation_y: float = 0.0
    camera_translation_z: float = 0.0
    surface_incidence_before: float = 0.0
    surface_incidence_after: float = 0.0
    surface_incidence_delta: float = 0.0
    surface_grazing_delta: float = 0.0
    surface_view_parallax: float = 0.0
    surface_tangent_x: float = 0.0
    surface_tangent_y: float = 0.0
    surface_normal_translation: float = 0.0
    surface_frame_quality: float = 0.0
    anchor_name: str = ""
    anchor_track_id: int | None = None
    matching_method: str = ""
    reference_type: str = "object"
    visibility_ratio: float = 1.0
    cycle_error: float = 0.0

    def feature_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.azimuth_delta,
                self.elevation_delta,
                self.log_radius_delta,
                self.displacement_x,
                self.displacement_y,
                self.displacement_z,
                self.rotation_x,
                self.rotation_y,
                self.rotation_z,
                self.inlier_ratio,
                self.normalized_residual,
                self.visibility_ratio,
                self.cycle_error,
                self.camera_rotation_x,
                self.camera_rotation_y,
                self.camera_rotation_z,
                self.camera_translation_x,
                self.camera_translation_y,
                self.camera_translation_z,
                self.surface_tangent_x,
                self.surface_tangent_y,
                self.surface_normal_translation,
                self.surface_frame_quality,
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_points(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    return np.all(np.isfinite(array), axis=1) & (np.linalg.norm(array, axis=1) > 1e-8)


def umeyama_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate target = scale * rotation * source + translation."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must be matching Nx3 arrays")
    if source.shape[0] < 3:
        raise ValueError("at least three correspondences are required")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if source_variance <= 1e-12:
        raise ValueError("degenerate source points")

    covariance = target_centered.T @ source_centered / source.shape[0]
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    scale = float(np.sum(singular * np.diag(correction)) / source_variance)
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("invalid similarity scale")
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def estimate_similarity_ransac(
    source: np.ndarray,
    target: np.ndarray,
    *,
    threshold_ratio: float = 0.06,
    iterations: int = 512,
    seed: int = 20260828,
) -> SimilarityEstimate:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = _finite_points(source) & _finite_points(target)
    source = source[valid]
    target = target[valid]
    if source.shape[0] < 6:
        raise ValueError("fewer than six valid 3D correspondences")

    target_center = np.median(target, axis=0)
    target_radius = np.linalg.norm(target - target_center, axis=1)
    target_scale = float(np.median(target_radius[target_radius > 1e-8]))
    if target_scale <= 1e-8:
        raise ValueError("degenerate target scale")
    threshold = max(1e-6, float(threshold_ratio) * target_scale)

    rng = np.random.default_rng(seed)
    best_inliers: np.ndarray | None = None
    best_residual = math.inf
    sample_size = min(5, source.shape[0])
    for _ in range(max(1, int(iterations))):
        sample = rng.choice(source.shape[0], size=sample_size, replace=False)
        try:
            scale, rotation, translation = umeyama_similarity(source[sample], target[sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        predicted = scale * (source @ rotation.T) + translation
        residual = np.linalg.norm(predicted - target, axis=1)
        inliers = residual <= threshold
        if int(inliers.sum()) < 4:
            continue
        median = float(np.median(residual[inliers]))
        if best_inliers is None or int(inliers.sum()) > int(best_inliers.sum()) or (
            int(inliers.sum()) == int(best_inliers.sum()) and median < best_residual
        ):
            best_inliers = inliers
            best_residual = median

    if best_inliers is None:
        raise ValueError("RANSAC found no valid similarity")
    scale, rotation, translation = umeyama_similarity(source[best_inliers], target[best_inliers])
    predicted = scale * (source @ rotation.T) + translation
    residual = np.linalg.norm(predicted - target, axis=1)
    refined = residual <= threshold
    if int(refined.sum()) >= 4:
        scale, rotation, translation = umeyama_similarity(source[refined], target[refined])
        best_inliers = refined
        predicted = scale * (source @ rotation.T) + translation
        residual = np.linalg.norm(predicted - target, axis=1)
    normalized = float(np.median(residual[best_inliers]) / target_scale)
    return SimilarityEstimate(scale, rotation, translation, best_inliers, normalized)


def relative_similarity_residual(
    background: SimilarityEstimate,
    foreground: SimilarityEstimate,
    *,
    reference_scale: float,
) -> RelativeSimilarityResidual:
    """Factor foreground motion relative to the static-scene camera transform.

    Both similarities map before-camera 3D points into the after-camera frame.
    Composing the foreground transform with the inverse background transform
    isolates rigid object motion in the before-scene coordinate system.
    """
    scale = float(foreground.scale / background.scale)
    rotation = background.rotation.T @ foreground.rotation
    translation = (
        background.rotation.T
        @ (foreground.translation - background.translation)
        / background.scale
    )
    rotation_vector, _ = cv2.Rodrigues(rotation)
    return RelativeSimilarityResidual(
        log_scale=float(math.log(max(scale, 1e-12))),
        rotation_vector=rotation_vector.reshape(3),
        translation=np.asarray(translation, dtype=np.float64).reshape(3)
        / max(1e-8, float(reference_scale)),
    )


def _bbox_mask(shape: tuple[int, int], bbox: Sequence[float], inset: float = 0.05) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = (float(value) for value in bbox)
    dx = max(0.0, x2 - x1) * inset
    dy = max(0.0, y2 - y1) * inset
    left = max(0, min(width - 1, int(round(x1 + dx))))
    top = max(0, min(height - 1, int(round(y1 + dy))))
    right = max(left + 1, min(width, int(round(x2 - dx))))
    bottom = max(top + 1, min(height, int(round(y2 - dy))))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:bottom, left:right] = 255
    return mask


def track_anchor_features(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    before_bbox: Sequence[float],
    after_bbox: Sequence[float],
    *,
    max_corners: int = 1200,
    backward_error: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    before_gray = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2GRAY)
    mask = _bbox_mask(before_gray.shape, before_bbox)
    points0 = cv2.goodFeaturesToTrack(
        before_gray,
        maxCorners=max_corners,
        qualityLevel=0.005,
        minDistance=4.0,
        mask=mask,
        blockSize=7,
    )
    if points0 is None or len(points0) < 8:
        raise ValueError("not enough anchor features")
    points1, status1, error1 = cv2.calcOpticalFlowPyrLK(
        before_gray,
        after_gray,
        points0,
        None,
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
    )
    points0_back, status0, _ = cv2.calcOpticalFlowPyrLK(
        after_gray,
        before_gray,
        points1,
        None,
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
    )
    p0 = points0.reshape(-1, 2)
    p1 = points1.reshape(-1, 2)
    p0_back = points0_back.reshape(-1, 2)
    valid = status1.reshape(-1).astype(bool) & status0.reshape(-1).astype(bool)
    if error1 is not None:
        valid &= error1.reshape(-1) < 40.0
    valid &= np.linalg.norm(p0 - p0_back, axis=1) <= float(backward_error)
    after_mask = _bbox_mask(after_gray.shape, after_bbox, inset=-0.03)
    x = np.clip(np.rint(p1[:, 0]).astype(int), 0, after_gray.shape[1] - 1)
    y = np.clip(np.rint(p1[:, 1]).astype(int), 0, after_gray.shape[0] - 1)
    valid &= after_mask[y, x] > 0
    if int(valid.sum()) < 8:
        raise ValueError("not enough bidirectionally consistent anchor tracks")
    return p0[valid], p1[valid]


def match_anchor_features_sift(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    before_bbox: Sequence[float],
    after_bbox: Sequence[float],
    *,
    max_features: int = 2400,
    ratio_threshold: float = 0.78,
) -> tuple[np.ndarray, np.ndarray]:
    before_gray = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2GRAY)
    before_mask = _bbox_mask(before_gray.shape, before_bbox, inset=0.02)
    after_mask = _bbox_mask(after_gray.shape, after_bbox, inset=-0.03)
    sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.015, edgeThreshold=14)
    keys0, descriptors0 = sift.detectAndCompute(before_gray, before_mask)
    keys1, descriptors1 = sift.detectAndCompute(after_gray, after_mask)
    if descriptors0 is None or descriptors1 is None or len(keys0) < 8 or len(keys1) < 8:
        raise ValueError("not enough SIFT anchor features")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(descriptors0, descriptors1, k=2)
    reverse = matcher.knnMatch(descriptors1, descriptors0, k=2)
    reverse_pairs = {
        (pair[0].queryIdx, pair[0].trainIdx)
        for pair in reverse
        if len(pair) == 2 and pair[0].distance < ratio_threshold * pair[1].distance
    }
    matches = [
        pair[0]
        for pair in forward
        if len(pair) == 2
        and pair[0].distance < ratio_threshold * pair[1].distance
        and (pair[0].trainIdx, pair[0].queryIdx) in reverse_pairs
    ]
    if len(matches) < 8:
        raise ValueError("not enough mutual-ratio SIFT matches")
    matches.sort(key=lambda item: item.distance)
    points0 = np.asarray([keys0[item.queryIdx].pt for item in matches], dtype=np.float64)
    points1 = np.asarray([keys1[item.trainIdx].pt for item in matches], dtype=np.float64)
    return points0, points1


def _sample_point_map(points: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    height, width = points.shape[:2]
    x = np.clip(np.rint(pixels[:, 0]).astype(int), 0, width - 1)
    y = np.clip(np.rint(pixels[:, 1]).astype(int), 0, height - 1)
    return np.asarray(points[y, x], dtype=np.float64)


def estimate_matched_similarity(
    before_points: np.ndarray,
    after_points: np.ndarray,
    matched_pixels: tuple[np.ndarray, np.ndarray],
    *,
    before_valid: np.ndarray | None = None,
    after_valid: np.ndarray | None = None,
    seed: int = 20260828,
) -> SimilarityEstimate:
    pixels0, pixels1 = matched_pixels
    points0 = _sample_point_map(before_points, pixels0)
    points1 = _sample_point_map(after_points, pixels1)
    valid = _finite_points(points0) & _finite_points(points1)
    if before_valid is not None:
        height, width = before_valid.shape[:2]
        x = np.clip(np.rint(pixels0[:, 0]).astype(int), 0, width - 1)
        y = np.clip(np.rint(pixels0[:, 1]).astype(int), 0, height - 1)
        valid &= np.asarray(before_valid, dtype=bool)[y, x]
    if after_valid is not None:
        height, width = after_valid.shape[:2]
        x = np.clip(np.rint(pixels1[:, 0]).astype(int), 0, width - 1)
        y = np.clip(np.rint(pixels1[:, 1]).astype(int), 0, height - 1)
        valid &= np.asarray(after_valid, dtype=bool)[y, x]
    points0 = points0[valid]
    points1 = points1[valid]
    if points0.shape[0] < 6:
        raise ValueError("fewer than six valid tracked 3D points")
    return estimate_similarity_ransac(points0, points1, seed=seed)


def _anchor_cloud(
    points: np.ndarray,
    valid_mask: np.ndarray | None,
    bbox: Sequence[float],
    *,
    max_points: int = 12000,
) -> np.ndarray:
    mask = _bbox_mask(points.shape[:2], bbox, inset=0.08).astype(bool)
    if valid_mask is not None:
        mask &= np.asarray(valid_mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("anchor bbox has no valid geometry")
    stride = max(1, int(math.ceil(len(xs) / max_points)))
    cloud = np.asarray(points[ys[::stride], xs[::stride]], dtype=np.float64)
    cloud = cloud[_finite_points(cloud)]
    if cloud.shape[0] < 32:
        raise ValueError("anchor cloud is too sparse")
    depth = cloud[:, 2]
    low, high = np.quantile(depth, [0.05, 0.80])
    cloud = cloud[(depth >= low) & (depth <= high)]
    if cloud.shape[0] < 24:
        raise ValueError("anchor depth filtering removed too many points")
    return cloud


def canonical_anchor_frame(cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    cloud = np.asarray(cloud, dtype=np.float64)
    center = np.median(cloud, axis=0)
    centered = cloud - center
    radius = np.linalg.norm(centered, axis=1)
    cutoff = float(np.quantile(radius, 0.90))
    centered = centered[radius <= cutoff]
    covariance = centered.T @ centered / max(1, centered.shape[0])
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]

    z_axis = vectors[:, -1]
    camera_direction = -center
    if float(np.dot(z_axis, camera_direction)) < 0.0:
        z_axis = -z_axis
    x_axis = vectors[:, 0] - float(np.dot(vectors[:, 0], z_axis)) * z_axis
    x_axis /= max(1e-12, float(np.linalg.norm(x_axis)))
    if float(np.dot(x_axis, np.asarray([1.0, 0.0, 0.0]))) < 0.0:
        x_axis = -x_axis
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(1e-12, float(np.linalg.norm(y_axis)))
    if float(np.dot(y_axis, np.asarray([0.0, -1.0, 0.0]))) < 0.0:
        x_axis = -x_axis
        y_axis = -y_axis
    basis = np.column_stack([x_axis, y_axis, z_axis])
    scale = float(max(1e-8, math.sqrt(max(values[0], 1e-12))))
    return center, basis, scale


def _spherical(vector: np.ndarray) -> tuple[float, float, float]:
    radius = float(np.linalg.norm(vector))
    if radius <= 1e-12:
        raise ValueError("zero-length camera-to-object vector")
    unit = vector / radius
    azimuth = math.atan2(float(unit[0]), float(unit[2]))
    elevation = math.asin(float(np.clip(unit[1], -1.0, 1.0)))
    return azimuth, elevation, radius


def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def inverse_camera_motion(
    similarity: SimilarityEstimate,
    *,
    reference_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert a static-scene Sim(3) into camera-frame motion.

    The similarity maps a tracked 3D point from the before camera frame into
    the after camera frame. Its inverse gives the after-camera pose in the
    before-camera coordinate system without requiring a canonical object
    center or object principal axes.
    """
    rotation = similarity.rotation.T
    translation = -(
        similarity.rotation.T @ similarity.translation
    ) / similarity.scale
    rotation_vector, _ = cv2.Rodrigues(rotation)
    normalized_translation = translation / max(1e-8, float(reference_scale))
    return rotation_vector.reshape(3), normalized_translation.reshape(3)


def surface_view_change(
    center_before: np.ndarray,
    normal_before: np.ndarray,
    similarity: SimilarityEstimate,
) -> tuple[float, float, float, float, float]:
    """Measure an axis-free view change around a tracked local surface.

    The patch center and normal are expressed in the before-camera frame.
    The fitted similarity transports the same physical patch into the
    after-camera frame.  Incidence therefore does not require a stable object
    center or principal axes and remains meaningful for everyday handheld
    observations where only a local workpiece surface is tracked.
    """

    center0 = np.asarray(center_before, dtype=np.float64).reshape(3)
    normal0 = np.asarray(normal_before, dtype=np.float64).reshape(3)
    normal_norm = float(np.linalg.norm(normal0))
    if normal_norm <= 1e-10:
        raise ValueError("surface normal is degenerate")
    normal0 /= normal_norm

    center1 = (
        similarity.scale * (similarity.rotation @ center0)
        + similarity.translation
    )
    normal1 = similarity.rotation @ normal0
    normal1 /= max(1e-10, float(np.linalg.norm(normal1)))

    view0 = -center0
    view1 = -center1
    view0 /= max(1e-10, float(np.linalg.norm(view0)))
    view1 /= max(1e-10, float(np.linalg.norm(view1)))

    incidence0 = abs(float(np.dot(normal0, view0)))
    incidence1 = abs(float(np.dot(normal1, view1)))
    incidence0 = float(np.clip(incidence0, 0.0, 1.0))
    incidence1 = float(np.clip(incidence1, 0.0, 1.0))
    grazing0 = math.sqrt(max(0.0, 1.0 - incidence0 * incidence0))
    grazing1 = math.sqrt(max(0.0, 1.0 - incidence1 * incidence1))

    # Express both viewing rays in the same physical patch frame before
    # measuring angular parallax.
    view1_in_before = similarity.rotation.T @ view1
    view1_in_before /= max(1e-10, float(np.linalg.norm(view1_in_before)))
    cosine = float(np.clip(np.dot(view0, view1_in_before), -1.0, 1.0))
    parallax = float(math.acos(cosine))
    return (
        incidence0,
        incidence1,
        incidence1 - incidence0,
        grazing1 - grazing0,
        parallax,
    )


def local_surface_camera_motion(
    center_before: np.ndarray,
    normal_before: np.ndarray,
    similarity: SimilarityEstimate,
) -> tuple[float, float, float, float]:
    """Express inverse camera translation in a tracked local-surface frame.

    The frame is anchored by a physical patch normal and the current viewing
    ray, not by an object centroid or object principal axes. It therefore
    remains defined when a handheld observation sees only part of a workpiece.
    The returned translation is directional; metric scale from monocular depth
    is intentionally discarded.
    """

    center = np.asarray(center_before, dtype=np.float64).reshape(3)
    normal = np.asarray(normal_before, dtype=np.float64).reshape(3)
    normal /= max(1e-10, float(np.linalg.norm(normal)))
    view_to_camera = -center
    view_to_camera /= max(1e-10, float(np.linalg.norm(view_to_camera)))
    if float(np.dot(normal, view_to_camera)) < 0.0:
        normal = -normal

    tangent_x = np.cross(normal, view_to_camera)
    frame_quality = float(np.linalg.norm(tangent_x))
    tangent_norm = frame_quality
    if tangent_norm <= 1e-4:
        camera_right = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        tangent_x = camera_right - float(np.dot(camera_right, normal)) * normal
        tangent_norm = float(np.linalg.norm(tangent_x))
    if tangent_norm <= 1e-4:
        camera_up = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
        tangent_x = camera_up - float(np.dot(camera_up, normal)) * normal
        tangent_norm = float(np.linalg.norm(tangent_x))
    tangent_x /= max(1e-10, float(np.linalg.norm(tangent_x)))
    tangent_y = np.cross(normal, tangent_x)
    tangent_y /= max(1e-10, float(np.linalg.norm(tangent_y)))

    camera_after = -(
        similarity.rotation.T @ similarity.translation
    ) / similarity.scale
    motion_norm = float(np.linalg.norm(camera_after))
    if motion_norm <= 1e-10:
        return 0.0, 0.0, 0.0, frame_quality
    direction = camera_after / motion_norm
    return (
        float(np.dot(direction, tangent_x)),
        float(np.dot(direction, tangent_y)),
        float(np.dot(direction, normal)),
        float(np.clip(frame_quality, 0.0, 1.0)),
    )


def relation_aligned_camera_motion(
    points: np.ndarray,
    valid_mask: np.ndarray | None,
    target_bbox: Sequence[float],
    reference_bbox: Sequence[float],
    camera_translation: Sequence[float],
) -> tuple[float, float, float, float]:
    """Project inverse camera motion into a part-to-reference surface frame.

    The signed tangent axis is the observed target-to-reference relation,
    projected onto the reference surface. It is constructed independently in
    each observation and therefore does not assume a fixed object center or
    a globally registered assistant/robot camera frame.
    """
    target_cloud = _anchor_cloud(points, valid_mask, target_bbox)
    reference_cloud = _anchor_cloud(points, valid_mask, reference_bbox)
    target_center = np.median(target_cloud, axis=0)
    reference_center, reference_basis, _ = canonical_anchor_frame(reference_cloud)
    normal = np.asarray(reference_basis[:, 2], dtype=np.float64)
    view_to_camera = -reference_center
    view_to_camera /= max(1e-10, float(np.linalg.norm(view_to_camera)))
    if float(np.dot(normal, view_to_camera)) < 0.0:
        normal = -normal

    relation = target_center - reference_center
    relation_norm = float(np.linalg.norm(relation))
    if relation_norm <= 1e-8:
        return 0.0, 0.0, 0.0, 0.0
    tangent = relation - float(np.dot(relation, normal)) * normal
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-8:
        return 0.0, 0.0, 0.0, 0.0
    axis = tangent / tangent_norm
    binormal = np.cross(normal, axis)
    binormal /= max(1e-10, float(np.linalg.norm(binormal)))

    motion = np.asarray(camera_translation, dtype=np.float64).reshape(3)
    motion_norm = float(np.linalg.norm(motion))
    if motion_norm <= 1e-10:
        return 0.0, 0.0, 0.0, 0.0
    direction = motion / motion_norm
    frame_quality = float(np.clip(tangent_norm / relation_norm, 0.0, 1.0))
    return (
        float(np.dot(direction, axis)),
        float(np.dot(direction, binormal)),
        float(np.dot(direction, normal)),
        frame_quality,
    )


def estimate_object_centric_transition(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    before_points: np.ndarray,
    after_points: np.ndarray,
    before_bbox: Sequence[float],
    after_bbox: Sequence[float],
    *,
    before_valid: np.ndarray | None = None,
    after_valid: np.ndarray | None = None,
    anchor_name: str = "",
    anchor_track_id: int | None = None,
    seed: int = 20260828,
    matched_pixels: tuple[np.ndarray, np.ndarray] | None = None,
    matching_method_override: str = "",
) -> ObjectCentricTransition:
    matching_method = matching_method_override or "bidirectional_lk"
    if matched_pixels is not None:
        pixels0, pixels1 = matched_pixels
    else:
        try:
            pixels0, pixels1 = track_anchor_features(
                before_bgr,
                after_bgr,
                before_bbox,
                after_bbox,
            )
        except ValueError:
            pixels0, pixels1 = match_anchor_features_sift(
                before_bgr,
                after_bgr,
                before_bbox,
                after_bbox,
            )
            matching_method = "mutual_ratio_sift"
    points0 = _sample_point_map(before_points, pixels0)
    points1 = _sample_point_map(after_points, pixels1)
    valid = _finite_points(points0) & _finite_points(points1)
    if before_valid is not None:
        height, width = before_valid.shape[:2]
        x = np.clip(np.rint(pixels0[:, 0]).astype(int), 0, width - 1)
        y = np.clip(np.rint(pixels0[:, 1]).astype(int), 0, height - 1)
        valid &= np.asarray(before_valid, dtype=bool)[y, x]
    if after_valid is not None:
        height, width = after_valid.shape[:2]
        x = np.clip(np.rint(pixels1[:, 0]).astype(int), 0, width - 1)
        y = np.clip(np.rint(pixels1[:, 1]).astype(int), 0, height - 1)
        valid &= np.asarray(after_valid, dtype=bool)[y, x]
    points0 = points0[valid]
    points1 = points1[valid]

    cloud = _anchor_cloud(before_points, before_valid, before_bbox)
    center, basis, anchor_scale = canonical_anchor_frame(cloud)
    similarity = estimate_similarity_ransac(points0, points1, seed=seed)
    (
        incidence_before,
        incidence_after,
        incidence_delta,
        grazing_delta,
        view_parallax,
    ) = surface_view_change(center, basis[:, 2], similarity)
    (
        surface_tangent_x,
        surface_tangent_y,
        surface_normal_translation,
        surface_frame_quality,
    ) = local_surface_camera_motion(center, basis[:, 2], similarity)

    camera_after = -(similarity.rotation.T @ similarity.translation) / similarity.scale
    camera_rotation_vector, camera_translation = inverse_camera_motion(
        similarity,
        reference_scale=anchor_scale,
    )
    before_vector = basis.T @ (-center)
    after_vector = basis.T @ (camera_after - center)
    azimuth0, elevation0, radius0 = _spherical(before_vector)
    azimuth1, elevation1, radius1 = _spherical(after_vector)
    displacement = basis.T @ camera_after / anchor_scale
    object_rotation = basis.T @ similarity.rotation.T @ basis
    rotation_vector, _ = cv2.Rodrigues(object_rotation)
    rotation_vector = rotation_vector.reshape(3)
    inlier_count = int(similarity.inliers.sum())

    return ObjectCentricTransition(
        azimuth_delta=_wrap_angle(azimuth1 - azimuth0),
        elevation_delta=float(elevation1 - elevation0),
        log_radius_delta=float(math.log(radius1 / radius0)),
        displacement_x=float(displacement[0]),
        displacement_y=float(displacement[1]),
        displacement_z=float(displacement[2]),
        rotation_x=float(rotation_vector[0]),
        rotation_y=float(rotation_vector[1]),
        rotation_z=float(rotation_vector[2]),
        inlier_ratio=float(inlier_count / max(1, points0.shape[0])),
        normalized_residual=similarity.normalized_residual,
        tracked_points=int(points0.shape[0]),
        inlier_points=inlier_count,
        anchor_scale=anchor_scale,
        camera_rotation_x=float(camera_rotation_vector[0]),
        camera_rotation_y=float(camera_rotation_vector[1]),
        camera_rotation_z=float(camera_rotation_vector[2]),
        camera_translation_x=float(camera_translation[0]),
        camera_translation_y=float(camera_translation[1]),
        camera_translation_z=float(camera_translation[2]),
        surface_incidence_before=incidence_before,
        surface_incidence_after=incidence_after,
        surface_incidence_delta=incidence_delta,
        surface_grazing_delta=grazing_delta,
        surface_view_parallax=view_parallax,
        surface_tangent_x=surface_tangent_x,
        surface_tangent_y=surface_tangent_y,
        surface_normal_translation=surface_normal_translation,
        surface_frame_quality=surface_frame_quality,
        anchor_name=str(anchor_name),
        anchor_track_id=anchor_track_id,
        matching_method=matching_method,
    )


def estimate_reference_centric_transition(
    before_points: np.ndarray,
    after_points: np.ndarray,
    matched_pixels: tuple[np.ndarray, np.ndarray],
    *,
    before_valid: np.ndarray | None = None,
    after_valid: np.ndarray | None = None,
    before_reference_bbox: Sequence[float] | None = None,
    after_reference_bbox: Sequence[float] | None = None,
    anchor_name: str = "",
    anchor_track_id: int | None = None,
    matching_method: str = "cotracker3",
    reference_type: str = "object",
    visibility_ratio: float = 1.0,
    cycle_error: float = 0.0,
    seed: int = 20260828,
) -> ObjectCentricTransition:
    """Recover camera motion in a stable object or scene reference frame.

    Correspondences may be tracked on the workpiece itself or on the static
    scene. When only an after-frame workpiece box is available, its cloud is
    transported into before-frame coordinates with the fitted scene Sim(3).
    This supports disocclusion without matching two different physical parts.
    """

    pixels0, pixels1 = matched_pixels
    points0 = _sample_point_map(before_points, pixels0)
    points1 = _sample_point_map(after_points, pixels1)
    valid = _finite_points(points0) & _finite_points(points1)
    if before_valid is not None:
        height, width = before_valid.shape[:2]
        x = np.clip(np.rint(pixels0[:, 0]).astype(int), 0, width - 1)
        y = np.clip(np.rint(pixels0[:, 1]).astype(int), 0, height - 1)
        valid &= np.asarray(before_valid, dtype=bool)[y, x]
    if after_valid is not None:
        height, width = after_valid.shape[:2]
        x = np.clip(np.rint(pixels1[:, 0]).astype(int), 0, width - 1)
        y = np.clip(np.rint(pixels1[:, 1]).astype(int), 0, height - 1)
        valid &= np.asarray(after_valid, dtype=bool)[y, x]
    points0 = points0[valid]
    points1 = points1[valid]
    if points0.shape[0] < 6:
        raise ValueError("fewer than six valid tracked 3D points")

    similarity = estimate_similarity_ransac(points0, points1, seed=seed)
    if before_reference_bbox is not None:
        cloud = _anchor_cloud(before_points, before_valid, before_reference_bbox)
    elif after_reference_bbox is not None:
        cloud_after = _anchor_cloud(after_points, after_valid, after_reference_bbox)
        cloud = (
            (cloud_after - similarity.translation) @ similarity.rotation
        ) / similarity.scale
    else:
        cloud = points0[similarity.inliers]
        if cloud.shape[0] < 12:
            raise ValueError("reference cloud is too sparse")

    center, basis, anchor_scale = canonical_anchor_frame(cloud)
    (
        incidence_before,
        incidence_after,
        incidence_delta,
        grazing_delta,
        view_parallax,
    ) = surface_view_change(center, basis[:, 2], similarity)
    (
        surface_tangent_x,
        surface_tangent_y,
        surface_normal_translation,
        surface_frame_quality,
    ) = local_surface_camera_motion(center, basis[:, 2], similarity)
    camera_after = -(similarity.rotation.T @ similarity.translation) / similarity.scale
    camera_rotation_vector, camera_translation = inverse_camera_motion(
        similarity,
        reference_scale=anchor_scale,
    )
    before_vector = basis.T @ (-center)
    after_vector = basis.T @ (camera_after - center)
    azimuth0, elevation0, radius0 = _spherical(before_vector)
    azimuth1, elevation1, radius1 = _spherical(after_vector)
    displacement = basis.T @ camera_after / anchor_scale
    camera_rotation = basis.T @ similarity.rotation.T @ basis
    rotation_vector, _ = cv2.Rodrigues(camera_rotation)
    rotation_vector = rotation_vector.reshape(3)
    inlier_count = int(similarity.inliers.sum())

    return ObjectCentricTransition(
        azimuth_delta=_wrap_angle(azimuth1 - azimuth0),
        elevation_delta=float(elevation1 - elevation0),
        log_radius_delta=float(math.log(radius1 / radius0)),
        displacement_x=float(displacement[0]),
        displacement_y=float(displacement[1]),
        displacement_z=float(displacement[2]),
        rotation_x=float(rotation_vector[0]),
        rotation_y=float(rotation_vector[1]),
        rotation_z=float(rotation_vector[2]),
        inlier_ratio=float(inlier_count / max(1, points0.shape[0])),
        normalized_residual=similarity.normalized_residual,
        tracked_points=int(points0.shape[0]),
        inlier_points=inlier_count,
        anchor_scale=anchor_scale,
        camera_rotation_x=float(camera_rotation_vector[0]),
        camera_rotation_y=float(camera_rotation_vector[1]),
        camera_rotation_z=float(camera_rotation_vector[2]),
        camera_translation_x=float(camera_translation[0]),
        camera_translation_y=float(camera_translation[1]),
        camera_translation_z=float(camera_translation[2]),
        surface_incidence_before=incidence_before,
        surface_incidence_after=incidence_after,
        surface_incidence_delta=incidence_delta,
        surface_grazing_delta=grazing_delta,
        surface_view_parallax=view_parallax,
        surface_tangent_x=surface_tangent_x,
        surface_tangent_y=surface_tangent_y,
        surface_normal_translation=surface_normal_translation,
        surface_frame_quality=surface_frame_quality,
        anchor_name=str(anchor_name),
        anchor_track_id=anchor_track_id,
        matching_method=str(matching_method),
        reference_type=str(reference_type),
        visibility_ratio=float(visibility_ratio),
        cycle_error=float(cycle_error),
    )


def select_anchor_pair(
    before_objects: Sequence[Mapping[str, Any]],
    after_objects: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Select a stable assembly anchor, preferring a shared housing track."""

    def priority(item: Mapping[str, Any]) -> tuple[int, int, float, float]:
        name = str(item.get("name", "")).lower()
        bbox = item.get("xyxy") or [0.0, 0.0, 0.0, 0.0]
        area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(
            0.0, float(bbox[3]) - float(bbox[1])
        )
        return (
            int("gearbox_housing" in name),
            int(bool(item.get("stable", False))),
            float(item.get("confidence", 0.0)),
            area,
        )

    def compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        left_name = str(left.get("name", "")).lower()
        right_name = str(right.get("name", "")).lower()
        if left_name == right_name and left_name:
            return True
        return (
            "gearbox_housing" in left_name and "gearbox_housing" in right_name
        )

    before_sorted = sorted(before_objects, key=priority, reverse=True)
    after_sorted = sorted(after_objects, key=priority, reverse=True)
    for before in before_sorted:
        before_track = before.get("track_id")
        if before_track in (None, ""):
            continue
        for after in after_sorted:
            if after.get("track_id") == before_track and compatible(before, after):
                return before, after
    for before in before_sorted:
        matches = [item for item in after_sorted if compatible(before, item)]
        if matches:
            return before, matches[0]
    if not before_sorted or not after_sorted:
        raise ValueError("no tracked assembly anchor at one or both endpoints")
    raise ValueError("no semantically compatible assembly anchor at both endpoints")
