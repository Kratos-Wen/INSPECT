"""Causal role-level gap recovery from local image keypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from ..core_types import Detection


ROLE_BY_CLASS = {
    "type_3_gear": "small_gear",
    "type_7_gear": "small_gear",
    "type_2_gear": "big_gear",
    "type_8_gear": "big_gear",
    "type_5_gearbox_cover": "cover",
    "type_6_gearbox_cover": "cover",
    "type_5_gearbox_housing": "housing",
    "type_6_gearbox_housing": "housing",
}


@dataclass
class _RoleTrack:
    role: str
    class_name: str
    box: tuple[float, float, float, float]
    confidence: float
    anchor_confidence: float
    points: np.ndarray
    track_id: int
    hits: int = 1
    age: int = 1
    gap: int = 0


class CausalKeypointRoleBridge:
    """Bridge short detector dropouts without committing an object identity.

    The bridge tracks texture points initialized inside a detected object box.
    Recovered boxes are role-level proposals only: they are explicitly marked
    identity-unsafe and therefore cannot satisfy exact-identity claim checks.
    Only the current and previous RGB frames are used.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        max_gap: int = 120,
        confidence_decay: float = 0.995,
        min_confidence: float = 0.08,
        max_width: int = 960,
        max_corners: int = 48,
        min_points: int = 6,
        quality_level: float = 0.01,
        min_distance: float = 5.0,
        fb_error: float = 2.5,
        ransac_threshold: float = 3.0,
        min_inlier_ratio: float = 0.50,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_gap = max(1, int(max_gap))
        self.confidence_decay = float(np.clip(confidence_decay, 0.0, 1.0))
        self.min_confidence = float(np.clip(min_confidence, 0.0, 1.0))
        self.max_width = max(160, int(max_width))
        self.max_corners = max(8, int(max_corners))
        self.min_points = max(3, int(min_points))
        self.quality_level = max(1e-5, float(quality_level))
        self.min_distance = max(1.0, float(min_distance))
        self.fb_error = max(0.1, float(fb_error))
        self.ransac_threshold = max(0.5, float(ransac_threshold))
        self.min_inlier_ratio = float(np.clip(min_inlier_ratio, 0.0, 1.0))
        self._previous_gray: np.ndarray | None = None
        self._scale = 1.0
        self._tracks: dict[str, _RoleTrack] = {}
        self._next_track_id = 100_000

    def update(self, frame_bgr: np.ndarray, detections: Iterable[Detection]) -> list[Detection]:
        rows = list(detections)
        if not self.enabled:
            return rows
        gray, scale = self._gray(frame_bgr)
        if self._previous_gray is not None and abs(scale - self._scale) > 1e-6:
            self.reset()
        observed: dict[str, Detection] = {}
        for detection in rows:
            role = ROLE_BY_CLASS.get(str(detection.name).strip().lower(), "")
            previous = observed.get(role)
            if role and (previous is None or float(detection.confidence) > float(previous.confidence)):
                observed[role] = detection

        recovered: list[Detection] = []
        if self._previous_gray is not None:
            for role, track in list(self._tracks.items()):
                if role in observed:
                    continue
                detection = self._propagate(track, self._previous_gray, gray, frame_bgr.shape[:2], scale)
                if detection is None:
                    self._tracks.pop(role, None)
                else:
                    recovered.append(detection)

        for role, detection in observed.items():
            scaled_box = self._scaled_box(detection.xyxy, scale)
            points = self._detect_points(gray, scaled_box)
            existing = self._tracks.get(role)
            track_id = existing.track_id if existing is not None else self._allocate_track_id()
            self._tracks[role] = _RoleTrack(
                role=role,
                class_name=str(detection.name),
                box=tuple(float(value) for value in detection.xyxy),
                confidence=float(detection.confidence),
                anchor_confidence=float(detection.confidence),
                points=points,
                track_id=track_id,
                hits=(existing.hits + 1) if existing is not None else 1,
                age=(existing.age + 1) if existing is not None else 1,
                gap=0,
            )

        self._previous_gray = gray
        self._scale = scale
        return rows + recovered

    def _propagate(
        self,
        track: _RoleTrack,
        previous_gray: np.ndarray,
        gray: np.ndarray,
        image_shape: tuple[int, int],
        scale: float,
    ) -> Detection | None:
        if track.gap >= self.max_gap or track.points.shape[0] < self.min_points:
            return None
        forward, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            gray,
            track.points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if forward is None or status is None:
            return None
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            previous_gray,
            forward,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if backward is None or back_status is None:
            return None
        fb = np.linalg.norm(track.points.reshape(-1, 2) - backward.reshape(-1, 2), axis=1)
        valid = (
            (status.reshape(-1) > 0)
            & (back_status.reshape(-1) > 0)
            & np.isfinite(fb)
            & (fb <= self.fb_error)
        )
        source = track.points.reshape(-1, 2)[valid]
        target = forward.reshape(-1, 2)[valid]
        if source.shape[0] < self.min_points:
            return None

        matrix, inliers = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            maxIters=1000,
            confidence=0.99,
            refineIters=10,
        )
        if matrix is None:
            delta = np.median(target - source, axis=0)
            matrix = np.asarray(
                [[1.0, 0.0, delta[0]], [0.0, 1.0, delta[1]]],
                dtype=np.float32,
            )
            inlier_mask = np.ones(source.shape[0], dtype=bool)
        else:
            inlier_mask = (
                inliers.reshape(-1).astype(bool)
                if inliers is not None
                else np.ones(source.shape[0], dtype=bool)
            )
        if int(inlier_mask.sum()) < self.min_points:
            return None
        inlier_ratio = float(inlier_mask.sum() / max(1, source.shape[0]))
        if inlier_ratio < self.min_inlier_ratio:
            return None
        a, b = float(matrix[0, 0]), float(matrix[0, 1])
        affine_scale = float((a * a + b * b) ** 0.5)
        if not 0.70 <= affine_scale <= 1.45:
            return None

        box_scaled = self._scaled_box(track.box, scale)
        corners = np.asarray(
            [
                [box_scaled[0], box_scaled[1]],
                [box_scaled[2], box_scaled[1]],
                [box_scaled[2], box_scaled[3]],
                [box_scaled[0], box_scaled[3]],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        warped = cv2.transform(corners, matrix).reshape(-1, 2)
        x1, y1 = np.min(warped, axis=0)
        x2, y2 = np.max(warped, axis=0)
        height, width = image_shape
        box = (
            float(np.clip(x1 / scale, 0.0, max(0, width - 1))),
            float(np.clip(y1 / scale, 0.0, max(0, height - 1))),
            float(np.clip(x2 / scale, 0.0, max(0, width - 1))),
            float(np.clip(y2 / scale, 0.0, max(0, height - 1))),
        )
        if box[2] - box[0] <= 2.0 or box[3] - box[1] <= 2.0:
            return None
        median_fb = float(np.median(fb[valid]))
        quality = float(
            np.clip(inlier_ratio, 0.0, 1.0)
            * np.exp(-median_fb / max(self.fb_error, 1e-6))
        )
        confidence = float(
            track.anchor_confidence
            * (self.confidence_decay ** float(track.gap + 1))
            * max(0.50, quality)
        )
        if confidence < self.min_confidence:
            return None

        next_points = target[inlier_mask].reshape(-1, 1, 2).astype(np.float32)
        track.box = box
        track.confidence = confidence
        track.points = next_points
        track.hits += 1
        track.age += 1
        track.gap += 1
        return Detection(
            name=track.class_name,
            xyxy=box,
            confidence=confidence,
            meta={
                "source": "causal_keypoint_role_bridge",
                "tracked": True,
                "tracker": "lk_forward_backward",
                "track_id": int(track.track_id),
                "track_hits": int(track.hits),
                "track_age": int(track.age),
                "track_velocity_px": float(
                    np.median(np.linalg.norm(target - source, axis=1))
                    / max(scale, 1e-6)
                ),
                "identity_safe": False,
                "proposal_only": True,
                "role_track_only": True,
                "semantic_role": track.role,
                "keypoint_gap": int(track.gap),
                "keypoint_count": int(next_points.shape[0]),
                "keypoint_inlier_ratio": inlier_ratio,
                "keypoint_fb_error": median_fb,
                "keypoint_quality": quality,
            },
        )

    def _gray(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = frame_bgr.shape[:2]
        scale = min(1.0, self.max_width / float(max(1, width)))
        frame = (
            frame_bgr
            if scale >= 0.999
            else cv2.resize(
                frame_bgr,
                (
                    max(1, int(round(width * scale))),
                    max(1, int(round(height * scale))),
                ),
            )
        )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scale

    def _detect_points(
        self,
        gray: np.ndarray,
        box: tuple[float, float, float, float],
    ) -> np.ndarray:
        height, width = gray.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        x1, x2 = max(0, min(width - 1, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height - 1, y1)), max(0, min(height, y2))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return np.empty((0, 1, 2), dtype=np.float32)
        inset_x = max(2, int(round(0.08 * (x2 - x1))))
        inset_y = max(2, int(round(0.08 * (y2 - y1))))
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[y1 + inset_y : y2 - inset_y, x1 + inset_x : x2 - inset_x] = 255
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask,
            blockSize=5,
        )
        return (
            points.astype(np.float32)
            if points is not None
            else np.empty((0, 1, 2), dtype=np.float32)
        )

    @staticmethod
    def _scaled_box(
        box: tuple[float, float, float, float],
        scale: float,
    ) -> tuple[float, float, float, float]:
        return tuple(float(value) * float(scale) for value in box)

    def _allocate_track_id(self) -> int:
        value = self._next_track_id
        self._next_track_id += 1
        return value

    def reset(self) -> None:
        self._previous_gray = None
        self._scale = 1.0
        self._tracks.clear()
        self._next_track_id = 100_000
