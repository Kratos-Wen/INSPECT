"""Temporal feature propagation for object-centric endpoint geometry."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def _rectangle_mask(shape: tuple[int, int], bbox: Sequence[float], expansion: float) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = (float(value) for value in bbox)
    dx = max(0.0, x2 - x1) * expansion
    dy = max(0.0, y2 - y1) * expansion
    left = max(0, min(width - 1, int(round(x1 - dx))))
    top = max(0, min(height - 1, int(round(y1 - dy))))
    right = max(left + 1, min(width, int(round(x2 + dx))))
    bottom = max(top + 1, min(height, int(round(y2 + dy))))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:bottom, left:right] = 255
    return mask


def track_features_through_sequence(
    frames_bgr: Sequence[np.ndarray],
    bboxes: Sequence[Sequence[float]],
    *,
    max_corners: int = 1800,
    backward_error: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate initial anchor features and return first/last pixel pairs."""

    if len(frames_bgr) != len(bboxes) or len(frames_bgr) < 2:
        raise ValueError("frames and bboxes must have the same length >= 2")
    first_gray = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
    mask = _rectangle_mask(first_gray.shape, bboxes[0], expansion=-0.03)
    initial = cv2.goodFeaturesToTrack(
        first_gray,
        maxCorners=max_corners,
        qualityLevel=0.003,
        minDistance=3.0,
        mask=mask,
        blockSize=7,
    )
    if initial is None or len(initial) < 12:
        raise ValueError("not enough initial features for sequential tracking")

    origins = initial.reshape(-1, 2)
    current = origins.copy()
    previous_gray = first_gray
    for frame, bbox in zip(frames_bgr[1:], bboxes[1:]):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        forward, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            gray,
            current.reshape(-1, 1, 2).astype(np.float32),
            None,
            winSize=(31, 31),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
        )
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            previous_gray,
            forward,
            None,
            winSize=(31, 31),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
        )
        next_points = forward.reshape(-1, 2)
        backward_points = backward.reshape(-1, 2)
        valid = forward_status.reshape(-1).astype(bool) & backward_status.reshape(-1).astype(bool)
        if forward_error is not None:
            valid &= forward_error.reshape(-1) < 50.0
        valid &= np.linalg.norm(current - backward_points, axis=1) <= backward_error
        region = _rectangle_mask(gray.shape, bbox, expansion=0.12)
        x = np.clip(np.rint(next_points[:, 0]).astype(int), 0, gray.shape[1] - 1)
        y = np.clip(np.rint(next_points[:, 1]).astype(int), 0, gray.shape[0] - 1)
        valid &= region[y, x] > 0
        origins = origins[valid]
        current = next_points[valid]
        if current.shape[0] < 8:
            raise ValueError("sequential tracking retained fewer than eight features")
        previous_gray = gray
    return origins, current
