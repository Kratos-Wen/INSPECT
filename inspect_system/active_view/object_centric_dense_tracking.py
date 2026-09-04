"""Dense temporal flow fallback for low-texture assembly anchors."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def _sample_flow(flow: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = flow.shape[:2]
    x = np.clip(np.rint(points[:, 0]).astype(int), 0, width - 1)
    y = np.clip(np.rint(points[:, 1]).astype(int), 0, height - 1)
    return np.asarray(flow[y, x], dtype=np.float64)


def _inside_bbox(points: np.ndarray, bbox: Sequence[float], expansion: float) -> np.ndarray:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    dx = max(0.0, x2 - x1) * expansion
    dy = max(0.0, y2 - y1) * expansion
    return (
        (points[:, 0] >= x1 - dx)
        & (points[:, 0] <= x2 + dx)
        & (points[:, 1] >= y1 - dy)
        & (points[:, 1] <= y2 + dy)
    )


def track_dense_flow_through_sequence(
    frames_bgr: Sequence[np.ndarray],
    bboxes: Sequence[Sequence[float]],
    *,
    image_scale: float = 0.5,
    grid_step: int = 14,
    backward_error: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    if len(frames_bgr) != len(bboxes) or len(frames_bgr) < 2:
        raise ValueError("frames and bboxes must have the same length >= 2")
    scale = float(image_scale)
    if not 0.1 <= scale <= 1.0:
        raise ValueError("image_scale must be in [0.1, 1.0]")

    scaled_frames = [
        cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        for frame in frames_bgr
    ]
    scaled_boxes = [[float(value) * scale for value in bbox] for bbox in bboxes]
    x1, y1, x2, y2 = scaled_boxes[0]
    step = max(4, int(grid_step))
    xs = np.arange(x1 + step * 0.5, x2 - step * 0.5 + 1e-6, step)
    ys = np.arange(y1 + step * 0.5, y2 - step * 0.5 + 1e-6, step)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("anchor bbox is too small for dense tracking")
    grid_x, grid_y = np.meshgrid(xs, ys)
    origins = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float64)
    current = origins.copy()
    if current.shape[0] < 16:
        raise ValueError("not enough dense anchor samples")

    forward_solver = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    backward_solver = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    for previous, frame, bbox in zip(scaled_frames[:-1], scaled_frames[1:], scaled_boxes[1:]):
        forward = forward_solver.calc(previous, frame, None)
        backward = backward_solver.calc(frame, previous, None)
        delta = _sample_flow(forward, current)
        next_points = current + delta
        reverse = _sample_flow(backward, next_points)
        valid = np.linalg.norm(delta + reverse, axis=1) <= float(backward_error)
        valid &= _inside_bbox(next_points, bbox, expansion=0.18)
        origins = origins[valid]
        current = next_points[valid]
        if current.shape[0] < 12:
            raise ValueError("dense flow retained fewer than twelve samples")
    return origins / scale, current / scale
