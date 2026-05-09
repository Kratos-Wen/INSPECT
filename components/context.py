"""Geometry-guided context selection."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from ..types import Detection, GeometryFrame


def _center_of(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def _depth_at(depth_map: np.ndarray, valid_mask: np.ndarray | None, cx: float, cy: float) -> float:
    height, width = depth_map.shape[:2]
    x = int(np.clip(cx, 0, width - 1))
    y = int(np.clip(cy, 0, height - 1))
    if valid_mask is not None and not bool(valid_mask[y, x]):
        return float("inf")
    return float(depth_map[y, x])


class DepthContextSelector:
    """Select local object context around the nearest component."""

    def __init__(self, tau_p: float = 80.0, tau_d: float = 0.12) -> None:
        self.tau_p = float(tau_p)
        self.tau_d = float(tau_d)

    def select(
        self,
        detections: List[Detection],
        geometry: GeometryFrame,
    ) -> Tuple[List[Detection], Optional[int]]:
        """Return relevant detections and the nearest detection index."""

        if not detections:
            return [], None

        depth_map = geometry.depth
        valid_mask = geometry.valid_mask
        enriched: list[tuple[int, float, float, float]] = []
        for index, detection in enumerate(detections):
            cx, cy = _center_of(detection.xyxy)
            enriched.append((index, cx, cy, _depth_at(depth_map, valid_mask, cx, cy)))

        nearest_index, nearest_x, nearest_y, nearest_depth = min(enriched, key=lambda item: item[3])
        relevant: List[Detection] = []
        for index, cx, cy, depth_value in enriched:
            if math.hypot(cx - nearest_x, cy - nearest_y) <= self.tau_p and abs(depth_value - nearest_depth) <= self.tau_d:
                relevant.append(detections[index])
        return relevant, nearest_index
