"""Geometry-aware scene graph construction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..types import Detection, GeometryFrame, SceneGraphFrame, SceneGraphRelation


def _bbox_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def _bbox_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return float(inter / max(1.0, area_a + area_b - inter))


def _horizontal_overlap_ratio(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    ax1, _, ax2, _ = first
    bx1, _, bx2, _ = second
    overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    width = min(max(1.0, ax2 - ax1), max(1.0, bx2 - bx1))
    return float(overlap / width)


def _edge_gap(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    dx = max(ax1 - bx2, bx1 - ax2, 0.0)
    dy = max(ay1 - by2, by1 - ay2, 0.0)
    return float((dx**2 + dy**2) ** 0.5)


def _mean_depth(
    box: tuple[float, float, float, float],
    depth: np.ndarray,
    valid_mask: Optional[np.ndarray],
) -> float:
    height, width = depth.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1 = max(0, min(width - 1, x1))
    x2 = max(x1 + 1, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(y1 + 1, min(height, y2))
    patch = depth[y1:y2, x1:x2]
    if valid_mask is not None:
        patch = patch[valid_mask[y1:y2, x1:x2]]
    if patch.size == 0:
        return float(np.mean(depth))
    return float(np.mean(patch))


@dataclass
class _ObjectSummary:
    index: int
    name: str
    depth: float
    center_x: float
    center_y: float
    top: float
    bottom: float
    box: tuple[float, float, float, float]


class GeometryAwareSceneGraphBuilder:
    """Infer lightweight spatial and support relations from detections plus geometry."""

    RELATION_TYPES = (
        "contacting",
        "supporting",
        "supported_by",
        "in_front_of",
        "behind",
        "left_of",
        "right_of",
        "above",
        "below",
        "overlapping",
    )

    def __init__(
        self,
        depth_margin: float = 0.08,
        contact_pixel_gap: float = 24.0,
        contact_depth_gap: float = 0.12,
        support_vertical_gap: float = 20.0,
        support_overlap_ratio: float = 0.25,
        max_relations: int = 24,
        enabled: bool = True,
    ) -> None:
        self.depth_margin = float(depth_margin)
        self.contact_pixel_gap = float(contact_pixel_gap)
        self.contact_depth_gap = float(contact_depth_gap)
        self.support_vertical_gap = float(support_vertical_gap)
        self.support_overlap_ratio = float(support_overlap_ratio)
        self.max_relations = max(1, int(max_relations))
        self.enabled = bool(enabled)

    def build(
        self,
        detections: List[Detection],
        geometry: GeometryFrame,
        relevant_detections: List[Detection],
        nearest_index: Optional[int],
    ) -> SceneGraphFrame:
        """Build a scene graph for the current frame."""

        if not self.enabled or len(detections) < 2:
            return SceneGraphFrame(relations=[], stats={"num_relations": 0.0}, extras={})

        summaries = self._summaries(detections, geometry)
        relations: List[SceneGraphRelation] = []
        for left_index in range(len(summaries)):
            for right_index in range(left_index + 1, len(summaries)):
                relations.extend(self._pairwise_relations(summaries[left_index], summaries[right_index]))

        relations.sort(key=lambda item: item.score, reverse=True)
        relations = relations[: self.max_relations]
        predicate_counts = Counter(relation.predicate for relation in relations)
        focus_relations = 0
        if nearest_index is not None:
            focus_relations = sum(
                1 for relation in relations if relation.subject_index == nearest_index or relation.object_index == nearest_index
            )
        avg_relation_score = float(np.mean([relation.score for relation in relations])) if relations else 0.0

        stats: Dict[str, float] = {"num_relations": float(len(relations)), "focus_relations": float(focus_relations)}
        for relation_type in self.RELATION_TYPES:
            stats[f"rel_{relation_type}"] = float(predicate_counts.get(relation_type, 0))
        stats["avg_relation_score"] = avg_relation_score

        extras = {
            "relevant_count": len(relevant_detections),
            "nearest_index": nearest_index,
        }
        return SceneGraphFrame(relations=relations, stats=stats, extras=extras)

    def _summaries(self, detections: List[Detection], geometry: GeometryFrame) -> List[_ObjectSummary]:
        valid_mask = geometry.valid_mask
        return [
            _ObjectSummary(
                index=index,
                name=str(detection.name).strip().lower(),
                depth=_mean_depth(detection.xyxy, geometry.depth, valid_mask),
                center_x=_bbox_center(detection.xyxy)[0],
                center_y=_bbox_center(detection.xyxy)[1],
                top=float(detection.xyxy[1]),
                bottom=float(detection.xyxy[3]),
                box=detection.xyxy,
            )
            for index, detection in enumerate(detections)
        ]

    def _pairwise_relations(self, first: _ObjectSummary, second: _ObjectSummary) -> List[SceneGraphRelation]:
        relations: List[SceneGraphRelation] = []
        depth_gap = abs(first.depth - second.depth)
        iou = _bbox_iou(first.box, second.box)
        edge_gap = _edge_gap(first.box, second.box)
        overlap_ratio = _horizontal_overlap_ratio(first.box, second.box)
        dx = first.center_x - second.center_x
        dy = first.center_y - second.center_y

        if depth_gap >= self.depth_margin:
            if first.depth < second.depth:
                relations.append(self._relation(first, "in_front_of", second, min(1.0, depth_gap / (3.0 * self.depth_margin))))
                relations.append(self._relation(second, "behind", first, min(1.0, depth_gap / (3.0 * self.depth_margin))))
            else:
                relations.append(self._relation(second, "in_front_of", first, min(1.0, depth_gap / (3.0 * self.depth_margin))))
                relations.append(self._relation(first, "behind", second, min(1.0, depth_gap / (3.0 * self.depth_margin))))

        horizontal_scale = max(1.0, abs(first.box[2] - first.box[0]), abs(second.box[2] - second.box[0]))
        if abs(dx) >= 0.2 * horizontal_scale:
            score = min(1.0, abs(dx) / max(1.0, 2.0 * horizontal_scale))
            if first.center_x < second.center_x:
                relations.append(self._relation(first, "left_of", second, score))
                relations.append(self._relation(second, "right_of", first, score))
            else:
                relations.append(self._relation(second, "left_of", first, score))
                relations.append(self._relation(first, "right_of", second, score))

        vertical_scale = max(1.0, abs(first.box[3] - first.box[1]), abs(second.box[3] - second.box[1]))
        if abs(dy) >= 0.2 * vertical_scale:
            score = min(1.0, abs(dy) / max(1.0, 2.0 * vertical_scale))
            if first.center_y < second.center_y:
                relations.append(self._relation(first, "above", second, score))
                relations.append(self._relation(second, "below", first, score))
            else:
                relations.append(self._relation(second, "above", first, score))
                relations.append(self._relation(first, "below", second, score))

        if iou >= 0.1:
            relations.append(self._relation(first, "overlapping", second, min(1.0, iou)))
            relations.append(self._relation(second, "overlapping", first, min(1.0, iou)))

        if edge_gap <= self.contact_pixel_gap and depth_gap <= self.contact_depth_gap:
            score = (1.0 - edge_gap / max(1.0, self.contact_pixel_gap)) * (1.0 - depth_gap / max(1e-6, self.contact_depth_gap))
            score = max(0.0, min(1.0, score))
            relations.append(self._relation(first, "contacting", second, score))
            relations.append(self._relation(second, "contacting", first, score))

        lower, upper = (first, second) if first.center_y > second.center_y else (second, first)
        support_gap = abs(upper.bottom - lower.top)
        if (
            overlap_ratio >= self.support_overlap_ratio
            and support_gap <= self.support_vertical_gap
            and depth_gap <= self.contact_depth_gap
        ):
            support_score = (
                overlap_ratio
                * (1.0 - support_gap / max(1.0, self.support_vertical_gap))
                * (1.0 - depth_gap / max(1e-6, self.contact_depth_gap))
            )
            support_score = max(0.0, min(1.0, support_score))
            relations.append(self._relation(lower, "supporting", upper, support_score))
            relations.append(self._relation(upper, "supported_by", lower, support_score))

        return [relation for relation in relations if relation.score > 0.05]

    def _relation(
        self,
        subject: _ObjectSummary,
        predicate: str,
        obj: _ObjectSummary,
        score: float,
    ) -> SceneGraphRelation:
        return SceneGraphRelation(
            subject_index=subject.index,
            object_index=obj.index,
            subject_name=subject.name,
            predicate=predicate,
            object_name=obj.name,
            score=float(score),
            extras={
                "subject_depth": subject.depth,
                "object_depth": obj.depth,
            },
        )
