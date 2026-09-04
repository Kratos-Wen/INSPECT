"""Geometry-aware scene graph construction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..core_types import Detection, GeometryFrame, SceneGraphFrame, SceneGraphRelation


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


def _bbox_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return float(max(1.0, x2 - x1) * max(1.0, y2 - y1))


def _intersection_area(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    return float(max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1))


def _center_inside(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> bool:
    cx, cy = _bbox_center(inner)
    x1, y1, x2, y2 = outer
    return bool(x1 <= cx <= x2 and y1 <= cy <= y2)


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


@dataclass(frozen=True)
class _DepthSummary:
    value: float
    quality: float
    valid_fraction: float
    relative_mad: float


def _robust_depth_summary(
    box: tuple[float, float, float, float],
    depth: np.ndarray,
    valid_mask: Optional[np.ndarray],
    confidence: Optional[np.ndarray],
    *,
    inner_ratio: float,
    max_relative_mad: float,
) -> _DepthSummary:
    height, width = depth.shape[:2]
    x1f, y1f, x2f, y2f = box
    inset_x = max(0.0, (x2f - x1f) * inner_ratio)
    inset_y = max(0.0, (y2f - y1f) * inner_ratio)
    x1, y1, x2, y2 = [int(round(value)) for value in (x1f + inset_x, y1f + inset_y, x2f - inset_x, y2f - inset_y)]
    x1 = max(0, min(width - 1, x1))
    x2 = max(x1 + 1, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(y1 + 1, min(height, y2))
    patch = np.asarray(depth[y1:y2, x1:x2], dtype=np.float32)
    finite = np.isfinite(patch) & (patch > 0.0)
    if valid_mask is not None:
        finite &= np.asarray(valid_mask[y1:y2, x1:x2]).astype(bool)
    valid_fraction = float(np.mean(finite)) if finite.size else 0.0
    values = patch[finite]
    if values.size == 0:
        global_values = np.asarray(depth, dtype=np.float32)
        global_values = global_values[np.isfinite(global_values) & (global_values > 0.0)]
        fallback = float(np.median(global_values)) if global_values.size else 0.0
        return _DepthSummary(fallback, 0.0, 0.0, 1.0)

    value = float(np.median(values))
    mad = float(np.median(np.abs(values - value)))
    relative_mad = mad / max(1e-6, abs(value))
    dispersion_quality = float(np.clip(1.0 - relative_mad / max(1e-6, max_relative_mad), 0.0, 1.0))
    confidence_quality = 1.0
    if confidence is not None and np.asarray(confidence).shape[:2] == depth.shape[:2]:
        confidence_values = np.asarray(confidence[y1:y2, x1:x2], dtype=np.float32)[finite]
        confidence_values = confidence_values[np.isfinite(confidence_values)]
        if confidence_values.size:
            median_confidence = max(0.0, float(np.median(confidence_values)))
            confidence_quality = median_confidence if median_confidence <= 1.0 else median_confidence / (1.0 + median_confidence)
    quality = float(np.clip(valid_fraction * dispersion_quality * confidence_quality, 0.0, 1.0))
    return _DepthSummary(value, quality, valid_fraction, relative_mad)


def _box_visibility(box: tuple[float, float, float, float], width: int, height: int) -> float:
    x1, y1, x2, y2 = box
    touches = int(x1 <= 1.0) + int(y1 <= 1.0) + int(x2 >= width - 1.0) + int(y2 >= height - 1.0)
    return float(1.0 if touches == 0 else 0.75 if touches == 1 else 0.50)


_HARD_PAIR_ROLE = {
    "type_3_gear": "small_gear",
    "type_7_gear": "small_gear",
    "type_2_gear": "big_gear",
    "type_8_gear": "big_gear",
    "type_5_gearbox_cover": "cover",
    "type_6_gearbox_cover": "cover",
    "type_5_gearbox_housing": "housing",
    "type_6_gearbox_housing": "housing",
}


def _identity_safe(detection: Detection) -> bool:
    meta = detection.meta if isinstance(detection.meta, dict) else {}
    return bool(meta.get("identity_safe", False))


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
    depth_quality: float
    depth_valid_fraction: float
    depth_relative_mad: float
    box_visibility: float


class GeometryAwareSceneGraphBuilder:
    """Infer lightweight spatial and support relations from detections plus geometry."""

    RELATION_TYPES = (
        "near",
        "contacting",
        "supporting",
        "supported_by",
        "inside",
        "aligned_with",
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
        visibility_calibration_enabled: bool = True,
        depth_margin: float = 0.08,
        depth_inner_ratio: float = 0.12,
        min_depth_valid_fraction: float = 0.35,
        max_relative_depth_mad: float = 0.20,
        contact_pixel_gap: float = 24.0,
        contact_depth_gap: float = 0.12,
        support_vertical_gap: float = 20.0,
        support_overlap_ratio: float = 0.25,
        hard_pair_dedupe_iou: float = 0.85,
        min_relation_score: float = 0.08,
        max_relations: int = 24,
        enabled: bool = True,
    ) -> None:
        self.visibility_calibration_enabled = bool(visibility_calibration_enabled)
        self.depth_margin = float(depth_margin)
        self.depth_inner_ratio = float(np.clip(depth_inner_ratio, 0.0, 0.40))
        self.min_depth_valid_fraction = float(np.clip(min_depth_valid_fraction, 0.0, 1.0))
        self.max_relative_depth_mad = max(1e-6, float(max_relative_depth_mad))
        self.contact_pixel_gap = float(contact_pixel_gap)
        self.contact_depth_gap = float(contact_depth_gap)
        self.support_vertical_gap = float(support_vertical_gap)
        self.support_overlap_ratio = float(support_overlap_ratio)
        self.hard_pair_dedupe_iou = float(np.clip(hard_pair_dedupe_iou, 0.0, 1.0))
        self.min_relation_score = max(0.0, float(min_relation_score))
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

        scene_detections, collapsed = self._collapse_hard_pair_nodes(detections)
        if len(scene_detections) < 2:
            return SceneGraphFrame(
                relations=[],
                stats={"num_relations": 0.0, "collapsed_hard_pair_nodes": float(collapsed)},
                extras={"collapsed_hard_pair_nodes": collapsed},
            )
        summaries = self._summaries(scene_detections, geometry)
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
        avg_depth_quality = float(np.mean([summary.depth_quality for summary in summaries])) if summaries else 0.0

        stats: Dict[str, float] = {
            "num_relations": float(len(relations)),
            "focus_relations": float(focus_relations),
            "collapsed_hard_pair_nodes": float(collapsed),
        }
        for relation_type in self.RELATION_TYPES:
            stats[f"rel_{relation_type}"] = float(predicate_counts.get(relation_type, 0))
        stats["avg_relation_score"] = avg_relation_score
        stats["avg_object_depth_quality"] = avg_depth_quality
        stats["low_depth_quality_objects"] = float(
            sum(summary.depth_valid_fraction < self.min_depth_valid_fraction for summary in summaries)
        )

        extras = {
            "relevant_count": len(relevant_detections),
            "nearest_index": nearest_index,
            "collapsed_hard_pair_nodes": collapsed,
            "visibility_calibration_enabled": self.visibility_calibration_enabled,
        }
        return SceneGraphFrame(relations=relations, stats=stats, extras=extras)

    def _collapse_hard_pair_nodes(self, detections: List[Detection]) -> tuple[List[Detection], int]:
        """Collapse duplicate class hypotheses for one physical role-level node.

        Raw class alternatives remain available to the claim verifier. Only the
        spatial scene graph is deduplicated, so overlapping hard-pair logits do
        not become fictitious object-object relations.
        """

        ranked = sorted(
            enumerate(detections),
            key=lambda item: (_identity_safe(item[1]), float(item[1].confidence)),
            reverse=True,
        )
        kept: List[tuple[int, Detection]] = []
        collapsed = 0
        for original_index, detection in ranked:
            role = _HARD_PAIR_ROLE.get(str(detection.name).strip().lower(), "")
            duplicate = False
            if role:
                for _, existing in kept:
                    existing_role = _HARD_PAIR_ROLE.get(str(existing.name).strip().lower(), "")
                    if existing_role == role and _bbox_iou(detection.xyxy, existing.xyxy) >= self.hard_pair_dedupe_iou:
                        duplicate = True
                        break
            if duplicate:
                collapsed += 1
                continue
            kept.append((original_index, detection))

        kept.sort(key=lambda item: item[0])
        output: List[Detection] = []
        for original_index, detection in kept:
            output.append(
                Detection(
                    name=detection.name,
                    xyxy=detection.xyxy,
                    confidence=detection.confidence,
                    meta={**dict(detection.meta), "scene_original_index": original_index},
                )
            )
        return output, collapsed

    def _summaries(self, detections: List[Detection], geometry: GeometryFrame) -> List[_ObjectSummary]:
        valid_mask = geometry.valid_mask
        height, width = geometry.depth.shape[:2]
        summaries = []
        for index, detection in enumerate(detections):
            depth = _robust_depth_summary(
                detection.xyxy,
                geometry.depth,
                valid_mask,
                geometry.confidence,
                inner_ratio=self.depth_inner_ratio,
                max_relative_mad=self.max_relative_depth_mad,
            )
            summaries.append(_ObjectSummary(
                index=int(detection.meta.get("scene_original_index", index)),
                name=str(detection.name).strip().lower(),
                depth=depth.value,
                center_x=_bbox_center(detection.xyxy)[0],
                center_y=_bbox_center(detection.xyxy)[1],
                top=float(detection.xyxy[1]),
                bottom=float(detection.xyxy[3]),
                box=detection.xyxy,
                depth_quality=depth.quality,
                depth_valid_fraction=depth.valid_fraction,
                depth_relative_mad=depth.relative_mad,
                box_visibility=_box_visibility(detection.xyxy, width, height),
            ))
        return summaries

    def _pairwise_relations(self, first: _ObjectSummary, second: _ObjectSummary) -> List[SceneGraphRelation]:
        relations: List[SceneGraphRelation] = []
        depth_gap = abs(first.depth - second.depth)
        iou = _bbox_iou(first.box, second.box)
        edge_gap = _edge_gap(first.box, second.box)
        overlap_ratio = _horizontal_overlap_ratio(first.box, second.box)
        dx = first.center_x - second.center_x
        dy = first.center_y - second.center_y
        first_area = _bbox_area(first.box)
        second_area = _bbox_area(second.box)
        min_area = max(1.0, min(first_area, second_area))
        intersection = _intersection_area(first.box, second.box)
        containment = float(intersection / min_area)
        near_gap = max(self.contact_pixel_gap * 3.0, 64.0)
        visibility_quality = float((first.box_visibility * second.box_visibility) ** 0.5)
        depth_quality = float((first.depth_quality * second.depth_quality) ** 0.5)
        if not self.visibility_calibration_enabled:
            visibility_quality = 1.0
            depth_quality = 1.0

        if edge_gap <= near_gap:
            score = (1.0 - edge_gap / max(1.0, near_gap)) * visibility_quality
            relations.append(self._relation(first, "near", second, score))
            relations.append(self._relation(second, "near", first, score))

        center_distance = float((dx * dx + dy * dy) ** 0.5)
        alignment_scale = max(1.0, min(abs(first.box[2] - first.box[0]), abs(second.box[2] - second.box[0])))
        if containment >= 0.18 and center_distance <= 0.55 * alignment_scale:
            score = max(0.0, min(1.0, 0.5 * containment + 0.5 * (1.0 - center_distance / max(1.0, 0.55 * alignment_scale)))) * visibility_quality
            relations.append(self._relation(first, "aligned_with", second, score))
            relations.append(self._relation(second, "aligned_with", first, score))

        if first_area <= second_area and _center_inside(first.box, second.box) and containment >= 0.35:
            score = max(0.0, min(1.0, containment)) * visibility_quality
            relations.append(self._relation(first, "inside", second, score))
        if second_area <= first_area and _center_inside(second.box, first.box) and containment >= 0.35:
            score = max(0.0, min(1.0, containment)) * visibility_quality
            relations.append(self._relation(second, "inside", first, score))

        depth_reliable = bool(
            not self.visibility_calibration_enabled
            or (
                first.depth_valid_fraction >= self.min_depth_valid_fraction
                and second.depth_valid_fraction >= self.min_depth_valid_fraction
            )
        )
        if depth_reliable and depth_gap >= self.depth_margin:
            if first.depth < second.depth:
                relations.append(self._relation(first, "in_front_of", second, min(1.0, depth_gap / (3.0 * self.depth_margin)) * depth_quality))
                relations.append(self._relation(second, "behind", first, min(1.0, depth_gap / (3.0 * self.depth_margin)) * depth_quality))
            else:
                relations.append(self._relation(second, "in_front_of", first, min(1.0, depth_gap / (3.0 * self.depth_margin)) * depth_quality))
                relations.append(self._relation(first, "behind", second, min(1.0, depth_gap / (3.0 * self.depth_margin)) * depth_quality))

        horizontal_scale = max(1.0, abs(first.box[2] - first.box[0]), abs(second.box[2] - second.box[0]))
        if abs(dx) >= 0.2 * horizontal_scale:
            score = min(1.0, abs(dx) / max(1.0, 2.0 * horizontal_scale)) * visibility_quality
            if first.center_x < second.center_x:
                relations.append(self._relation(first, "left_of", second, score))
                relations.append(self._relation(second, "right_of", first, score))
            else:
                relations.append(self._relation(second, "left_of", first, score))
                relations.append(self._relation(first, "right_of", second, score))

        vertical_scale = max(1.0, abs(first.box[3] - first.box[1]), abs(second.box[3] - second.box[1]))
        if abs(dy) >= 0.2 * vertical_scale:
            score = min(1.0, abs(dy) / max(1.0, 2.0 * vertical_scale)) * visibility_quality
            if first.center_y < second.center_y:
                relations.append(self._relation(first, "above", second, score))
                relations.append(self._relation(second, "below", first, score))
            else:
                relations.append(self._relation(second, "above", first, score))
                relations.append(self._relation(first, "below", second, score))

        if iou >= 0.1:
            relations.append(self._relation(first, "overlapping", second, min(1.0, iou) * visibility_quality))
            relations.append(self._relation(second, "overlapping", first, min(1.0, iou) * visibility_quality))

        if depth_reliable and edge_gap <= self.contact_pixel_gap and depth_gap <= self.contact_depth_gap:
            score = (1.0 - edge_gap / max(1.0, self.contact_pixel_gap)) * (1.0 - depth_gap / max(1e-6, self.contact_depth_gap))
            score = max(0.0, min(1.0, score)) * depth_quality
            relations.append(self._relation(first, "contacting", second, score))
            relations.append(self._relation(second, "contacting", first, score))

        lower, upper = (first, second) if first.center_y > second.center_y else (second, first)
        support_gap = abs(upper.bottom - lower.top)
        if (
            overlap_ratio >= self.support_overlap_ratio
            and support_gap <= self.support_vertical_gap
            and depth_reliable
            and depth_gap <= self.contact_depth_gap
        ):
            support_score = (
                overlap_ratio
                * (1.0 - support_gap / max(1.0, self.support_vertical_gap))
                * (1.0 - depth_gap / max(1e-6, self.contact_depth_gap))
            )
            support_score = max(0.0, min(1.0, support_score)) * depth_quality
            relations.append(self._relation(lower, "supporting", upper, support_score))
            relations.append(self._relation(upper, "supported_by", lower, support_score))

        return [relation for relation in relations if relation.score >= self.min_relation_score]

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
                "subject_depth_quality": subject.depth_quality,
                "object_depth_quality": obj.depth_quality,
                "subject_depth_valid_fraction": subject.depth_valid_fraction,
                "object_depth_valid_fraction": obj.depth_valid_fraction,
                "subject_box_visibility": subject.box_visibility,
                "object_box_visibility": obj.box_visibility,
            },
        )
