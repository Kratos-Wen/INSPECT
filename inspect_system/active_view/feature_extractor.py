"""Dependency-light structured feature helpers for future learned rankers."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

from .ontology import normalize_claim, normalize_key


def bbox_area_ratio(bbox_xyxy: Sequence[float] | None, frame_shape_hw: Sequence[int] | None) -> float:
    if not bbox_xyxy or len(bbox_xyxy) < 4 or not frame_shape_hw or len(frame_shape_hw) < 2:
        return 0.0
    h, w = float(frame_shape_hw[0]), float(frame_shape_hw[1])
    if h <= 0 or w <= 0:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy[:4]]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / (w * h)


def bbox_center_offset(bbox_xyxy: Sequence[float] | None, frame_shape_hw: Sequence[int] | None) -> tuple[float, float]:
    if not bbox_xyxy or len(bbox_xyxy) < 4 or not frame_shape_hw or len(frame_shape_hw) < 2:
        return (0.0, 0.0)
    h, w = float(frame_shape_hw[0]), float(frame_shape_hw[1])
    if h <= 0 or w <= 0:
        return (0.0, 0.0)
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy[:4]]
    return (((x1 + x2) * 0.5 / w) - 0.5, ((y1 + y2) * 0.5 / h) - 0.5)


def structured_features(payload: Mapping[str, object]) -> Dict[str, object]:
    """Normalize common runtime fields into a compact feature dictionary."""

    frame_shape = payload.get("frame_shape_hw") or payload.get("frame_shape") or []
    bbox = payload.get("target_bbox_xyxy") or payload.get("bbox_xyxy") or []
    off_x, off_y = bbox_center_offset(bbox if isinstance(bbox, Sequence) else None, frame_shape if isinstance(frame_shape, Sequence) else None)
    return {
        "claim_id": normalize_claim(payload.get("claim_id", "")),
        "missing_evidence": normalize_key(payload.get("missing_evidence", "")),
        "target_class": normalize_key(payload.get("target_class", "")),
        "target_conf": float(payload.get("target_conf", 0.0) or 0.0),
        "hard_pair_margin": float(payload.get("hard_pair_margin", 0.0) or 0.0),
        "bbox_area_ratio": bbox_area_ratio(bbox if isinstance(bbox, Sequence) else None, frame_shape if isinstance(frame_shape, Sequence) else None),
        "bbox_center_offset_x": off_x,
        "bbox_center_offset_y": off_y,
        "slot_visibility": float(payload.get("slot_visibility", 0.0) or 0.0),
        "relation_visibility": float(payload.get("relation_visibility", 0.0) or 0.0),
        "occlusion_score": float(payload.get("occlusion_score", 0.0) or 0.0),
        "blur_score": float(payload.get("blur_score", 0.0) or 0.0),
    }
