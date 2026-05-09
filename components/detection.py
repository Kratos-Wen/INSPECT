"""Detection component wrappers and box-level post-processing."""

from __future__ import annotations

from collections import defaultdict
from typing import List, Union

import numpy as np

from mica_glasses.core.yolo import YOLODetector as LegacyYOLODetector

from ..types import Detection


def _clip_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1 = float(max(0.0, min(x1, max(0, width - 1))))
    x2 = float(max(0.0, min(x2, max(0, width - 1))))
    y1 = float(max(0.0, min(y1, max(0, height - 1))))
    y2 = float(max(0.0, min(y2, max(0, height - 1))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


class YOLODetectorComponent:
    """Thin wrapper around the existing Ultralytics YOLO backend with live-safe box filtering."""

    def __init__(
        self,
        weights: str,
        device: Union[int, str] = "cpu",
        nms_iou: float = 0.4,
        use_builtin_tta: bool = False,
        min_box_area_ratio: float = 0.0002,
        max_box_area_ratio: float = 0.60,
        max_aspect_ratio: float = 6.0,
        reject_multi_border_boxes: bool = True,
        border_margin_px: int = 4,
        max_per_class: int = 2,
        dedupe_iou: float = 0.65,
    ) -> None:
        parsed_device: Union[int, str]
        if isinstance(device, str) and device.isdigit():
            parsed_device = int(device)
        else:
            parsed_device = device
        self.detector = LegacyYOLODetector(
            weights=weights,
            device=parsed_device,
            nms_iou=nms_iou,
            use_builtin_tta=use_builtin_tta,
        )
        self.min_box_area_ratio = float(min_box_area_ratio)
        self.max_box_area_ratio = float(max_box_area_ratio)
        self.max_aspect_ratio = float(max_aspect_ratio)
        self.reject_multi_border_boxes = bool(reject_multi_border_boxes)
        self.border_margin_px = max(0, int(border_margin_px))
        self.max_per_class = max(1, int(max_per_class))
        self.dedupe_iou = float(dedupe_iou)

    def detect(self, frame_bgr: np.ndarray, conf: float, tta: bool) -> List[Detection]:
        """Run object detection and convert outputs into typed dataclasses."""

        raw = self.detector.detect(frame_bgr, conf=conf, tta=tta)
        height, width = frame_bgr.shape[:2]
        detections: List[Detection] = []
        for item in raw:
            box = _clip_box(tuple(float(x) for x in item["xyxy"]), width, height)
            x1, y1, x2, y2 = box
            box_w = max(0.0, x2 - x1)
            box_h = max(0.0, y2 - y1)
            area_ratio = float((box_w * box_h) / max(1.0, float(width * height)))
            detections.append(
                Detection(
                    name=str(item["name"]).strip().lower(),
                    xyxy=box,
                    confidence=float(item.get("conf", 0.0)),
                    meta={
                        "area_ratio": area_ratio,
                        "source": "yolo",
                    },
                )
            )
        return self._filter_detections(detections, width=width, height=height)

    def _filter_detections(self, detections: List[Detection], width: int, height: int) -> List[Detection]:
        if not detections:
            return []

        filtered: List[Detection] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            x1, y1, x2, y2 = detection.xyxy
            box_w = max(0.0, x2 - x1)
            box_h = max(0.0, y2 - y1)
            if box_w <= 1.0 or box_h <= 1.0:
                continue
            area_ratio = float((box_w * box_h) / max(1.0, float(width * height)))
            if area_ratio < self.min_box_area_ratio or area_ratio > self.max_box_area_ratio:
                continue
            aspect_ratio = max(box_w / max(1.0, box_h), box_h / max(1.0, box_w))
            if aspect_ratio > self.max_aspect_ratio:
                continue
            if self.reject_multi_border_boxes and self._touch_count(detection.xyxy, width, height) >= 3:
                continue
            filtered.append(
                Detection(
                    name=detection.name,
                    xyxy=detection.xyxy,
                    confidence=detection.confidence,
                    meta={
                        **detection.meta,
                        "area_ratio": area_ratio,
                        "aspect_ratio": float(aspect_ratio),
                    },
                )
            )

        grouped: dict[str, List[Detection]] = defaultdict(list)
        for item in filtered:
            grouped[item.name].append(item)

        deduped = []
        for class_items in grouped.values():
            kept = self._greedy_dedupe(sorted(class_items, key=lambda item: item.confidence, reverse=True))
            deduped.extend(kept[: self.max_per_class])
        deduped.sort(key=lambda item: item.confidence, reverse=True)
        return deduped

    def _greedy_dedupe(self, detections: List[Detection]) -> List[Detection]:
        kept: List[Detection] = []
        for detection in detections:
            if any(_iou(detection.xyxy, existing.xyxy) >= self.dedupe_iou for existing in kept):
                continue
            kept.append(detection)
        return kept

    def _touch_count(self, box: tuple[float, float, float, float], width: int, height: int) -> int:
        x1, y1, x2, y2 = box
        margin = float(self.border_margin_px)
        count = 0
        if x1 <= margin:
            count += 1
        if y1 <= margin:
            count += 1
        if x2 >= max(0.0, width - 1 - margin):
            count += 1
        if y2 >= max(0.0, height - 1 - margin):
            count += 1
        return count
