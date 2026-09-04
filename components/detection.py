"""Detection component wrappers and box-level post-processing."""

from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Union

import numpy as np

from ..core_types import Detection
from .keypoint_roles import CausalKeypointRoleBridge


class _UltralyticsYOLOBackend:
    """Small local Ultralytics adapter returning the detector schema used by INSPECT."""

    def __init__(
        self,
        weights: str,
        device: Union[int, str] = "cpu",
        nms_iou: float = 0.4,
        use_builtin_tta: bool = False,
        end2end: Optional[bool] = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on optional runtime package.
            raise RuntimeError(
                "Ultralytics is required for YOLO detection. Install it with `pip install ultralytics`."
            ) from exc

        self.model = YOLO(weights)
        self.device = device
        self.nms_iou = float(nms_iou)
        self.use_builtin_tta = bool(use_builtin_tta)
        self.end2end = end2end

    def detect(self, frame_bgr: np.ndarray, conf: float, tta: bool) -> List[dict[str, object]]:
        kwargs = {
            "source": frame_bgr,
            "conf": float(conf),
            "iou": self.nms_iou,
            "device": self.device,
            "augment": bool(tta or self.use_builtin_tta),
            "verbose": False,
        }
        if self.end2end is not None:
            kwargs["end2end"] = bool(self.end2end)
        results = self.model.predict(**kwargs)
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        names = getattr(result, "names", None) or getattr(self.model, "names", {}) or {}
        detections: List[dict[str, object]] = []
        for box in boxes:
            cls_id = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else -1
            if isinstance(names, dict):
                name = names.get(cls_id, str(cls_id))
            else:
                name = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
            detections.append(
                {
                    "name": str(name),
                    "xyxy": [float(value) for value in box.xyxy[0].tolist()],
                    "conf": float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0,
                }
            )
        return detections

    def track(
        self,
        frame_bgr: np.ndarray,
        conf: float,
        tta: bool,
        tracker: str = "botsort.yaml",
        persist: bool = True,
    ) -> List[dict[str, object]]:
        kwargs = {
            "source": frame_bgr,
            "conf": float(conf),
            "iou": self.nms_iou,
            "device": self.device,
            "augment": bool(tta or self.use_builtin_tta),
            "tracker": str(tracker),
            "persist": bool(persist),
            "verbose": False,
        }
        if self.end2end is not None:
            kwargs["end2end"] = bool(self.end2end)
        results = self.model.track(**kwargs)
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        names = getattr(result, "names", None) or getattr(self.model, "names", {}) or {}
        track_ids = getattr(boxes, "id", None)
        detections: List[dict[str, object]] = []
        for index, box in enumerate(boxes):
            cls_id = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else -1
            if isinstance(names, dict):
                name = names.get(cls_id, str(cls_id))
            else:
                name = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
            track_id = -1
            if track_ids is not None:
                try:
                    track_id = int(track_ids[index].item())
                except Exception:
                    track_id = -1
            detections.append(
                {
                    "name": str(name),
                    "xyxy": [float(value) for value in box.xyxy[0].tolist()],
                    "conf": float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0,
                    "track_id": track_id,
                }
            )
        return detections

    def reset_trackers(self) -> None:
        """Best-effort reset for Ultralytics persistent trackers."""

        predictor = getattr(self.model, "predictor", None)
        if predictor is None:
            return
        trackers = getattr(predictor, "trackers", None)
        if trackers:
            for tracker in trackers:
                try:
                    tracker.reset()
                except Exception:
                    continue
        vid_path = getattr(predictor, "vid_path", None)
        if isinstance(vid_path, list):
            predictor.vid_path = [None] * len(vid_path)
        try:
            predictor._feats = None
        except Exception:
            pass


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
    """Thin wrapper around Ultralytics YOLO with live-safe box filtering."""

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
        track_with_model: bool = False,
        tracker_config: str = "botsort.yaml",
        track_class_smoothing_alpha: float = 0.82,
        track_class_switch_margin: float = 0.12,
        identity_commit_conf: float = 0.50,
        identity_commit_track_margin: float = 0.12,
        role_bridge_enabled: bool = False,
        role_bridge_max_gap: int = 120,
        role_bridge_confidence_decay: float = 0.995,
        role_bridge_min_confidence: float = 0.08,
        role_bridge_max_width: int = 960,
        role_bridge_min_points: int = 6,
        role_bridge_fb_error: float = 2.5,
        end2end: Optional[bool] = None,
    ) -> None:
        parsed_device: Union[int, str]
        if isinstance(device, str) and device.isdigit():
            parsed_device = int(device)
        else:
            parsed_device = device
        self.detector = _UltralyticsYOLOBackend(
            weights=weights,
            device=parsed_device,
            nms_iou=nms_iou,
            use_builtin_tta=use_builtin_tta,
            end2end=end2end,
        )
        self.min_box_area_ratio = float(min_box_area_ratio)
        self.max_box_area_ratio = float(max_box_area_ratio)
        self.max_aspect_ratio = float(max_aspect_ratio)
        self.reject_multi_border_boxes = bool(reject_multi_border_boxes)
        self.border_margin_px = max(0, int(border_margin_px))
        self.max_per_class = max(1, int(max_per_class))
        self.dedupe_iou = float(dedupe_iou)
        self.track_with_model = bool(track_with_model)
        self.tracker_config = str(tracker_config or "botsort.yaml")
        self.track_class_smoothing_alpha = float(np.clip(track_class_smoothing_alpha, 0.0, 0.98))
        self.track_class_switch_margin = max(0.0, float(track_class_switch_margin))
        self.identity_commit_conf = float(np.clip(identity_commit_conf, 0.0, 1.0))
        self.identity_commit_track_margin = float(np.clip(identity_commit_track_margin, 0.0, 1.0))
        self._model_track_stats: dict[int, dict[str, object]] = {}
        self.role_bridge = CausalKeypointRoleBridge(
            enabled=role_bridge_enabled,
            max_gap=role_bridge_max_gap,
            confidence_decay=role_bridge_confidence_decay,
            min_confidence=role_bridge_min_confidence,
            max_width=role_bridge_max_width,
            min_points=role_bridge_min_points,
            fb_error=role_bridge_fb_error,
        )

    def detect(self, frame_bgr: np.ndarray, conf: float, tta: bool) -> List[Detection]:
        """Run object detection and convert outputs into typed dataclasses."""

        if self.track_with_model:
            raw = self.detector.track(
                frame_bgr,
                conf=conf,
                tta=tta,
                tracker=self.tracker_config,
                persist=True,
            )
        else:
            raw = self.detector.detect(frame_bgr, conf=conf, tta=tta)
        height, width = frame_bgr.shape[:2]
        detections: List[Detection] = []
        for item in raw:
            box = _clip_box(tuple(float(x) for x in item["xyxy"]), width, height)
            x1, y1, x2, y2 = box
            box_w = max(0.0, x2 - x1)
            box_h = max(0.0, y2 - y1)
            area_ratio = float((box_w * box_h) / max(1.0, float(width * height)))
            raw_name = str(item["name"]).strip().lower()
            track_id = int(item.get("track_id", -1))
            confidence = float(item.get("conf", 0.0))
            name, track_meta = self._update_model_track_stats(
                name=raw_name,
                track_id=track_id,
                box=box,
                confidence=confidence,
            )
            track_margin = float(track_meta.get("track_class_margin", 1.0) or 0.0)
            class_consistent = bool(not track_meta or raw_name == name)
            identity_safe = bool(
                confidence >= self.identity_commit_conf
                and track_margin >= self.identity_commit_track_margin
                and class_consistent
            )
            detections.append(
                Detection(
                    name=name,
                    xyxy=box,
                    confidence=confidence,
                    meta={
                        "area_ratio": area_ratio,
                        "source": "yolo",
                        "raw_name": raw_name,
                        "tracked": bool(self.track_with_model and int(item.get("track_id", -1)) >= 0),
                        "track_id": track_id,
                        "tracker": self.tracker_config if self.track_with_model else "",
                        "identity_safe": identity_safe,
                        "identity_commit_conf": self.identity_commit_conf,
                        "identity_commit_track_margin": self.identity_commit_track_margin,
                        "proposal_only": not identity_safe,
                        **track_meta,
                    },
                )
            )
        filtered = self._filter_detections(detections, width=width, height=height)
        return self.role_bridge.update(frame_bgr, filtered)

    def _update_model_track_stats(
        self,
        name: str,
        track_id: int,
        box: tuple[float, float, float, float],
        confidence: float,
    ) -> tuple[str, dict[str, object]]:
        if not self.track_with_model or track_id < 0:
            return name, {}
        key = int(track_id)
        center = self._center(box)
        previous = self._model_track_stats.get(key)
        if previous is None:
            class_scores = {str(name): float(confidence)}
            self._model_track_stats[key] = {
                "hits": 1,
                "age": 1,
                "center": center,
                "velocity": 0.0,
                "class_scores": class_scores,
                "smoothed_name": str(name),
            }
            return str(name), {
                "track_hits": 1,
                "track_age": 1,
                "track_velocity_px": 0.0,
                "smoothed_name": str(name),
                "class_smoothed": False,
                "class_scores": class_scores,
                "track_class_posterior": 1.0,
                "track_class_margin": 1.0,
            }
        old_center = previous.get("center", center)
        velocity = float(((center[0] - old_center[0]) ** 2 + (center[1] - old_center[1]) ** 2) ** 0.5)
        hits = int(previous.get("hits", 1)) + 1
        age = int(previous.get("age", 1)) + 1
        class_scores = {
            str(class_name): float(score) * self.track_class_smoothing_alpha
            for class_name, score in dict(previous.get("class_scores", {}) or {}).items()
        }
        class_scores[str(name)] = float(class_scores.get(str(name), 0.0)) + float(confidence)
        ranked = sorted(class_scores.items(), key=lambda item: item[1], reverse=True)
        top_name, top_score = ranked[0]
        runner_score = ranked[1][1] if len(ranked) > 1 else 0.0
        score_total = max(1e-8, sum(max(0.0, float(score)) for _, score in ranked))
        top_posterior = float(top_score / score_total)
        class_margin = float((top_score - runner_score) / score_total)
        previous_name = str(previous.get("smoothed_name", name))
        previous_score = float(class_scores.get(previous_name, 0.0))
        if previous_name and top_name != previous_name and top_score - previous_score < self.track_class_switch_margin:
            smoothed_name = previous_name
        elif top_score - runner_score < self.track_class_switch_margin and previous_name:
            smoothed_name = previous_name
        else:
            smoothed_name = top_name
        class_scores = dict(sorted(class_scores.items(), key=lambda item: item[1], reverse=True)[:5])
        self._model_track_stats[key] = {
            "hits": hits,
            "age": age,
            "center": center,
            "velocity": velocity,
            "class_scores": class_scores,
            "smoothed_name": smoothed_name,
        }
        return smoothed_name, {
            "track_hits": hits,
            "track_age": age,
            "track_velocity_px": velocity,
            "smoothed_name": smoothed_name,
            "class_smoothed": bool(smoothed_name != name),
            "class_scores": class_scores,
            "track_class_posterior": top_posterior,
            "track_class_margin": class_margin,
        }

    def reset(self) -> None:
        """Reset temporal detector state at episode or camera-view boundaries."""

        self._model_track_stats.clear()
        self.role_bridge.reset()
        try:
            self.detector.reset_trackers()
        except Exception:
            pass

    @staticmethod
    def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, y2 = box
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)

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
