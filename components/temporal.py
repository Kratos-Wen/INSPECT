"""Temporal fusion and lightweight tracking for live detections."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..types import Detection


def _iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Compute IoU between two XYXY boxes."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
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


class WindowIoUFusion:
    """Confidence-weighted IoU fusion over a sliding window."""

    def __init__(self, window: int = 5, iou_thr: float = 0.5) -> None:
        self.window = max(1, int(window))
        self.iou_thr = float(iou_thr)
        self.history: Deque[List[Detection]] = deque(maxlen=self.window)

    def update(self, detections: List[Detection]) -> List[Detection]:
        """Update the temporal history and return fused detections."""

        self.history.append(list(detections))
        if not self.history:
            return []

        grouped: dict[str, list[Detection]] = {}
        for frame_detections in self.history:
            for detection in frame_detections:
                grouped.setdefault(detection.name, []).append(detection)

        fused: List[Detection] = []
        for name, items in grouped.items():
            used = [False] * len(items)
            for anchor_index, anchor in enumerate(items):
                if used[anchor_index]:
                    continue
                cluster = [anchor]
                used[anchor_index] = True
                for candidate_index in range(anchor_index + 1, len(items)):
                    if used[candidate_index]:
                        continue
                    candidate = items[candidate_index]
                    if _iou(anchor.xyxy, candidate.xyxy) >= self.iou_thr:
                        cluster.append(candidate)
                        used[candidate_index] = True

                weights = np.array([max(1e-6, item.confidence) for item in cluster], dtype=np.float32)
                boxes = np.array([item.xyxy for item in cluster], dtype=np.float32)
                fused_box = tuple(float(x) for x in (boxes * weights[:, None]).sum(axis=0) / weights.sum())
                fused_confidence = float(weights.mean())
                fused.append(Detection(name=name, xyxy=fused_box, confidence=fused_confidence))
        return fused


@dataclass
class _TrackState:
    track_id: int
    name: str
    xyxy: tuple[float, float, float, float]
    confidence: float
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    confirmed: bool = False


class ByteTrackLiteFusion:
    """A lightweight offline-friendly ByteTrack-style tracker for live detections."""

    def __init__(
        self,
        track_high_thresh: float = 0.32,
        track_low_thresh: float = 0.10,
        new_track_thresh: float = 0.40,
        match_iou_thr: float = 0.25,
        lost_buffer: int = 8,
        min_confirmed_hits: int = 2,
        smooth_alpha: float = 0.70,
    ) -> None:
        self.track_high_thresh = float(track_high_thresh)
        self.track_low_thresh = float(track_low_thresh)
        self.new_track_thresh = float(new_track_thresh)
        self.match_iou_thr = float(match_iou_thr)
        self.lost_buffer = max(1, int(lost_buffer))
        self.min_confirmed_hits = max(1, int(min_confirmed_hits))
        self.smooth_alpha = float(np.clip(smooth_alpha, 0.0, 0.98))
        self._tracks: List[_TrackState] = []
        self._next_track_id = 1

    def update(self, detections: List[Detection]) -> List[Detection]:
        """Track detections online and return the currently visible confirmed tracks."""

        for track in self._tracks:
            track.age += 1
            track.time_since_update += 1

        high_detections = [det for det in detections if det.confidence >= self.track_high_thresh]
        low_detections = [
            det for det in detections
            if self.track_low_thresh <= det.confidence < self.track_high_thresh
        ]

        matches, unmatched_track_indices, unmatched_high_indices = self._match(self._tracks, high_detections)
        for track_index, det_index in matches:
            self._update_track(self._tracks[track_index], high_detections[det_index])

        second_stage_tracks = [self._tracks[index] for index in unmatched_track_indices]
        low_matches, still_unmatched_local, _ = self._match(second_stage_tracks, low_detections)
        for local_track_index, det_index in low_matches:
            self._update_track(second_stage_tracks[local_track_index], low_detections[det_index])

        unmatched_track_indices = [unmatched_track_indices[index] for index in still_unmatched_local]
        for det_index in unmatched_high_indices:
            detection = high_detections[det_index]
            if detection.confidence >= self.new_track_thresh:
                self._tracks.append(self._new_track(detection))

        self._tracks = [track for track in self._tracks if track.time_since_update <= self.lost_buffer]
        visible_tracks = [track for track in self._tracks if track.confirmed and track.time_since_update == 0]
        return [self._to_detection(track) for track in visible_tracks]

    def _match(
        self,
        tracks: List[_TrackState],
        detections: List[Detection],
    ) -> tuple[List[tuple[int, int]], List[int], List[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        matches: List[tuple[int, int]] = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_detections = set(range(len(detections)))

        class_names = sorted({track.name for track in tracks} | {det.name for det in detections})
        for class_name in class_names:
            track_indices = [index for index, track in enumerate(tracks) if track.name == class_name]
            detection_indices = [index for index, det in enumerate(detections) if det.name == class_name]
            if not track_indices or not detection_indices:
                continue

            cost_matrix = np.ones((len(track_indices), len(detection_indices)), dtype=np.float32)
            for row, track_index in enumerate(track_indices):
                for col, det_index in enumerate(detection_indices):
                    overlap = _iou(tracks[track_index].xyxy, detections[det_index].xyxy)
                    if overlap >= self.match_iou_thr:
                        cost_matrix[row, col] = 1.0 - overlap
                    else:
                        cost_matrix[row, col] = 1e3

            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            for row, col in zip(row_indices.tolist(), col_indices.tolist()):
                if cost_matrix[row, col] >= 1e2:
                    continue
                track_index = track_indices[row]
                det_index = detection_indices[col]
                matches.append((track_index, det_index))
                unmatched_tracks.discard(track_index)
                unmatched_detections.discard(det_index)

        return matches, sorted(unmatched_tracks), sorted(unmatched_detections)

    def _new_track(self, detection: Detection) -> _TrackState:
        track = _TrackState(
            track_id=self._next_track_id,
            name=detection.name,
            xyxy=detection.xyxy,
            confidence=float(detection.confidence),
            hits=1,
            age=1,
            time_since_update=0,
            confirmed=self.min_confirmed_hits <= 1,
        )
        self._next_track_id += 1
        return track

    def _update_track(self, track: _TrackState, detection: Detection) -> None:
        previous = np.asarray(track.xyxy, dtype=np.float32)
        current = np.asarray(detection.xyxy, dtype=np.float32)
        blended = self.smooth_alpha * previous + (1.0 - self.smooth_alpha) * current
        track.xyxy = tuple(float(value) for value in blended.tolist())
        track.confidence = float(max(detection.confidence, 0.65 * track.confidence + 0.35 * detection.confidence))
        track.hits += 1
        track.time_since_update = 0
        track.confirmed = track.confirmed or track.hits >= self.min_confirmed_hits

    def _to_detection(self, track: _TrackState) -> Detection:
        return Detection(
            name=track.name,
            xyxy=track.xyxy,
            confidence=float(track.confidence),
            meta={
                "track_id": int(track.track_id),
                "track_hits": int(track.hits),
                "track_age": int(track.age),
                "tracked": True,
                "tracker": "bytetrack_lite",
            },
        )
