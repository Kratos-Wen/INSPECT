"""Temporal fusion and lightweight tracking for live detections."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..core_types import Detection


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
                representative = max(cluster, key=lambda item: item.confidence)
                meta = dict(representative.meta or {})
                if any("identity_safe" in (item.meta or {}) for item in cluster):
                    meta["identity_safe"] = all(
                        bool((item.meta or {}).get("identity_safe", False)) for item in cluster
                    )
                    meta["proposal_only"] = not bool(meta["identity_safe"])
                fused.append(
                    Detection(
                        name=name,
                        xyxy=fused_box,
                        confidence=fused_confidence,
                        meta=meta,
                    )
                )
        return fused

    def reset(self) -> None:
        """Clear temporal state at episode or camera-view boundaries."""

        self.history.clear()


class IdentityTemporalFusion:
    """Pass-through temporal fusion used when the detector already tracks objects."""

    def update(self, detections: List[Detection]) -> List[Detection]:
        return list(detections)

    def reset(self) -> None:
        """Identity fusion has no internal state."""

        return None


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
    velocity_px: float = 0.0
    last_detection_confidence: float = 0.0
    last_observed_name: str = ""
    class_scores: dict[str, float] = field(default_factory=dict)


class ByteTrackLiteFusion:
    """Lightweight causal tracker with optional hard-pair identity belief."""

    def __init__(
        self,
        track_high_thresh: float = 0.32,
        track_low_thresh: float = 0.10,
        new_track_thresh: float = 0.40,
        match_iou_thr: float = 0.25,
        lost_buffer: int = 8,
        min_confirmed_hits: int = 2,
        smooth_alpha: float = 0.70,
        identity_groups: Sequence[Sequence[str]] | None = None,
        identity_smoothing_alpha: float = 0.82,
        identity_switch_margin: float = 0.12,
        identity_commit_conf: float = 0.50,
        identity_commit_margin: float = 0.12,
        cross_identity_match_penalty: float = 0.04,
        preserve_current_detections: bool = False,
    ) -> None:
        self.track_high_thresh = float(track_high_thresh)
        self.track_low_thresh = float(track_low_thresh)
        self.new_track_thresh = float(new_track_thresh)
        self.match_iou_thr = float(match_iou_thr)
        self.lost_buffer = max(1, int(lost_buffer))
        self.min_confirmed_hits = max(1, int(min_confirmed_hits))
        self.smooth_alpha = float(np.clip(smooth_alpha, 0.0, 0.98))
        self.identity_smoothing_alpha = float(np.clip(identity_smoothing_alpha, 0.0, 0.98))
        self.identity_switch_margin = max(0.0, float(identity_switch_margin))
        self.identity_commit_conf = float(np.clip(identity_commit_conf, 0.0, 1.0))
        self.identity_commit_margin = float(np.clip(identity_commit_margin, 0.0, 1.0))
        self.cross_identity_match_penalty = max(0.0, float(cross_identity_match_penalty))
        self.preserve_current_detections = bool(preserve_current_detections)
        self._identity_group_by_name: dict[str, str] = {}
        self._identity_members: dict[str, tuple[str, ...]] = {}
        for index, raw_group in enumerate(identity_groups or []):
            members = tuple(dict.fromkeys(str(name).strip().lower() for name in raw_group if str(name).strip()))
            if len(members) < 2:
                continue
            group_id = f"identity_group_{index}"
            self._identity_members[group_id] = members
            for name in members:
                if name in self._identity_group_by_name:
                    raise ValueError(f"Identity class {name!r} occurs in more than one group")
                self._identity_group_by_name[name] = group_id
        self._tracks: List[_TrackState] = []
        self._next_track_id = 1

    def reset(self) -> None:
        """Clear all active tracks for a new episode or discontinuous view."""

        self._tracks.clear()
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

        track_by_detection: dict[int, _TrackState] = {}
        matches, unmatched_track_indices, unmatched_high_indices = self._match(self._tracks, high_detections)
        for track_index, det_index in matches:
            track = self._tracks[track_index]
            detection = high_detections[det_index]
            self._update_track(track, detection)
            track_by_detection[id(detection)] = track

        second_stage_tracks = [self._tracks[index] for index in unmatched_track_indices]
        low_matches, still_unmatched_local, _ = self._match(second_stage_tracks, low_detections)
        for local_track_index, det_index in low_matches:
            track = second_stage_tracks[local_track_index]
            detection = low_detections[det_index]
            self._update_track(track, detection)
            track_by_detection[id(detection)] = track

        unmatched_track_indices = [unmatched_track_indices[index] for index in still_unmatched_local]
        for det_index in unmatched_high_indices:
            detection = high_detections[det_index]
            if detection.confidence >= self.new_track_thresh:
                track = self._new_track(detection)
                self._tracks.append(track)
                track_by_detection[id(detection)] = track

        self._tracks = [track for track in self._tracks if track.time_since_update <= self.lost_buffer]
        if self.preserve_current_detections:
            return [
                self._to_detection(track_by_detection[id(detection)], observation=detection)
                if id(detection) in track_by_detection
                else self._untracked_detection(detection)
                for detection in detections
            ]
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

        identity_keys = sorted(
            {self._identity_key(track.name) for track in tracks}
            | {self._identity_key(det.name) for det in detections}
        )
        for identity_key in identity_keys:
            track_indices = [
                index for index, track in enumerate(tracks)
                if self._identity_key(track.name) == identity_key
            ]
            detection_indices = [
                index for index, det in enumerate(detections)
                if self._identity_key(det.name) == identity_key
            ]
            if not track_indices or not detection_indices:
                continue

            cost_matrix = np.ones((len(track_indices), len(detection_indices)), dtype=np.float32)
            for row, track_index in enumerate(track_indices):
                for col, det_index in enumerate(detection_indices):
                    overlap = _iou(tracks[track_index].xyxy, detections[det_index].xyxy)
                    if overlap >= self.match_iou_thr:
                        identity_penalty = (
                            self.cross_identity_match_penalty
                            if tracks[track_index].name != detections[det_index].name
                            else 0.0
                        )
                        cost_matrix[row, col] = 1.0 - overlap + identity_penalty
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
        group_members = self._group_members(detection.name)
        class_scores = {name: 0.0 for name in group_members}
        class_scores[detection.name] = max(1e-6, float(detection.confidence))
        track = _TrackState(
            track_id=self._next_track_id,
            name=detection.name,
            xyxy=detection.xyxy,
            confidence=float(detection.confidence),
            hits=1,
            age=1,
            time_since_update=0,
            confirmed=self.min_confirmed_hits <= 1,
            velocity_px=0.0,
            last_detection_confidence=float(detection.confidence),
            last_observed_name=str(detection.name),
            class_scores=class_scores,
        )
        self._next_track_id += 1
        return track

    def _update_track(self, track: _TrackState, detection: Detection) -> None:
        old_center = self._center(track.xyxy)
        new_center = self._center(detection.xyxy)
        previous = np.asarray(track.xyxy, dtype=np.float32)
        current = np.asarray(detection.xyxy, dtype=np.float32)
        blended = self.smooth_alpha * previous + (1.0 - self.smooth_alpha) * current
        track.xyxy = tuple(float(value) for value in blended.tolist())
        track.velocity_px = float(((new_center[0] - old_center[0]) ** 2 + (new_center[1] - old_center[1]) ** 2) ** 0.5)
        track.confidence = float(max(detection.confidence, 0.65 * track.confidence + 0.35 * detection.confidence))
        track.last_detection_confidence = float(detection.confidence)
        track.last_observed_name = str(detection.name)
        self._update_identity_belief(track, detection)
        track.hits += 1
        track.time_since_update = 0
        track.confirmed = track.confirmed or track.hits >= self.min_confirmed_hits

    def _to_detection(
        self,
        track: _TrackState,
        observation: Detection | None = None,
    ) -> Detection:
        track_quality = min(1.0, 0.55 * float(track.confidence) + 0.45 * min(1.0, track.hits / max(1.0, self.min_confirmed_hits + 3.0)))
        posterior = self._normalized_class_scores(track)
        ordered = sorted(posterior.items(), key=lambda item: item[1], reverse=True)
        top_score = float(ordered[0][1]) if ordered else 0.0
        top_name = str(ordered[0][0]) if ordered else str(track.name)
        second_score = float(ordered[1][1]) if len(ordered) > 1 else 0.0
        identity_margin = max(0.0, top_score - second_score)
        output_name = str(observation.name) if observation is not None else str(track.name)
        output_confidence = float(observation.confidence) if observation is not None else float(track.confidence)
        output_box = observation.xyxy if observation is not None else track.xyxy
        identity_safe = bool(
            track.last_detection_confidence >= self.identity_commit_conf
            and identity_margin >= self.identity_commit_margin
            and track.hits >= self.min_confirmed_hits
            and output_name == top_name
        )
        return Detection(
            name=output_name,
            xyxy=output_box,
            confidence=output_confidence,
            meta={
                "track_id": int(track.track_id),
                "track_hits": int(track.hits),
                "track_age": int(track.age),
                "track_velocity_px": float(track.velocity_px),
                "track_quality": float(track_quality),
                "last_detection_confidence": float(track.last_detection_confidence),
                "track_confirmed": bool(track.confirmed),
                "tracked": True,
                "tracker": "counterfactual_bytetrack_lite" if self._identity_group_by_name else "bytetrack_lite",
                "raw_name": str(track.last_observed_name or track.name),
                "smoothed_name": str(track.name),
                "track_class_scores": posterior,
                "track_class_margin": float(identity_margin),
                "track_identity_hypothesis": top_name,
                "identity_safe": identity_safe,
                "proposal_only": not identity_safe,
            },
        )

    def _untracked_detection(self, detection: Detection) -> Detection:
        meta = dict(detection.meta or {})
        if detection.name in self._identity_group_by_name:
            meta["identity_safe"] = False
            meta["proposal_only"] = True
        meta.update(
            {
                "tracked": False,
                "track_id": -1,
                "tracker": "counterfactual_bytetrack_lite",
                "raw_name": str(detection.name),
                "smoothed_name": str(detection.name),
                "track_confirmed": False,
            }
        )
        return Detection(
            name=detection.name,
            xyxy=detection.xyxy,
            confidence=float(detection.confidence),
            meta=meta,
        )

    def _update_identity_belief(self, track: _TrackState, detection: Detection) -> None:
        members = self._group_members(track.name)
        if detection.name not in members:
            return
        for name in members:
            previous = float(track.class_scores.get(name, 0.0))
            observation = float(detection.confidence) if name == detection.name else 0.0
            track.class_scores[name] = (
                self.identity_smoothing_alpha * previous
                + (1.0 - self.identity_smoothing_alpha) * observation
            )
        posterior = self._normalized_class_scores(track)
        if not posterior:
            return
        top_name, top_score = max(posterior.items(), key=lambda item: item[1])
        current_score = float(posterior.get(track.name, 0.0))
        if top_name == track.name or top_score - current_score >= self.identity_switch_margin:
            track.name = str(top_name)

    def _identity_key(self, name: str) -> str:
        normalized = str(name).strip().lower()
        return self._identity_group_by_name.get(normalized, f"class:{normalized}")

    def _group_members(self, name: str) -> tuple[str, ...]:
        normalized = str(name).strip().lower()
        group_id = self._identity_group_by_name.get(normalized)
        return self._identity_members.get(group_id, (normalized,))

    @staticmethod
    def _normalized_class_scores(track: _TrackState) -> dict[str, float]:
        total = sum(max(0.0, float(value)) for value in track.class_scores.values())
        if total <= 1e-8:
            return {str(track.name): 1.0}
        return {
            str(name): float(max(0.0, score) / total)
            for name, score in track.class_scores.items()
        }

    @staticmethod
    def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, y2 = box
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)
