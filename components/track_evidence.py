"""Track-level evidence aggregation for procedural state verification."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, List

from ..core_types import Detection, TrackEvidenceFrame


def _slug(text: object) -> str:
    value = str(text or "").strip().lower().replace("-", "_").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value).strip("_")


class TrackEvidenceAggregator:
    """Convert tracked detections into stable object hypotheses.

    The detector remains responsible for proposing object identities. The
    tracker turns those frame-level proposals into temporally grounded object
    hypotheses that can support state verification, review triggers, memory,
    and robot-view evidence export.
    """

    def __init__(
        self,
        enabled: bool = True,
        stable_hits: int = 3,
        stable_confidence: float = 0.35,
        role_track_min_quality: float = 0.50,
        motion_px_threshold: float = 4.0,
        max_tracks: int = 24,
    ) -> None:
        self.enabled = bool(enabled)
        self.stable_hits = max(1, int(stable_hits))
        self.stable_confidence = float(stable_confidence)
        self.role_track_min_quality = float(role_track_min_quality)
        self.motion_px_threshold = max(0.0, float(motion_px_threshold))
        self.max_tracks = max(1, int(max_tracks))

    def build(self, detections: Iterable[Detection]) -> TrackEvidenceFrame:
        """Return a compact summary of currently visible object tracks."""

        if not self.enabled:
            return TrackEvidenceFrame(objects=[], extras={"enabled": False})

        objects = []
        track_counts: Counter[str] = Counter()
        stable_track_counts: Counter[str] = Counter()
        track_ids: dict[str, list[int]] = defaultdict(list)
        track_confidences: dict[str, float] = {}
        track_hits: dict[str, int] = {}
        track_ages: dict[str, int] = {}
        track_motion: dict[str, float] = {}
        evidence_keys = set()

        for detection in list(detections)[: self.max_tracks]:
            meta = dict(detection.meta or {})
            if not bool(meta.get("tracked", False)):
                continue
            name = _slug(detection.name)
            if not name:
                continue
            try:
                track_id = int(meta.get("track_id", -1))
            except (TypeError, ValueError):
                track_id = -1
            if track_id < 0:
                continue
            hits = int(meta.get("track_hits", 1) or 1)
            age = int(meta.get("track_age", hits) or hits)
            confidence = float(detection.confidence)
            motion = float(meta.get("track_velocity_px", meta.get("track_motion_px", 0.0)) or 0.0)
            stable = self._is_stable(meta, hits, confidence)
            key = f"{name}#{track_id}"

            track_counts[name] += 1
            track_ids[name].append(track_id)
            track_confidences[key] = confidence
            track_hits[key] = hits
            track_ages[key] = age
            track_motion[key] = motion
            evidence_keys.add(f"track:object:{name}")
            evidence_keys.add(f"track:id:{name}:{track_id}")
            if stable:
                stable_track_counts[name] += 1
                evidence_keys.add(f"track:stable_object:{name}")
            if motion >= self.motion_px_threshold:
                evidence_keys.add(f"track:motion:{name}")

            objects.append(
                {
                    "name": name,
                    "track_id": track_id,
                    "confidence": confidence,
                    "hits": hits,
                    "age": age,
                    "stable": stable,
                    "velocity_px": motion,
                    "xyxy": [float(value) for value in detection.xyxy],
                    "source": str(meta.get("tracker", "tracker")),
                }
            )

        objects.sort(key=lambda item: (not bool(item["stable"]), str(item["name"]), int(item["track_id"])))
        return TrackEvidenceFrame(
            objects=objects,
            track_counts={key: int(value) for key, value in sorted(track_counts.items())},
            stable_track_counts={key: int(value) for key, value in sorted(stable_track_counts.items())},
            track_ids={key: sorted(int(item) for item in value) for key, value in sorted(track_ids.items())},
            track_confidences={key: float(value) for key, value in sorted(track_confidences.items())},
            track_hits={key: int(value) for key, value in sorted(track_hits.items())},
            track_ages={key: int(value) for key, value in sorted(track_ages.items())},
            track_motion={key: float(value) for key, value in sorted(track_motion.items())},
            evidence_keys=sorted(evidence_keys),
            extras={
                "enabled": True,
                "stable_hits": self.stable_hits,
                "stable_confidence": self.stable_confidence,
                "role_track_min_quality": self.role_track_min_quality,
                "motion_px_threshold": self.motion_px_threshold,
            },
        )

    def stable_detections(self, detections: Iterable[Detection]) -> List[Detection]:
        """Filter detections to tracks stable enough for state rules."""

        stable: List[Detection] = []
        for detection in detections:
            meta = dict(detection.meta or {})
            if not bool(meta.get("tracked", False)):
                continue
            hits = int(meta.get("track_hits", 1) or 1)
            if self._is_stable(meta, hits, float(detection.confidence)):
                stable.append(detection)
        return stable

    def _is_stable(self, meta: dict[str, object], hits: int, confidence: float) -> bool:
        if hits < self.stable_hits:
            return False
        if bool(meta.get("role_track_only", False)):
            quality = float(meta.get("keypoint_quality", 0.0) or 0.0)
            return quality >= self.role_track_min_quality
        return confidence >= self.stable_confidence

    def reset(self) -> None:
        """Reserved reset hook for symmetry with stateful perception modules."""

        return None
