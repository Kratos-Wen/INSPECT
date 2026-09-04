"""Generic scene-level evidence aggregation for procedural verification."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional

from ..core_types import Detection, InteractionEvidence, SceneEvidenceFrame, SceneGraphFrame, TrackEvidenceFrame


def _slug(text: object) -> str:
    value = str(text or "").strip().lower().replace("-", "_").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value).strip("_")


class ProceduralSceneEvidenceTracker:
    """Fuse object, track, interaction, and relation cues into scene evidence.

    This module intentionally does not encode a table, station, fixture, or
    other layout-specific prior. It summarizes what the stream currently
    supports about procedural evidence: visible objects, stable object tracks,
    active/moving objects, object relations, and changes in those relations.
    """

    def __init__(
        self,
        enabled: bool = True,
        relation_change_memory: int = 24,
        max_objects: int = 32,
        max_relations: int = 48,
        max_changes: int = 32,
    ) -> None:
        self.enabled = bool(enabled)
        self.relation_change_memory = max(1, int(relation_change_memory))
        self.max_objects = max(1, int(max_objects))
        self.max_relations = max(1, int(max_relations))
        self.max_changes = max(1, int(max_changes))
        self._previous_relation_keys: set[str] = set()
        self._previous_visible_objects: set[str] = set()
        self._previous_active_objects: set[str] = set()

    def build(
        self,
        *,
        detections: Iterable[Detection],
        scene_graph: SceneGraphFrame,
        track_evidence: TrackEvidenceFrame,
        interaction_evidence: InteractionEvidence,
    ) -> SceneEvidenceFrame:
        """Return a layout-agnostic procedural scene evidence snapshot."""

        if not self.enabled:
            return SceneEvidenceFrame(extras={"enabled": False})

        detection_list = list(detections)
        visible_counts = Counter(_slug(detection.name) for detection in detection_list if _slug(detection.name))
        visible_objects = sorted(visible_counts)

        stable_objects = sorted(
            name for name, count in dict(track_evidence.stable_track_counts).items() if int(count) > 0 and _slug(name)
        )
        moving_objects = sorted(
            {
                key.split(":", 2)[2]
                for key in track_evidence.evidence_keys
                if str(key).startswith("track:motion:") and len(str(key).split(":", 2)) == 3
            }
        )

        active_objects = set()
        active = _slug(interaction_evidence.active_object)
        if active:
            active_objects.add(active)
        for name, count in dict(interaction_evidence.contact_counts).items():
            if int(count) > 0 and _slug(name):
                active_objects.add(_slug(name))
        active_objects_sorted = sorted(active_objects)

        relation_rows = []
        relation_keys = []
        for relation in list(scene_graph.relations)[: self.max_relations]:
            subject = _slug(relation.subject_name)
            predicate = _slug(relation.predicate)
            obj = _slug(relation.object_name)
            if not subject or not predicate or not obj:
                continue
            if subject == obj:
                continue
            key = f"relation:{subject}:{predicate}:{obj}"
            relation_keys.append(key)
            relation_rows.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "score": float(relation.score),
                }
            )

        current_relation_keys = set(relation_keys)
        current_visible_objects = set(visible_objects)
        current_active_objects = set(active_objects_sorted)

        relation_started = sorted(current_relation_keys - self._previous_relation_keys)[: self.max_changes]
        relation_ended = sorted(self._previous_relation_keys - current_relation_keys)[: self.max_changes]
        object_appeared = sorted(current_visible_objects - self._previous_visible_objects)[: self.max_changes]
        object_disappeared = sorted(self._previous_visible_objects - current_visible_objects)[: self.max_changes]
        active_changed = sorted(current_active_objects - self._previous_active_objects)[: self.max_changes]

        relation_change_keys = [
            f"scene:relation_started:{key.split(':', 1)[1]}" for key in relation_started
        ] + [
            f"scene:relation_ended:{key.split(':', 1)[1]}" for key in relation_ended
        ]
        transition_keys = [
            *(f"scene:object_appeared:{name}" for name in object_appeared),
            *(f"scene:object_disappeared:{name}" for name in object_disappeared),
            *(f"scene:active_object_changed:{name}" for name in active_changed),
            *(f"scene:object_moving:{name}" for name in moving_objects),
        ]
        if interaction_evidence.transition_likelihood >= 0.5:
            transition_keys.append("scene:transition_likely")
        phase = _slug(interaction_evidence.contact_phase)
        if phase and phase != "none":
            transition_keys.append(f"scene:contact_phase:{phase}")

        evidence_keys = set()
        evidence_keys.update(f"scene:visible_object:{name}" for name in visible_objects)
        evidence_keys.update(f"scene:stable_object:{name}" for name in stable_objects)
        evidence_keys.update(f"scene:active_object:{name}" for name in active_objects_sorted)
        evidence_keys.update(f"scene:moving_object:{name}" for name in moving_objects)
        evidence_keys.update(relation_keys)
        evidence_keys.update(relation_change_keys)
        evidence_keys.update(transition_keys)

        object_rows = []
        for detection in detection_list[: self.max_objects]:
            name = _slug(detection.name)
            if not name:
                continue
            meta = dict(detection.meta or {})
            object_rows.append(
                {
                    "name": name,
                    "confidence": float(detection.confidence),
                    "xyxy": [float(value) for value in detection.xyxy],
                    "track_id": meta.get("track_id", None),
                    "tracked": bool(meta.get("tracked", False)),
                }
            )

        self._previous_relation_keys = set(list(current_relation_keys)[-self.relation_change_memory :])
        self._previous_visible_objects = current_visible_objects
        self._previous_active_objects = current_active_objects

        return SceneEvidenceFrame(
            evidence_keys=sorted(evidence_keys),
            visible_objects=visible_objects,
            stable_objects=stable_objects,
            active_objects=active_objects_sorted,
            moving_objects=moving_objects,
            relation_keys=sorted(relation_keys),
            relation_change_keys=sorted(relation_change_keys),
            transition_keys=sorted(transition_keys),
            objects=object_rows,
            relations=relation_rows,
            extras={
                "enabled": True,
                "num_visible_objects": len(visible_objects),
                "num_stable_objects": len(stable_objects),
                "num_active_objects": len(active_objects_sorted),
                "num_moving_objects": len(moving_objects),
                "num_relations": len(relation_keys),
                "num_relation_changes": len(relation_change_keys),
            },
        )

    def reset(self) -> None:
        """Reset temporal scene evidence at episode or stream boundaries."""

        self._previous_relation_keys.clear()
        self._previous_visible_objects.clear()
        self._previous_active_objects.clear()
