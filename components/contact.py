"""Hand/tool-object contact estimation for interaction evidence."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..core_types import Detection, GeometryFrame, HandObjectContact, InteractionEvidence, SceneGraphFrame, SegmentationMask


def _name(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _area(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def _intersection(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def _expand(box: Tuple[float, float, float, float], margin: float) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (x1 - margin, y1 - margin, x2 + margin, y2 + margin)


def _edge_distance(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return float((dx * dx + dy * dy) ** 0.5)


def _median_depth(geometry: GeometryFrame, box: Tuple[float, float, float, float]) -> Optional[float]:
    depth = getattr(geometry, "depth", None)
    if depth is None or not isinstance(depth, np.ndarray) or depth.size == 0:
        return None
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = depth[y1:y2, x1:x2]
    crop = crop[np.isfinite(crop)]
    if crop.size == 0:
        return None
    return float(np.median(crop))


class HandObjectContactEstimator:
    """Estimate interaction evidence from detections, geometry, and temporal contact phases."""

    def __init__(
        self,
        enabled: bool = True,
        hand_names: Iterable[str] = ("hand", "left_hand", "right_hand", "glove"),
        tool_names: Iterable[str] = ("screwdriver", "wrench", "pliers", "tool"),
        exclude_object_names: Iterable[str] = ("person", "arm", "hand", "left_hand", "right_hand", "glove"),
        contact_margin_px: float = 18.0,
        near_margin_px: float = 48.0,
        depth_contact_gap: float = 0.12,
        min_contact_score: float = 0.35,
        history: int = 6,
    ) -> None:
        self.enabled = bool(enabled)
        self.hand_names = {_name(item) for item in hand_names}
        self.tool_names = {_name(item) for item in tool_names}
        self.exclude_object_names = {_name(item) for item in exclude_object_names}
        self.contact_margin_px = float(contact_margin_px)
        self.near_margin_px = float(near_margin_px)
        self.depth_contact_gap = float(depth_contact_gap)
        self.min_contact_score = float(min_contact_score)
        self.history = max(2, int(history))
        self._score_history: Dict[Tuple[str, str], Deque[float]] = defaultdict(lambda: deque(maxlen=self.history))

    def estimate(
        self,
        detections: List[Detection],
        geometry: GeometryFrame,
        scene_graph: Optional[SceneGraphFrame] = None,
        relevant_detections: Optional[List[Detection]] = None,
        segmentation_masks: Optional[List[SegmentationMask]] = None,
    ) -> InteractionEvidence:
        if not self.enabled:
            return InteractionEvidence(contacts=[])
        actors = self._actor_indices(detections)
        objects = self._object_indices(detections)
        contacts: List[HandObjectContact] = []
        mask_contacts = self._mask_contacts(segmentation_masks or [])
        for actor_index in actors:
            actor = detections[actor_index]
            for object_index in objects:
                if actor_index == object_index:
                    continue
                obj = detections[object_index]
                contact = self._score_pair(actor_index, actor, object_index, obj, geometry)
                if contact.contact_score >= self.min_contact_score or contact.phase == "release":
                    contacts.append(contact)
        contacts.extend(mask_contacts)

        contacts.sort(key=lambda item: item.contact_score, reverse=True)
        primary_contact = self._primary_contact(contacts)
        active_object = primary_contact.object_name if primary_contact is not None else ""
        contact_phase = primary_contact.phase if primary_contact is not None else "none"
        interaction_target = self._interaction_target(active_object, scene_graph, relevant_detections)
        transition_likelihood = self._transition_likelihood(contacts)
        contact_counts = Counter(contact.object_name for contact in contacts)
        contact_facts = [(contact.hand_name, "contact", contact.object_name) for contact in contacts]
        return InteractionEvidence(
            contacts=contacts,
            active_object=active_object,
            interaction_target=interaction_target,
            contact_phase=contact_phase,
            transition_likelihood=transition_likelihood,
            contact_counts={str(key): int(value) for key, value in contact_counts.items()},
            contact_facts=contact_facts,
            extras={
                "num_actors": len(actors),
                "num_objects": len(objects),
                "contacts": [asdict(contact) for contact in contacts[:8]],
            },
        )

    def _actor_indices(self, detections: List[Detection]) -> List[int]:
        indices = []
        for index, detection in enumerate(detections):
            name = _name(detection.name)
            if name in self.hand_names or name in self.tool_names or "hand" in name or "glove" in name:
                indices.append(index)
        return indices

    def _object_indices(self, detections: List[Detection]) -> List[int]:
        indices = []
        for index, detection in enumerate(detections):
            name = _name(detection.name)
            if name in self.exclude_object_names or name in self.hand_names or name in self.tool_names:
                continue
            indices.append(index)
        return indices

    def _score_pair(
        self,
        hand_index: int,
        hand: Detection,
        object_index: int,
        obj: Detection,
        geometry: GeometryFrame,
    ) -> HandObjectContact:
        hand_box = tuple(float(v) for v in hand.xyxy)
        object_box = tuple(float(v) for v in obj.xyxy)
        inter = _intersection(_expand(hand_box, self.contact_margin_px), object_box)
        object_area = max(1.0, _area(object_box))
        hand_area = max(1.0, _area(hand_box))
        overlap_ratio = float(inter / min(object_area, hand_area))
        distance_px = _edge_distance(hand_box, object_box)
        distance_score = max(0.0, 1.0 - distance_px / max(1.0, self.near_margin_px))

        hand_depth = _median_depth(geometry, hand_box)
        object_depth = _median_depth(geometry, object_box)
        if hand_depth is None or object_depth is None:
            depth_gap = 0.0
            depth_score = 0.5
        else:
            depth_gap = abs(float(hand_depth) - float(object_depth))
            depth_score = max(0.0, 1.0 - depth_gap / max(1e-6, self.depth_contact_gap))

        contact_score = float(min(1.0, 0.52 * overlap_ratio + 0.33 * distance_score + 0.15 * depth_score))
        pair_key = (_name(hand.name), _name(obj.name))
        phase = self._phase(pair_key, contact_score)
        return HandObjectContact(
            hand_name=_name(hand.name),
            object_name=_name(obj.name),
            contact_score=contact_score,
            distance_px=float(distance_px),
            overlap_ratio=float(overlap_ratio),
            depth_gap=float(depth_gap),
            phase=phase,
            hand_index=int(hand_index),
            object_index=int(object_index),
            extras={
                "hand_confidence": float(hand.confidence),
                "object_confidence": float(obj.confidence),
                "distance_score": float(distance_score),
                "depth_score": float(depth_score),
            },
        )

    def _mask_contacts(self, masks: List[SegmentationMask]) -> List[HandObjectContact]:
        if not masks:
            return []
        actors = [mask for mask in masks if _name(mask.name) in self.hand_names or "hand" in _name(mask.name) or "glove" in _name(mask.name)]
        objects = [
            mask
            for mask in masks
            if _name(mask.name) not in self.exclude_object_names
            and _name(mask.name) not in self.hand_names
            and _name(mask.name) not in self.tool_names
        ]
        contacts: List[HandObjectContact] = []
        for actor in actors:
            actor_mask = np.asarray(actor.mask).astype(bool) if actor.mask is not None else None
            for obj in objects:
                obj_mask = np.asarray(obj.mask).astype(bool) if obj.mask is not None else None
                if actor_mask is None or obj_mask is None or actor_mask.shape != obj_mask.shape:
                    continue
                inter = float(np.logical_and(actor_mask, obj_mask).sum())
                actor_area = float(actor_mask.sum())
                obj_area = float(obj_mask.sum())
                if min(actor_area, obj_area) <= 0:
                    continue
                overlap = inter / min(actor_area, obj_area)
                distance = _edge_distance(tuple(float(v) for v in actor.xyxy), tuple(float(v) for v in obj.xyxy))
                distance_score = max(0.0, 1.0 - distance / max(1.0, self.near_margin_px))
                score = float(min(1.0, 0.70 * overlap + 0.30 * distance_score))
                phase = self._phase((_name(actor.name), _name(obj.name)), score)
                if score < self.min_contact_score and phase != "release":
                    continue
                contacts.append(
                    HandObjectContact(
                        hand_name=_name(actor.name),
                        object_name=_name(obj.name),
                        contact_score=score,
                        distance_px=float(distance),
                        overlap_ratio=float(overlap),
                        depth_gap=0.0,
                        phase=phase,
                        hand_index=-1,
                        object_index=-1,
                        source=f"{actor.source}+{obj.source}",
                        extras={
                            "actor_score": float(actor.score),
                            "object_score": float(obj.score),
                            "actor_object_id": actor.object_id,
                            "object_object_id": obj.object_id,
                            "mask_contact": True,
                        },
                    )
                )
        return contacts

    def _primary_contact(self, contacts: List[HandObjectContact]) -> Optional[HandObjectContact]:
        if not contacts:
            return None
        phase_priority = {"release": 3.0, "contact": 2.0, "approach": 1.0, "none": 0.0}
        return max(
            contacts,
            key=lambda item: (
                phase_priority.get(str(item.phase), 0.0),
                float(item.contact_score),
            ),
        )

    def _phase(self, pair_key: Tuple[str, str], contact_score: float) -> str:
        history = self._score_history[pair_key]
        previous = history[-1] if history else 0.0
        history.append(float(contact_score))
        if contact_score >= self.min_contact_score and previous < self.min_contact_score:
            return "approach"
        if contact_score >= self.min_contact_score:
            return "contact"
        if previous >= self.min_contact_score and contact_score < self.min_contact_score:
            return "release"
        return "none"

    @staticmethod
    def _interaction_target(
        active_object: str,
        scene_graph: Optional[SceneGraphFrame],
        relevant_detections: Optional[List[Detection]],
    ) -> str:
        if not active_object:
            return ""
        active = _name(active_object)
        for relation in getattr(scene_graph, "relations", []) if scene_graph is not None else []:
            subject = _name(relation.subject_name)
            obj = _name(relation.object_name)
            if subject == active:
                return f"{subject}:{_name(relation.predicate)}:{obj}"
            if obj == active:
                return f"{subject}:{_name(relation.predicate)}:{obj}"
        if relevant_detections:
            for detection in relevant_detections:
                name = _name(detection.name)
                if name != active:
                    return f"{active}:near:{name}"
        return active

    @staticmethod
    def _transition_likelihood(contacts: List[HandObjectContact]) -> float:
        if not contacts:
            return 0.0
        best = contacts[0]
        phase_bonus = {"approach": 0.15, "contact": 0.25, "release": 0.35}.get(best.phase, 0.0)
        return float(min(1.0, best.contact_score + phase_bonus))

    def reset(self) -> None:
        """Clear temporal contact phases for a new episode or view stream."""

        self._score_history.clear()
