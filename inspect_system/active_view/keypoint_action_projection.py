"""Project inverse-camera keypoint motion onto relative lattice action tokens."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np

from .object_centric_evidence_memory import ObjectCentricMotion
from .object_centric_reveal import camera_motion_from_lattice
from .view_lattice import (
    ORBIT_ACTIONS,
    SIX_VIEWS,
    ViewNode,
    direction_token,
)


def _unit(values: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.size != 3 or norm <= 1e-8 or not np.all(np.isfinite(vector)):
        return None
    return vector / norm


def translation_direction(motion: ObjectCentricMotion) -> np.ndarray | None:
    return _unit(
        np.asarray(
            [
                motion.camera_translation_x,
                motion.camera_translation_y,
                motion.camera_translation_z,
            ],
            dtype=np.float64,
        )
    )


def lattice_action_prototypes(
    views: Mapping[str, ViewNode] | None = None,
) -> Dict[str, tuple[np.ndarray, ...]]:
    table = views or SIX_VIEWS
    grouped: Dict[str, list[np.ndarray]] = {action: [] for action in ORBIT_ACTIONS}
    for current_view in table:
        for candidate_view in table:
            if current_view == candidate_view:
                continue
            action = direction_token(current_view, candidate_view, table)
            if action not in grouped:
                continue
            direction = translation_direction(
                camera_motion_from_lattice(
                    current_view,
                    candidate_view,
                    table,
                )
            )
            if direction is not None:
                grouped[action].append(direction)
    return {
        action: tuple(vectors)
        for action, vectors in grouped.items()
        if vectors
    }


@dataclass(frozen=True)
class KeypointActionProjection:
    action: str
    confidence: float
    scores: Dict[str, float]


def project_inverse_camera_motion(
    motion: ObjectCentricMotion,
    *,
    views: Mapping[str, ViewNode] | None = None,
    temperature: float = 0.15,
) -> KeypointActionProjection | None:
    query = translation_direction(motion)
    if query is None:
        return None
    prototypes = lattice_action_prototypes(views)
    scores = {
        action: max(float(np.dot(query, prototype)) for prototype in vectors)
        for action, vectors in prototypes.items()
    }
    if not scores:
        return None
    best_action = max(scores, key=lambda action: (scores[action], action))
    scale = max(1e-6, float(temperature))
    maximum = max(scores.values())
    exponentials = {
        action: math.exp((score - maximum) / scale)
        for action, score in scores.items()
    }
    denominator = sum(exponentials.values())
    confidence = exponentials[best_action] / max(1e-12, denominator)
    return KeypointActionProjection(
        action=best_action,
        confidence=float(confidence),
        scores=scores,
    )
