"""Feature encoding for structured episodic memory."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Dict, Iterable, List, Optional

import numpy as np

from ..types import Detection, GeometryFrame, SceneGraphFrame, StepPrediction
from .types import MemoryObservation, Signature


def detection_signature(detections: Iterable[Detection]) -> Signature:
    """Return a canonical count signature for a detection list."""

    counts = Counter(str(detection.name).strip().lower() for detection in detections)
    return tuple(sorted((name, int(count)) for name, count in counts.items() if name))


def relation_signature(scene_graph: SceneGraphFrame | None) -> Signature:
    """Return a sparse relation signature from the current scene graph."""

    if scene_graph is None:
        return ()
    counts = Counter(
        f"{relation.subject_name}|{relation.predicate}|{relation.object_name}"
        for relation in scene_graph.relations
    )
    return tuple(sorted((token, int(count)) for token, count in counts.items() if token))


def summarize_geometry(
    geometry: GeometryFrame,
    detections: List[Detection],
    relevant_detections: List[Detection],
    nearest_index: Optional[int],
) -> Dict[str, float]:
    """Compress dense geometry into a small retrieval-friendly descriptor."""

    depth = geometry.depth
    if depth.size == 0:
        return {
            "valid_ratio": 0.0,
            "depth_mean": 0.0,
            "depth_std": 0.0,
            "nearest_depth": 0.0,
            "num_detections": float(len(detections)),
            "num_relevant": float(len(relevant_detections)),
            "nearest_area": 0.0,
        }

    valid_mask = geometry.valid_mask
    if valid_mask is None:
        valid_mask = np.ones(depth.shape[:2], dtype=bool)
    valid_mask = valid_mask.astype(bool)
    valid_depth = depth[valid_mask]
    if valid_depth.size == 0:
        valid_depth = depth.reshape(-1)

    depth_mean = float(np.mean(valid_depth))
    depth_std = float(np.std(valid_depth))
    nearest_depth = depth_mean
    nearest_area = 0.0

    if nearest_index is not None and 0 <= nearest_index < len(detections):
        detection = detections[nearest_index]
        height, width = depth.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in detection.xyxy]
        x1 = max(0, min(width - 1, x1))
        x2 = max(x1 + 1, min(width, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(y1 + 1, min(height, y2))
        crop_depth = depth[y1:y2, x1:x2]
        crop_mask = valid_mask[y1:y2, x1:x2]
        crop_valid = crop_depth[crop_mask]
        if crop_valid.size:
            nearest_depth = float(np.mean(crop_valid))
        nearest_area = float((x2 - x1) * (y2 - y1) / max(1, width * height))

    return {
        "valid_ratio": float(np.mean(valid_mask)),
        "depth_mean": depth_mean,
        "depth_std": depth_std,
        "nearest_depth": nearest_depth,
        "num_detections": float(len(detections)),
        "num_relevant": float(len(relevant_detections)),
        "nearest_area": nearest_area,
    }


def summarize_step_consensus(
    expert_predictions: Dict[str, StepPrediction],
    steps: List[str],
) -> Dict[str, object]:
    """Aggregate expert outputs into a lightweight pre-fusion consensus summary."""

    normalized_steps = [str(step).strip().upper() for step in steps]
    raw_scores: Dict[str, float] = {step_id: 0.0 for step_id in normalized_steps}
    if not expert_predictions:
        return {
            "step_id": normalized_steps[0] if normalized_steps else "",
            "confidence": 0.0,
            "margin": 0.0,
            "disagreement": 0.0,
            "scores": raw_scores,
        }

    dense_vectors: Dict[str, np.ndarray] = {}
    for expert_name, prediction in expert_predictions.items():
        vector = np.array([float(prediction.scores.get(step_id, 0.0)) for step_id in normalized_steps], dtype=np.float32)
        dense_vectors[expert_name] = vector
        for index, step_id in enumerate(normalized_steps):
            raw_scores[step_id] += float(vector[index])

    for step_id in normalized_steps:
        raw_scores[step_id] /= float(len(expert_predictions))

    ordered = sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)
    top_step = ordered[0][0] if ordered else (normalized_steps[0] if normalized_steps else "")
    margin = float(ordered[0][1] - ordered[1][1]) if len(ordered) > 1 else 0.0

    values = np.array([score for _, score in ordered], dtype=np.float32)
    if values.size > 0:
        values = values - values.max()
        probs = np.exp(values)
        probs = probs / max(1e-6, float(probs.sum()))
        confidence = float(probs[0])
    else:
        confidence = 0.0

    disagreement = 0.0
    if len(dense_vectors) > 1:
        pairwise = []
        for first, second in combinations(dense_vectors.values(), 2):
            pairwise.append(float(np.mean(np.abs(first - second))))
        disagreement = float(np.mean(pairwise)) if pairwise else 0.0

    return {
        "step_id": top_step,
        "confidence": confidence,
        "margin": margin,
        "disagreement": disagreement,
        "scores": raw_scores,
    }


class StructuredMemoryEncoder:
    """Encode observations into numeric vectors and sparse tokens."""

    GEOMETRY_KEYS = (
        "valid_ratio",
        "depth_mean",
        "depth_std",
        "nearest_depth",
        "num_detections",
        "num_relevant",
        "nearest_area",
    )
    RELATION_TYPES = (
        "contacting",
        "supporting",
        "supported_by",
        "in_front_of",
        "behind",
        "left_of",
        "right_of",
        "above",
        "below",
        "overlapping",
    )

    def __init__(self, steps: List[str], component_names: List[str], expert_names: List[str]) -> None:
        self.steps = [str(step).strip().upper() for step in steps]
        self.component_names = [str(name).strip().lower() for name in component_names if str(name).strip()]
        self.expert_names = [str(name).strip().lower() for name in expert_names if str(name).strip()]

    def encode(self, observation: MemoryObservation) -> np.ndarray:
        """Encode one observation into a normalized feature vector."""

        features: List[float] = []
        for expert_name in self.expert_names:
            expert_scores = observation.expert_scores.get(expert_name, {})
            for step_id in self.steps:
                features.append(float(expert_scores.get(step_id, 0.0)))
            features.append(float(observation.expert_confidences.get(expert_name, 0.0)))

        signature_counts = Counter(dict(observation.signature))
        relevant_counts = Counter(dict(observation.relevant_signature))
        relation_counts = Counter(
            token.split("|", 2)[1]
            for token, count in observation.relation_signature
            for _ in range(max(0, int(count)))
            if "|" in token
        )
        total_signature = max(1, sum(signature_counts.values()))
        total_relevant = max(1, sum(relevant_counts.values()))
        for component_name in self.component_names:
            features.append(float(signature_counts.get(component_name, 0)) / float(total_signature))
        for component_name in self.component_names:
            features.append(float(relevant_counts.get(component_name, 0)) / float(total_relevant))
        total_relations = max(1, sum(relation_counts.values()))
        for relation_type in self.RELATION_TYPES:
            features.append(float(relation_counts.get(relation_type, 0)) / float(total_relations))

        for key in self.GEOMETRY_KEYS:
            features.append(float(observation.geometry_stats.get(key, 0.0)))
        features.append(float(observation.scene_graph_stats.get("num_relations", 0.0)))
        features.append(float(observation.scene_graph_stats.get("rel_contacting", 0.0)))
        features.append(float(observation.scene_graph_stats.get("rel_supporting", 0.0)))
        features.append(float(observation.scene_graph_stats.get("rel_supported_by", 0.0)))
        features.append(float(observation.scene_graph_stats.get("avg_relation_score", 0.0)))

        for step_id in self.steps:
            features.append(1.0 if observation.prev_step == step_id else 0.0)

        features.extend(float(value) for value in observation.visual_embedding)

        vector = np.array(features, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector = vector / norm
        return vector

    def tokens(self, observation: MemoryObservation) -> List[str]:
        """Extract sparse retrieval tokens from an observation."""

        tokens = []
        for name, count in observation.signature:
            tokens.append(f"sig:{name}")
            tokens.append(f"sig:{name}:{count}")
        for name, count in observation.relevant_signature:
            tokens.append(f"rel:{name}")
            tokens.append(f"rel:{name}:{count}")
        for token, count in observation.relation_signature:
            tokens.append(f"graph:{token}")
            tokens.append(f"graph:{token}:{count}")
        if observation.prev_step:
            tokens.append(f"prev:{observation.prev_step}")
        tokens.append(f"ensemble:{observation.ensemble_step}")
        for expert_name in self.expert_names:
            if expert_name in observation.expert_steps:
                tokens.append(f"expert:{expert_name}:{observation.expert_steps[expert_name]}")
        return sorted(set(tokens))
