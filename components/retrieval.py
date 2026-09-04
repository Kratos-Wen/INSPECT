"""Retrieval-based step experts built on a shared visual embedding space."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..core_types import RetrievalInput, StepPrediction
from .visual_embedding import SharedVisualEncoder


def _normalize_scores(raw_scores: Dict[str, float]) -> Dict[str, float]:
    values = list(raw_scores.values())
    if not values:
        return {}
    high = max(values)
    low = min(values)
    if abs(high - low) < 1e-6:
        if high <= 0.0:
            return {step_id: 0.0 for step_id in raw_scores}
        return {step_id: 0.5 for step_id in raw_scores}
    return {step_id: float((value - low) / (high - low)) for step_id, value in raw_scores.items()}


@dataclass
class GalleryItem:
    step_id: str
    path: str
    embedding: np.ndarray


class DummyRetrievalExpert:
    """Fallback retrieval expert used when no gallery is available."""

    def __init__(self, steps: List[str]) -> None:
        self.steps = [str(step).strip().upper() for step in steps]

    def predict(self, payload: object) -> StepPrediction:
        scores = {step_id: 0.0 for step_id in self.steps}
        return StepPrediction(
            step_id=self.steps[0],
            confidence=0.0,
            scores=scores,
            extras={"reason": "empty_gallery"},
        )


class GalleryRetrievalExpert:
    """Reference-gallery expert that scores all steps in a shared visual space."""

    def __init__(
        self,
        root: str,
        steps: List[str],
        exts: List[str],
        embed_mode: str = "hybrid-4",
        topk: int = 5,
        encoder: Optional[SharedVisualEncoder] = None,
        negative_weight: float = 0.65,
        negative_margin: float = 0.03,
    ) -> None:
        self.root = Path(root)
        self.steps = [str(step).strip().upper() for step in steps]
        self.exts = {str(ext).lower() for ext in exts}
        self.embed_mode = str(embed_mode).strip().lower()
        self.topk = max(1, int(topk))
        self.encoder = encoder or SharedVisualEncoder(mode=self.embed_mode)
        self.negative_weight = float(negative_weight)
        self.negative_margin = float(negative_margin)
        self.items: List[GalleryItem] = []
        self.negative_items: List[GalleryItem] = []

    def build(self) -> int:
        """Index the gallery and return the number of indexed images."""

        if not self.root.exists():
            raise FileNotFoundError(f"Gallery root not found: {self.root}")
        self.items.clear()
        self.negative_items.clear()
        for step_dir in sorted(self.root.iterdir()):
            if not step_dir.is_dir():
                continue
            step_id = self._parse_step_id(step_dir.name)
            is_negative = step_id is None and self._is_negative_folder(step_dir.name)
            if step_id is None and not is_negative:
                continue
            for path in step_dir.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in self.exts:
                    continue
                image = cv2.imread(str(path))
                if image is None:
                    continue
                embedding = self.encoder.encode_image(image)
                item = GalleryItem(step_id=step_id or "__NEGATIVE__", path=str(path), embedding=embedding)
                if is_negative:
                    self.negative_items.append(item)
                else:
                    self.items.append(item)
        return len(self.items) + len(self.negative_items)

    def predict(self, payload: object) -> StepPrediction:
        """Predict dense step scores from the current frame or structured query."""

        if not self.items:
            return DummyRetrievalExpert(self.steps).predict(payload)

        frame_bgr, focus_detection = self._resolve_query(payload)
        query = self.encoder.encode_query(frame_bgr, focus_detection=focus_detection)
        matrix = np.stack([item.embedding for item in self.items], axis=0)
        similarities = matrix @ query
        negative_score: Optional[float] = None
        if self.negative_items:
            negative_matrix = np.stack([item.embedding for item in self.negative_items], axis=0)
            negative_values = negative_matrix @ query
            negative_score = float(np.max(negative_values)) if negative_values.size else None

        aggregated: Dict[str, List[float]] = {step_id: [] for step_id in self.steps}
        for item, similarity in zip(self.items, similarities):
            aggregated.setdefault(item.step_id, []).append(float(similarity))

        raw_scores: Dict[str, float] = {}
        for step_id in self.steps:
            values = sorted(aggregated.get(step_id, []), reverse=True)
            if not values:
                raw_scores[step_id] = -1.0
                continue
            raw_scores[step_id] = float(np.mean(values[: self.topk]))

        best_positive = max(raw_scores.values()) if raw_scores else -1.0
        invalid_like = False
        if negative_score is not None:
            invalid_like = negative_score >= best_positive + self.negative_margin
            if invalid_like:
                scores = {step_id: 0.0 for step_id in self.steps}
                top_step = self.steps[0]
                return StepPrediction(
                    step_id=top_step,
                    confidence=0.0,
                    scores=scores,
                    extras={
                        "raw_scores": raw_scores,
                        "embed_mode": self.embed_mode,
                        "negative_score": negative_score,
                        "invalid_like": True,
                    },
                )
            for step_id, value in list(raw_scores.items()):
                penalty = max(0.0, negative_score - value + self.negative_margin)
                raw_scores[step_id] = float(value - self.negative_weight * penalty)

        scores = _normalize_scores(raw_scores)
        top_step = max(scores, key=scores.get) if scores else self.steps[0]
        return StepPrediction(
            step_id=top_step,
            confidence=float(scores.get(top_step, 0.0)),
            scores=scores,
            extras={
                "raw_scores": raw_scores,
                "embed_mode": self.embed_mode,
                "negative_score": negative_score,
                "num_negative_items": len(self.negative_items),
                "invalid_like": False,
            },
        )

    def _resolve_query(self, payload: object) -> tuple[np.ndarray, Optional[object]]:
        if isinstance(payload, RetrievalInput):
            return payload.frame_bgr, payload.focus_detection
        if isinstance(payload, np.ndarray):
            return payload, None
        raise TypeError("GalleryRetrievalExpert expects either a RetrievalInput or a BGR frame.")

    def _parse_step_id(self, folder_name: str) -> Optional[str]:
        name = folder_name.strip()
        lower = name.lower()
        if lower.startswith("step"):
            digits = "".join(character for character in name[4:] if character.isdigit())
            if digits:
                step_id = f"S{int(digits)}"
                if step_id in self.steps:
                    return step_id
        upper = name.upper()
        if upper in self.steps:
            return upper
        return None

    @staticmethod
    def _is_negative_folder(folder_name: str) -> bool:
        normalized = folder_name.strip().lower().replace("-", "_").replace(" ", "_")
        return normalized in {
            "wrong",
            "invalid",
            "negative",
            "negatives",
            "hard_negative",
            "hard_negatives",
            "no_step",
            "no_step4",
            "_negative",
        } or normalized.startswith("wrong_") or normalized.startswith("negative_") or normalized.startswith("no_step")
