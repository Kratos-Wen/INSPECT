"""Retrieval-based step experts built on a shared visual embedding space."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..types import RetrievalInput, StepPrediction
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
    ) -> None:
        self.root = Path(root)
        self.steps = [str(step).strip().upper() for step in steps]
        self.exts = {str(ext).lower() for ext in exts}
        self.embed_mode = str(embed_mode).strip().lower()
        self.topk = max(1, int(topk))
        self.encoder = encoder or SharedVisualEncoder(mode=self.embed_mode)
        self.items: List[GalleryItem] = []

    def build(self) -> int:
        """Index the gallery and return the number of indexed images."""

        if not self.root.exists():
            raise FileNotFoundError(f"Gallery root not found: {self.root}")
        self.items.clear()
        for step_dir in sorted(self.root.iterdir()):
            if not step_dir.is_dir():
                continue
            step_id = self._parse_step_id(step_dir.name)
            if step_id is None:
                continue
            for path in step_dir.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in self.exts:
                    continue
                image = cv2.imread(str(path))
                if image is None:
                    continue
                embedding = self.encoder.encode_image(image)
                self.items.append(GalleryItem(step_id=step_id, path=str(path), embedding=embedding))
        return len(self.items)

    def predict(self, payload: object) -> StepPrediction:
        """Predict dense step scores from the current frame or structured query."""

        if not self.items:
            return DummyRetrievalExpert(self.steps).predict(payload)

        frame_bgr, focus_detection = self._resolve_query(payload)
        query = self.encoder.encode_query(frame_bgr, focus_detection=focus_detection)
        matrix = np.stack([item.embedding for item in self.items], axis=0)
        similarities = matrix @ query

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

        scores = _normalize_scores(raw_scores)
        top_step = max(scores, key=scores.get) if scores else self.steps[0]
        return StepPrediction(
            step_id=top_step,
            confidence=float(scores.get(top_step, 0.0)),
            scores=scores,
            extras={"raw_scores": raw_scores, "embed_mode": self.embed_mode},
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
