"""Adaptive continuous observation primitives for assistant evidence traces."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


FEATURE_NAMES = (
    "tangent_x",
    "tangent_y",
    "radial_log_scale",
    "parallax_x",
    "parallax_y",
    "log_track_support",
    "log_frame_span",
    "motion_confidence",
)


def _number(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


@dataclass(frozen=True)
class ObservationTransition:
    """A daily-use observation change before projection to a robot action set."""

    values: tuple[float, ...]
    motion_source: str = "unknown"
    relative_action: str = "unknown"

    def as_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float64)

    @classmethod
    def from_event(cls, event: Mapping[str, Any]) -> "ObservationTransition":
        metadata = event.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        stats = metadata.get("auto_relative_action_stats", {})
        if not isinstance(stats, Mapping):
            stats = {}

        dx = _number(stats.get("median_dx"))
        dy = _number(stats.get("median_dy"))
        outer_dx = _number(stats.get("outer_median_dx"), dx)
        outer_dy = _number(stats.get("outer_median_dy"), dy)
        scale = max(1e-6, _number(stats.get("scale_ratio"), 1.0))
        tracks = max(0.0, _number(stats.get("num_tracks")))
        before = _number(event.get("before_frame"))
        after = _number(event.get("after_frame"), before)
        confidence = max(
            0.0,
            min(
                1.0,
                _number(
                    metadata.get(
                        "relative_action_confidence",
                        metadata.get("action_confidence", 0.0),
                    )
                ),
            ),
        )
        return cls(
            values=(
                dx,
                dy,
                math.log(scale),
                outer_dx - dx,
                outer_dy - dy,
                math.log1p(tracks),
                math.log1p(abs(after - before)),
                confidence,
            ),
            motion_source=str(stats.get("motion_source", "unknown") or "unknown"),
            relative_action=str(metadata.get("relative_action", "unknown") or "unknown"),
        )


@dataclass
class RobustFeatureScaler:
    median: List[float] = field(default_factory=list)
    scale: List[float] = field(default_factory=list)
    clip: float = 6.0

    def fit(self, values: Sequence[Sequence[float]]) -> "RobustFeatureScaler":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError("RobustFeatureScaler requires a non-empty 2D array")
        median = np.median(array, axis=0)
        q25, q75 = np.percentile(array, [25.0, 75.0], axis=0)
        robust_sigma = (q75 - q25) / 1.349
        fallback = np.std(array, axis=0)
        scale = np.where(robust_sigma > 1e-6, robust_sigma, fallback)
        scale = np.where(scale > 1e-6, scale, 1.0)
        self.median = median.tolist()
        self.scale = scale.tolist()
        return self

    def transform(self, values: Sequence[float] | Sequence[Sequence[float]]) -> np.ndarray:
        if not self.median or not self.scale:
            raise RuntimeError("RobustFeatureScaler has not been fit")
        array = np.asarray(values, dtype=np.float64)
        result = (array - np.asarray(self.median)) / np.asarray(self.scale)
        return np.clip(result, -float(self.clip), float(self.clip))

    def to_dict(self) -> Dict[str, Any]:
        return {"median": self.median, "scale": self.scale, "clip": self.clip}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RobustFeatureScaler":
        return cls(
            median=[float(value) for value in payload.get("median", [])],
            scale=[float(value) for value in payload.get("scale", [])],
            clip=float(payload.get("clip", 6.0)),
        )


@dataclass
class AdaptiveObservationBasis:
    """A bounded RBF dictionary that evolves with verified assistant use."""

    max_elements: int = 16
    novelty_radius: float = 1.25
    bandwidth: float = 1.0
    scaler: RobustFeatureScaler = field(default_factory=RobustFeatureScaler)
    centers: List[List[float]] = field(default_factory=list)
    support: List[float] = field(default_factory=list)

    @staticmethod
    def _distance(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(left - right))))

    def _merge_closest(self) -> None:
        if len(self.centers) < 2:
            return
        best = (float("inf"), 0, 1)
        for first in range(len(self.centers)):
            for second in range(first + 1, len(self.centers)):
                distance = self._distance(
                    np.asarray(self.centers[first]),
                    np.asarray(self.centers[second]),
                )
                if distance < best[0]:
                    best = (distance, first, second)
        _, first, second = best
        first_mass = float(self.support[first])
        second_mass = float(self.support[second])
        total = max(1e-12, first_mass + second_mass)
        merged = (
            first_mass * np.asarray(self.centers[first])
            + second_mass * np.asarray(self.centers[second])
        ) / total
        self.centers[first] = merged.tolist()
        self.support[first] = total
        del self.centers[second]
        del self.support[second]

    def _update_standardized(self, vector: np.ndarray, weight: float = 1.0) -> int:
        mass = max(1e-6, float(weight))
        if not self.centers:
            self.centers.append(vector.tolist())
            self.support.append(mass)
            return 0
        distances = [
            self._distance(vector, np.asarray(center, dtype=np.float64))
            for center in self.centers
        ]
        nearest = int(np.argmin(distances))
        if distances[nearest] > float(self.novelty_radius):
            if len(self.centers) >= int(self.max_elements):
                self._merge_closest()
            self.centers.append(vector.tolist())
            self.support.append(mass)
            return len(self.centers) - 1
        previous = float(self.support[nearest])
        updated = (
            previous * np.asarray(self.centers[nearest], dtype=np.float64)
            + mass * vector
        ) / (previous + mass)
        self.centers[nearest] = updated.tolist()
        self.support[nearest] = previous + mass
        return nearest

    def fit(
        self,
        transitions: Iterable[ObservationTransition],
        weights: Sequence[float] | None = None,
    ) -> "AdaptiveObservationBasis":
        rows = list(transitions)
        if not rows:
            raise ValueError("AdaptiveObservationBasis requires at least one transition")
        raw = [row.values for row in rows]
        self.scaler.fit(raw)
        self.centers = []
        self.support = []
        sample_weights = list(weights) if weights is not None else [1.0] * len(rows)
        if len(sample_weights) != len(rows):
            raise ValueError("weights must match transitions")
        for vector, weight in zip(self.scaler.transform(raw), sample_weights):
            self._update_standardized(np.asarray(vector), float(weight))
        return self

    def partial_fit(self, transition: ObservationTransition, weight: float = 1.0) -> int:
        if not self.scaler.median:
            self.scaler.fit([transition.values])
        standardized = self.scaler.transform(transition.values)
        return self._update_standardized(np.asarray(standardized), weight)

    def transform(self, transition: ObservationTransition) -> np.ndarray:
        if not self.centers:
            return np.zeros((0,), dtype=np.float64)
        vector = self.scaler.transform(transition.values)
        distances = np.asarray(
            [self._distance(vector, np.asarray(center)) for center in self.centers]
        )
        bandwidth = max(1e-6, float(self.bandwidth))
        return np.exp(-0.5 * np.square(distances / bandwidth))

    def nearest(self, transition: ObservationTransition) -> tuple[int, float]:
        if not self.centers:
            return -1, float("inf")
        vector = self.scaler.transform(transition.values)
        distances = [self._distance(vector, np.asarray(center)) for center in self.centers]
        index = int(np.argmin(distances))
        return index, float(distances[index])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": "adaptive_observation_basis",
            "feature_names": list(FEATURE_NAMES),
            "max_elements": self.max_elements,
            "novelty_radius": self.novelty_radius,
            "bandwidth": self.bandwidth,
            "scaler": self.scaler.to_dict(),
            "centers": self.centers,
            "support": self.support,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AdaptiveObservationBasis":
        feature_names = tuple(payload.get("feature_names", FEATURE_NAMES))
        if feature_names != FEATURE_NAMES:
            raise ValueError("Adaptive observation basis feature schema mismatch")
        return cls(
            max_elements=int(payload.get("max_elements", 16)),
            novelty_radius=float(payload.get("novelty_radius", 1.25)),
            bandwidth=float(payload.get("bandwidth", 1.0)),
            scaler=RobustFeatureScaler.from_dict(payload.get("scaler", {})),
            centers=[
                [float(value) for value in center]
                for center in payload.get("centers", [])
            ],
            support=[float(value) for value in payload.get("support", [])],
        )
