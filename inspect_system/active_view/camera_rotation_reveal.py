"""Candidate calibration from keypoint-derived inverse camera rotation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .object_centric_evidence_memory import ObjectCentricMotion
from .object_centric_reveal import camera_motion_from_lattice
from .ontology import normalize_key
from .reveal_model import PriorTableRevealModel
from .view_lattice import ViewNode


def camera_rotation_features(
    motion: ObjectCentricMotion,
    *,
    rotation_scale: float = math.radians(15.0),
) -> np.ndarray:
    scale = max(1e-8, float(rotation_scale))
    rotation = np.clip(
        np.asarray(
            [
                motion.camera_rotation_x,
                motion.camera_rotation_y,
                motion.camera_rotation_z,
            ],
            dtype=np.float64,
        )
        / scale,
        -3.0,
        3.0,
    )
    return np.concatenate(
        (
            np.asarray([1.0]),
            rotation,
            np.abs(rotation),
            np.asarray([max(0.05, min(1.0, motion.quality))]),
        )
    )


class CameraRotationGainRevealModel(PriorTableRevealModel):
    """Reweight lattice candidates with an assistant-only rotation-gain model."""

    def __init__(
        self,
        base_model: PriorTableRevealModel,
        coefficients: Sequence[float],
        *,
        gain_scale: float = 0.10,
        target_min: float = -0.25,
        target_max: float = 0.25,
        calibration_strength: float = 1.0,
        rotation_scale: float = math.radians(15.0),
    ) -> None:
        super().__init__(
            alpha=float(base_model.alpha),
            default_probability=float(base_model.default_probability),
            counts={key: dict(value) for key, value in base_model.counts.items()},
            metadata={
                **dict(base_model.metadata),
                "model_type": "camera_rotation_gain_reveal",
                "uses_robot_view_training": False,
                "uses_candidate_view_images": False,
                "uses_ground_truth_pose": False,
            },
        )
        self.base_model = base_model
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.gain_scale = max(1e-6, float(gain_scale))
        self.target_min = float(target_min)
        self.target_max = float(target_max)
        self.calibration_strength = max(0.0, float(calibration_strength))
        self.rotation_scale = max(1e-8, float(rotation_scale))
        expected = camera_rotation_features(
            ObjectCentricMotion(0.0, 0.0),
            rotation_scale=self.rotation_scale,
        ).shape[0]
        if self.coefficients.shape != (expected,):
            raise ValueError(
                f"Expected {expected} rotation coefficients, got "
                f"{self.coefficients.shape}."
            )

    def action_distribution(
        self,
        claim_id: str,
        evidence_role: str,
        context: Mapping[str, Any] | None = None,
    ) -> Dict[str, float]:
        return self.base_model.action_distribution(
            claim_id,
            evidence_role,
            context=context,
        )

    def candidate_affordance_factor(
        self,
        claim_id: str,
        evidence_role: str,
        current_view: str,
        candidate_view: str,
        views: Mapping[str, ViewNode],
        context: Mapping[str, Any] | None = None,
    ) -> float:
        del claim_id, evidence_role, context
        motion = camera_motion_from_lattice(
            current_view,
            candidate_view,
            views,
        )
        features = camera_rotation_features(
            motion,
            rotation_scale=self.rotation_scale,
        )
        gain = float(features @ self.coefficients)
        gain = min(self.target_max, max(self.target_min, gain))
        factor = math.exp(
            self.calibration_strength * gain / self.gain_scale
        )
        return min(4.0, max(0.25, factor))

    def probability(
        self,
        claim_id: str,
        evidence_role: str,
        action: str,
        context: Mapping[str, Any] | None = None,
    ) -> float:
        return self.action_distribution(
            claim_id,
            evidence_role,
            context=context,
        ).get(normalize_key(action), 0.0)

    def counterfactual_relevance(
        self,
        claim_id: str,
        evidence_role: str,
        context: Mapping[str, Any] | None = None,
    ) -> float:
        function = getattr(self.base_model, "counterfactual_relevance", None)
        return (
            float(function(claim_id, evidence_role, context=context))
            if callable(function)
            else 1.0
        )

    def transport_confidence(
        self,
        claim_id: str,
        evidence_role: str,
        context: Mapping[str, Any] | None = None,
    ) -> float:
        function = getattr(self.base_model, "transport_confidence", None)
        return (
            float(function(claim_id, evidence_role, context=context))
            if callable(function)
            else 1.0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": "camera_rotation_gain_reveal",
            "base_model": self.base_model.to_dict(),
            "coefficients": self.coefficients.tolist(),
            "gain_scale": self.gain_scale,
            "target_min": self.target_min,
            "target_max": self.target_max,
            "calibration_strength": self.calibration_strength,
            "rotation_scale": self.rotation_scale,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "CameraRotationGainRevealModel":
        if payload.get("model_type") != "camera_rotation_gain_reveal":
            raise ValueError("Camera-rotation reveal schema mismatch.")
        from .object_centric_reveal import ObjectCentricCalibratedRevealModel

        return cls(
            base_model=ObjectCentricCalibratedRevealModel._load_base(
                dict(payload.get("base_model", {}))
            ),
            coefficients=list(payload.get("coefficients", [])),
            gain_scale=float(payload.get("gain_scale", 0.10)),
            target_min=float(payload.get("target_min", -0.25)),
            target_max=float(payload.get("target_max", 0.25)),
            calibration_strength=float(
                payload.get("calibration_strength", 1.0)
            ),
            rotation_scale=float(
                payload.get("rotation_scale", math.radians(15.0))
            ),
        )

    def save(self, path: str | Path) -> None:
        super().save(path)
