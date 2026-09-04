"""Assistant-trained evidence-gain calibration for lattice camera motion."""

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


GEOMETRY_FIELDS = (
    "azimuth_delta", "elevation_delta", "log_radius_delta",
    "displacement_x", "displacement_y", "displacement_z",
    "rotation_x", "rotation_y", "rotation_z", "inlier_ratio",
    "normalized_residual", "visibility_ratio", "cycle_error",
    "camera_rotation_x", "camera_rotation_y", "camera_rotation_z",
    "camera_translation_x", "camera_translation_y", "camera_translation_z",
    "surface_incidence_before", "surface_incidence_after",
    "surface_incidence_delta", "surface_grazing_delta",
    "surface_view_parallax", "surface_tangent_x", "surface_tangent_y",
    "surface_normal_translation", "surface_frame_quality",
    "relation_axis_translation", "relation_binormal_translation",
    "relation_normal_translation", "relation_frame_quality",
)


def _motion_vector(motion: ObjectCentricMotion) -> np.ndarray:
    values = {
        "azimuth_delta": motion.azimuth,
        "elevation_delta": motion.elevation,
        "log_radius_delta": motion.log_radius,
        "rotation_z": motion.rotation_z,
        "camera_rotation_x": motion.camera_rotation_x,
        "camera_rotation_y": motion.camera_rotation_y,
        "camera_rotation_z": motion.camera_rotation_z,
        "camera_translation_x": motion.camera_translation_x,
        "camera_translation_y": motion.camera_translation_y,
        "camera_translation_z": motion.camera_translation_z,
        "surface_incidence_before": motion.surface_incidence_before,
        "surface_incidence_after": motion.surface_incidence_after,
        "surface_incidence_delta": motion.surface_incidence_delta,
        "surface_grazing_delta": motion.surface_grazing_delta,
        "surface_view_parallax": motion.surface_view_parallax,
        "surface_tangent_x": motion.surface_tangent_x,
        "surface_tangent_y": motion.surface_tangent_y,
        "surface_normal_translation": motion.surface_normal_translation,
        "surface_frame_quality": motion.surface_frame_quality,
        "relation_axis_translation": motion.relation_axis_translation,
        "relation_binormal_translation": motion.relation_binormal_translation,
        "relation_normal_translation": motion.relation_normal_translation,
        "relation_frame_quality": motion.relation_frame_quality,
    }
    return np.asarray([float(values.get(field, 0.0)) for field in GEOMETRY_FIELDS])


def camera_pose_features(
    motion: ObjectCentricMotion,
    median: Sequence[float],
    scale: Sequence[float],
) -> np.ndarray:
    """Match the compact pose design used by assistant-side nested LOOV."""
    center = np.asarray(median, dtype=np.float64)
    spread = np.asarray(scale, dtype=np.float64)
    if center.shape != (len(GEOMETRY_FIELDS),) or spread.shape != center.shape:
        raise ValueError("Camera-motion normalization has an invalid shape.")
    standardized = np.clip((_motion_vector(motion) - center) / spread, -5.0, 5.0)
    index = {name: offset for offset, name in enumerate(GEOMETRY_FIELDS)}
    pose = standardized[
        [
            index["camera_rotation_y"],
            index["camera_translation_x"],
            index["camera_rotation_x"],
            index["camera_translation_y"],
            index["camera_rotation_z"],
            index["camera_translation_z"],
        ]
    ]
    compact = np.concatenate(
        (
            pose,
            np.abs(pose[:4]),
            np.asarray([pose[0] * -pose[1], pose[2] * -pose[3], 1.0]),
        )
    )
    return np.concatenate(([1.0], compact))


class CameraMotionGainRevealModel(PriorTableRevealModel):
    """Reweight candidates with assistant-derived continuous motion gain."""

    def __init__(
        self,
        base_model: PriorTableRevealModel,
        coefficients: Sequence[float],
        median: Sequence[float],
        scale: Sequence[float],
        *,
        gain_scale: float = 0.10,
        target_min: float = -0.25,
        target_max: float = 0.25,
        calibration_strength: float = 1.0,
        calibration_mode: str = "negative_only",
    ) -> None:
        super().__init__(
            alpha=float(base_model.alpha),
            default_probability=float(base_model.default_probability),
            counts={key: dict(value) for key, value in base_model.counts.items()},
            metadata={
                **dict(base_model.metadata),
                "model_type": "camera_motion_gain_reveal",
                "uses_robot_view_training": False,
                "uses_candidate_view_images": False,
                "uses_ground_truth_pose": False,
                "stay_anchored": True,
            },
        )
        self.base_model = base_model
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.median = np.asarray(median, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.gain_scale = max(1e-6, float(gain_scale))
        self.target_min = float(target_min)
        self.target_max = float(target_max)
        self.calibration_strength = max(0.0, float(calibration_strength))
        self.calibration_mode = str(calibration_mode)
        if self.calibration_mode not in {"signed", "negative_only"}:
            raise ValueError("Unknown camera-motion calibration mode.")
        expected = camera_pose_features(
            ObjectCentricMotion(0.0, 0.0), self.median, self.scale
        ).shape[0]
        if self.coefficients.shape != (expected,):
            raise ValueError(
                f"Expected {expected} camera-motion coefficients, got "
                f"{self.coefficients.shape}."
            )

    def action_distribution(
        self,
        claim_id: str,
        evidence_role: str,
        context: Mapping[str, Any] | None = None,
    ) -> Dict[str, float]:
        return self.base_model.action_distribution(
            claim_id, evidence_role, context=context
        )

    def _gain(self, motion: ObjectCentricMotion) -> float:
        value = float(
            camera_pose_features(motion, self.median, self.scale)
            @ self.coefficients
        )
        return min(self.target_max, max(self.target_min, value))

    def candidate_affordance_factor(
        self,
        claim_id: str,
        evidence_role: str,
        current_view: str,
        candidate_view: str,
        views: Mapping[str, ViewNode],
        context: Mapping[str, Any] | None = None,
    ) -> float:
        del claim_id
        role_normals = dict((context or {}).get("role_surface_normals_world") or {})
        relation_frames = dict((context or {}).get("role_relation_frames_world") or {})
        motion = camera_motion_from_lattice(
            current_view,
            candidate_view,
            views,
            surface_normal_world=(context or {}).get("surface_normal_world")
            or role_normals.get(normalize_key(evidence_role)),
            relation_frame_world=relation_frames.get(normalize_key(evidence_role)),
        )
        stay = camera_motion_from_lattice(current_view, current_view, views)
        relative_gain = self._gain(motion) - self._gain(stay)
        if self.calibration_mode == "negative_only":
            relative_gain = min(0.0, relative_gain)
        factor = math.exp(
            self.calibration_strength * relative_gain / self.gain_scale
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
            claim_id, evidence_role, context=context
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
            "model_type": "camera_motion_gain_reveal",
            "base_model": self.base_model.to_dict(),
            "coefficients": self.coefficients.tolist(),
            "median": self.median.tolist(),
            "scale": self.scale.tolist(),
            "gain_scale": self.gain_scale,
            "target_min": self.target_min,
            "target_max": self.target_max,
            "calibration_strength": self.calibration_strength,
            "calibration_mode": self.calibration_mode,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CameraMotionGainRevealModel":
        if payload.get("model_type") != "camera_motion_gain_reveal":
            raise ValueError("Camera-motion reveal schema mismatch.")
        from .object_centric_reveal import ObjectCentricCalibratedRevealModel

        return cls(
            base_model=ObjectCentricCalibratedRevealModel._load_base(
                dict(payload.get("base_model", {}))
            ),
            coefficients=list(payload.get("coefficients", [])),
            median=list(payload.get("median", [])),
            scale=list(payload.get("scale", [])),
            gain_scale=float(payload.get("gain_scale", 0.10)),
            target_min=float(payload.get("target_min", -0.25)),
            target_max=float(payload.get("target_max", 0.25)),
            calibration_strength=float(payload.get("calibration_strength", 1.0)),
            calibration_mode=str(payload.get("calibration_mode", "negative_only")),
        )

    def save(self, path: str | Path) -> None:
        super().save(path)
