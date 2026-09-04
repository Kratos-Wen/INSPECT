"""Candidate-level reveal calibration from object-centric assistant motion."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping

import cv2
import numpy as np

from .counterfactual_transport import (
    counterfactual_mixture,
    infer_counterfactual_family,
)
from .evidence_transport import action_family
from .evidence_affordance_memory import EvidenceAffordanceContext
from .object_centric_evidence_memory import (
    ObjectCentricEvidenceMemory,
    ObjectCentricMotion,
)
from .ontology import normalize_key
from .reveal_model import PriorTableRevealModel
from .view_lattice import ViewNode, delta, direction_token
from .ray_visibility import camera_position, camera_to_world_rotation
from .session_ray_field import SessionRayEvidenceField


def _logit(probability: float) -> float:
    value = min(1.0 - 1e-5, max(1e-5, float(probability)))
    return math.log(value / (1.0 - value))


def _unit(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    return values / norm if norm > 1e-10 else np.zeros_like(values)


def camera_motion_from_lattice(
    current_view: str,
    candidate_view: str,
    views: Mapping[str, ViewNode],
    surface_normal_world: Any | None = None,
    relation_frame_world: Mapping[str, Any] | None = None,
) -> ObjectCentricMotion:
    """Express a calibrated candidate pose in the current camera frame."""
    current_to_world = camera_to_world_rotation(views[current_view])
    candidate_to_world = camera_to_world_rotation(views[candidate_view])
    camera_rotation = current_to_world.T @ candidate_to_world
    rotation_vector, _ = cv2.Rodrigues(camera_rotation)
    translation = current_to_world.T @ (
        camera_position(views[candidate_view])
        - camera_position(views[current_view])
    )
    norm = float(np.linalg.norm(translation))
    if norm > 1e-8:
        translation = translation / norm
    rotation_vector = rotation_vector.reshape(3)
    yaw, elevation = delta(current_view, candidate_view, views)
    normal = None
    if surface_normal_world is not None:
        values = np.asarray(surface_normal_world, dtype=np.float64).reshape(-1)
        if values.size >= 3 and np.all(np.isfinite(values[:3])):
            candidate = values[:3]
            if float(np.linalg.norm(candidate)) > 1e-8:
                normal = candidate / float(np.linalg.norm(candidate))
    current_ray = camera_position(views[current_view])
    candidate_ray = camera_position(views[candidate_view])
    surface_tangent_x = 0.0
    surface_tangent_y = 0.0
    surface_normal_translation = 0.0
    surface_frame_quality = 0.0
    relation_axis_translation = 0.0
    relation_binormal_translation = 0.0
    relation_normal_translation = 0.0
    relation_frame_quality = 0.0
    if normal is not None:
        if float(np.dot(normal, current_ray)) < 0.0:
            normal = -normal
        tangent_axis = np.cross(normal, current_ray)
        surface_frame_quality = float(np.linalg.norm(tangent_axis))
        if surface_frame_quality <= 1e-4:
            camera_right = camera_to_world_rotation(views[current_view])[:, 0]
            tangent_axis = (
                camera_right
                - float(np.dot(camera_right, normal)) * normal
            )
        tangent_axis /= max(1e-10, float(np.linalg.norm(tangent_axis)))
        vertical_axis = np.cross(normal, tangent_axis)
        vertical_axis /= max(1e-10, float(np.linalg.norm(vertical_axis)))
        displacement_world = candidate_ray - current_ray
        displacement_norm = float(np.linalg.norm(displacement_world))
        if displacement_norm > 1e-10:
            displacement_world /= displacement_norm
            surface_tangent_x = float(
                np.dot(displacement_world, tangent_axis)
            )
            surface_tangent_y = float(
                np.dot(displacement_world, vertical_axis)
            )
            surface_normal_translation = float(
                np.dot(displacement_world, normal)
            )
    relation_frame = dict(relation_frame_world or {})
    relation_axis = np.asarray(
        relation_frame.get("axis_world", []), dtype=np.float64
    ).reshape(-1)
    relation_binormal = np.asarray(
        relation_frame.get("binormal_world", []), dtype=np.float64
    ).reshape(-1)
    relation_normal = np.asarray(
        relation_frame.get("normal_world", []), dtype=np.float64
    ).reshape(-1)
    if (
        relation_axis.size >= 3
        and relation_binormal.size >= 3
        and relation_normal.size >= 3
    ):
        displacement_world = candidate_ray - current_ray
        displacement_norm = float(np.linalg.norm(displacement_world))
        if displacement_norm > 1e-10:
            displacement_world /= displacement_norm
            relation_axis_translation = float(
                np.dot(displacement_world, _unit(relation_axis[:3]))
            )
            relation_binormal_translation = float(
                np.dot(displacement_world, _unit(relation_binormal[:3]))
            )
            relation_normal_translation = float(
                np.dot(displacement_world, _unit(relation_normal[:3]))
            )
            relation_frame_quality = float(
                np.clip(relation_frame.get("quality", 0.0), 0.0, 1.0)
            )
    before_incidence = abs(float(np.dot(normal, current_ray))) if normal is not None else 0.0
    after_incidence = abs(float(np.dot(normal, candidate_ray))) if normal is not None else 0.0
    incidence_delta = after_incidence - before_incidence
    ray_dot = float(np.clip(np.dot(current_ray, candidate_ray), -1.0, 1.0))
    return ObjectCentricMotion(
        azimuth=math.radians(yaw),
        elevation=math.radians(elevation),
        rotation_z=float(rotation_vector[2]),
        camera_rotation_x=float(rotation_vector[0]),
        camera_rotation_y=float(rotation_vector[1]),
        camera_rotation_z=float(rotation_vector[2]),
        camera_translation_x=float(translation[0]),
        camera_translation_y=float(translation[1]),
        camera_translation_z=float(translation[2]),
        surface_incidence_before=before_incidence,
        surface_incidence_after=after_incidence,
        surface_incidence_delta=incidence_delta,
        surface_grazing_delta=-incidence_delta,
        surface_view_parallax=float(math.acos(ray_dot)),
        surface_tangent_x=surface_tangent_x,
        surface_tangent_y=surface_tangent_y,
        surface_normal_translation=surface_normal_translation,
        surface_frame_quality=float(
            np.clip(surface_frame_quality, 0.0, 1.0)
        ),
        relation_axis_translation=relation_axis_translation,
        relation_binormal_translation=relation_binormal_translation,
        relation_normal_translation=relation_normal_translation,
        relation_frame_quality=relation_frame_quality,
        quality=1.0,
        reference_type="scene_transport",
    )


class ObjectCentricCalibratedRevealModel(PriorTableRevealModel):
    """Calibrate candidate displacements without observing candidate images."""

    def __init__(
        self,
        base_model: PriorTableRevealModel,
        memory: ObjectCentricEvidenceMemory,
        family_memories: Mapping[str, ObjectCentricEvidenceMemory] | None = None,
        calibration_strength: float = 0.5,
        online_calibration_strength: float | None = None,
        support_strength: float = 2.0,
        gain_scale: float = 0.10,
        geometry_action_conditioning: str = 'full',
        gain_weight: float = 1.0,
        geometry_scope: str = "all_candidates",
        session_ray_field: SessionRayEvidenceField | None = None,
    ) -> None:
        super().__init__(
            alpha=float(base_model.alpha),
            default_probability=float(base_model.default_probability),
            counts={key: dict(value) for key, value in base_model.counts.items()},
            metadata={
                **dict(base_model.metadata),
                "model_type": "object_centric_calibrated_reveal",
                "base_model_type": str(
                    base_model.metadata.get(
                        "model_type", "prior_table_relative_reveal"
                    )
                ),
                "uses_robot_view_training": False,
                "uses_candidate_view_images": False,
            },
        )
        self.base_model = base_model
        self.memory = memory
        self.family_memories = {
            normalize_key(key): value
            for key, value in dict(family_memories or {}).items()
            if normalize_key(key)
        }
        self.calibration_strength = max(0.0, float(calibration_strength))
        self.online_calibration_strength = (
            self.calibration_strength
            if online_calibration_strength is None
            else max(0.0, float(online_calibration_strength))
        )
        self.support_strength = max(1e-6, float(support_strength))
        self.gain_scale = max(1e-6, float(gain_scale))
        self.gain_weight = max(0.0, float(gain_weight))
        if geometry_action_conditioning not in {'full', 'identifiable'}:
            raise ValueError(
                'Unknown geometry action conditioning: '
                f'{geometry_action_conditioning}'
            )
        self.geometry_action_conditioning = geometry_action_conditioning
        self.session_ray_field = session_ray_field or SessionRayEvidenceField()
        if geometry_scope not in {"all_candidates", "top_action_family"}:
            raise ValueError(f"Unknown object-centric geometry scope: {geometry_scope}")
        self.geometry_scope = geometry_scope

    def action_distribution(
        self,
        claim_id: str,
        evidence_role: str,
        context: Mapping[str, Any] | None = None,
    ) -> Dict[str, float]:
        return self.base_model.action_distribution(
            claim_id, evidence_role, context=context
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
        session_id = str((context or {}).get("session_id", "")).strip() or None
        action = normalize_key(direction_token(current_view, candidate_view, views))
        normalized_role = normalize_key(evidence_role)
        if self.geometry_scope == "top_action_family":
            distribution = self.base_model.action_distribution(
                claim_id,
                normalized_role,
                context=context,
            )
            family_mass: Dict[str, float] = {}
            for relative_action, probability in distribution.items():
                family = action_family(relative_action)
                if family:
                    family_mass[family] = (
                        family_mass.get(family, 0.0) + float(probability)
                    )
            candidate_family = action_family(action)
            if (
                not candidate_family
                or not family_mass
                or family_mass.get(candidate_family, 0.0)
                < max(family_mass.values()) - 1e-12
            ):
                return 1.0
        role_normals = dict((context or {}).get("role_surface_normals_world") or {})
        surface_normal = role_normals.get(normalized_role)
        if surface_normal is None:
            surface_normal = (context or {}).get("surface_normal_world")
        role_relation_frames = dict(
            (context or {}).get("role_relation_frames_world") or {}
        )
        relation_frame = role_relation_frames.get(normalized_role)
        motion = camera_motion_from_lattice(
            current_view,
            candidate_view,
            views,
            surface_normal_world=surface_normal,
            relation_frame_world=relation_frame,
        )
        mixture = counterfactual_mixture(
            context,
            infer_counterfactual_family(claim_id, evidence_role, context),
        )
        global_log_ratio = 0.0
        global_gain_delta = 0.0
        global_geometry_support = 0.0
        online_log_delta = 0.0
        online_gain_delta = 0.0
        online_support = 0.0
        for family, mixture_weight in mixture.items():
            family_key = normalize_key(family)
            memory = self.family_memories.get(family_key, self.memory)
            geometry_mode = normalize_key(memory.kernel_config.geometry_mode)
            geometry_action = (
                'geometry_invariant'
                if self.geometry_action_conditioning == 'identifiable'
                and geometry_mode in {'parallax', 'axis_invariant'}
                else action
            )
            evidence_context = EvidenceAffordanceContext(
                # Unsigned kernels cannot identify left from right. Keep that
                # ordering in the base policy and calibrate only the motion
                # magnitude supported by assistant traces.
                action=geometry_action,
                role=normalized_role,
                counterfactual=family_key,
                claim=claim_id,
            )
            global_prior = memory.discrete.predict(evidence_context)
            global_prediction = memory.predict(evidence_context, motion)
            global_log_ratio += mixture_weight * (
                _logit(global_prediction.helpful_probability)
                - _logit(global_prior.helpful_probability)
            )
            global_gain_delta += mixture_weight * (
                global_prediction.expected_gain - global_prior.expected_gain
            )
            global_geometry_support += mixture_weight * max(
                0.0,
                global_prediction.effective_support
                - global_prior.effective_support,
            )

            if session_id:
                session_prediction = memory.predict(
                    evidence_context,
                    motion,
                    session_id=session_id,
                )
                online_log_delta += mixture_weight * (
                    _logit(session_prediction.helpful_probability)
                    - _logit(global_prediction.helpful_probability)
                )
                online_gain_delta += mixture_weight * (
                    session_prediction.expected_gain
                    - global_prediction.expected_gain
                )
                online_support += mixture_weight * max(
                    0.0,
                    session_prediction.effective_support
                    - global_prediction.effective_support,
                )

        global_reliability = global_geometry_support / (
            global_geometry_support + self.support_strength
        )
        online_reliability = online_support / (
            online_support + self.support_strength
        )
        global_delta = (
            global_log_ratio
            + self.gain_weight * global_gain_delta / self.gain_scale
        )
        online_delta = (
            online_log_delta
            + self.gain_weight * online_gain_delta / self.gain_scale
        )
        exponent = (
            self.calibration_strength * global_reliability * global_delta
            + self.online_calibration_strength
            * online_reliability
            * online_delta
        )
        destination_factor = self.session_ray_field.factor(
            session_id=session_id,
            claim_id=claim_id,
            evidence_role=normalized_role,
            counterfactual_weights=mixture,
            candidate_view=candidate_view,
            views=views,
        )
        base_factor_function = getattr(
            self.base_model,
            'candidate_affordance_factor',
            None,
        )
        base_factor = (
            float(
                base_factor_function(
                    claim_id,
                    evidence_role,
                    current_view,
                    candidate_view,
                    views,
                    context=context,
                )
            )
            if callable(base_factor_function)
            else 1.0
        )
        factor = base_factor * math.exp(exponent) * destination_factor
        return min(4.0, max(0.25, factor))

    def update_online_destination(
        self,
        *,
        session_id: str,
        claim_id: str,
        evidence_role: str,
        selected_view: str,
        views: Mapping[str, ViewNode],
        signed_gain: float,
        weight: float,
        context: Mapping[str, Any] | None = None,
    ) -> int:
        mixture = counterfactual_mixture(
            context,
            infer_counterfactual_family(claim_id, evidence_role, context),
        )
        updates = 0
        for family, family_weight in mixture.items():
            updates += int(
                self.session_ray_field.update(
                    session_id=session_id,
                    claim_id=claim_id,
                    evidence_role=evidence_role,
                    counterfactual_family=family,
                    view_id=selected_view,
                    views=views,
                    signed_gain=signed_gain,
                    weight=max(0.0, float(weight)) * max(0.0, float(family_weight)),
                )
            )
        return updates

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
            'geometry_action_conditioning': self.geometry_action_conditioning,
            "model_type": "object_centric_calibrated_reveal",
            "base_model": self.base_model.to_dict(),
            "memory": self.memory.to_dict(),
            "family_memories": {
                key: memory.to_dict()
                for key, memory in sorted(self.family_memories.items())
            },
            "calibration_strength": self.calibration_strength,
            "online_calibration_strength": self.online_calibration_strength,
            "support_strength": self.support_strength,
            "gain_scale": self.gain_scale,
            "gain_weight": self.gain_weight,
            "geometry_scope": self.geometry_scope,
            "session_ray_field": self.session_ray_field.to_dict(),
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def _load_base(payload: Mapping[str, Any]) -> PriorTableRevealModel:
        model_type = str((payload.get("metadata") or {}).get("model_type", ""))
        if (
            payload.get("model_type") == "object_centric_calibrated_reveal"
            or model_type == "object_centric_calibrated_reveal"
        ):
            return ObjectCentricCalibratedRevealModel.from_dict(payload)
        if model_type == "counterfactual_evidence_transport":
            from .counterfactual_transport import CounterfactualEvidenceTransportModel

            return CounterfactualEvidenceTransportModel.from_dict(payload)
        if model_type == "hierarchical_equivariant_evidence_transport":
            from .evidence_transport import HierarchicalEvidenceTransportModel

            return HierarchicalEvidenceTransportModel.from_dict(payload)
        if model_type == "affordance_calibrated_reveal":
            from .affordance_calibrated_reveal import AffordanceCalibratedRevealModel

            return AffordanceCalibratedRevealModel.from_dict(payload)
        if model_type == "camera_motion_gain_reveal":
            from .camera_motion_reveal import CameraMotionGainRevealModel

            return CameraMotionGainRevealModel.from_dict(payload)
        return PriorTableRevealModel.from_dict(payload)

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ObjectCentricCalibratedRevealModel":
        if payload.get("model_type") != "object_centric_calibrated_reveal":
            raise ValueError("Object-centric reveal schema mismatch")
        return cls(
            base_model=cls._load_base(dict(payload.get("base_model", {}))),
            memory=ObjectCentricEvidenceMemory.from_dict(
                dict(payload.get("memory", {}))
            ),
            family_memories={
                str(key): ObjectCentricEvidenceMemory.from_dict(dict(value))
                for key, value in dict(payload.get("family_memories", {})).items()
            },
            calibration_strength=float(payload.get("calibration_strength", 0.5)),
            online_calibration_strength=float(
                payload.get(
                    "online_calibration_strength",
                    payload.get("calibration_strength", 0.5),
                )
            ),
            support_strength=float(payload.get("support_strength", 2.0)),
            gain_scale=float(payload.get("gain_scale", 0.10)),
            gain_weight=float(payload.get("gain_weight", 1.0)),
            geometry_scope=str(payload.get("geometry_scope", "all_candidates")),
            geometry_action_conditioning=str(
                payload.get('geometry_action_conditioning', 'full')
            ),
            session_ray_field=SessionRayEvidenceField.from_dict(
                dict(payload.get("session_ray_field", {}))
            ),
        )
