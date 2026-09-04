"""Continual evidence-affordance memory in an object-centric motion frame."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from .evidence_affordance_memory import (
    EvidenceAffordanceConfig,
    EvidenceAffordanceContext,
    EvidenceAffordanceMemory,
    EvidenceAffordancePrediction,
    OutcomeStatistics,
)


def _finite(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


@dataclass(frozen=True)
class ObjectCentricMotion:
    """Relative camera motion recovered around a tracked workpiece reference."""

    azimuth: float
    elevation: float
    log_radius: float = 0.0
    rotation_z: float = 0.0
    camera_rotation_x: float = 0.0
    camera_rotation_y: float = 0.0
    camera_rotation_z: float = 0.0
    camera_translation_x: float = 0.0
    camera_translation_y: float = 0.0
    camera_translation_z: float = 0.0
    surface_incidence_before: float = 0.0
    surface_incidence_after: float = 0.0
    surface_incidence_delta: float = 0.0
    surface_grazing_delta: float = 0.0
    surface_view_parallax: float = 0.0
    surface_tangent_x: float = 0.0
    surface_tangent_y: float = 0.0
    surface_normal_translation: float = 0.0
    surface_frame_quality: float = 0.0
    relation_axis_translation: float = 0.0
    relation_binormal_translation: float = 0.0
    relation_normal_translation: float = 0.0
    relation_frame_quality: float = 0.0
    quality: float = 1.0
    reference_type: str = "unknown"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ObjectCentricMotion":
        inlier_ratio = min(1.0, max(0.0, _finite(payload.get("inlier_ratio"), 0.0)))
        visibility = min(1.0, max(0.0, _finite(payload.get("visibility_ratio"), 0.0)))
        residual = max(0.0, _finite(payload.get("normalized_residual"), 1.0))
        cycle = max(0.0, _finite(payload.get("cycle_error"), 1.0))
        if "quality" in payload:
            quality = _finite(payload.get("quality"), 0.05)
        else:
            quality = math.sqrt(inlier_ratio * visibility)
            quality *= math.exp(-residual / 0.08)
            quality *= math.exp(-cycle / 4.0)
        return cls(
            azimuth=_wrap_angle(
                _finite(payload.get("azimuth_delta", payload.get("azimuth")))
            ),
            elevation=_finite(payload.get("elevation_delta", payload.get("elevation"))),
            log_radius=_finite(
                payload.get("log_radius_delta", payload.get("log_radius"))
            ),
            rotation_z=_wrap_angle(_finite(payload.get("rotation_z"))),
            camera_rotation_x=_wrap_angle(_finite(payload.get("camera_rotation_x"))),
            camera_rotation_y=_wrap_angle(_finite(payload.get("camera_rotation_y"))),
            camera_rotation_z=_wrap_angle(_finite(payload.get("camera_rotation_z"))),
            camera_translation_x=_finite(payload.get("camera_translation_x")),
            camera_translation_y=_finite(payload.get("camera_translation_y")),
            camera_translation_z=_finite(payload.get("camera_translation_z")),
            surface_incidence_before=_finite(payload.get("surface_incidence_before")),
            surface_incidence_after=_finite(payload.get("surface_incidence_after")),
            surface_incidence_delta=_finite(payload.get("surface_incidence_delta")),
            surface_grazing_delta=_finite(payload.get("surface_grazing_delta")),
            surface_view_parallax=_finite(payload.get("surface_view_parallax")),
            surface_tangent_x=_finite(payload.get("surface_tangent_x")),
            surface_tangent_y=_finite(payload.get("surface_tangent_y")),
            surface_normal_translation=_finite(
                payload.get("surface_normal_translation")
            ),
            surface_frame_quality=min(
                1.0,
                max(0.0, _finite(payload.get("surface_frame_quality"))),
            ),
            relation_axis_translation=_finite(payload.get("relation_axis_translation")),
            relation_binormal_translation=_finite(
                payload.get("relation_binormal_translation")
            ),
            relation_normal_translation=_finite(
                payload.get("relation_normal_translation")
            ),
            relation_frame_quality=min(
                1.0,
                max(0.0, _finite(payload.get("relation_frame_quality"))),
            ),
            quality=min(1.0, max(0.05, quality)),
            reference_type=str(payload.get("reference_type", "unknown")),
        )


@dataclass(frozen=True)
class ObjectCentricKernelConfig:
    """Hyperparameters selected only inside the training-video folds."""

    azimuth_bandwidth: float = 0.45
    elevation_bandwidth: float = 0.55
    radius_bandwidth: float = 0.45
    rotation_bandwidth: float = 0.65
    incidence_bandwidth: float = 0.25
    parallax_bandwidth: float = 0.45
    geometry_mode: str = "direction"
    semantic_backoff: float = 0.35
    unrelated_backoff: float = 0.08
    reference_mismatch: float = 0.75
    kernel_prior_strength: float = 2.0
    session_boost: float = 1.5
    merge_radius: float = 0.45
    max_atoms: int = 256


@dataclass
class ObjectCentricEvidenceAtom:
    context: EvidenceAffordanceContext
    motion: ObjectCentricMotion
    outcome: OutcomeStatistics = field(default_factory=OutcomeStatistics)
    center_weight: float = 0.0
    session_id: str | None = None

    def update_center(self, motion: ObjectCentricMotion, weight: float) -> None:
        mass = max(0.0, float(weight)) * max(0.05, motion.quality)
        if mass <= 0.0:
            return
        old = self.center_weight
        total = old + mass
        if old <= 0.0:
            self.motion = motion
            self.center_weight = mass
            return

        def circular(left: float, right: float) -> float:
            x = old * math.cos(left) + mass * math.cos(right)
            y = old * math.sin(left) + mass * math.sin(right)
            return math.atan2(y, x)

        self.motion = ObjectCentricMotion(
            azimuth=circular(self.motion.azimuth, motion.azimuth),
            elevation=(old * self.motion.elevation + mass * motion.elevation) / total,
            log_radius=(old * self.motion.log_radius + mass * motion.log_radius)
            / total,
            rotation_z=circular(self.motion.rotation_z, motion.rotation_z),
            camera_rotation_x=circular(
                self.motion.camera_rotation_x,
                motion.camera_rotation_x,
            ),
            camera_rotation_y=circular(
                self.motion.camera_rotation_y,
                motion.camera_rotation_y,
            ),
            camera_rotation_z=circular(
                self.motion.camera_rotation_z,
                motion.camera_rotation_z,
            ),
            camera_translation_x=(
                old * self.motion.camera_translation_x
                + mass * motion.camera_translation_x
            )
            / total,
            camera_translation_y=(
                old * self.motion.camera_translation_y
                + mass * motion.camera_translation_y
            )
            / total,
            camera_translation_z=(
                old * self.motion.camera_translation_z
                + mass * motion.camera_translation_z
            )
            / total,
            surface_incidence_before=(
                old * self.motion.surface_incidence_before
                + mass * motion.surface_incidence_before
            )
            / total,
            surface_incidence_after=(
                old * self.motion.surface_incidence_after
                + mass * motion.surface_incidence_after
            )
            / total,
            surface_incidence_delta=(
                old * self.motion.surface_incidence_delta
                + mass * motion.surface_incidence_delta
            )
            / total,
            surface_grazing_delta=(
                old * self.motion.surface_grazing_delta
                + mass * motion.surface_grazing_delta
            )
            / total,
            surface_view_parallax=(
                old * self.motion.surface_view_parallax
                + mass * motion.surface_view_parallax
            )
            / total,
            surface_tangent_x=(
                old * self.motion.surface_tangent_x + mass * motion.surface_tangent_x
            )
            / total,
            surface_tangent_y=(
                old * self.motion.surface_tangent_y + mass * motion.surface_tangent_y
            )
            / total,
            surface_normal_translation=(
                old * self.motion.surface_normal_translation
                + mass * motion.surface_normal_translation
            )
            / total,
            surface_frame_quality=(
                old * self.motion.surface_frame_quality
                + mass * motion.surface_frame_quality
            )
            / total,
            quality=(old * self.motion.quality + mass * motion.quality) / total,
            reference_type=(
                self.motion.reference_type
                if self.motion.reference_type == motion.reference_type
                else "mixed"
            ),
        )
        self.center_weight = total


@dataclass
class ObjectCentricEvidenceMemory:
    """Bounded online kernel memory over keypoint-derived relative viewpoints."""

    discrete_config: EvidenceAffordanceConfig = field(
        default_factory=EvidenceAffordanceConfig
    )
    kernel_config: ObjectCentricKernelConfig = field(
        default_factory=ObjectCentricKernelConfig
    )
    discrete: EvidenceAffordanceMemory = field(init=False)
    atoms: list[ObjectCentricEvidenceAtom] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.discrete = EvidenceAffordanceMemory(config=self.discrete_config)

    def _distance_square(
        self, left: ObjectCentricMotion, right: ObjectCentricMotion
    ) -> float:
        cfg = self.kernel_config
        if cfg.geometry_mode in {"camera_pose", "surface_hybrid"}:
            rotation_left = (
                left.camera_rotation_x,
                left.camera_rotation_y,
                left.camera_rotation_z,
            )
            rotation_right = (
                right.camera_rotation_x,
                right.camera_rotation_y,
                right.camera_rotation_z,
            )
            terms = [
                (_wrap_angle(left_value - right_value) / cfg.rotation_bandwidth) ** 2
                for left_value, right_value in zip(
                    rotation_left,
                    rotation_right,
                )
            ]
            translation_left = (
                left.camera_translation_x,
                left.camera_translation_y,
                left.camera_translation_z,
            )
            translation_right = (
                right.camera_translation_x,
                right.camera_translation_y,
                right.camera_translation_z,
            )

            def unit(values: tuple[float, float, float]) -> tuple[float, float, float]:
                norm = math.sqrt(sum(value * value for value in values))
                if norm <= 1e-8:
                    return (0.0, 0.0, 0.0)
                return tuple(value / norm for value in values)

            terms.extend(
                ((left_value - right_value) / cfg.radius_bandwidth) ** 2
                for left_value, right_value in zip(
                    unit(translation_left),
                    unit(translation_right),
                )
            )
            if cfg.geometry_mode == "surface_hybrid":
                terms.extend(
                    [
                        (
                            (
                                left.surface_incidence_before
                                - right.surface_incidence_before
                            )
                            / cfg.incidence_bandwidth
                        )
                        ** 2,
                        (
                            (
                                left.surface_incidence_after
                                - right.surface_incidence_after
                            )
                            / cfg.incidence_bandwidth
                        )
                        ** 2,
                        (
                            (
                                left.surface_incidence_delta
                                - right.surface_incidence_delta
                            )
                            / cfg.incidence_bandwidth
                        )
                        ** 2,
                        (
                            (left.surface_view_parallax - right.surface_view_parallax)
                            / cfg.parallax_bandwidth
                        )
                        ** 2,
                    ]
                )
        elif cfg.geometry_mode == "local_surface_motion":
            quality = math.sqrt(
                max(0.05, left.surface_frame_quality)
                * max(0.05, right.surface_frame_quality)
            )
            terms = [
                quality
                * (
                    (left.surface_tangent_x - right.surface_tangent_x)
                    / cfg.radius_bandwidth
                )
                ** 2,
                quality
                * (
                    (left.surface_tangent_y - right.surface_tangent_y)
                    / cfg.radius_bandwidth
                )
                ** 2,
                quality
                * (
                    (left.surface_normal_translation - right.surface_normal_translation)
                    / cfg.radius_bandwidth
                )
                ** 2,
                (
                    (left.surface_incidence_delta - right.surface_incidence_delta)
                    / cfg.incidence_bandwidth
                )
                ** 2,
                (
                    (left.surface_view_parallax - right.surface_view_parallax)
                    / cfg.parallax_bandwidth
                )
                ** 2,
            ]
        elif cfg.geometry_mode == "relation_surface_motion":
            quality = math.sqrt(
                max(0.05, left.relation_frame_quality)
                * max(0.05, right.relation_frame_quality)
            )
            terms = [
                quality
                * (
                    (left.relation_axis_translation - right.relation_axis_translation)
                    / cfg.radius_bandwidth
                )
                ** 2,
                quality
                * (
                    (
                        left.relation_binormal_translation
                        - right.relation_binormal_translation
                    )
                    / cfg.radius_bandwidth
                )
                ** 2,
                quality
                * (
                    (
                        left.relation_normal_translation
                        - right.relation_normal_translation
                    )
                    / cfg.radius_bandwidth
                )
                ** 2,
                (
                    (left.surface_view_parallax - right.surface_view_parallax)
                    / cfg.parallax_bandwidth
                )
                ** 2,
            ]
        elif cfg.geometry_mode == "spherical_bearing":
            left_norm = math.hypot(left.azimuth, left.elevation)
            right_norm = math.hypot(right.azimuth, right.elevation)
            if left_norm <= 1e-6 or right_norm <= 1e-6:
                terms = [((left_norm - right_norm) / cfg.azimuth_bandwidth) ** 2]
            else:
                left_bearing = math.atan2(left.elevation, left.azimuth)
                right_bearing = math.atan2(right.elevation, right.azimuth)
                terms = [
                    (_wrap_angle(left_bearing - right_bearing) / cfg.rotation_bandwidth)
                    ** 2
                ]
        elif cfg.geometry_mode == "parallax":
            left_angle = math.hypot(left.azimuth, left.elevation)
            right_angle = math.hypot(right.azimuth, right.elevation)
            terms = [((left_angle - right_angle) / cfg.azimuth_bandwidth) ** 2]
        elif cfg.geometry_mode == "axis_invariant":
            terms = [
                ((abs(left.azimuth) - abs(right.azimuth)) / cfg.azimuth_bandwidth) ** 2,
                ((abs(left.elevation) - abs(right.elevation)) / cfg.elevation_bandwidth)
                ** 2,
            ]
        else:
            terms = [
                (_wrap_angle(left.azimuth - right.azimuth) / cfg.azimuth_bandwidth)
                ** 2,
                ((left.elevation - right.elevation) / cfg.elevation_bandwidth) ** 2,
            ]
        if cfg.geometry_mode in {"direction_radius", "direction_radius_roll"}:
            terms.append(
                ((left.log_radius - right.log_radius) / cfg.radius_bandwidth) ** 2
            )
        if cfg.geometry_mode in {"direction_roll", "direction_radius_roll"}:
            terms.append(
                (
                    _wrap_angle(left.rotation_z - right.rotation_z)
                    / cfg.rotation_bandwidth
                )
                ** 2
            )
        return float(sum(terms))

    def _context_weight(
        self, query: EvidenceAffordanceContext, atom: EvidenceAffordanceContext
    ) -> float:
        cfg = self.kernel_config
        if query.semantic_key == atom.semantic_key:
            value = 1.0
        elif query.role == atom.role or query.counterfactual == atom.counterfactual:
            value = cfg.semantic_backoff
        else:
            value = cfg.unrelated_backoff
        if query.claim == atom.claim:
            value *= 1.15
        return value

    def _kernel_weight(
        self,
        context: EvidenceAffordanceContext,
        motion: ObjectCentricMotion,
        atom: ObjectCentricEvidenceAtom,
        session_id: str | None,
    ) -> float:
        if atom.session_id is not None and atom.session_id != session_id:
            return 0.0
        distance = self._distance_square(motion, atom.motion)
        weight = math.exp(-0.5 * distance) * self._context_weight(context, atom.context)
        if (
            self.kernel_config.geometry_mode != "spherical_bearing"
            and motion.reference_type != atom.motion.reference_type
            and "mixed" not in {motion.reference_type, atom.motion.reference_type}
        ):
            weight *= self.kernel_config.reference_mismatch
        weight *= math.sqrt(max(0.05, motion.quality) * max(0.05, atom.motion.quality))
        if session_id and atom.session_id == session_id:
            weight *= self.kernel_config.session_boost
        return weight

    def predict(
        self,
        context: EvidenceAffordanceContext,
        motion: ObjectCentricMotion,
        session_id: str | None = None,
    ) -> EvidenceAffordancePrediction:
        prior = self.discrete.predict(context, session_id=session_id)
        positive = 0.0
        negative = 0.0
        gain_sum = 0.0
        support = 0.0
        for atom in self.atoms:
            similarity = self._kernel_weight(context, motion, atom, session_id)
            if similarity < 1e-6 or atom.outcome.weight <= 0.0:
                continue
            positive += similarity * atom.outcome.positive_weight
            negative += similarity * atom.outcome.negative_weight
            gain_sum += similarity * atom.outcome.gain_sum
            support += similarity * atom.outcome.weight

        strength = max(1e-6, self.kernel_config.kernel_prior_strength)
        probability = (positive + strength * prior.helpful_probability) / (
            positive + negative + strength
        )
        gain = (gain_sum + strength * prior.expected_gain) / (support + strength)
        probability = min(1.0 - 1e-6, max(1e-6, probability))
        total_support = prior.effective_support + support
        uncertainty = (
            4.0 * probability * (1.0 - probability) / math.sqrt(1.0 + total_support)
        )
        return EvidenceAffordancePrediction(
            helpful_probability=probability,
            expected_gain=float(gain),
            uncertainty=float(uncertainty),
            effective_support=float(total_support),
            scope="object_kernel+" + prior.scope if support > 0.0 else prior.scope,
        )

    def _nearest_mergeable(
        self,
        context: EvidenceAffordanceContext,
        motion: ObjectCentricMotion,
        session_id: str | None,
    ) -> ObjectCentricEvidenceAtom | None:
        candidates = [
            atom
            for atom in self.atoms
            if atom.context.exact_key == context.exact_key
            and atom.session_id == session_id
        ]
        if not candidates or self.kernel_config.merge_radius <= 0.0:
            return None
        atom = min(
            candidates, key=lambda item: self._distance_square(motion, item.motion)
        )
        distance = math.sqrt(self._distance_square(motion, atom.motion))
        return atom if distance <= self.kernel_config.merge_radius else None

    def _prune(self) -> None:
        limit = max(1, int(self.kernel_config.max_atoms))
        while len(self.atoms) > limit:
            victim = min(
                range(len(self.atoms)),
                key=lambda index: (
                    self.atoms[index].outcome.weight,
                    self.atoms[index].outcome.update_count,
                    self.atoms[index].center_weight,
                ),
            )
            del self.atoms[victim]

    def update(
        self,
        context: EvidenceAffordanceContext,
        motion: ObjectCentricMotion,
        signed_gain: float,
        weight: float = 1.0,
        session_id: str | None = None,
        update_shared: bool = True,
    ) -> None:
        self.discrete.update(
            context,
            signed_gain,
            weight,
            session_id=session_id,
            update_shared=update_shared,
        )
        # Held-out-session feedback remains identifiable as local adaptation;
        # offline training examples use ``None`` and form the shared field.
        atom_session = session_id if session_id else None
        atom = self._nearest_mergeable(context, motion, atom_session)
        if atom is None:
            atom = ObjectCentricEvidenceAtom(
                context=context,
                motion=motion,
                session_id=atom_session,
            )
            self.atoms.append(atom)
        atom.update_center(motion, weight)
        atom.outcome.update(signed_gain, weight * max(0.05, motion.quality))
        self._prune()

    def fit(
        self,
        examples: Iterable[
            tuple[EvidenceAffordanceContext, ObjectCentricMotion, float, float]
        ],
    ) -> "ObjectCentricEvidenceMemory":
        for context, motion, gain, weight in examples:
            self.update(context, motion, gain, weight)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "discrete": self.discrete.to_dict(),
            "kernel_config": self.kernel_config.__dict__,
            "atoms": [
                {
                    "context": atom.context.__dict__,
                    "motion": atom.motion.__dict__,
                    "outcome": atom.outcome.to_dict(),
                    "center_weight": atom.center_weight,
                    "session_id": atom.session_id,
                }
                for atom in self.atoms
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObjectCentricEvidenceMemory":
        discrete = EvidenceAffordanceMemory.from_dict(dict(payload.get("discrete", {})))
        model = cls(
            discrete_config=discrete.config,
            kernel_config=ObjectCentricKernelConfig(
                **dict(payload.get("kernel_config", {}))
            ),
        )
        model.discrete = discrete
        for item in payload.get("atoms", []):
            atom = ObjectCentricEvidenceAtom(
                context=EvidenceAffordanceContext.from_mapping(
                    dict(item.get("context", {}))
                ),
                motion=ObjectCentricMotion.from_mapping(dict(item.get("motion", {}))),
                outcome=OutcomeStatistics.from_dict(dict(item.get("outcome", {}))),
                center_weight=float(item.get("center_weight", 0.0)),
                session_id=item.get("session_id"),
            )
            model.atoms.append(atom)
        model._prune()
        return model

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ObjectCentricEvidenceMemory":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
