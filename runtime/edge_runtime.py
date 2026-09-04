"""Epistemic compute routing and low-overhead stage profiling."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, Tuple

from ..config import EdgeRuntimeConfig


def _normalized(values: Iterable[str]) -> set[str]:
    return {str(value).strip().lower().replace(" ", "_") for value in values if str(value).strip()}


class StageProfiler:
    """Collect wall-clock latency by stage without changing inference outputs."""

    def __init__(self, *, enabled: bool, synchronize_cuda: bool = False) -> None:
        self.enabled = bool(enabled)
        self.synchronize_cuda = bool(synchronize_cuda)
        self._started = time.perf_counter()
        self._stage_ms: Dict[str, float] = {}

    def _sync(self) -> None:
        if not self.synchronize_cuda:
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            return

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._sync()
        started = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            key = str(stage).strip().lower()
            self._stage_ms[key] = self._stage_ms.get(key, 0.0) + elapsed_ms

    def finish(self) -> Dict[str, float]:
        if not self.enabled:
            return {}
        self._sync()
        values = {key: round(value, 4) for key, value in self._stage_ms.items()}
        values["total"] = round((time.perf_counter() - self._started) * 1000.0, 4)
        return values


@dataclass(frozen=True)
class ComputeContext:
    """Runtime evidence state used to schedule optional computation."""

    frame_index: int
    claim_state: str = "insufficient"
    missing_roles: Tuple[str, ...] = ()
    missing_role_scores: Tuple[Tuple[str, float], ...] = ()
    query_intent: str = ""
    step_confidence: float = 0.0
    stable_tracks: bool = False
    visual_change: bool = True
    interaction_transition: bool = False
    epistemic_gain: float = 0.0


@dataclass(frozen=True)
class ComputePlan:
    """One edge-runtime decision; it never changes verifier semantics."""

    run_detection: bool
    run_geometry: bool
    run_segmentation: bool
    run_retrieval: bool
    run_memory_embedding: bool
    response_tier: str
    target_latency_ms: float
    acquisition_mode: str = "monitor"
    external_observation_recommended: bool = False
    stage_evidence_values: Tuple[Tuple[str, float], ...] = field(default_factory=tuple)
    reasons: Tuple[str, ...] = field(default_factory=tuple)


class EpistemicComputeRouter:
    """Route compute toward evidence that can change the current claim state."""

    def __init__(self, config: EdgeRuntimeConfig) -> None:
        self.config = config
        self._detection_roles = _normalized(config.force_detection_roles)
        self._retrieval_roles = _normalized(config.force_retrieval_roles)
        self._geometry_roles = _normalized(config.force_geometry_roles)
        self._segmentation_roles = _normalized(config.force_segmentation_roles)
        self._structured_intents = _normalized(config.structured_response_intents)
        self._small_llm_intents = _normalized(config.small_llm_intents)
        self.reset()

    def reset(self) -> None:
        self._last_missing_roles: Tuple[str, ...] = ()
        self._persistent_missing_frames = 0

    @staticmethod
    def _role_values(context: ComputeContext) -> Dict[str, float]:
        values = {
            str(role).strip().lower().replace(" ", "_"): max(0.0, min(1.0, float(score)))
            for role, score in context.missing_role_scores
            if str(role).strip()
        }
        for role in _normalized(context.missing_roles):
            values.setdefault(role, 1.0)
        return values

    @staticmethod
    def _stage_value(role_values: Dict[str, float], supported_roles: set[str]) -> float:
        return max((value for role, value in role_values.items() if role in supported_roles), default=0.0)

    def plan(self, context: ComputeContext) -> ComputePlan:
        frame_index = max(0, int(context.frame_index))
        claim_state = str(context.claim_state or "insufficient").strip().lower()
        query_intent = str(context.query_intent or "").strip().lower()
        missing = _normalized(context.missing_roles)
        role_values = self._role_values(context)
        uncertain = claim_state in {"ins", "insufficient", "pending", "unresolved"}
        stable = bool(context.stable_tracks) and not bool(context.visual_change)
        missing_key = tuple(sorted(missing))
        stalled = float(context.epistemic_gain) < max(
            0.0,
            float(self.config.min_epistemic_gain),
        )
        if (
            uncertain
            and missing_key
            and missing_key == self._last_missing_roles
            and stalled
            and not context.interaction_transition
        ):
            self._persistent_missing_frames += 1
        else:
            self._persistent_missing_frames = 0
        self._last_missing_roles = missing_key
        cold_start = frame_index <= max(0, int(self.config.cold_start_frames))
        external_observation = bool(
            uncertain
            and missing_key
            and self._persistent_missing_frames >= max(1, int(self.config.persistent_missing_patience))
        )

        detection_stride = max(1, int(self.config.detection_interval_stable))
        geometry_stride = max(1, int(self.config.geometry_interval_stable))
        retrieval_stride = max(1, int(self.config.retrieval_interval_stable))
        memory_stride = max(1, int(self.config.memory_embedding_interval_stable))

        threshold = max(0.0, float(self.config.min_stage_evidence_value))
        detection_value = self._stage_value(role_values, self._detection_roles)
        retrieval_value = self._stage_value(role_values, self._retrieval_roles)
        geometry_value = self._stage_value(role_values, self._geometry_roles)
        segmentation_value = self._stage_value(role_values, self._segmentation_roles)
        history_value = max(
            (value for role, value in role_values.items() if role == "history_precondition"),
            default=0.0,
        )

        relation_query = query_intent in {"object_relation", "why_not_progressing"}
        identity_query = query_intent in {"component_info", "object_count", "object_presence"}
        memory_query = query_intent in {"history_feedback", "history_step", "memory_context"}
        generic_uncertainty = uncertain and not missing

        role_detection = (
            (uncertain and (detection_value >= threshold or generic_uncertainty))
            or identity_query
        )
        role_retrieval = (
            (uncertain and (retrieval_value >= threshold or generic_uncertainty))
            or identity_query
        )
        role_geometry = (uncertain and geometry_value >= threshold) or relation_query
        role_segmentation = (uncertain and segmentation_value >= threshold) or relation_query
        role_memory = (uncertain and history_value >= threshold) or memory_query

        allow_role_compute = not external_observation
        run_detection = bool(
            cold_start
            or context.visual_change
            or context.interaction_transition
            or (allow_role_compute and role_detection)
            or (stable and frame_index % detection_stride == 0)
        )
        run_geometry = (
            cold_start
            or bool(context.interaction_transition)
            or (allow_role_compute and role_geometry)
            or (stable and frame_index % geometry_stride == 0)
        )
        run_segmentation = bool(allow_role_compute and role_segmentation)
        run_retrieval = (
            cold_start
            or (allow_role_compute and role_retrieval)
            or (uncertain and float(context.step_confidence) < 0.55)
            or (stable and frame_index % retrieval_stride == 0)
        )
        run_memory_embedding = bool(
            cold_start
            or (allow_role_compute and role_memory)
            or (stable and frame_index % memory_stride == 0)
        )

        if not query_intent:
            response_tier = "none"
            target_latency = float(self.config.target_perception_ms)
        elif query_intent in self._structured_intents:
            response_tier = "structured"
            target_latency = float(self.config.target_answer_ms)
        elif query_intent in self._small_llm_intents:
            response_tier = "small_llm"
            target_latency = float(self.config.target_answer_ms)
        else:
            response_tier = "medium_llm"
            target_latency = float(self.config.target_answer_ms)

        reasons = []
        if uncertain:
            reasons.append("claim_insufficient")
        if uncertain and geometry_value >= threshold:
            reasons.append("missing_geometric_role")
        if uncertain and detection_value >= threshold:
            reasons.append("missing_detection_role")
        if uncertain and retrieval_value >= threshold:
            reasons.append("missing_identity_role")
        if uncertain and history_value >= threshold:
            reasons.append("missing_history_role")
        if relation_query:
            reasons.append("relation_query")
        if context.interaction_transition:
            reasons.append("interaction_transition")
        if stable:
            reasons.append("stable_scene_budgeting")
        if cold_start:
            reasons.append("cold_start")
        if external_observation:
            reasons.append("persistent_missing_evidence")
        if stalled and uncertain and missing_key:
            reasons.append("no_epistemic_gain")
        if response_tier != "none":
            reasons.append(f"response:{response_tier}")

        stage_values = (
            ("detection", float(detection_value)),
            ("retrieval", float(retrieval_value)),
            ("geometry", float(geometry_value)),
            ("segmentation", float(segmentation_value)),
            ("memory", float(history_value)),
        )
        any_role_compute = any(
            value >= threshold
            for _, value in stage_values
        )
        acquisition_mode = (
            "external_observation"
            if external_observation
            else "internal_compute"
            if cold_start or context.visual_change or context.interaction_transition or any_role_compute
            else "monitor"
        )
        return ComputePlan(
            run_detection=run_detection,
            run_geometry=run_geometry,
            run_segmentation=run_segmentation,
            run_retrieval=run_retrieval,
            run_memory_embedding=run_memory_embedding,
            response_tier=response_tier,
            target_latency_ms=target_latency,
            acquisition_mode=acquisition_mode,
            external_observation_recommended=external_observation,
            stage_evidence_values=stage_values,
            reasons=tuple(reasons),
        )
