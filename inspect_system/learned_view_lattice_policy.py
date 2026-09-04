"""Runtime adapter from the shared verifier to the learned view selector."""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .active_view.active_selector import ActiveSelector
from .active_view.decidability_gated_transport import (
    ClaimStructuredEvidenceTransportSelector,
)
from .active_view.decidability_transport import (
    claim_decidability,
    load_typed_requirement_counts,
    mix_typed_requirement_weights,
    typed_claim_id,
)
from .active_view.counterfactual_transport import (
    counterfactual_mixture,
    infer_counterfactual_family,
)
from .active_view.counterfactual_evidence_belief import (
    CounterfactualEvidenceBeliefConfig,
    CounterfactualEvidenceBeliefMemory,
)
from .active_view.evidence_affordance_memory import EvidenceAffordanceContext
from .active_view.evidence_state import EvidenceItem, EvidenceState
from .active_view.object_centric_reveal import (
    ObjectCentricCalibratedRevealModel,
    camera_motion_from_lattice,
)
from .active_view.ontology import (
    infer_role_from_fields,
    normalize_claim,
    normalize_key,
    roles_for_claim,
)
from .active_view.requirement_model import (
    RequirementCalibration,
    blend_session_requirement_weights,
    load_requirement_counts,
    mix_requirement_weights,
)
from .active_view.reveal_model import PriorTableRevealModel
from .active_view.view_lattice import SIX_VIEWS, direction_token
from .types import (
    ActiveObservationDecision,
    ProceduralEvidenceGraph,
    RobotObservation,
    VerificationResult,
    ViewCandidate,
)


_CONTEXT_KEYS = (
    "candidate_role_factors",
    "preservation_role_factors",
    "surface_normal_world",
    "role_surface_normals_world",
    "role_relation_frames_world",
    "counterfactual_family",
    "counterfactual_scores",
)


@dataclass
class LearnedViewLatticePolicy:
    """Select a fixed robot view from assistant-derived R and pi.

    Candidate images and evaluation utilities are deliberately absent from
    this adapter. Geometry fields are accepted only when derived from the
    current observation.
    """

    selector: Any
    requirement_counts: Mapping[str, Mapping[str, float]]
    requirement_calibration: RequirementCalibration
    typed_requirement_counts: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )
    typed_requirement_enabled: bool = False
    selector_mode: str = "legacy"
    unified_claim_evidence_ledger: bool = False
    pareto_evidence_checkpoint: bool = False
    counterfactual_conditioned_requirement: bool = False
    online_update_enabled: bool = True
    online_update_weight: float = 1.0
    online_requirement_scale: float = 1.0
    online_requirement_blend: float = 1.0
    online_requirement_prior_strength: float = 0.05
    online_gain_deadband: float = 0.05
    session_requirement_counts: Dict[str, Dict[str, Dict[str, float]]] = field(
        default_factory=dict,
        repr=False,
    )
    session_role_evidence: Dict[
        str, Dict[str, Dict[str, Dict[str, float | str]]]
    ] = field(default_factory=dict, repr=False)
    session_evidence_checkpoints: Dict[str, Dict[str, Dict[str, Any]]] = field(
        default_factory=dict,
        repr=False,
    )
    session_visited_views: Dict[str, set[str]] = field(default_factory=dict, repr=False)
    online_update_log: list[Dict[str, Any]] = field(default_factory=list, repr=False)
    claim_belief_enabled: bool = False
    claim_belief_config: CounterfactualEvidenceBeliefConfig = field(
        default_factory=CounterfactualEvidenceBeliefConfig,
        repr=False,
    )
    session_claim_beliefs: Dict[
        str, CounterfactualEvidenceBeliefMemory
    ] = field(default_factory=dict, repr=False)

    @classmethod
    def from_paths(
        cls,
        *,
        reveal_model: str | Path,
        requirement_report: str | Path | None = None,
        requirement_calibration: str | Path | None = None,
        candidates: Sequence[ViewCandidate] | None = None,
        requirement_beta: float = 2.0,
        requirement_strength: float = 1.0,
        requirement_temperature: float = 1.0,
        counterfactual_conditioned_requirement: bool = False,
        lambda_cost: float = 0.05,
        tau_view: float = 0.02,
        online_update_enabled: bool = True,
        online_update_weight: float = 1.0,
        online_requirement_scale: float = 1.0,
        online_requirement_blend: float = 1.0,
        online_requirement_prior_strength: float = 0.05,
        online_gain_deadband: float = 0.05,
        claim_belief_enabled: bool = False,
        belief_support_threshold: float = 0.35,
        belief_contradiction_threshold: float = 0.45,
        belief_margin_threshold: float = 0.05,
        belief_role_threshold: float = 0.65,
        belief_conflict_threshold: float = 0.20,
        selector_mode: str = "legacy",
        typed_requirement_enabled: bool = False,
        unified_claim_evidence_ledger: bool = False,
        pareto_evidence_checkpoint: bool = False,
    ) -> "LearnedViewLatticePolicy":
        model = PriorTableRevealModel.load(reveal_model)
        candidate_ids = {
            str(candidate.view_id)
            for candidate in (candidates or [])
            if str(candidate.view_id) in SIX_VIEWS
        }
        views = {
            view_id: node
            for view_id, node in SIX_VIEWS.items()
            if not candidate_ids or view_id in candidate_ids
        }
        calibration = RequirementCalibration.load(
            requirement_calibration,
            beta=requirement_beta,
            strength=requirement_strength,
            temperature=requirement_temperature,
        )
        normalized_selector_mode = str(selector_mode).strip().lower()
        if normalized_selector_mode not in {"legacy", "claim_structured"}:
            raise ValueError(f"Unknown selector mode: {selector_mode}")
        selector = (
            ClaimStructuredEvidenceTransportSelector(
                model=model,
                views=views,
                lambda_cost=float(lambda_cost),
                tau_view=float(tau_view),
            )
            if normalized_selector_mode == "claim_structured"
            else ActiveSelector(
                model=model,
                views=views,
                lambda_cost=float(lambda_cost),
                tau_view=float(tau_view),
            )
        )
        return cls(
            selector=selector,
            requirement_counts=load_requirement_counts(
                requirement_report,
                counterfactual_conditioned=bool(
                    counterfactual_conditioned_requirement
                ),
            ),
            requirement_calibration=calibration,
            typed_requirement_counts=(
                load_typed_requirement_counts(requirement_report)
                if typed_requirement_enabled and requirement_report is not None
                else {}
            ),
            typed_requirement_enabled=bool(typed_requirement_enabled),
            selector_mode=normalized_selector_mode,
            unified_claim_evidence_ledger=bool(unified_claim_evidence_ledger),
            pareto_evidence_checkpoint=bool(pareto_evidence_checkpoint),
            counterfactual_conditioned_requirement=bool(
                counterfactual_conditioned_requirement
            ),
            online_update_enabled=bool(online_update_enabled),
            online_update_weight=max(0.0, float(online_update_weight)),
            online_requirement_scale=max(0.0, float(online_requirement_scale)),
            online_requirement_blend=_clamp01(online_requirement_blend),
            online_requirement_prior_strength=max(
                1e-9, float(online_requirement_prior_strength)
            ),
            online_gain_deadband=max(0.0, float(online_gain_deadband)),
            claim_belief_enabled=bool(claim_belief_enabled),
            claim_belief_config=CounterfactualEvidenceBeliefConfig(
                support_threshold=float(belief_support_threshold),
                contradiction_threshold=float(belief_contradiction_threshold),
                margin_threshold=float(belief_margin_threshold),
                role_threshold=float(belief_role_threshold),
                conflict_threshold=float(belief_conflict_threshold),
            ),
        )

    @classmethod
    def from_claim_structured_paths(
        cls,
        **kwargs: Any,
    ) -> "LearnedViewLatticePolicy":
        """Build the adopted successor without changing legacy call sites."""

        return cls.from_paths(
            **{
                **kwargs,
                "selector_mode": "claim_structured",
                "typed_requirement_enabled": True,
            }
        )

    def accumulate_verification(
        self,
        observation: RobotObservation,
        verification: VerificationResult,
    ) -> VerificationResult:
        """Fuse evidence within one inspection session and claim only."""

        if not self.claim_belief_enabled:
            return verification
        session_id = self._session_id(observation, verification)
        memory = self.session_claim_beliefs.setdefault(
            session_id,
            CounterfactualEvidenceBeliefMemory(config=self.claim_belief_config),
        )
        return memory.observe(verification)

    def _apply_requirement_model(
        self,
        state: EvidenceState,
        session_id: str,
    ) -> EvidenceState:
        if self.typed_requirement_enabled:
            state = mix_typed_requirement_weights(
                state,
                self.requirement_counts,
                self.typed_requirement_counts,
                beta=self.requirement_calibration.beta,
                temperature=self.requirement_calibration.temperature,
            )
        else:
            state = mix_requirement_weights(
                state,
                self._effective_requirement_counts(session_id),
                beta=self.requirement_calibration.beta,
                strength=self.requirement_calibration.strength,
                temperature=self.requirement_calibration.temperature,
                preserve_scores=True,
            )
        return blend_session_requirement_weights(
            state,
            self.session_requirement_counts.get(session_id, {}).get(
                state.claim_id,
                {},
            ),
            blend=self.online_requirement_blend,
            prior_strength=self.online_requirement_prior_strength,
        )

    @staticmethod
    def _state_decidability(state: EvidenceState) -> float:
        return LearnedViewLatticePolicy._state_decidability_components(state)[0]

    @staticmethod
    def _state_decidability_components(
        state: EvidenceState,
    ) -> tuple[float, Dict[str, float]]:
        values = {
            normalize_key(item.evidence_role or item.name): _clamp01(
                float(item.score) / max(1e-9, float(item.threshold))
            )
            for item in state.items
        }
        score, clauses = claim_decidability(state, values)
        return float(score), {str(key): float(value) for key, value in clauses.items()}

    def _evidence_ledger_key(self, state: EvidenceState) -> str:
        return self._evidence_ledger_key_from_fields(
            state.claim_id, state.counterfactual_family
        )

    def _evidence_ledger_key_from_fields(
        self,
        claim_id: str,
        counterfactual_family: str,
    ) -> str:
        claim = (
            typed_claim_id(claim_id)
            if self.typed_requirement_enabled
            else normalize_claim(claim_id)
        )
        if self.unified_claim_evidence_ledger:
            return claim
        family = normalize_key(counterfactual_family) or "__any__"
        return f"{claim}|{family}"

    def _remember_evidence(
        self,
        *,
        session_id: str,
        view_id: str,
        state: EvidenceState,
    ) -> EvidenceState:
        """Retain role evidence and the best model-scored observed view."""

        key = self._evidence_ledger_key(state)
        current_score, current_clauses = self._state_decidability_components(state)
        checkpoints = self.session_evidence_checkpoints.setdefault(session_id, {})
        previous = checkpoints.get(key)
        should_replace = previous is None or current_score > float(
            (previous or {}).get("score", -1.0)
        )
        if should_replace and previous is not None and self.pareto_evidence_checkpoint:
            previous_clauses = dict(previous.get("clauses") or {})
            should_replace = all(
                current_clauses.get(name, 0.0) + 1e-9 >= float(value)
                for name, value in previous_clauses.items()
            )
        if should_replace:
            checkpoints[key] = {
                "view_id": view_id,
                "score": current_score,
                "clauses": current_clauses,
            }

        session = self.session_role_evidence.setdefault(session_id, {})
        ledger = session.setdefault(key, {})
        for item in state.items:
            role = normalize_key(item.evidence_role or item.name)
            stored = ledger.get(role)
            if stored is None or float(item.score) > float(stored.get("score", -1.0)):
                ledger[role] = {
                    "score": float(item.score),
                    "threshold": float(item.threshold),
                    "view_id": view_id,
                }
        fused_items = []
        for item in state.items:
            role = normalize_key(item.evidence_role or item.name)
            stored = ledger.get(role, {})
            fused_items.append(
                EvidenceItem(
                    name=item.name,
                    evidence_role=item.evidence_role,
                    score=max(float(item.score), float(stored.get("score", 0.0))),
                    threshold=float(item.threshold),
                    importance=float(item.importance),
                )
            )
        return EvidenceState(
            claim_id=state.claim_id,
            items=fused_items,
            claim_score=state.claim_score,
            contradiction_score=state.contradiction_score,
            margin=state.margin,
            counterfactual_family=state.counterfactual_family,
            counterfactual_scores=dict(state.counterfactual_scores),
            current_utility_proxy=state.current_utility_proxy,
        )

    def best_evidence_view(
        self,
        session_id: str,
        claim_id: str,
        counterfactual_family: str = "",
    ) -> str:
        key = self._evidence_ledger_key_from_fields(
            claim_id, counterfactual_family
        )
        return str(
            self.session_evidence_checkpoints.get(session_id, {})
            .get(key, {})
            .get("view_id", "")
        )

    def remember_observation_evidence(
        self,
        observation: RobotObservation,
        verification: VerificationResult,
        graph: Optional[ProceduralEvidenceGraph] = None,
    ) -> EvidenceState:
        """Record an acquired observation even when no further move is requested."""

        session_id = self._session_id(observation, verification)
        state = self._evidence_state(observation, verification, graph)
        state = self._apply_requirement_model(state, session_id)
        if self.selector_mode == "claim_structured":
            state = self._remember_evidence(
                session_id=session_id,
                view_id=str(observation.view_id),
                state=state,
            )
        return state

    def select_view(
        self,
        observation: RobotObservation,
        verification: VerificationResult,
        graph: Optional[ProceduralEvidenceGraph] = None,
    ) -> ActiveObservationDecision:
        session_id = self._session_id(observation, verification)
        visited_views = self.session_visited_views.setdefault(session_id, set())
        visited_views.add(str(observation.view_id))
        verification = self.accumulate_verification(observation, verification)
        state = self.remember_observation_evidence(
            observation, verification, graph
        )
        if verification.verified:
            return ActiveObservationDecision(
                observation_id=verification.observation_id,
                selected_view="",
                utility=float(verification.confidence),
                expected_observed_evidence=[],
                unresolved_evidence=[],
                candidate_scores={},
                metadata={
                    "policy": (
                        "claim_structured_evidence_transport"
                        if self.selector_mode == "claim_structured"
                        else "assistant_trace_guided_lattice_v2"
                    ),
                    "selector_action": "stay",
                    "selector_reason": "claim_resolved_by_counterfactual_belief",
                    "claim_id": state.claim_id,
                    "current_view": observation.view_id,
                    "session_id": session_id,
                    "counterfactual_evidence_belief": dict(
                        verification.metadata.get(
                            "counterfactual_evidence_belief", {}
                        )
                    ),
                    "uses_candidate_view_images": False,
                    "uses_robot_utility_labels": False,
                },
            )
        selection = self.selector.select(
            current_view=observation.view_id,
            evidence_state=state,
            visited_views=sorted(visited_views),
            candidate_context=self._candidate_context(
                observation, verification, session_id
            ),
        )
        selected_view = selection.selected_view if selection.action == "move" else ""
        candidate_scores = {
            str(item.get("view_id", "")): float(item.get("score", 0.0))
            for item in selection.ranked_views
            if item.get("view_id")
        }
        return ActiveObservationDecision(
            observation_id=verification.observation_id,
            selected_view=selected_view,
            utility=float(selection.score),
            expected_observed_evidence=list(selection.missing_evidence),
            unresolved_evidence=list(selection.missing_evidence),
            candidate_scores=candidate_scores,
            metadata={
                "policy": (
                    "claim_structured_evidence_transport"
                    if self.selector_mode == "claim_structured"
                    else "assistant_trace_guided_lattice_v2"
                ),
                "selector_action": selection.action,
                "selector_reason": selection.reason,
                "claim_id": state.claim_id,
                "current_view": observation.view_id,
                "session_id": session_id,
                "online_update_enabled": bool(self.online_update_enabled),
                "evidence_state": state.to_dict(),
                "ranked_views": selection.ranked_views,
                "reveal_model_type": str(
                    self.selector.model.metadata.get(
                        "model_type", type(self.selector.model).__name__
                    )
                ),
                "requirement_beta": self.requirement_calibration.beta,
                "requirement_strength": self.requirement_calibration.strength,
                "requirement_temperature": self.requirement_calibration.temperature,
                "requirement_conditioning": (
                    "typed_claim"
                    if self.typed_requirement_enabled
                    else (
                        "claim_and_counterfactual_family"
                        if self.counterfactual_conditioned_requirement
                        else "claim_only"
                    )
                ),
                "online_requirement_blend": self.online_requirement_blend,
                "online_requirement_prior_strength": self.online_requirement_prior_strength,
                "uses_candidate_view_images": False,
                "uses_robot_utility_labels": False,
                "training_signal": "assistant_uncertainty_resolution_traces",
            },
        )

    def _evidence_state(
        self,
        observation: RobotObservation,
        verification: VerificationResult,
        graph: Optional[ProceduralEvidenceGraph],
    ) -> EvidenceState:
        metadata = {**observation.metadata, **verification.metadata}
        claim_id = self._claim_id(metadata, verification.predicted_state)
        role_priors = roles_for_claim(claim_id)
        missing_raw = list(verification.missing_evidence)
        missing_raw.extend(_as_list(metadata.get("missing_roles")))
        missing_roles = {
            infer_role_from_fields(claim_id=claim_id, missing_evidence=value)
            for value in missing_raw
            if str(value).strip()
        }
        explicit_scores = _score_map(metadata.get("evidence_role_scores"))
        explicit_scores.update(_score_map(metadata.get("role_scores")))
        coverage = _clamp01(verification.evidence_coverage)
        items = []
        for role, importance in role_priors.items():
            score = explicit_scores.get(
                role, 0.0 if role in missing_roles else coverage
            )
            items.append(
                EvidenceItem(
                    name=role,
                    evidence_role=role,
                    score=_clamp01(score),
                    threshold=_clamp01(metadata.get("role_threshold", 0.65)),
                    importance=float(importance),
                )
            )
        support_score = float(
            metadata.get(
                "support_score",
                verification.confidence if verification.verified else 0.0,
            )
        )
        contradiction_score = float(metadata.get("contradiction_score", 0.0))
        return EvidenceState(
            claim_id=claim_id,
            items=items,
            claim_score=_clamp01(support_score),
            contradiction_score=_clamp01(contradiction_score),
            margin=float(
                metadata.get("counterfactual_margin", metadata.get("margin", 0.0))
                or 0.0
            ),
            counterfactual_family=str(metadata.get("counterfactual_family", "")),
            counterfactual_scores=_score_map(metadata.get("counterfactual_scores")),
        )

    def _claim_id(self, metadata: Mapping[str, Any], fallback: str) -> str:
        for key in ("claim_id", "claim_type", "claim", "state_claim", "target_claim"):
            if metadata.get(key):
                return (
                    typed_claim_id(metadata[key])
                    if self.typed_requirement_enabled
                    else normalize_claim(metadata[key])
                )
        return (
            typed_claim_id(fallback)
            if self.typed_requirement_enabled
            else normalize_claim(fallback)
        )

    @staticmethod
    def _session_id(
        observation: RobotObservation,
        verification: VerificationResult,
    ) -> str:
        metadata = {**observation.metadata, **verification.metadata}
        for key in ("session_id", "episode_id", "trial_id", "setup_id", "video"):
            value = str(metadata.get(key, "")).strip()
            if value:
                return value
        observation_id = str(
            observation.observation_id or verification.observation_id
        ).strip()
        for view_id in SIX_VIEWS:
            for separator in ("_", "-", "/"):
                suffix = f"{separator}{view_id}"
                if observation_id.endswith(suffix):
                    return observation_id[: -len(suffix)]
        return observation_id or "robot_session"

    def _effective_requirement_counts(
        self,
        session_id: str,
    ) -> Mapping[str, Mapping[str, float]]:
        local = self.session_requirement_counts.get(session_id)
        if not local:
            return self.requirement_counts
        if self.online_requirement_blend > 0.0:
            return self.requirement_counts
        merged = {
            str(claim): {str(role): float(value) for role, value in values.items()}
            for claim, values in self.requirement_counts.items()
        }
        for claim, values in local.items():
            bucket = merged.setdefault(str(claim), {})
            for role, value in values.items():
                bucket[str(role)] = float(bucket.get(str(role), 0.0)) + float(value)
        return merged

    @staticmethod
    def _online_trust(verification: VerificationResult) -> float:
        metadata = verification.metadata
        for key in (
            "online_update_trust",
            "verification_trust",
            "evidence_trust",
            "source_trust",
        ):
            if metadata.get(key) not in (None, ""):
                return _clamp01(metadata[key])
        trust = max(0.0, float(verification.evidence_coverage))
        if verification.verified:
            trust = max(trust, float(verification.confidence))
        if verification.recommended_action != "observe":
            trust = max(trust, 0.5)
        return _clamp01(trust)

    @staticmethod
    def _role_gain(
        before: EvidenceItem,
        after: EvidenceItem | None,
        after_missing: set[str],
    ) -> float:
        after_score = float(after.score) if after is not None else 0.0
        scale = max(0.10, float(before.threshold))
        delta = (after_score - float(before.score)) / scale
        role = normalize_key(before.evidence_role or before.name)
        if after is not None and after.observed:
            delta = max(delta, 0.35)
        elif role not in after_missing:
            delta = max(delta, 0.20)
        return max(-1.0, min(1.0, float(delta)))

    @staticmethod
    def _transition_stability(
        before_observation: RobotObservation,
        before_verification: VerificationResult,
        after_observation: RobotObservation,
        after_verification: VerificationResult,
    ) -> tuple[float, str]:
        """Reject updates that may be explained by a workpiece state change."""

        before = {**before_observation.metadata, **before_verification.metadata}
        after = {**after_observation.metadata, **after_verification.metadata}
        for key in (
            "state_changed",
            "workpiece_changed",
            "object_reoriented",
            "manipulation_between_views",
            "non_transferable_transition",
        ):
            if bool(before.get(key)) or bool(after.get(key)):
                return 0.0, key
        before_claim = typed_claim_id(
            before.get("claim_id", before_verification.predicted_state)
        )
        after_claim = typed_claim_id(
            after.get("claim_id", after_verification.predicted_state)
        )
        if before_claim and after_claim and before_claim != after_claim:
            return 0.0, "claim_changed"
        stability = min(
            _clamp01(before.get("evidence_stability", 1.0)),
            _clamp01(after.get("evidence_stability", 1.0)),
        )
        return stability, "stable_view_change"

    def update_after_observation(
        self,
        *,
        before_observation: RobotObservation,
        before_verification: VerificationResult,
        decision: ActiveObservationDecision,
        after_observation: RobotObservation,
        after_verification: VerificationResult,
    ) -> Dict[str, Any]:
        """Update session-local transport from causal evidence deltas.

        Partial score changes update the reveal memory. The requirement model
        is updated only when the claim becomes fully resolved.
        """

        model = self.selector.model
        if not self.online_update_enabled:
            return {"applied": False, "reason": "online_update_disabled"}
        if not isinstance(model, ObjectCentricCalibratedRevealModel):
            return {"applied": False, "reason": "reveal_model_is_not_object_centric"}
        current_view = str(before_observation.view_id)
        selected_view = str(after_observation.view_id or decision.selected_view)
        if (
            current_view not in self.selector.views
            or selected_view not in self.selector.views
            or current_view == selected_view
        ):
            return {"applied": False, "reason": "invalid_or_stationary_transition"}

        before_decision = normalize_key(
            before_verification.metadata.get("decision", "")
        )
        after_decision = normalize_key(
            after_verification.metadata.get("decision", "")
        )
        before_is_insufficient = (
            before_decision in {"insufficient", "unresolved"}
            or (
                not before_decision
                and not before_verification.verified
                and before_verification.recommended_action == "observe"
            )
        )
        after_is_resolved = (
            after_decision in {"supported", "contradicted"}
            or (
                not after_decision
                and (
                    after_verification.verified
                    or after_verification.recommended_action != "observe"
                )
            )
        )
        if not before_is_insufficient:
            return {
                "applied": False,
                "reason": "source_not_insufficient",
                "role_updates": [],
                "destination_ray_updates": 0,
                "requirement_mass": 0.0,
                "gain_deadband": float(self.online_gain_deadband),
            }
        stability, stability_reason = self._transition_stability(
            before_observation,
            before_verification,
            after_observation,
            after_verification,
        )
        if stability <= 0.0:
            return {
                "applied": False,
                "reason": f"non_causal_transition:{stability_reason}",
                "role_updates": [],
                "destination_ray_updates": 0,
                "requirement_mass": 0.0,
                "gain_deadband": float(self.online_gain_deadband),
            }

        session_id = self._session_id(before_observation, before_verification)
        before_state = self._evidence_state(
            before_observation,
            before_verification,
            None,
        )
        after_state = self._evidence_state(
            after_observation,
            after_verification,
            None,
        )
        if self.selector_mode == "claim_structured":
            remembered_after = self._apply_requirement_model(after_state, session_id)
            self._remember_evidence(
                session_id=session_id,
                view_id=selected_view,
                state=remembered_after,
            )
        after_items = {
            normalize_key(item.evidence_role or item.name): item
            for item in after_state.items
        }
        after_missing = {
            normalize_key(item.evidence_role or item.name)
            for item in after_state.missing()
        }
        decision_roles = {
            normalize_key(value)
            for value in decision.unresolved_evidence
            if str(value).strip()
        }
        missing_before = [
            item
            for item in before_state.missing()
            if not decision_roles
            or normalize_key(item.evidence_role or item.name) in decision_roles
        ]
        if not missing_before:
            return {"applied": False, "reason": "no_selected_missing_roles"}

        context = self._candidate_context(
            before_observation,
            before_verification,
            session_id,
        )
        role_normals = dict(context.get("role_surface_normals_world") or {})
        role_relation_frames = dict(context.get("role_relation_frames_world") or {})
        action = normalize_key(
            direction_token(
                current_view,
                selected_view,
                self.selector.views,
            )
        )
        trust = self._online_trust(after_verification) * stability * max(
            0.0,
            float(self.online_update_weight),
        )
        if trust <= 0.0:
            return {"applied": False, "reason": "zero_update_trust"}

        role_gains: Dict[str, float] = {}
        role_updates: list[Dict[str, Any]] = []
        destination_updates = 0
        for before_item in missing_before:
            role = normalize_key(before_item.evidence_role or before_item.name)
            gain = self._role_gain(
                before_item,
                after_items.get(role),
                after_missing,
            )
            if abs(gain) < float(self.online_gain_deadband):
                continue
            role_gains[role] = gain
            surface_normal = role_normals.get(role)
            if surface_normal is None:
                surface_normal = context.get("surface_normal_world")
            relation_frame = role_relation_frames.get(role)
            motion = camera_motion_from_lattice(
                current_view,
                selected_view,
                self.selector.views,
                surface_normal_world=surface_normal,
                relation_frame_world=relation_frame,
            )
            fallback_family = infer_counterfactual_family(
                before_state.claim_id,
                role,
                context,
            )
            family_weights = counterfactual_mixture(context, fallback_family)
            for family, family_weight in family_weights.items():
                family_key = normalize_key(family)
                memory = model.family_memories.get(family_key, model.memory)
                evidence_context = EvidenceAffordanceContext(
                    action=action,
                    role=role,
                    counterfactual=family_key,
                    claim=before_state.claim_id,
                )
                update_weight = trust * max(0.0, float(family_weight))
                if update_weight <= 0.0:
                    continue
                memory.update(
                    evidence_context,
                    motion,
                    signed_gain=gain,
                    weight=update_weight,
                    session_id=session_id,
                    update_shared=False,
                )
                role_updates.append(
                    {
                        "role": role,
                        "counterfactual_family": family_key,
                        "signed_gain": gain,
                        "weight": update_weight,
                    }
                )
            destination_updates += model.update_online_destination(
                session_id=session_id,
                claim_id=before_state.claim_id,
                evidence_role=role,
                selected_view=selected_view,
                views=self.selector.views,
                signed_gain=gain,
                weight=trust,
                context=context,
            )

        positive_gains = {
            role: max(0.0, gain) for role, gain in role_gains.items() if gain > 0.0
        }
        positive_mass = sum(positive_gains.values())
        requirement_mass = 0.0
        if (
            after_is_resolved
            and positive_mass > 1e-9
            and self.online_requirement_scale > 0.0
        ):
            requirement_mass = trust * max(0.0, float(self.online_requirement_scale))
            session_bucket = self.session_requirement_counts.setdefault(
                session_id,
                {},
            )
            claim_bucket = session_bucket.setdefault(before_state.claim_id, {})
            for role, gain in positive_gains.items():
                increment = requirement_mass * gain / positive_mass
                claim_bucket[role] = float(claim_bucket.get(role, 0.0)) + increment

        summary = {
            "applied": bool(role_updates or destination_updates),
            "reason": (
                "causal_evidence_delta"
                if role_updates or destination_updates
                else "no_role_evidence_change"
            ),
            "source": "post_move_reverification",
            "update_scope": (
                "transport_and_requirement"
                if after_is_resolved
                else "transport_only_partial"
            ),
            "claim_resolved_after_move": bool(after_is_resolved),
            "evidence_stability": stability,
            "stability_reason": stability_reason,
            "session_id": session_id,
            "claim_id": before_state.claim_id,
            "current_view": current_view,
            "selected_view": selected_view,
            "relative_action": action,
            "trust": trust,
            "gain_deadband": float(self.online_gain_deadband),
            "role_updates": role_updates,
            "destination_ray_updates": destination_updates,
            "requirement_mass": requirement_mass,
            "uses_candidate_view_images": False,
            "uses_robot_utility_labels": False,
        }
        self.online_update_log.append(summary)
        if len(self.online_update_log) > 512:
            del self.online_update_log[:-512]
        return summary

    def save_online_state(self, reveal_model_path: str | Path) -> Dict[str, str]:
        """Persist the adapted reveal model and its session-local R sidecar."""

        output = Path(reveal_model_path)
        self.selector.model.save(output)
        suffix = output.suffix or ".json"
        requirements_output = output.with_name(
            f"{output.stem}.session_requirements{suffix}"
        )
        requirements_output.write_text(
            json.dumps(
                {
                    "schema": "inspect_online_requirement_state_v1",
                    "session_requirement_counts": self.session_requirement_counts,
                    "num_updates": len(self.online_update_log),
                    "uses_candidate_view_images": False,
                    "uses_robot_utility_labels": False,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {
            "reveal_model": str(output),
            "session_requirements": str(requirements_output),
        }

    @staticmethod
    def _candidate_context(
        observation: RobotObservation,
        verification: VerificationResult,
        session_id: str = "",
    ) -> Dict[str, Any]:
        metadata = {**observation.metadata, **verification.metadata}
        if metadata.get("candidate_images_used") or metadata.get(
            "robot_utility_labels_used"
        ):
            raise RuntimeError(
                "Learned lattice policy received forbidden candidate-image or robot-utility context."
            )
        context = {key: metadata[key] for key in _CONTEXT_KEYS if key in metadata}
        if session_id:
            context["session_id"] = session_id
        return context


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _score_map(value: object) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): float(score) for key, score in value.items()}


def _clamp01(value: object) -> float:
    return max(0.0, min(1.0, float(value)))
