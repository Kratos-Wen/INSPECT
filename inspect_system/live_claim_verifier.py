"""Causal claim verification for the online assistant runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, Tuple

from .evidence_scorer import (
    PrototypeEvidenceScorer,
    det_conf,
    det_name,
    extract_relation_features,
    identity_safe,
    normalize_claim,
)

if TYPE_CHECKING:
    from .causal_evidence_bank import TriageOutput


CLAIM_BY_STEP = {
    "S1": "component_present",
    "S2": "small_gear_inserted",
    "S3": "big_gear_inserted",
    "S4": "cover_seated",
}

PREREQUISITES_BY_STEP = {
    "S2": (),
    "S3": ("small_gear_inserted",),
    "S4": ("small_gear_inserted", "big_gear_inserted"),
}

STEP_BY_CLAIM = {claim: step for step, claim in CLAIM_BY_STEP.items()}

FAMILY_CLASSES = {
    "A": {
        "type_5_gearbox_housing": 2.0,
        "type_5_gearbox_cover": 1.25,
        "type_3_gear": 0.6,
        "type_8_gear": 0.6,
    },
    "B": {
        "type_6_gearbox_housing": 2.0,
        "type_6_gearbox_cover": 1.25,
        "type_7_gear": 0.6,
        "type_2_gear": 0.6,
    },
}


@dataclass(frozen=True)
class LiveClaimDecision:
    """One causal verifier output exposed to memory and the assistant."""

    frame_index: int
    proposed_step: str
    committed_step: str
    claim_id: str
    product_family: str
    state: str
    support_score: float
    contradiction_score: float
    visibility_score: float
    counterfactual_margin: float
    admissible: bool
    memory_ready: bool
    missing_roles: Tuple[str, ...] = ()
    features: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "frame_index": int(self.frame_index),
            "proposed_step": self.proposed_step,
            "committed_step": self.committed_step,
            "claim_id": self.claim_id,
            "product_family": self.product_family,
            "state": self.state,
            "support_score": float(self.support_score),
            "contradiction_score": float(self.contradiction_score),
            "visibility_score": float(self.visibility_score),
            "counterfactual_margin": float(self.counterfactual_margin),
            "admissible": bool(self.admissible),
            "memory_ready": bool(self.memory_ready),
            "missing_roles": list(self.missing_roles),
            "features": {str(key): float(value) for key, value in self.features.items()},
        }


class LiveClaimVerifier:
    """Apply the frozen assistant-trained scorer with causal runtime state."""

    def __init__(
        self,
        scorer: Optional[PrototypeEvidenceScorer],
        *,
        support_threshold: float = 0.35,
        contradiction_threshold: float = 0.35,
        counterfactual_margin: float = 0.0,
        admissibility_gate_enabled: bool = True,
        memory_gate_enabled: bool = True,
        specialized_counterfactual_enabled: bool = True,
        prerequisite_bootstrap_enabled: bool = True,
        prerequisite_confirmation_frames: int = 2,
        ema_decay: float = 0.55,
        require_step_match_for_support: bool = True,
        product_family: str = "",
        family_min_confidence: float = 0.08,
        family_margin: float = 0.03,
        family_confirmation_frames: int = 2,
    ) -> None:
        self.scorer = scorer
        self.support_threshold = float(support_threshold)
        self.contradiction_threshold = float(contradiction_threshold)
        self.counterfactual_margin = float(counterfactual_margin)
        self.admissibility_gate_enabled = bool(admissibility_gate_enabled)
        self.memory_gate_enabled = bool(memory_gate_enabled)
        self.specialized_counterfactual_enabled = bool(specialized_counterfactual_enabled)
        self.prerequisite_bootstrap_enabled = bool(prerequisite_bootstrap_enabled)
        self.prerequisite_confirmation_frames = max(1, int(prerequisite_confirmation_frames))
        self.ema_decay = max(0.0, min(0.999, float(ema_decay)))
        self.require_step_match_for_support = bool(require_step_match_for_support)
        configured_family = str(product_family or "").strip().upper()
        self.configured_product_family = configured_family if configured_family in FAMILY_CLASSES else ""
        self.family_min_confidence = max(0.0, float(family_min_confidence))
        self.family_margin = max(0.0, float(family_margin))
        self.family_confirmation_frames = max(1, int(family_confirmation_frames))
        self.reset()

    def reset(self) -> None:
        self._ema: Dict[str, Tuple[float, float]] = {}
        self._margin_ema: Dict[str, float] = {}
        self._claim_states: Dict[str, str] = {}
        self._committed_step = ""
        self._product_family = self.configured_product_family
        self._family_candidate = ""
        self._family_candidate_hits = 0
        self._prerequisite_candidate_hits: Dict[str, int] = {}

    @property
    def committed_step(self) -> str:
        return self._committed_step

    @property
    def product_family(self) -> str:
        return self._product_family

    def infer_product_family(self, detections: Sequence[Any]) -> str:
        """Update and return the causal product-family estimate once per frame."""

        return self._infer_product_family(detections)

    def confirm_step(self, step_id: str) -> None:
        """Commit explicit trusted operator feedback without changing scorer weights."""

        step = self._normalize_step(step_id)
        if step not in CLAIM_BY_STEP:
            return
        claim = CLAIM_BY_STEP[step]
        self._claim_states[claim] = "supported"
        self._commit(step)

    def verify(
        self,
        detections: Sequence[Any],
        *,
        frame_index: int,
        proposed_step: str,
        proposal_confidence: float,
        image_shape: Optional[Tuple[int, int]] = None,
        learned_triage: Optional["TriageOutput"] = None,
        product_family: Optional[str] = None,
    ) -> LiveClaimDecision:
        step = self._normalize_step(proposed_step)
        claim = CLAIM_BY_STEP.get(step, normalize_claim("", step))
        supplied_product = str(product_family or "").strip().upper()
        product = (
            supplied_product
            if supplied_product in FAMILY_CLASSES
            else self._infer_product_family(detections)
        )
        step_match = step == self._normalize_step(proposed_step) and float(proposal_confidence) > 0.0

        if step == "S1":
            raw_support = max((det_conf(item) for item in detections), default=0.0)
            support, contradiction = self._update_ema(claim, raw_support, 0.0)
            admissible = True
            memory_ready = True
            actual_counterfactual_margin = support - contradiction
            state = self._decide(
                support,
                contradiction,
                actual_counterfactual_margin,
                admissible,
                memory_ready,
                step_match,
            )
            visibility = raw_support
            features: Dict[str, float] = {
                "top_conf": raw_support,
                "det_count": float(len(detections)),
            }
        elif step not in {"S2", "S3", "S4"} or not product:
            support, contradiction = self._update_ema(claim, 0.0, 0.0)
            admissible = self._is_admissible(step)
            memory_ready = self._memory_ready(step)
            state = "insufficient"
            actual_counterfactual_margin = support - contradiction
            visibility = max((det_conf(item) for item in detections), default=0.0)
            features = {"top_conf": visibility, "det_count": float(len(detections))}
        elif learned_triage is not None:
            bootstrapped = self._bootstrap_prerequisites(
                detections,
                step=step,
                product=product,
                image_shape=image_shape,
            )
            support = float(learned_triage.support_score)
            contradiction = float(learned_triage.contradiction_score)
            visibility = float(learned_triage.visibility_score)
            admissible = self._is_admissible(step)
            memory_ready = self._memory_ready(step)
            actual_counterfactual_margin = float(learned_triage.posterior_margin)
            state = str(learned_triage.state)
            if not admissible:
                state = "contradicted"
            elif state == "supported" and not memory_ready:
                state = "insufficient"
            elif (
                state == "supported"
                and self.require_step_match_for_support
                and not step_match
            ):
                state = "insufficient"
            features = dict(learned_triage.features)
            features["learned_insufficient_score"] = float(
                learned_triage.insufficient_score
            )
            features["prerequisite_bootstrap_count"] = float(bootstrapped)
            features["counterfactual_margin"] = actual_counterfactual_margin
            features["learned_triage"] = 1.0
        else:
            if self.scorer is None:
                raise RuntimeError(
                    "Claim verification requires a learned triage output "
                    "or a prototype evidence scorer"
                )
            bootstrapped = self._bootstrap_prerequisites(
                detections,
                step=step,
                product=product,
                image_shape=image_shape,
            )
            features = extract_relation_features(
                detections,
                claim_id=claim,
                step_id=step,
                product=product,
                image_shape=image_shape,
            )
            raw_scores = self.scorer.score_features(features, claim_id=claim, step_id=step)
            support, contradiction = self._update_ema(
                claim,
                float(raw_scores.get("support_score", 0.0)),
                float(raw_scores.get("contradiction_score", 0.0)),
            )
            visibility = float(raw_scores.get("visibility_score", 0.0))
            admissible = self._is_admissible(step)
            memory_ready = self._memory_ready(step)
            counterfactual_support = self._counterfactual_support(
                detections,
                claim=claim,
                step=step,
                product=product,
                image_shape=image_shape,
            )
            actual_counterfactual_margin = self._update_margin_ema(
                claim,
                float(raw_scores.get("support_score", 0.0)) - counterfactual_support,
            )
            features = dict(features)
            features["prerequisite_bootstrap_count"] = float(bootstrapped)
            features["counterfactual_support"] = counterfactual_support
            features["counterfactual_margin"] = actual_counterfactual_margin
            state = self._decide(
                support,
                contradiction,
                actual_counterfactual_margin,
                admissible,
                memory_ready,
                step_match,
            )

        features = dict(features)
        features["decision_admissible"] = float(admissible)
        features["decision_memory_ready"] = float(memory_ready)
        features["decision_step_match"] = float(step_match)

        if state in {"supported", "contradicted"}:
            self._claim_states[claim] = state
        if state == "supported":
            self._commit(step)

        missing_roles = self._missing_roles(
            step=step,
            state=state,
            product=product,
            features=features,
            support=support,
            contradiction=contradiction,
        )
        return LiveClaimDecision(
            frame_index=int(frame_index),
            proposed_step=step,
            committed_step=self._committed_step,
            claim_id=claim,
            product_family=product,
            state=state,
            support_score=support,
            contradiction_score=contradiction,
            visibility_score=visibility,
            counterfactual_margin=actual_counterfactual_margin,
            admissible=admissible,
            memory_ready=memory_ready,
            missing_roles=missing_roles,
            features=features,
        )

    @staticmethod
    def _normalize_step(step_id: object) -> str:
        text = str(step_id or "").strip().upper()
        if text.startswith("STEP") and text[4:].isdigit():
            return f"S{text[4:]}"
        return text

    def _update_ema(self, claim: str, support: float, contradiction: float) -> Tuple[float, float]:
        current = (max(0.0, min(1.0, support)), max(0.0, min(1.0, contradiction)))
        previous = self._ema.get(claim)
        if previous is None:
            smoothed = current
        else:
            decay = self.ema_decay
            smoothed = (
                decay * previous[0] + (1.0 - decay) * current[0],
                decay * previous[1] + (1.0 - decay) * current[1],
            )
        self._ema[claim] = smoothed
        return smoothed

    def _update_margin_ema(self, claim: str, margin: float) -> float:
        current = max(-1.0, min(1.0, float(margin)))
        previous = self._margin_ema.get(claim)
        smoothed = current if previous is None else self.ema_decay * previous + (1.0 - self.ema_decay) * current
        self._margin_ema[claim] = smoothed
        return smoothed

    def _decide(
        self,
        support: float,
        contradiction: float,
        actual_counterfactual_margin: float,
        admissible: bool,
        memory_ready: bool,
        step_match: bool,
    ) -> str:
        margin = self.counterfactual_margin
        if not admissible:
            return "contradicted"
        if contradiction >= self.contradiction_threshold and contradiction >= support + margin:
            return "contradicted"
        if (
            support >= self.support_threshold
            and support >= contradiction + margin
            and actual_counterfactual_margin >= margin
        ):
            if not memory_ready:
                return "insufficient"
            if self.require_step_match_for_support and not step_match:
                return "insufficient"
            return "supported"
        return "insufficient"

    def _is_admissible(self, step: str) -> bool:
        if not self.admissibility_gate_enabled:
            return True
        prerequisites = PREREQUISITES_BY_STEP.get(step, ())
        return not any(self._claim_states.get(claim) == "contradicted" for claim in prerequisites)

    def _memory_ready(self, step: str) -> bool:
        if not self.memory_gate_enabled:
            return True
        prerequisites = PREREQUISITES_BY_STEP.get(step, ())
        return all(self._claim_states.get(claim) == "supported" for claim in prerequisites)

    def _bootstrap_prerequisites(
        self,
        detections: Sequence[Any],
        *,
        step: str,
        product: str,
        image_shape: Optional[Tuple[int, int]],
    ) -> int:
        """Recover visible prerequisites when assistance starts mid-task.

        Only current RGB-derived evidence is used. An explicitly contradicted
        prerequisite is never overwritten by this bootstrap path.
        """

        if not self.prerequisite_bootstrap_enabled or not self.memory_gate_enabled:
            return 0
        committed = 0
        for claim in PREREQUISITES_BY_STEP.get(step, ()):
            if self._claim_states.get(claim) in {"supported", "contradicted"}:
                continue
            prerequisite_step = STEP_BY_CLAIM.get(claim, "")
            if not prerequisite_step:
                continue
            features = extract_relation_features(
                detections,
                claim_id=claim,
                step_id=prerequisite_step,
                product=product,
                image_shape=image_shape,
            )
            scores = self.scorer.score_features(
                features,
                claim_id=claim,
                step_id=prerequisite_step,
            )
            support = float(scores.get("support_score", 0.0))
            contradiction = float(scores.get("contradiction_score", 0.0))
            counterfactual = self._counterfactual_support(
                detections,
                claim=claim,
                step=prerequisite_step,
                product=product,
                image_shape=image_shape,
            )
            visible_support = (
                support >= self.support_threshold
                and support >= contradiction + self.counterfactual_margin
                and support - counterfactual >= self.counterfactual_margin
            )
            hits = self._prerequisite_candidate_hits.get(claim, 0)
            hits = hits + 1 if visible_support else 0
            self._prerequisite_candidate_hits[claim] = hits
            if hits < self.prerequisite_confirmation_frames:
                continue
            self._claim_states[claim] = "supported"
            self._commit(prerequisite_step)
            committed += 1
        return committed

    def _counterfactual_support(
        self,
        detections: Sequence[Any],
        *,
        claim: str,
        step: str,
        product: str,
        image_shape: Optional[Tuple[int, int]],
    ) -> float:
        if not self.specialized_counterfactual_enabled:
            return 0.0
        alternative = {"A": "B", "B": "A"}.get(product, "")
        if not alternative:
            return 0.0
        features = extract_relation_features(
            detections,
            claim_id=claim,
            step_id=step,
            product=alternative,
            image_shape=image_shape,
        )
        scores = self.scorer.score_features(features, claim_id=claim, step_id=step)
        return max(0.0, min(1.0, float(scores.get("support_score", 0.0))))

    def _commit(self, step: str) -> None:
        if step not in CLAIM_BY_STEP:
            return
        order = {name: index for index, name in enumerate(CLAIM_BY_STEP)}
        if not self._committed_step or order.get(step, -1) >= order.get(self._committed_step, -1):
            self._committed_step = step

    def _infer_product_family(self, detections: Sequence[Any]) -> str:
        if self.configured_product_family:
            return self.configured_product_family
        if self._product_family:
            return self._product_family

        scores = {family: 0.0 for family in FAMILY_CLASSES}
        for item in detections:
            if not identity_safe(item):
                continue
            name = det_name(item)
            confidence = det_conf(item)
            for family, weighted_classes in FAMILY_CLASSES.items():
                scores[family] += confidence * float(weighted_classes.get(name, 0.0))
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        best_family, best_score = ranked[0]
        runner_up = ranked[1][1]
        if best_score < self.family_min_confidence or best_score - runner_up < self.family_margin:
            self._family_candidate = ""
            self._family_candidate_hits = 0
            return ""
        if best_family == self._family_candidate:
            self._family_candidate_hits += 1
        else:
            self._family_candidate = best_family
            self._family_candidate_hits = 1
        if self._family_candidate_hits >= self.family_confirmation_frames:
            self._product_family = best_family
        return self._product_family

    def _missing_roles(
        self,
        *,
        step: str,
        state: str,
        product: str,
        features: Mapping[str, float],
        support: float,
        contradiction: float,
    ) -> Tuple[str, ...]:
        roles = []

        def add(role: str) -> None:
            if role not in roles:
                roles.append(role)

        if step == "S1":
            if float(features.get("top_conf", 0.0)) < self.support_threshold:
                add("object_presence")
                add("occlusion_recovery")
            return tuple(roles)
        if not product:
            add("identity_disambiguation")
            add("occlusion_recovery")
            return tuple(roles)

        target_role_conf = float(features.get("target_role_conf", 0.0))
        target_conf = float(features.get("target_conf", 0.0))
        target_proposal = float(features.get("target_proposal_conf", 0.0))
        housing_conf = float(features.get("housing_role_conf", 0.0))
        containment = float(features.get("containment_score", 0.0))
        if target_role_conf < 0.10:
            add("object_presence")
            add("occlusion_recovery")
        if target_conf < max(0.10, 0.65 * target_proposal):
            add("identity_disambiguation")
        if housing_conf < 0.10:
            add("occlusion_recovery")

        if step in {"S2", "S3"}:
            if containment < 0.20:
                add("containment")
                add("slot_relation")
                add("insertion")
            if step == "S3" and state == "insufficient" and containment >= 0.20:
                add("alignment")
        elif step == "S4":
            if containment < 0.20:
                add("boundary_visibility")
                add("alignment")
                add("contact_verification")

        if state == "insufficient" and not roles:
            if abs(support - contradiction) < max(0.05, self.counterfactual_margin):
                add("claim_disambiguation")
            else:
                add("occlusion_recovery")
        if not self._memory_ready(step):
            add("history_precondition")
        return tuple(roles)
