"""Run robot closed-loop evaluation with the frozen evidence-availability gate.

This wrapper keeps the base evaluator unchanged and applies the detector and
relation thresholds fixed by the pre-existing protocol: confidence 0.25,
hard-pair margin 0.08, and relation evidence 0.18.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

try:
    import evaluate_robot_closed_loop as base
except ModuleNotFoundError:
    from scripts import evaluate_robot_closed_loop as base
from inspect_system.active_view.evidence_state import EvidenceState


IDENTITY_CONFIDENCE_THRESHOLD = 0.25
IDENTITY_MARGIN_THRESHOLD = 0.08
RELATION_EVIDENCE_THRESHOLD = 0.18


def apply_evidence_availability_gate(
    prediction: Mapping[str, Any],
    *,
    identity_confidence_threshold: float = IDENTITY_CONFIDENCE_THRESHOLD,
    identity_margin_floor: float = -IDENTITY_MARGIN_THRESHOLD,
    counterfactual_identity_threshold: float | None = None,
    counterfactual_margin_floor: float | None = None,
    relation_evidence_threshold: float = RELATION_EVIDENCE_THRESHOLD,
    allow_relation_absence_contradiction: bool = True,
) -> Dict[str, Any]:
    """Apply a positive-evidence gate without changing scorer outputs.

    The frozen protocol permits a small negative identity margin and treats
    visible-but-low relation evidence as a contradiction cue. Stricter commit
    protocols can require a positive identity margin and explicit
    counterfactual identity evidence instead, while discovery remains frozen.
    """
    gated_prediction = dict(prediction)
    features = dict(gated_prediction.get("features") or {})
    role_scores = dict(gated_prediction.get("role_scores") or {})
    target_conf = float(features.get("target_conf", 0.0))
    wrong_conf = float(features.get("wrong_same_role_conf", 0.0))
    housing_conf = float(features.get("housing_role_conf", 0.0))
    identity_margin = float(features.get("identity_margin", 0.0))
    alternative_is_reliable = bool(
        counterfactual_identity_threshold is not None
        and wrong_conf >= float(counterfactual_identity_threshold)
    )
    effective_margin_floor = float(
        counterfactual_margin_floor
        if alternative_is_reliable and counterfactual_margin_floor is not None
        else identity_margin_floor
    )

    identity_available = bool(
        target_conf >= float(identity_confidence_threshold)
        and identity_margin >= effective_margin_floor
    )
    relation_available = bool(
        max(
            float(role_scores.get("insertion_verification_view", 0.0)),
            float(role_scores.get("containment_verification_view", 0.0)),
            float(role_scores.get("slot_relation_view", 0.0)),
            float(role_scores.get("gap_visibility_view", 0.0)),
            float(role_scores.get("boundary_alignment_view", 0.0)),
            float(role_scores.get("contact_verification_view", 0.0)),
        )
        >= float(relation_evidence_threshold)
    )
    wrong_identity_visible = bool(
        wrong_conf
        >= max(
            float(identity_confidence_threshold),
            float(counterfactual_identity_threshold or 0.0),
        )
        and wrong_conf >= target_conf + IDENTITY_MARGIN_THRESHOLD
        and housing_conf >= 0.10
    )
    incomplete_relation_visible = bool(
        target_conf >= 0.10
        and housing_conf >= 0.10
        and max(
            float(role_scores.get("insertion_verification_view", 0.0)),
            float(role_scores.get("containment_verification_view", 0.0)),
            float(role_scores.get("slot_relation_view", 0.0)),
        )
        < float(relation_evidence_threshold)
    )
    support_available = identity_available and relation_available
    contradiction_available = wrong_identity_visible or bool(
        allow_relation_absence_contradiction and incomplete_relation_visible
    )
    decision = str(gated_prediction.get("decision", "insufficient"))
    if decision == "supported" and not support_available:
        gated_prediction["decision"] = "insufficient"
    elif decision == "contradicted" and not contradiction_available:
        gated_prediction["decision"] = "insufficient"
    gated_prediction.update(
        {
            "identity_available": identity_available,
            "relation_available": relation_available,
            "wrong_identity_visible": wrong_identity_visible,
            "incomplete_relation_visible": incomplete_relation_visible,
            "support_evidence_available": support_available,
            "contradiction_evidence_available": contradiction_available,
            "evidence_availability_gate": True,
            "evidence_availability_gate_config": {
                "identity_confidence": float(identity_confidence_threshold),
                "identity_margin_floor": float(identity_margin_floor),
                "counterfactual_identity_threshold": counterfactual_identity_threshold,
                "counterfactual_margin_floor": counterfactual_margin_floor,
                "alternative_identity_reliable": alternative_is_reliable,
                "effective_identity_margin_floor": effective_margin_floor,
                "relation_evidence": float(relation_evidence_threshold),
                "allow_relation_absence_contradiction": bool(
                    allow_relation_absence_contradiction
                ),
            },
        }
    )
    return gated_prediction


def gated_claim_evidence(*args: Any, **kwargs: Any) -> Tuple[EvidenceState, Dict[str, Any]]:
    state, prediction = _ORIGINAL(*args, **kwargs)
    return state, apply_evidence_availability_gate(prediction)


def argument_value(flag: str) -> str:
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return ""


_ORIGINAL = base.claim_evidence
base.claim_evidence = gated_claim_evidence


if __name__ == "__main__":
    base.main()
    output = argument_value("--output-json")
    if output:
        path = Path(output)
        payload: Mapping[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        result = dict(payload)
        result["evidence_availability_gate"] = {
            "identity_confidence": IDENTITY_CONFIDENCE_THRESHOLD,
            "identity_margin": IDENTITY_MARGIN_THRESHOLD,
            "relation_evidence": RELATION_EVIDENCE_THRESHOLD,
            "calibration_source": "frozen assistant/detector operating protocol",
        }
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
