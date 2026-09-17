"""Evaluate the assistant-trained causal evidence bank on frozen robot views."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2

import evaluate_robot_closed_loop as base
import evaluate_robot_closed_loop_gated as gated
from core_types import (
    Detection,
    EvidenceToken,
    SceneGraphFrame,
    SceneGraphRelation,
    StepPrediction,
)
from inspect_system.causal_evidence_bank import CausalEvidenceBank
from inspect_system.evidence_scorer import PrototypeEvidenceScorer, normalize_claim


STEPS = ("S1", "S2", "S3", "S4")


def detections_from_record(
    raw: Mapping[str, Any],
    *,
    metadata_key: str = "detections",
) -> list[Detection]:
    metadata = dict(raw.get("metadata") or {})
    return [
        Detection(
            name=str(item.get("name", "")),
            xyxy=tuple(float(value) for value in item.get("xyxy", (0, 0, 0, 0))),
            confidence=float(item.get("confidence", 0.0)),
            meta=dict(item.get("meta") or {}),
        )
        for item in metadata.get(metadata_key) or []
        if len(item.get("xyxy") or []) == 4
    ]


def scene_graph_from_record(raw: Mapping[str, Any]) -> SceneGraphFrame:
    metadata = dict(raw.get("metadata") or {})
    scene = dict(metadata.get("scene_evidence") or {})
    relations = [
        SceneGraphRelation(
            subject_index=-1,
            object_index=-1,
            subject_name=str(item.get("subject", "")),
            predicate=str(item.get("predicate", "")),
            object_name=str(item.get("object", "")),
            score=float(item.get("score", 0.0)),
        )
        for item in scene.get("relations") or []
    ]
    extras = dict(scene.get("extras") or {})
    stats = {
        "focus_relations": float(extras.get("num_relations", len(relations))),
        "relation_density": min(1.0, len(relations) / 20.0),
    }
    return SceneGraphFrame(relations=relations, stats=stats, extras=extras)


def evidence_token_from_record(
    raw: Mapping[str, Any],
    *,
    active_step: str,
) -> EvidenceToken:
    metadata = dict(raw.get("metadata") or {})
    track = dict(metadata.get("track_evidence") or {})
    return EvidenceToken(
        frame_index=int(metadata.get("frame_index") or 0),
        prev_step=active_step,
        visible_counts={
            str(key): int(value)
            for key, value in dict(raw.get("visible_counts") or {}).items()
        },
        relevant_counts={
            str(key): int(value)
            for key, value in dict(raw.get("relevant_counts") or {}).items()
        },
        relation_counts={
            str(key): int(value)
            for key, value in dict(raw.get("relation_counts") or {}).items()
        },
        relation_facts=[tuple(value) for value in raw.get("relation_facts") or []],
        state_scores={step: float(step == active_step) for step in STEPS},
        retrieval_scores={},
        memory_scores={},
        state_confidence=1.0,
        retrieval_confidence=0.0,
        memory_confidence=0.0,
        has_visual_evidence=bool(metadata.get("detections")),
        track_counts={
            str(key): int(value)
            for key, value in dict(track.get("track_counts") or {}).items()
        },
        stable_track_counts={
            str(key): int(value)
            for key, value in dict(track.get("stable_track_counts") or {}).items()
        },
    )


def rule_prediction(active_step: str) -> StepPrediction:
    return StepPrediction(
        step_id=active_step,
        confidence=1.0,
        scores={step: float(step == active_step) for step in STEPS},
        extras={"active_claim_supplied_equally": True},
    )


def summarize(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    total = len(rows)
    supported = [row for row in rows if row["truth"] == "supported"]
    contradicted = [row for row in rows if row["truth"] == "contradicted"]
    decisions = [str(row[key]) for row in rows]

    def recall(items: Sequence[Mapping[str, Any]], label: str) -> float:
        return (
            sum(str(row[key]) == label for row in items) / len(items)
            if items
            else 0.0
        )

    return {
        "views": total,
        "valid_support_recall": recall(supported, "supported"),
        "contradiction_recall": recall(contradicted, "contradicted"),
        "false_support": (
            sum(
                str(row[key]) == "supported" and row["truth"] != "supported"
                for row in rows
            )
            / max(1, sum(row["truth"] != "supported" for row in rows))
        ),
        "resolved_accuracy": (
            sum(str(row[key]) == row["truth"] for row in rows) / max(1, total)
        ),
        "decision_counts": dict(Counter(decisions)),
    }


def gated_causal_decision(
    state: str,
    prototype: Mapping[str, Any],
) -> str:
    support_available = bool(prototype.get("support_evidence_available", False))
    contradiction_available = bool(
        prototype.get("contradiction_evidence_available", False)
    )
    if state == "supported":
        return (
            "supported"
            if support_available and not contradiction_available
            else "insufficient"
        )
    if state == "contradicted":
        return "contradicted" if contradiction_available else "insufficient"
    return "insufficient"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--trial-gt", type=Path, required=True)
    parser.add_argument("--prototype-scorer", type=Path, required=True)
    parser.add_argument("--proposal-model", type=Path, required=True)
    parser.add_argument("--triage-model", type=Path, required=True)
    parser.add_argument(
        "--dinov2-repo",
        type=Path,
        default=Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-rows-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_by_trial = {row["trial_id"]: row for row in base.read_csv(args.trial_gt)}
    scorer = PrototypeEvidenceScorer.load(args.prototype_scorer)
    bank = CausalEvidenceBank(
        args.proposal_model,
        args.triage_model,
        dinov2_repo=args.dinov2_repo,
        device=args.device,
        load_encoder_on_start=True,
    )
    rows: list[dict[str, Any]] = []
    for raw in base.read_jsonl(args.observations):
        metadata = dict(raw.get("metadata") or {})
        trial_id = str(metadata.get("trial_id", ""))
        gt = gt_by_trial.get(trial_id)
        if not gt or str(gt.get("target_step", "")).lower() not in base.ACTIVE_STEPS:
            continue
        active_step = base._step_state(str(gt.get("target_step", "")))
        claim = normalize_claim(gt.get("claim_id"), active_step)
        product = str(gt.get("product_variant", ""))
        truth = str(gt.get("claim_outcome", "")).strip().lower()
        rgb_path = Path(str(metadata.get("rgb_path", "")))
        frame = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Could not read robot RGB frame: {rgb_path}")
        detections = detections_from_record(raw)
        role_detections = detections_from_record(
            raw,
            metadata_key="role_detections",
        )
        scene_graph = scene_graph_from_record(raw)
        token = evidence_token_from_record(raw, active_step=active_step)
        bank.reset()
        fusion = bank.propose(
            frame,
            detections,
            scene_graph,
            token,
            rule_prediction(active_step),
        )
        triage = bank.triage(
            detections,
            scene_graph,
            token,
            fusion,
            claim=claim,
            step=active_step,
            product=product,
            image_shape=frame.shape[:2],
            role_detections=role_detections,
        )
        _, prototype = gated.gated_claim_evidence(
            raw,
            gt=gt,
            scorer=scorer,
            support_threshold=0.35,
            contradiction_threshold=0.45,
            margin_threshold=0.05,
            role_threshold=0.65,
        )
        rows.append(
            {
                "trial_id": trial_id,
                "view_id": str(raw.get("view_id", "")),
                "observation_id": str(raw.get("observation_id", "")),
                "truth": truth,
                "prototype_decision": str(prototype.get("decision", "insufficient")),
                "causal_bank_decision": triage.state,
                "causal_bank_gated_decision": gated_causal_decision(
                    triage.state,
                    prototype,
                ),
                "causal_bank_scores": {
                    "supported": triage.support_score,
                    "contradicted": triage.contradiction_score,
                    "insufficient": triage.insufficient_score,
                },
                "proposal_step": fusion.step_id,
                "proposal_confidence": fusion.confidence,
                "diagnostics": dict(bank.diagnostics),
            }
        )

    payload = {
        "protocol": {
            "views_are_independent": True,
            "active_claim_supplied_equally": True,
            "assistant_training_only": True,
            "robot_view_labels_used_for_training_or_calibration": False,
            "candidate_view_images_used": False,
            "device": str(bank.device),
            "proposal_model": str(args.proposal_model),
            "triage_model": str(args.triage_model),
        },
        "results": {
            "Prototype Evidence Scorer": summarize(rows, "prototype_decision"),
            "Causal Evidence Bank": summarize(rows, "causal_bank_decision"),
            "Causal Evidence Bank + Counterfactual Availability Gate": summarize(
                rows,
                "causal_bank_gated_decision",
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output_rows_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

