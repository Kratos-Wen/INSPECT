"""Replay a frozen view policy with the assistant-trained shared verifier.

The policy rows fix the selected view before this script reads any selected
image. Robot labels and oracle utilities are used only for final metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2

import evaluate_robot_closed_loop as base
from evaluate_robot_causal_evidence_bank import (
    detections_from_record,
    evidence_token_from_record,
    rule_prediction,
    scene_graph_from_record,
)
from inspect_system.causal_evidence_bank import CausalEvidenceBank, TriageOutput
from inspect_system.evidence_scorer import normalize_claim


def evaluate_observation(
    bank: CausalEvidenceBank,
    raw: Mapping[str, Any],
    gt: Mapping[str, Any],
    *,
    force_appearance_refresh: bool,
) -> TriageOutput:
    active_step = base._step_state(str(gt.get("target_step", "")))
    claim = normalize_claim(gt.get("claim_id"), active_step)
    product = str(gt.get("product_variant", ""))
    metadata = dict(raw.get("metadata") or {})
    rgb_path = Path(str(metadata.get("rgb_path", "")))
    frame = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Could not read robot RGB frame: {rgb_path}")
    detections = detections_from_record(raw)
    role_detections = detections_from_record(raw, metadata_key="role_detections")
    scene_graph = scene_graph_from_record(raw)
    token = evidence_token_from_record(raw, active_step=active_step)
    fusion = bank.propose(
        frame,
        detections,
        scene_graph,
        token,
        rule_prediction(active_step),
        force_appearance_refresh=force_appearance_refresh,
    )
    return bank.triage(
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


def score_summary(rows: Sequence[Mapping[str, Any]], decision_key: str) -> dict[str, Any]:
    total = len(rows)
    supported = [row for row in rows if row["truth"] == "supported"]
    contradicted = [row for row in rows if row["truth"] == "contradicted"]
    resolved = [row for row in rows if row[decision_key] != "insufficient"]
    fully_verifiable = [row for row in rows if int(row["selected_utility"]) == 2]

    def recall(items: Sequence[Mapping[str, Any]], label: str) -> float:
        return (
            sum(row[decision_key] == label for row in items) / len(items)
            if items
            else 0.0
        )

    def correct(items: Sequence[Mapping[str, Any]]) -> int:
        return sum(row[decision_key] == row["truth"] for row in items)

    return {
        "trials": total,
        "support_recall": recall(supported, "supported"),
        "contradiction_recall": recall(contradicted, "contradicted"),
        "false_support": recall(contradicted, "supported"),
        "false_contradiction": recall(supported, "contradicted"),
        "resolved_coverage": len(resolved) / max(1, total),
        "resolved_precision": correct(resolved) / max(1, len(resolved)),
        "correct_resolution": correct(rows) / max(1, total),
        "fully_verifiable_trials": len(fully_verifiable),
        "correct_resolution_at_u2": correct(fully_verifiable)
        / max(1, len(fully_verifiable)),
        "decision_counts": dict(Counter(row[decision_key] for row in rows)),
    }


def output_record(triage: TriageOutput) -> dict[str, Any]:
    return {
        "state": triage.state,
        "scores": {
            "supported": triage.support_score,
            "contradicted": triage.contradiction_score,
            "insufficient": triage.insufficient_score,
        },
        "posterior_margin": triage.posterior_margin,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--trial-gt", type=Path, required=True)
    parser.add_argument("--policy-rows", type=Path, required=True)
    parser.add_argument("--policy-key", default="INSPECT")
    parser.add_argument("--proposal-model", type=Path, required=True)
    parser.add_argument("--triage-model", type=Path, required=True)
    parser.add_argument(
        "--dinov2-repo",
        type=Path,
        default=Path.home()
        / ".cache"
        / "torch"
        / "hub"
        / "facebookresearch_dinov2_main",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-rows-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_by_trial = {row["trial_id"]: row for row in base.read_csv(args.trial_gt)}
    observations: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for raw in base.read_jsonl(args.observations):
        metadata = dict(raw.get("metadata") or {})
        trial_id = str(metadata.get("trial_id", ""))
        view_id = str(raw.get("view_id") or metadata.get("view_id") or "")
        if trial_id and view_id:
            observations[trial_id][view_id] = raw

    policy_payload = json.loads(args.policy_rows.read_text(encoding="utf-8"))
    policy_rows = policy_payload.get(args.policy_key)
    if not isinstance(policy_rows, list):
        raise KeyError(f"Policy key not found in rows payload: {args.policy_key}")

    bank = CausalEvidenceBank(
        args.proposal_model,
        args.triage_model,
        dinov2_repo=args.dinov2_repo,
        device=args.device,
        load_encoder_on_start=True,
    )
    rows: list[dict[str, Any]] = []
    for policy_row in policy_rows:
        trial_id = str(policy_row.get("trial_id", ""))
        gt = gt_by_trial.get(trial_id)
        if not gt:
            raise KeyError(f"Missing trial GT: {trial_id}")
        current_view = str(policy_row.get("current_view", ""))
        selected_view = str(policy_row.get("selected_view", ""))
        view_map = observations.get(trial_id, {})
        current_raw = view_map.get(current_view)
        selected_raw = view_map.get(selected_view)
        if current_raw is None or selected_raw is None:
            raise KeyError(
                f"Missing observation for {trial_id}: {current_view}->{selected_view}"
            )

        bank.reset()
        current = evaluate_observation(
            bank,
            current_raw,
            gt,
            force_appearance_refresh=False,
        )
        if selected_view == current_view:
            causal_selected = current
        else:
            causal_selected = evaluate_observation(
                bank,
                selected_raw,
                gt,
                force_appearance_refresh=True,
            )

        bank.reset()
        framewise_selected = evaluate_observation(
            bank,
            selected_raw,
            gt,
            force_appearance_refresh=False,
        )
        truth = str(gt.get("claim_outcome", "")).strip().lower()
        rows.append(
            {
                "trial_id": trial_id,
                "claim_id": str(gt.get("claim_id", "")),
                "truth": truth,
                "current_view": current_view,
                "selected_view": selected_view,
                "moved": selected_view != current_view,
                "current_utility": int(policy_row.get("current_utility", 0)),
                "selected_utility": int(policy_row.get("selected_utility", 0)),
                "current_decision": current.state,
                "selected_framewise_decision": framewise_selected.state,
                "selected_causal_decision": causal_selected.state,
                "current": output_record(current),
                "selected_framewise": output_record(framewise_selected),
                "selected_causal": output_record(causal_selected),
            }
        )

    payload = {
        "protocol": {
            "policy_rows_are_frozen_before_image_replay": True,
            "candidate_view_images_used_for_selection": False,
            "assistant_training_only": True,
            "robot_labels_used_for_training_or_calibration": False,
            "robot_labels_and_oracle_utilities_used_for_metrics_only": True,
            "camera_move_forces_appearance_refresh": True,
            "policy_key": args.policy_key,
            "device": str(bank.device),
            "observations": str(args.observations),
            "trial_gt": str(args.trial_gt),
            "policy_rows": str(args.policy_rows),
            "proposal_model": str(args.proposal_model),
            "triage_model": str(args.triage_model),
        },
        "policy_utility": {
            "selected_utility": sum(row["selected_utility"] for row in rows)
            / max(1, len(rows)),
            "gain": sum(
                row["selected_utility"] - row["current_utility"] for row in rows
            )
            / max(1, len(rows)),
            "resolve_at_1": sum(row["selected_utility"] == 2 for row in rows)
            / max(1, len(rows)),
        },
        "verifier": {
            "current_only": score_summary(rows, "current_decision"),
            "selected_only": score_summary(rows, "selected_framewise_decision"),
            "current_to_selected": score_summary(rows, "selected_causal_decision"),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output_rows_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
