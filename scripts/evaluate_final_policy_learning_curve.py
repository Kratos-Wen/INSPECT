"""Evaluate event-count sensitivity with the final object-centered selector.

The requirement model and geometric calibration remain frozen. The event
budget refers only to the discrete relative policy, not all supervision.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "scripts"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
from evaluate_assistance_event_learning_curve import coverage_order
from inspect_system.active_view.evidence_transport import HierarchicalEvidenceTransportModel
from inspect_system.active_view.trace_event_miner import MinedEvent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def fingerprint(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-rows", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    protocol = read(args.reference)["protocol"]
    inputs = protocol["input_fingerprints"]
    for name, value in inputs.items():
        if value and fingerprint(value["path"])["sha256"] != value["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")
    source = read(inputs["reveal_model"]["path"])
    base = source["base_model"]
    events = [MinedEvent(**row) for row in read(args.training_report)["trainable_events"]]
    budgets = [0, 2, 4, 8, 13, 18, len(events)]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    receipt = {
        "scope": "Discrete relative-policy event count in the final selector; R and geometry frozen",
        "not_total_assistant_supervision_budget": True,
        "zero_events_still_has_geometry_and_R_supervision": True,
        "seed_semantics": "Assistant-event subset seed, not optimization initialization",
        "budgets": budgets, "subset_seeds": list(range(5)),
        "reference": fingerprint(args.reference), "training_report": fingerprint(args.training_report),
        "script": fingerprint(__file__),
        "evaluator": fingerprint(ROOT / "scripts/evaluate_decidability_gated_transport.py"),
        "robot_labels_used_for_fitting_or_selection": False,
    }
    write(out / "protocol.json", receipt)
    runs = []
    for budget in budgets:
        for seed in ([0] if budget in (0, len(events)) else range(5)):
            selected = coverage_order(events, seed)[:budget]
            model = HierarchicalEvidenceTransportModel(**{
                k: base[k] for k in ("base_concentration", "hierarchy_strength", "risk_beta", "gain_power", "confidence_scale")
            }).fit_events(selected)
            payload = copy.deepcopy(source)
            payload["base_model"] = model.to_dict()
            payload["metadata"]["num_training_events"] = budget
            payload["metadata"]["num_training_videos"] = len({e.video for e in selected})
            stem = f"events_{budget:02d}_seed_{seed:02d}"
            model_path = out / (stem + "_model.json")
            result_path = out / (stem + ".json")
            rows_path = out / (stem + "_rows.json")
            write(model_path, payload)
            command = [sys.executable, str(ROOT / "scripts/evaluate_decidability_gated_transport.py")]
            for flag, name in (
                ("observations", "observations"), ("trial-gt", "trial_gt"),
                ("evidence-scorer", "evidence_scorer"), ("requirement-report", "requirement_report"),
                ("requirement-calibration", "requirement_calibration"),
                ("current-geometry", "current_geometry"), ("evaluation-utility-csv", "evaluation_utility"),
            ):
                command += ["--" + flag, inputs[name]["path"]]
            command += ["--reveal-model", str(model_path), "--output-json", str(result_path),
                        "--output-rows-json", str(rows_path), "--only-variant", "INSPECT"]
            flags = {"support-threshold": .35, "contradiction-threshold": .45,
                     "margin-threshold": .05, "role-threshold": .65, "partial-threshold": .35,
                     "lambda-cost": protocol["lambda_cost"], "tau-view": protocol["tau_view"]}
            for flag, name in (
                ("commit-identity-confidence", "identity_confidence"),
                ("commit-identity-margin", "identity_margin_floor"),
                ("commit-alternative-identity-confidence", "counterfactual_identity_threshold"),
                ("commit-counterfactual-margin", "counterfactual_margin_floor"),
                ("commit-relation-threshold", "relation_evidence"),
            ):
                flags[flag] = protocol["commit_gate"][name]
            for flag, value in flags.items():
                command += ["--" + flag, str(value)]
            run_receipt = {"command": command, "protocol": receipt, "model": fingerprint(model_path)}
            receipt_path = out / (stem + "_receipt.json")
            reusable = result_path.exists() and rows_path.exists() and receipt_path.exists() and read(receipt_path) == run_receipt
            if not reusable:
                completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
                (out / (stem + ".log")).write_text(completed.stdout + completed.stderr, encoding="utf-8")
                if completed.returncode:
                    raise RuntimeError(f"Evaluation failed for {stem}; see log")
                write(receipt_path, run_receipt)
            row = read(result_path)["results"]["INSPECT"]
            runs.append({"events": budget, "subset_seed": seed,
                         "event_ids": [e.event_id for e in selected], "metrics": row})
            write(out / "runs.json", runs)
            print(f"{stem}: utility={row['selected_utility']:.6f}, resolve={row['resolve_at_1']:.6f}", flush=True)
    full_rows = read(out / f"events_{len(events):02d}_seed_00_rows.json")["INSPECT"]
    reference_rows = read(args.reference_rows)["INSPECT"]
    fields = ("trial_id", "current_view", "selected_view", "selected_utility", "resolve_at_1")
    endpoint_matches = [[r[f] for f in fields] for r in full_rows] == [[r[f] for f in fields] for r in reference_rows]
    summary = []
    for budget in budgets:
        subset = [r for r in runs if r["events"] == budget]
        metrics = {}
        for metric in ("selected_utility", "gain", "resolve_at_1", "regret"):
            values = [r["metrics"][metric] for r in subset]
            metrics[metric] = {"mean": statistics.fmean(values),
                               "sample_sd": statistics.stdev(values) if len(values) > 1 else None}
        summary.append({"events": budget, "subset_runs": len(subset), **metrics})
    write(out / "summary.json", {"protocol": receipt, "full_endpoint_matches": endpoint_matches, "curve": summary})
    if not endpoint_matches:
        raise AssertionError("Full-budget endpoint does not reproduce final policy choices")


if __name__ == "__main__":
    main()
