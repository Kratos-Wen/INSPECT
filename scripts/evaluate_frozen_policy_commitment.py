"""Apply one frozen verifier to all frozen view policies on matched trials."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "scripts"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from evaluate_selected_view_shared_verifier import evaluate_observation, output_record
from evaluate_robot_closed_loop import read_csv, read_jsonl
from inspect_system.causal_evidence_bank import CausalEvidenceBank
from inspect_system.active_view.view_lattice import candidate_views, transition_cost


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def fingerprint(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def score(rows, field):
    total = sum(r["weight"] for r in rows)
    def mass(predicate):
        return sum(r["weight"] for r in rows if predicate(r))
    def fraction(predicate, denominator=total):
        return mass(predicate) / denominator if denominator else None
    truth_sup = mass(lambda r: r["truth"] == "supported")
    truth_con = mass(lambda r: r["truth"] == "contradicted")
    resolved = mass(lambda r: r[field] != "insufficient")
    utility_key = "current_utility" if field == "current_decision" else "selected_utility"
    u2 = mass(lambda r: r[utility_key] == 2)
    return {
        "trial_mass": total,
        "correct_resolution": fraction(lambda r: r[field] == r["truth"]),
        "support_recall": fraction(lambda r: r[field] == "supported" and r["truth"] == "supported", truth_sup),
        "contradiction_recall": fraction(lambda r: r[field] == "contradicted" and r["truth"] == "contradicted", truth_con),
        "false_support": fraction(lambda r: r[field] == "supported" and r["truth"] == "contradicted", truth_con),
        "false_contradiction": fraction(lambda r: r[field] == "contradicted" and r["truth"] == "supported", truth_sup),
        "resolved_coverage": resolved / total,
        "resolved_precision": fraction(lambda r: r[field] == r["truth"], resolved),
        "fully_verifiable_mass": u2,
        "correct_resolution_at_u2": fraction(lambda r: r[utility_key] == 2 and r[field] == r["truth"], u2),
        "decision_mass": {s: mass(lambda r: r[field] == s) for s in ("supported", "contradicted", "insufficient")},
    }


def action_candidates(name, record, utilities):
    trial, current, selected = record["trial_id"], record["current_view"], record["selected_view"]
    if selected == "ORACLE":
        views = sorted([current, *candidate_views(current)])
        best = max(utilities[trial, view] for view in views)
        return [view for view in views if utilities[trial, view] == best]
    if selected != "EXPECTED":
        return [selected]
    if not record["triggered"]:
        return [current]
    views = candidate_views(current)
    if name == "Shortest-Move":
        cost = min(transition_cost(current, view) for view in views)
        return [view for view in views if transition_cost(current, view) == cost]
    if name == "Uniform Non-current":
        return views
    raise ValueError(f"Unknown expected-action policy: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy-rows", type=Path, required=True)
    parser.add_argument("--final-rows", type=Path, required=True)
    parser.add_argument("--component-rows", type=Path, required=True)
    parser.add_argument("--proposal-model", type=Path, required=True)
    parser.add_argument("--triage-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    reference = read(args.reference)["protocol"]
    inputs = reference["input_fingerprints"]
    for name, value in inputs.items():
        if value and fingerprint(value["path"])["sha256"] != value["sha256"]:
            raise ValueError(f"Reference changed: {name}")
    policy_rows = read(args.policy_rows)
    policy_rows["INSPECT"] = read(args.final_rows)["INSPECT"]
    for name, rows in read(args.component_rows).items():
        if name != "INSPECT":
            policy_rows[name] = rows
    gt = {r["trial_id"]: r for r in read_csv(Path(inputs["trial_gt"]["path"]))}
    observations = defaultdict(dict)
    for row in read_jsonl(Path(inputs["observations"]["path"])):
        meta = row.get("metadata", {})
        observations[meta["trial_id"]][row["view_id"]] = row
    utilities = {}
    for row in read_csv(Path(inputs["evaluation_utility"]["path"])):
        if row.get("human_utility_0_1_2") in ("0", "1", "2"):
            utilities[row["trial_id"], row["view_id"]] = int(row["human_utility_0_1_2"])
    provenance = {
        "protocol": "Frozen policy choices, then identical current/selected-view verifier replay",
        "perception": "Frozen predicted detector/MoGe/scene-graph cache",
        "appearance": "RGB re-encoded on GPU; refresh forced after a camera move",
        "active_claim": "Task-specified annotated active claim; not end-to-end claim discovery",
        "policy_inputs": ["target_step", "claim_id", "product_variant"],
        "GT_history_or_feedback_injected": False,
        "robot_labels_used_for_training_or_calibration": False,
        "candidate_images_used_for_selection": False,
        "thresholds_retuned": False,
        "uniform": "Exact expectation over non-current views under the frozen common trigger",
        "shortest": "Exact expectation over minimum-transition-cost ties under the same trigger",
        "oracle": "Human-utility maximum with uniform tie averaging; evaluation-only reference",
        "state_reset": "Every trial starts independently; no future-view state carries across trials",
        "device": args.device,
        "input_receipts": [fingerprint(p) for p in (args.reference, args.policy_rows, args.final_rows, args.component_rows, args.proposal_model, args.triage_model)],
        "code_receipts": [fingerprint(p) for p in (Path(__file__), ROOT / "scripts/evaluate_selected_view_shared_verifier.py", ROOT / "inspect_system/causal_evidence_bank.py")],
    }
    write(args.output_dir / "protocol.json", provenance)
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    cache = args.output_dir / "cache" / signature[:16]
    cache.mkdir(parents=True, exist_ok=True)
    bank = CausalEvidenceBank(args.proposal_model, args.triage_model,
        dinov2_repo=Path.home()/".cache/torch/hub/facebookresearch_dinov2_main",
        device=args.device, load_encoder_on_start=True)
    if str(bank.device).startswith("cpu") and args.device.startswith("cuda"):
        raise RuntimeError("GPU requested but unavailable")
    results = {}
    expected_keys = {(r["trial_id"], r["current_view"]) for r in policy_rows["INSPECT"]}
    for name, rows in policy_rows.items():
        if {(r["trial_id"], r["current_view"]) for r in rows} != expected_keys or len(rows) != 132:
            raise AssertionError(f"Unmatched policy: {name}")
        outputs = []
        for record in rows:
            trial, current = record["trial_id"], record["current_view"]
            spec = {k: gt[trial][k] for k in ("target_step", "claim_id", "product_variant")}
            candidates = action_candidates(name, record, utilities)
            for target in candidates:
                path = cache / f"{trial}_{current}_{target}.json"
                if path.exists():
                    computed = read(path)
                else:
                    started = time.perf_counter()
                    bank.reset()
                    first = evaluate_observation(bank, observations[trial][current], spec, force_appearance_refresh=False)
                    after = first if target == current else evaluate_observation(bank, observations[trial][target], spec, force_appearance_refresh=True)
                    bank.reset()
                    alone = evaluate_observation(bank, observations[trial][target], spec, force_appearance_refresh=False)
                    computed = {"trial_id": trial, "current_view": current, "selected_view": target,
                        "current_decision": first.state, "selected_decision": alone.state, "sequential_decision": after.state,
                        "current": output_record(first), "selected": output_record(alone), "sequential": output_record(after),
                        "fresh_replay_seconds": time.perf_counter()-started}
                    write(path, computed)
                outputs.append({**computed, "weight": 1/len(candidates), "truth": gt[trial]["claim_outcome"],
                    "current_utility": utilities[trial, current], "selected_utility": utilities[trial, target]})
        total = sum(r["weight"] for r in outputs)
        metrics = {
            "current": score(outputs, "current_decision"),
            "selected": score(outputs, "selected_decision"),
            "sequential": score(outputs, "sequential_decision"),
            "selected_utility": sum(r["weight"]*r["selected_utility"] for r in outputs)/total,
            "resolve_at_1": sum(r["weight"]*(r["selected_utility"] == 2) for r in outputs)/total,
        }
        expected_utility = sum(float(r["selected_utility"]) for r in rows)/132
        if not math.isclose(metrics["selected_utility"], expected_utility, abs_tol=1e-9):
            raise AssertionError(f"Utility mismatch: {name}")
        results[name] = metrics
        slug = hashlib.sha256(name.encode()).hexdigest()[:12]
        write(args.output_dir / f"rows_{slug}.json", {"policy": name, "rows": outputs})
        write(args.output_dir / "summary.json", {"protocol": provenance, "results": results})
        print(f"{name}: U={metrics['selected_utility']:.6f} correct={metrics['selected']['correct_resolution']:.6f} sequential={metrics['sequential']['correct_resolution']:.6f}", flush=True)


if __name__ == "__main__":
    main()
