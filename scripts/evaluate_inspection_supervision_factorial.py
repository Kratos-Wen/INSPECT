"""Remove R, discrete-motion, and geometric records under one frozen protocol.

This isolates inspection supervision, not detector/verifier training. All
hyperparameters, current-image predictions, KB rules, and screening are fixed.
"""
from __future__ import annotations

import argparse
import os
import copy
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Frozen evaluation artifacts from the paper runs, released separately from the
# code. Point INSPECT_ARTIFACTS_ROOT at your local copy (default: artifacts/).
# Missing inputs raise FileNotFoundError with the exact expected path.
ARTIFACTS = Path(os.environ.get("INSPECT_ARTIFACTS_ROOT", "artifacts"))

sys.path.insert(0, str(ROOT))
from inspect_system.active_view.evidence_transport import HierarchicalEvidenceTransportModel
from inspect_system.active_view.object_centric_evidence_memory import ObjectCentricEvidenceMemory
from inspect_system.active_view.object_centric_reveal import ObjectCentricCalibratedRevealModel


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def empty_memory(memory):
    return ObjectCentricEvidenceMemory(
        copy.deepcopy(memory.discrete_config), copy.deepcopy(memory.kernel_config)
    ).fit([])


def mask_model(source, *, discrete: bool, geometry: bool):
    payload = copy.deepcopy(source)
    if not geometry:
        wrapper = ObjectCentricCalibratedRevealModel.from_dict(copy.deepcopy(source))
        wrapper.memory = empty_memory(wrapper.memory)
        wrapper.family_memories = {key: empty_memory(value)
                                   for key, value in wrapper.family_memories.items()}
        empty = wrapper.to_dict()
        payload["memory"] = empty["memory"]
        payload["family_memories"] = empty["family_memories"]
    if not discrete:
        base = source["base_model"]
        model = HierarchicalEvidenceTransportModel(**{
            key: base[key] for key in ("base_concentration", "hierarchy_strength",
                                      "risk_beta", "gain_power", "confidence_scale")
        }).fit_events([])
        payload["base_model"] = model.to_dict()
        payload["metadata"]["num_training_events"] = 0
        payload["metadata"]["num_training_videos"] = 0
    return payload


def metrics(rows):
    n = len(rows)
    correct = sum(r["correct_resolution"] for r in rows)
    wrong = sum(r["false_support"] + r["false_contradiction"] for r in rows)
    abstain = sum(r["defer"] for r in rows)
    assert correct + wrong + abstain == n
    return {
        "trials": n, "triggered": sum(r["triggered"] for r in rows),
        "moves": sum(r["moved"] for r in rows),
        "utility": statistics.fmean(r["selected_utility"] for r in rows),
        "verifiable_at_1": statistics.fmean(r["resolve_at_1"] for r in rows),
        "correct": correct, "incorrect": wrong, "abstain": abstain,
        "correct_decision_rate": correct / n,
        "commit_precision": correct / (correct + wrong) if correct + wrong else None,
        "new_correct_on_moved": sum(r["correct_resolution"] for r in rows if r["moved"]),
    }


def paired_effect(low, high):
    keys = lambda rows: {(r["trial_id"], r["current_view"]): r for r in rows}
    a, b = keys(low), keys(high)
    assert a.keys() == b.keys()
    differences = {}
    for key in a:
        differences.setdefault(key[0], []).append(b[key]["selected_utility"] - a[key]["selected_utility"])
    setup_gains = {key: statistics.fmean(values) for key, values in differences.items()}
    return {
        "utility_delta": statistics.fmean(b[k]["selected_utility"] - a[k]["selected_utility"] for k in a),
        "verifiable_delta": statistics.fmean(b[k]["resolve_at_1"] - a[k]["resolve_at_1"] for k in a),
        "changed_views": sum(a[k]["selected_view"] != b[k]["selected_view"] for k in a),
        "setup_signs": {name: sum(test(value) for value in setup_gains.values())
                        for name, test in (("positive", lambda x: x > 0),
                                           ("tie", lambda x: x == 0),
                                           ("negative", lambda x: x < 0))},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    reference = ARTIFACTS / "external_comparison/inspect_final_stack_132.json"
    reference_rows = ARTIFACTS / "external_comparison/inspect_final_stack_rows_132.json"
    reference_command = ARTIFACTS / "controls/frozen_reference_receipt.json"
    proto = read(reference)["protocol"]
    inputs = proto["input_fingerprints"]
    for value in inputs.values():
        if value:
            assert sha(value["path"]) == value["sha256"], value["path"]
    source = read(inputs["reveal_model"]["path"])
    original_command = read(reference_command)["command"]
    evaluator = ROOT / "scripts/evaluate_decidability_gated_transport.py"
    assert sha(evaluator) == read(reference_command)["protocol"]["evaluator_sha256"]
    empty_requirements = out / "ontology_only_requirements.json"
    save(empty_requirements, {"requirement_events": [], "trainable_events": [],
                              "uses_robot_view_training": False})
    arms = {}
    # Persist all interventions before invoking an evaluator or reading outcomes.
    for requirement, discrete, geometry in itertools.product((0, 1), repeat=3):
        name = f"R{requirement}_D{discrete}_G{geometry}"
        model_path = out / f"{name}_model.json"
        save(model_path, mask_model(source, discrete=bool(discrete), geometry=bool(geometry)))
        req_path = Path(inputs["requirement_report"]["path"]) if requirement else empty_requirements
        command = list(original_command)
        command[0] = sys.executable
        command[1] = str(evaluator)
        for flag, value in {
            "--reveal-model": str(model_path),
            "--requirement-report": str(req_path),
            "--output-json": str(out / f"{name}.json"),
            "--output-rows-json": str(out / f"{name}_rows.json"),
        }.items():
            command[command.index(flag) + 1] = value
        arms[name] = {
            "R_records": 115 if requirement else 0,
            "D_records": 26 if discrete else 0,
            "G_records": 47 if geometry else 0,
            "model_sha256": sha(model_path), "requirement_sha256": sha(req_path),
            "command": command,
        }
    protocol = {
        "scope": "Inspection-supervision removal; detector and verifier training retained",
        "perception": "Frozen predicted detector/MoGe/scene-graph caches; not new neural inference",
        "active_claim": "Task supplied; not end-to-end claim discovery",
        "all_hyperparameters_frozen": True, "robot_labels_used_for_fitting": False,
        "candidate_images_used_for_selection": False,
        "no_records": "Empty empirical counts/memories with unchanged KB, priors and screening",
        "calibration_scope": "Source-selected hyperparameters retained; not a data-free baseline",
        "input_fingerprints": inputs, "script_sha256": sha(__file__),
        "evaluator_sha256": sha(evaluator), "arms": arms,
    }
    save(out / "protocol.json", protocol)
    rows_by_arm, results = {}, {}
    for name, arm in arms.items():
        model_path = out / f"{name}_model.json"
        reuse = None
        if name == "R1_D1_G1":
            assert read(model_path) == source
            reuse = reference_rows
        elif name == "R1_D0_G1":
            prefix = ARTIFACTS / "learning_curve/events_00_seed_00"
            existing_model = Path(str(prefix) + "_model.json")
            if existing_model.exists() and read(existing_model) == read(model_path):
                existing_receipt = read(Path(str(prefix) + "_receipt.json"))
                assert existing_receipt["protocol"]["evaluator"]["sha256"] == sha(evaluator)
                reuse = Path(str(prefix) + "_rows.json")
        receipt_path = out / f"{name}_receipt.json"
        rows_path = out / f"{name}_rows.json"
        if reuse:
            rows = read(reuse)["INSPECT"]
            save(rows_path, {"INSPECT": rows})
            save(receipt_path, {**arm, "reused_rows": str(reuse), "reused_rows_sha256": sha(reuse)})
        elif (receipt_path.exists() and rows_path.exists()
              and read(receipt_path).get("model_sha256") == arm["model_sha256"]
              and read(receipt_path).get("requirement_sha256") == arm["requirement_sha256"]
              and read(receipt_path).get("command") == arm["command"]
              and read(receipt_path).get("evaluator_sha256") == sha(evaluator)):
            rows = read(rows_path)["INSPECT"]
        else:
            result = subprocess.run(arm["command"], cwd=ROOT, capture_output=True, text=True)
            (out / f"{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
            if result.returncode:
                raise RuntimeError(f"{name} failed; see its log")
            rows = read(rows_path)["INSPECT"]
            save(receipt_path, {**arm, "evaluator_sha256": sha(evaluator),
                                "rows_sha256": sha(rows_path)})
        assert len(rows) == 132
        rows_by_arm[name] = rows
        results[name] = metrics(rows)
        save(out / "runs.json", results)
        print(name, json.dumps(results[name]), flush=True)
    full = rows_by_arm["R1_D1_G1"]
    reference_keys = [(r["trial_id"], r["current_view"], r["triggered"], r["current_utility"]) for r in full]
    for rows in rows_by_arm.values():
        assert [(r["trial_id"], r["current_view"], r["triggered"], r["current_utility"]) for r in rows] == reference_keys
    effects = {}
    for axis, label in enumerate(("R", "D", "G")):
        for fixed in itertools.product((0, 1), repeat=2):
            bits = list(fixed)
            bits.insert(axis, 0)
            low = "R%d_D%d_G%d" % tuple(bits)
            bits[axis] = 1
            high = "R%d_D%d_G%d" % tuple(bits)
            effects[f"{label}: {low} -> {high}"] = paired_effect(rows_by_arm[low], rows_by_arm[high])
    save(out / "summary.json", {"protocol": protocol, "results": results,
                                "paired_effects": effects, "all_trial_keys_and_triggers_match": True})
    lines = ["# Frozen Inspection-Supervision Factorial", "",
             "R: requirement records; D: discrete motion records; G: geometric records.",
             "All eight arms retain the same KB, perception, thresholds and clause screening.",
             "These are cached-perception decision replays, not live robot trials.", "",
             "| Arm | Utility | Verifiable@1 | Moves | Correct | Incorrect | Abstain |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in results.items():
        lines.append(f"| {name} | {row['utility']:.3f} | {row['verifiable_at_1']:.3f} | {row['moves']} | {row['correct']} | {row['incorrect']} | {row['abstain']} |")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
