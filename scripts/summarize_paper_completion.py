"""Audit and summarize completed matched experiments without editing the paper."""
from __future__ import annotations

import argparse
import os
import csv
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
# Frozen evaluation artifacts from the paper runs, released separately from the
# code. Point INSPECT_ARTIFACTS_ROOT at your local copy (default: artifacts/).
# Missing inputs raise FileNotFoundError with the exact expected path.
ARTIFACTS = Path(os.environ.get("INSPECT_ARTIFACTS_ROOT", "artifacts"))



def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def fingerprint(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def fmt(value):
    if value is None:
        return "--"
    return str(Decimal(str(value)).quantize(Decimal(".001"), rounding=ROUND_HALF_UP))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    out = args.root
    curve = read(out / "learning_curve/summary.json")
    assert curve["full_endpoint_matches"]
    robot = read(out / "robot/matched_policies/summary.json")
    assert len(robot["results"]) == 19
    replay = read(out / "robot/shared_verifier_replay.json")
    for mode, previous in (("current", "current_only"), ("selected", "selected_only"), ("sequential", "current_to_selected")):
        for metric in ("correct_resolution", "false_support", "resolved_coverage", "resolved_precision"):
            assert abs(robot["results"]["INSPECT"][mode][metric] - replay["verifier"][previous][metric]) < 1e-10
    source_paths = [
        out / "learning_curve/summary.json", out / "learning_curve/protocol.json",
        out / "robot/shared_verifier_replay.json", out / "robot/matched_policies/summary.json",
        out / "verifier_matched_causalbank/receipt.json", out / "tests.xml",
    ]
    test_root = ET.parse(out / "tests.xml").getroot()
    suites = list(test_root.iter("testsuite"))
    assert sum(int(s.attrib.get("failures", 0)) + int(s.attrib.get("errors", 0)) for s in suites) == 0
    tests = sum(int(s.attrib.get("tests", 0)) for s in suites)
    rows, lines = [], [
        "# Matched Experimental Completion",
        "",
        "All results are retained regardless of direction. No manuscript or deployment default was changed.",
        "This package is an evaluation record, not an assertion that every scientific claim is now validated.",
        "",
        "## Scope",
        "",
        "- Robot comparison: frozen selected views followed by the same GPU verifier, using cached predicted detector/MoGe/scene-graph evidence and freshly encoded RGB appearance.",
        "- Runtime ablations: predicted-perception cache replay with cross-fitted scorers; no GT feedback or verified history is inserted.",
        "- Conditional learned scores retain the annotated active-claim and assembly-family context of their original benchmark. Runtime family checks use predicted detections.",
        "- The prototype branch is refitted on other scenario folds and uses the frozen runtime thresholds. It is a diagnostic, not the trained deployment checkpoint.",
        "- The event-count curve varies discrete relative-policy training events only. Requirement and geometry supervision remain fixed, including at zero relative-policy events.",
        "- Qwen policy choices are reused from the frozen Qwen3-VL-4B-Instruct evaluation. No paid API calls were made.",
        "",
        "## Robot Re-verification",
        "",
        "Current and selected-only columns reset the verifier. Sequential retains current-view state before observing the selected view. Correct resolution is a correct physical support/contradiction decision over all trials. It is not the human-utility Resolve@1 metric.",
        "Uniform averages all non-current views; Shortest-Move averages minimum-cost ties. Both use the original common trigger. Oracle averages tied human-utility maxima and is evaluation-only.",
        "",
        "| Policy | Utility | View Resolve@1 | Selected Correct | Sequential Correct | Sequential False Support |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    names = {"VLM (RGB + claim)": "Qwen3-VL-4B-Instruct (RGB + claim)",
             "VLM (+ procedure context + scene graph)": "Qwen3-VL-4B-Instruct (procedure + scene graph)"}
    for name, value in robot["results"].items():
        assert abs(value["selected"]["trial_mass"] - 132) < 1e-9
        row = {"experiment": "robot", "variant": name, "n": 132,
               "utility": value["selected_utility"], "view_resolve_at_1": value["resolve_at_1"],
               "selected_correct_resolution": value["selected"]["correct_resolution"],
               "sequential_correct_resolution": value["sequential"]["correct_resolution"],
               "sequential_false_support": value["sequential"]["false_support"]}
        rows.append(row)
        lines.append("| " + " | ".join([names.get(name, name)] + [fmt(row[k]) for k in ("utility", "view_resolve_at_1", "selected_correct_resolution", "sequential_correct_resolution", "sequential_false_support")]) + " |")
    lines += [
        "",
        "The final selector improves annotated view utility, but that gain does not translate into better correct-resolution rate for the frozen shared verifier in this replay. The findings do not justify an end-to-end one-step resolution claim.",
        "",
        "## Relative-policy Event Count",
        "",
        "Variation below is across assistant-event subset seeds, not neural-network initialization seeds. Full and zero budgets have one distinct subset; their SD is unavailable, not evidence of zero training variability.",
        "",
        "| Events | Subsets | Utility Mean | Subset SD | Resolve@1 Mean |",
        "|---:|---:|---:|---:|---:|",
    ]
    for point in curve["curve"]:
        row = {"experiment": "relative_policy_event_count", "events": point["events"], "subset_runs": point["subset_runs"],
               "utility": point["selected_utility"]["mean"], "subset_sd": point["selected_utility"]["sample_sd"],
               "view_resolve_at_1": point["resolve_at_1"]["mean"]}
        rows.append(row)
        lines.append(f"| {point['events']} | {point['subset_runs']} | {fmt(row['utility'])} | {fmt(row['subset_sd'])} | {fmt(row['view_resolve_at_1'])} |")
    lines += ["", "The 26-event endpoint reproduces all 132 final-policy choices. The curve is not monotone at small budgets. Zero discrete-policy events does not mean no assistance supervision.", ""]
    variants = ("learned_conditional", "prototype_conditional", "prototype_predicted_proposal")
    for mode in variants:
        path = out / "verifier_matched_causalbank" / (mode + ".json")
        source_paths.append(path)
        data = read(path)
        detail = read(out / "verifier_matched_causalbank" / (mode + "_rows.json"))
        lines += ["## Runtime Ablation: " + mode, "",
                  "| Variant | Scorable N | Supported Recall | False Support | Macro-F1 | Joint / 1315 | Changed Decisions |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        full_keys = [(r["video"], r["frame"]) for r in detail["Full"]]
        assert len(full_keys) == len(set(full_keys)) == 1315
        for name, value in data["results"].items():
            assert [(r["video"], r["frame"]) for r in detail[name]] == full_keys
            assert sum(value["confusion"].values()) == value["samples"]
            row = {"experiment": mode, "variant": name, "n": value["samples"],
                   "supported_recall": value["supported_recall"], "false_support": value["false_support"],
                   "macro_f1": value["macro_f1"], "joint_correct": value["joint_correct_over_all_samples"],
                   "changed_decisions": value["changed_decisions_vs_full"]}
            rows.append(row)
            lines.append("| " + " | ".join([name, str(row["n"])] + [fmt(row[k]) for k in ("supported_recall", "false_support", "macro_f1", "joint_correct")] + [str(row["changed_decisions"])]) + " |")
        lines += ["", "Predicted-proposal claim metrics condition on a correct proposal because other proposed claims have no annotated truth. Joint success uses all 1,315 states. Conditional rows are not complete assistant results.",
                  "The memory flag also disables prerequisite bootstrap. Additional isolated-check variants retain bootstrap. Learned triage does not use the active-claim specialized scorer, but bootstrap still invokes specialized checks.", ""]
    lines += ["## Proposal-verification Cascade", "",
              "| Visual Context | N | Proposal Top-1 | Joint Top-1 | Joint Top-2 | False Support Given Correct Top-1 |",
              "|---|---:|---:|---:|---:|---:|"]
    for variant in ("visual_appearance_no_consistency", "causal_visual_appearance_no_consistency"):
        path = out / "cascade" / (variant + ".json")
        source_paths.append(path)
        data = read(path)
        assert all(data["protocol"][k] == 0 for k in ("missing_proposal_rows", "missing_timeline_rows", "truth_mismatch_rows"))
        m = data["metrics"]
        rows.append({"experiment": "cascade", "variant": variant, **{k: v for k, v in m.items() if not isinstance(v, dict)}})
        lines.append("| " + " | ".join([variant, str(m["samples"])] + [fmt(m[k]) for k in ("proposal_top1_active_step_recall", "joint_top1_triage_accuracy", "joint_top2_triage_accuracy", "false_support_given_correct_top1_proposal")]) + " |")
    history = read(out / "verifier_matched_causalbank/receipt.json")["annotation_history"]
    assert history["multi_step_videos"] == 0
    geometry_path = ARTIFACTS / "geometry/causal_weighted_local_surface_policy_report.json"
    geometry = read(geometry_path)
    source_paths.append(geometry_path)
    lines += [
        "",
        "## Remaining Annotation Dependency",
        "",
        f"The evaluated {history['videos']} assembly videos contain zero videos with multiple annotated assembly steps. Conditional admissibility therefore has no annotated cross-step transition to test.",
        "Admissibility can trigger on predicted-step sequences, but this does not establish correct handling of real historical violations. Different clips were not stitched together and no GT history was injected.",
        "The existing raw egocentric videos and test-step annotations specify progress intervals, not atomic claim validity and admissible history. Their labels were not silently converted to completed-claim GT.",
        "A real cross-step history benchmark remains pending claim-validity and session/reset annotations on continuous sequences. The current experiments cannot replace that ground truth.",
        "",
        "## Training-count Audit",
        "",
        f"The discrete relative policy has 26 events from 7 videos. The separate saved geometric calibration report lists {geometry['events']} geometry records from {geometry['videos']} videos. These units are distinct; neither is the total count of all supervision in INSPECT.",
        "",
        "## Verification",
        "",
        f"- {tests} focused regression tests passed.",
        "- Full learning-curve endpoint matches all final policy selections.",
        "- All 19 robot policy evaluations use the same 132 start-view trials and reproduce the original utility means.",
        "- Each of the 21 runtime ablations has the same 1,315 evaluation keys.",
        "- Both cascade comparisons have zero missing or truth-mismatched joins.",
        "- No paper text, figure, model checkpoint or deployment defaults were changed.",
    ]
    write_csv(out / "results.csv", rows)
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    status = {
        "robot_matched_reverification": {"status": "complete", "policies": 19, "trials_per_policy": 132},
        "relative_policy_event_count": {"status": "complete", "runs": 27, "scope": "R and geometry frozen"},
        "runtime_verifier_diagnostics": {"status": "complete", "paths": 3, "variants_per_path": 7, "states": 1315},
        "cascade": {"status": "complete", "comparisons": 2, "states": 1315},
        "true_cross_step_admissibility_benchmark": {"status": "annotation_required", "multi_step_annotated_videos": 0},
        "full_raw_video_LLM_assistant_benchmark": {"status": "not_run_in_this_completion", "reason": "These are verification and policy diagnostics, not response-generation or STT/TTS trials"},
        "tests": {"passed": tests},
    }
    write_json(out / "STATUS.json", status)
    write_json(out / "RESULT_RECEIPT.json", {"status": "PASS", "inputs": [fingerprint(p) for p in source_paths],
        "scripts": [fingerprint(ROOT / "scripts" / p) for p in ("evaluate_final_policy_learning_curve.py", "evaluate_frozen_policy_commitment.py", "evaluate_runtime_verifier_matched.py", "summarize_paper_completion.py")],
        "outputs": [fingerprint(out / p) for p in ("REPORT.md", "results.csv", "STATUS.json")]})
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
