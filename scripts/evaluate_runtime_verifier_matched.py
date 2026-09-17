"""Replay runtime verifier ablations on frozen, predicted visual evidence.

Conditional and predicted-proposal protocols are separate. Incorrect proposed
claims receive no invented labels. Every prototype excludes its evaluation
scenario fold. Learned triage uses the original cross-fitted predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "scripts"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
from evaluate_active_claim_cascade import load_timeline, timeline_at
from inspect_system.evidence_scorer import PrototypeEvidenceScorer, extract_relation_features
from inspect_system.live_claim_verifier import LiveClaimVerifier
from inspect_system.causal_evidence_bank import TriageOutput

STATES = ("supported", "contradicted", "insufficient")


class InstrumentedVerifier(LiveClaimVerifier):
    """Count execution paths without changing runtime decisions."""

    def __init__(self, *args, **kwargs):
        self.execution = Counter()
        self.in_bootstrap = False
        self.readiness_check_only_disabled = kwargs.pop("readiness_check_only_disabled", False)
        self.active_counterfactual_only_disabled = kwargs.pop("active_counterfactual_only_disabled", False)
        super().__init__(*args, **kwargs)

    def _memory_ready(self, step):
        if self.readiness_check_only_disabled:
            return True
        return super()._memory_ready(step)

    def _bootstrap_prerequisites(self, *args, **kwargs):
        self.in_bootstrap = True
        try:
            return super()._bootstrap_prerequisites(*args, **kwargs)
        finally:
            self.in_bootstrap = False

    def _counterfactual_support(self, *args, **kwargs):
        if self.active_counterfactual_only_disabled and not self.in_bootstrap:
            return 0.0
        if self.specialized_counterfactual_enabled:
            name = "counterfactual_bootstrap_calls" if self.in_bootstrap else "counterfactual_direct_calls"
            self.execution[name] += 1
        return super()._counterfactual_support(*args, **kwargs)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def receipt(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def key(video, frame):
    return Path(video).name.lower(), int(float(frame))


def normalize(value):
    return "insufficient" if value == "unresolved" else str(value)


def ratio(n, d):
    return n / d if d else None


def metrics(rows):
    counts = Counter((r["truth"], r["prediction"]) for r in rows)
    result = {"samples": len(rows), "confusion": {f"{a}->{b}": counts[a, b] for a in STATES for b in STATES}}
    f1 = []
    for label in STATES:
        truth_n = sum(counts[label, b] for b in STATES)
        pred_n = sum(counts[a, label] for a in STATES)
        tp = counts[label, label]
        result[label + "_recall"] = ratio(tp, truth_n)
        result[label + "_precision"] = ratio(tp, pred_n)
        f1.append(2 * tp / (truth_n + pred_n) if truth_n + pred_n else 0.0)
    result["false_support"] = ratio(sum(r["truth"] != "supported" and r["prediction"] == "supported" for r in rows),
                                    sum(r["truth"] != "supported" for r in rows))
    result["macro_f1"] = sum(f1) / 3 if rows else None
    result["accuracy"] = ratio(sum(r["truth"] == r["prediction"] for r in rows), len(rows))
    result["resolved_coverage"] = ratio(sum(r["prediction"] != "insufficient" for r in rows), len(rows))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--verifier-predictions", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    timeline = load_timeline(args.timeline)
    predictions = [r for r in csv_rows(args.verifier_predictions) if r["variant"] == "causal_visual_appearance_no_consistency"]
    lookup = {key(r["video"], r["frame"]): r for r in predictions}
    if len(lookup) != len(predictions):
        raise ValueError("Duplicate evaluation keys")
    proposals = {key(r["video"], r["frame"]): r for r in jsonl(args.proposals)}
    folds = {Path(r["video"]).name.lower(): int(r["outer_fold"]) for r in predictions}
    observations, input_paths = {}, []
    for run in read(args.summary):
        video = Path(run["video"]).name.lower()
        if video not in folds:
            continue
        if run["feedback"] != 0 or run["returncode"] != 0:
            raise ValueError("Replay has feedback or failed observations")
        path = ROOT / run["run_dir"] / "iterations.jsonl"
        input_paths.append(receipt(path))
        observations[video] = sorted(jsonl(path), key=lambda r: r["frame_index"])
    samples = []
    annotated_steps = defaultdict(set)
    for video, rows in observations.items():
        for raw in rows:
            r = lookup.get(key(video, raw["frame_index"]))
            if r is None:
                continue
            gt = timeline_at(timeline, video, raw["frame_index"])
            if gt is None or normalize(gt["outcome"]) != normalize(r["truth"]):
                raise ValueError(f"Timeline mismatch: {video}:{raw['frame_index']}")
            annotated_steps[video].add(gt["step_id"])
            if normalize(r["truth"]) == "insufficient":
                continue
            features = extract_relation_features(
                raw.get("fused_detections") or raw.get("raw_detections") or [],
                claim_id=gt["claim_id"], step_id=gt["step_id"], product=gt["assembly_set"],
                image_shape=tuple(raw.get("frame_shape") or (1080, 1920))[:2])
            samples.append({"video": video, "claim_id": gt["claim_id"], "step_id": gt["step_id"],
                            "label": "support" if normalize(r["truth"]) == "supported" else "contradiction",
                            "features": features})
    scorers = {}
    for fold in sorted(set(folds.values())):
        training = [r for r in samples if folds[r["video"]] != fold]
        scorers[fold] = PrototypeEvidenceScorer.train(training)
        scorers[fold].metadata["excluded_evaluation_videos"] = sorted(v for v, f in folds.items() if f == fold)
        scorers[fold].metadata["training_videos"] = sorted({r["video"] for r in training})
        scorers[fold].save(args.output_dir / f"prototype_fold_{fold}.json")
    variants = {"Full": {}, "Without admissibility": {"admissibility_gate_enabled": False},
                "Without memory readiness": {"memory_gate_enabled": False},
                "Without specialized counterfactual": {"specialized_counterfactual_enabled": False},
                "Without readiness check, bootstrap retained": {"readiness_check_only_disabled": True},
                "Without active-claim counterfactual, bootstrap retained": {"active_counterfactual_only_disabled": True},
                "Without prerequisite bootstrap": {"prerequisite_bootstrap_enabled": False}}
    receipt_payload = {
        "script": receipt(__file__), "runtime": receipt(ROOT / "inspect_system/live_claim_verifier.py"),
        "inputs": [receipt(p) for p in (args.summary, args.timeline, args.verifier_predictions, args.proposals)],
        "perception_logs": input_paths,
        "annotation_history": {"videos": len(annotated_steps),
            "multi_step_videos": sum(len(v) > 1 for v in annotated_steps.values()),
            "steps_by_video": {k: sorted(v) for k, v in annotated_steps.items()}},
    }
    write(args.output_dir / "receipt.json", receipt_payload)
    for mode in ("learned_conditional", "prototype_conditional", "prototype_predicted_proposal"):
        variants_rows, activities = {}, {}
        for name, flags in variants.items():
            output, triggers = [], Counter()
            for video, raw_rows in observations.items():
                verifier = InstrumentedVerifier(scorers[folds[video]],
                    support_threshold=.35, contradiction_threshold=.35,
                    counterfactual_margin=0.0, ema_decay=.55, **flags)
                for raw in raw_rows:
                    frame = raw["frame_index"]
                    r = lookup.get(key(video, frame))
                    gt = timeline_at(timeline, video, frame) if r is not None else None
                    proposal = proposals.get(key(video, frame))
                    if mode.endswith("predicted_proposal"):
                        if proposal is None:
                            if r is not None:
                                raise ValueError(f"Missing cross-fitted proposal: {video}:{frame}")
                            continue
                        step, confidence = proposal["proposed_step"], float(proposal["confidence"])
                    else:
                        if r is None:
                            continue
                        step, confidence = gt["step_id"], 1.0
                    learned = None
                    if mode == "learned_conditional":
                        sp, cp, ip = (float(r[k]) for k in ("support_probability", "contradiction_probability", "insufficient_probability"))
                        learned = TriageOutput(normalize(r["prediction"]), sp, cp, ip, 1-ip, sp-max(cp, ip), {})
                    decision = verifier.verify(
                        raw.get("fused_detections") or raw.get("raw_detections") or [],
                        frame_index=frame, proposed_step=step, proposal_confidence=confidence,
                        image_shape=tuple(raw.get("frame_shape") or (1080, 1920))[:2],
                        learned_triage=learned)
                    triggers["observations"] += 1
                    triggers["inadmissible"] += not decision.admissible
                    triggers["memory_not_ready"] += not decision.memory_ready
                    triggers["negative_margin"] += decision.counterfactual_margin < 0
                    if r is not None:
                        correct_proposal = step == gt["step_id"]
                        output.append({"video": video, "frame": frame, "fold": folds[video],
                            "truth": normalize(r["truth"]), "prediction": decision.state,
                            "active_step": gt["step_id"], "proposed_step": step,
                            "proposal_correct": correct_proposal,
                            "joint_correct": correct_proposal and decision.state == normalize(r["truth"]),
                            "family": decision.product_family, "committed_step": decision.committed_step,
                            "admissible": decision.admissible, "memory_ready": decision.memory_ready,
                            "counterfactual_margin": decision.counterfactual_margin})
                triggers.update(verifier.execution)
            if len(output) != len(predictions):
                raise AssertionError(f"Mismatched samples: {len(output)}")
            variants_rows[name], activities[name] = output, dict(triggers)
        full = variants_rows["Full"]
        results = {}
        for name, rows in variants_rows.items():
            # Wrong proposed claims have no truth labels in this dataset.
            matched = [r for r in rows if r["proposal_correct"]]
            value = metrics(matched)
            value["joint_correct_over_all_samples"] = sum(r["joint_correct"] for r in rows) / len(rows)
            value["total_samples"] = len(rows)
            value["changed_decisions_vs_full"] = sum(a["prediction"] != b["prediction"] for a, b in zip(rows, full))
            value["activity"] = activities[name]
            results[name] = value
        protocol = {
            "runtime_path": mode, "annotated_active_claim": not mode.endswith("predicted_proposal"),
            "scorable_claim_metrics_condition_on_correct_proposal": True,
            "GT_feedback_or_history_injected": False, "family_inferred_from_predictions": True,
            "perception_cache_replay": True, "raw_perception_rerun": False, "LLM_called": False,
            "prototype_training": "Other outer scenario folds only",
            "prototype_training_supervision": "Annotated claim outcomes on training folds, not online user feedback",
            "thresholds": {"support": .35, "contradiction": .35, "margin": 0.0, "ema": .55},
            "threshold_selection_this_run": "None; existing runtime operating point",
            "specialized_counterfactual_usage": "Prerequisite bootstrap only" if mode == "learned_conditional" else "Active claim and prerequisite bootstrap",
            "variant_switches": variants,
            "memory_flag_also_disables_bootstrap": True,
            "independent_check_ablations_also_reported": True,
            "reset_at_video_boundary": True, "cross_step_history_synthesized": False,
            "multi_step_annotated_videos": receipt_payload["annotation_history"]["multi_step_videos"],
        }
        write(args.output_dir / f"{mode}.json", {"protocol": protocol, "results": results})
        write(args.output_dir / f"{mode}_rows.json", variants_rows)
        print(json.dumps({"path": mode, "results": results}), flush=True)


if __name__ == "__main__":
    main()
