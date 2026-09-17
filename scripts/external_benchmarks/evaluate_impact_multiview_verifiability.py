"""Evaluate frozen IMPACT claim verifiability across synchronized views.

The state head, active-claim manifest, query offset, and selective cutoff are
fixed before this evaluation. Candidate-view predictions are used only to
measure view sufficiency; they are never inputs to a view-selection policy.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from impact_state_evidence_head import StateEvidenceHead, WORLD_VALUES
from impact_query_truth import relabel_query


VIEWS = ("ego", "front", "left", "right", "top")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def feature_path(front_dir: Path, multiview_root: Path, recording: str, view: str) -> Path:
    root = front_dir if view == "front" else multiview_root / view
    path = root / f"{recording}_{view}.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def decision_from_probability(probability: np.ndarray, cutoff: float) -> dict[str, Any]:
    p_wrong, p_absent, p_support = (float(value) for value in probability)
    counterfactual = max(p_wrong, p_absent)
    raw_index = int(np.argmax(probability))
    raw_value = int(WORLD_VALUES[raw_index])
    raw_decision = "supported" if p_support >= counterfactual else "contradicted"
    confidence = p_support - counterfactual if raw_decision == "supported" else counterfactual
    decision = raw_decision if confidence >= cutoff else "insufficient"
    return {
        "p_installed_wrongly": p_wrong,
        "p_not_installed": p_absent,
        "p_installed_correctly": p_support,
        "p_counterfactual": counterfactual,
        "support_margin": p_support - counterfactual,
        "raw_world_value": raw_value,
        "raw_claim_decision": raw_decision,
        "decision_confidence": confidence,
        "claim_prediction": decision,
    }


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def view_metrics(rows: list[Mapping[str, Any]]) -> dict[str, float | int]:
    answered = [row for row in rows if row["claim_prediction"] != "insufficient"]
    supported = [row for row in rows if row["semantic_outcome"] == "supported"]
    contradicted = [row for row in rows if row["semantic_outcome"] == "contradicted"]
    return {
        "n": len(rows),
        "utility": float(np.mean([float(row["utility"]) for row in rows])),
        "full_verifiability": safe_div(sum(int(row["utility"]) == 2 for row in rows), len(rows)),
        "partial_verifiability": safe_div(sum(int(row["utility"]) == 1 for row in rows), len(rows)),
        "coverage": safe_div(len(answered), len(rows)),
        "selective_accuracy": safe_div(
            sum(row["claim_prediction"] == row["semantic_outcome"] for row in answered),
            len(answered),
        ),
        "valid_support_recall": safe_div(
            sum(row["claim_prediction"] == "supported" for row in supported), len(supported)
        ),
        "invalid_block": safe_div(
            sum(row["claim_prediction"] != "supported" for row in contradicted), len(contradicted)
        ),
        "false_support": safe_div(
            sum(row["claim_prediction"] == "supported" for row in contradicted), len(contradicted)
        ),
    }


def aggregate_events(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["recording_id"]), int(row["source_frame"]), int(row["component_id"]))
        grouped[key][str(row["view"])] = row

    events = []
    for key, by_view in sorted(grouped.items()):
        missing = sorted(set(VIEWS) - set(by_view))
        if missing:
            raise RuntimeError(f"Missing views for {key}: {missing}")
        ego = int(by_view["ego"]["utility"])
        noncurrent = [int(by_view[view]["utility"]) for view in VIEWS if view != "ego"]
        all_utilities = [int(by_view[view]["utility"]) for view in VIEWS]
        exemplar = by_view["ego"]
        events.append(
            {
                "recording_id": key[0],
                "source_frame": key[1],
                "component_id": key[2],
                "component_name": exemplar["component_name"],
                "semantic_outcome": exemplar["semantic_outcome"],
                "ego_utility": ego,
                "uniform_noncurrent_utility": float(np.mean(noncurrent)),
                "oracle_utility": max(all_utilities),
                "oracle_gain": max(all_utilities) - ego,
                "improvable": int(max(noncurrent) > ego),
                "ego_insufficient": int(by_view["ego"]["claim_prediction"] == "insufficient"),
                "resolved_by_another_view": int(
                    by_view["ego"]["claim_prediction"] == "insufficient"
                    and max(noncurrent) == 2
                ),
                "utilities": {view: int(by_view[view]["utility"]) for view in VIEWS},
            }
        )
    return events


def event_metrics(events: list[Mapping[str, Any]]) -> dict[str, float | int]:
    ego_insufficient = [event for event in events if int(event["ego_insufficient"]) == 1]
    return {
        "n": len(events),
        "recordings": len({str(event["recording_id"]) for event in events}),
        "ego_utility": float(np.mean([float(event["ego_utility"]) for event in events])),
        "uniform_noncurrent_utility": float(
            np.mean([float(event["uniform_noncurrent_utility"]) for event in events])
        ),
        "oracle_utility": float(np.mean([float(event["oracle_utility"]) for event in events])),
        "oracle_gain_over_ego": float(np.mean([float(event["oracle_gain"]) for event in events])),
        "ego_full_verifiability": safe_div(
            sum(int(event["ego_utility"]) == 2 for event in events), len(events)
        ),
        "oracle_full_verifiability": safe_div(
            sum(int(event["oracle_utility"]) == 2 for event in events), len(events)
        ),
        "improvable_fraction": safe_div(sum(int(event["improvable"]) for event in events), len(events)),
        "ego_insufficient_n": len(ego_insufficient),
        "insufficient_resolved_by_another_view": safe_div(
            sum(int(event["resolved_by_another_view"]) for event in ego_insufficient),
            len(ego_insufficient),
        ),
    }


def clustered_bootstrap(
    events: list[Mapping[str, Any]], iterations: int, seed: int
) -> dict[str, list[float]]:
    by_recording: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        by_recording[str(event["recording_id"])].append(event)
    recordings = sorted(by_recording)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        sample = rng.choice(recordings, size=len(recordings), replace=True)
        sampled_events = [event for recording in sample for event in by_recording[str(recording)]]
        metrics = event_metrics(sampled_events)
        for name in (
            "ego_utility",
            "uniform_noncurrent_utility",
            "oracle_utility",
            "oracle_gain_over_ego",
            "ego_full_verifiability",
            "oracle_full_verifiability",
            "improvable_fraction",
            "insufficient_resolved_by_another_view",
        ):
            values[name].append(float(metrics[name]))
    return {
        name: [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]
        for name, samples in values.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims-jsonl", type=Path, required=True)
    parser.add_argument("--front-feature-dir", type=Path, required=True)
    parser.add_argument("--multiview-feature-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, required=True)
    parser.add_argument("--post-frames", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    claims = read_jsonl(args.claims_jsonl)
    if not claims:
        raise RuntimeError("No active claims")
    if {str(row["split"]) for row in claims} != {"test"}:
        raise RuntimeError("Multiview evaluation requires only official test rows")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = StateEvidenceHead(
        feature_dim=int(checkpoint["feature_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        components=int(checkpoint["components"]),
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    feature_cache: dict[tuple[str, str], np.ndarray] = {}
    annotation_cache = {}
    indexed = []
    for row in claims:
        recording = str(row["recording_id"])
        reference_key = (recording, "front")
        if reference_key not in feature_cache:
            feature_cache[reference_key] = np.load(
                feature_path(args.front_feature_dir, args.multiview_feature_root, recording, "front"),
                mmap_mode="r",
            )
        reference_features = feature_cache[reference_key]
        reference_frame = min(
            max(int(row["source_frame"]) + args.post_frames, 0),
            reference_features.shape[1] - 1,
        )
        for view in VIEWS:
            cache_key = (recording, view)
            if cache_key not in feature_cache:
                feature_cache[cache_key] = np.load(
                    feature_path(args.front_feature_dir, args.multiview_feature_root, recording, view),
                    mmap_mode="r",
                )
            features = feature_cache[cache_key]
            if features.ndim != 2 or features.shape[0] != int(checkpoint["feature_dim"]):
                raise ValueError(f"Unexpected feature shape for {cache_key}: {features.shape}")
            # Ego is 25 Hz while the static IMPACT streams are 30 Hz. Mapping
            # by normalized recording time keeps all views at one instant.
            frame = int(
                round(
                    reference_frame
                    * max(features.shape[1] - 1, 0)
                    / max(reference_features.shape[1] - 1, 1)
                )
            )
            indexed.append(
                {
                    "row": row,
                    "view": view,
                    "frame": frame,
                    "reference_frame": reference_frame,
                    "feature": np.asarray(features[:, frame], dtype=np.float32),
                }
            )

    output_rows = []
    with torch.inference_mode():
        for start in range(0, len(indexed), args.batch_size):
            group = indexed[start : start + args.batch_size]
            tensor = torch.from_numpy(np.stack([item["feature"] for item in group])).to(device)
            probabilities = torch.softmax(model(tensor), dim=-1).cpu().numpy()
            for item, component_probabilities in zip(group, probabilities):
                row = item["row"]
                component = int(row["component_id"])
                decision = decision_from_probability(component_probabilities[component], args.cutoff)
                annotation_path = str(row["source_annotation"])
                if annotation_path not in annotation_cache:
                    annotation_cache[annotation_path] = json.loads(
                        Path(annotation_path).read_text(encoding="utf-8")
                    )
                output_rows.append(
                    relabel_query({
                        **row,
                        "view": item["view"],
                        "query_frame": item["frame"],
                        "reference_query_frame": item["reference_frame"],
                        **decision,
                    }, annotation_cache[annotation_path])
                )

    events = aggregate_events(output_rows)
    summary = {
        "protocol": {
            "dataset": "IMPACT-v1.1",
            "split": "official test only",
            "active_claims": str(args.claims_jsonl),
            "views": list(VIEWS),
            "feature": "official frozen VideoMAEv2",
            "head_checkpoint": str(args.checkpoint),
            "head_best_epoch": int(checkpoint.get("best_epoch", -1)),
            "post_frames": args.post_frames,
            "truth_time_basis": "front ASR at reference_query_frame, not source event",
            "view_time_alignment": "normalized recording time (ego 25 Hz; static views 30 Hz)",
            "selective_cutoff": args.cutoff,
            "utility_definition": {
                "0": "raw claim direction is incorrect",
                "1": "raw claim direction is correct but confidence is below the frozen cutoff",
                "2": "selective claim decision is correct and above the frozen cutoff",
            },
            "candidate_views_used_by_policy": False,
            "purpose": "cross-view verifiability diagnostic, not view-policy evaluation",
        },
        "per_view": {
            view: view_metrics([row for row in output_rows if row["view"] == view]) for view in VIEWS
        },
        "cross_view": event_metrics(events),
        "clustered_recording_bootstrap_95ci": clustered_bootstrap(
            events, args.bootstrap_iterations, args.seed
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "multiview_claim_scores.jsonl", output_rows)
    write_jsonl(args.output_dir / "multiview_event_summary.jsonl", events)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
