"""Train a deployable claim-triage model from assistant replay evidence.

Model and refresh-schedule selection use grouped out-of-fold predictions from
assistant sessions only. The final estimator is fitted on all assistant
training sessions after selection; robot-view labels and candidate images are
never read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for directory in (ROOT, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from evaluate_procedural_claim_triage_calibrated import summarize_predictions  # noqa: E402
from evaluate_relation_aware_claim_verifier import (  # noqa: E402
    LABEL_TO_INDEX,
    OUTCOMES,
    align_probabilities,
    build_samples,
    causal_filter_probabilities,
    choose_causal_operating_point,
    decide,
    feature_vector,
    grouped_oof_probabilities,
    make_model,
    PROPOSAL_FEATURE_VARIANTS,
    sample_weights,
    scenario_group,
)


DEPLOYABLE_VARIANTS = (
    "causal_appearance_no_consistency",
    "causal_visual_appearance_no_consistency",
)


def ordered_features(
    feature_groups: dict[str, list[str]],
    variant: str,
) -> list[str]:
    names = list(feature_groups["detection"])
    names.extend(feature_groups["scene_relation"])
    if variant in PROPOSAL_FEATURE_VARIANTS:
        names.extend(feature_groups["proposal"])
    names.extend(feature_groups["appearance"])
    return names


def refresh_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing refresh metadata: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("task_labels_used") is not False:
        raise ValueError(f"Refresh schedule is not label-free: {path}")
    if payload.get("ground_truth_boxes_used") is not False:
        raise ValueError(f"Refresh schedule uses ground-truth boxes: {path}")
    if payload.get("future_frames_used") is not False:
        raise ValueError(f"Refresh schedule uses future frames: {path}")
    return payload


def deployment_score(metrics: dict[str, Any], false_support_cap: float) -> float:
    macro = float(metrics.get("triage_macro_f1") or 0.0)
    support = float(metrics.get("support_accuracy") or 0.0)
    contradiction = float(metrics.get("contradiction_recall") or 0.0)
    false_support = float(metrics.get("false_accept_rate_on_non_supported") or 0.0)
    score = macro + 0.12 * support + 0.08 * contradiction
    if false_support > float(false_support_cap):
        score -= 2.0 + 10.0 * (false_support - float(false_support_cap))
    return score


def projection_matrix(raw_dim: int, projection_dim: int) -> np.ndarray:
    dimension = min(int(projection_dim), int(raw_dim))
    rng = np.random.default_rng(20260828)
    return (
        rng.standard_normal((int(raw_dim), dimension), dtype=np.float32)
        / np.sqrt(float(dimension))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--timeline-csv", type=Path, required=True)
    parser.add_argument("--proposal-jsonl", type=Path)
    parser.add_argument(
        "--variant",
        choices=DEPLOYABLE_VARIANTS,
        default="causal_appearance_no_consistency",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=2,
        metavar=("NAME", "EMBEDDING_NPZ"),
        required=True,
    )
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--false-support-cap", type=float, default=0.04)
    parser.add_argument("--max-refresh-rate", type=float, default=0.30)
    parser.add_argument("--ema-decay", type=float, default=0.60)
    parser.add_argument("--proposal-top-k", type=int, default=2)
    args = parser.parse_args()
    if args.variant in PROPOSAL_FEATURE_VARIANTS and args.proposal_jsonl is None:
        parser.error(f"--proposal-jsonl is required for variant {args.variant}")

    candidates: list[dict[str, Any]] = []
    selected_samples = None
    selected_feature_groups = None
    selected_path = None
    selected_operating_point = None
    selected_key = (-float("inf"), -float("inf"), "")

    for raw_name, raw_path in args.candidate:
        name = str(raw_name).strip()
        path = Path(raw_path)
        metadata = refresh_metadata(path)
        refresh_rate = float(metadata.get("refresh_rate", 1.0))
        samples, feature_groups = build_samples(
            args.summary_csv,
            args.timeline_csv,
            float(args.ema_decay),
            int(args.proposal_top_k),
            "assembly",
            path,
            args.proposal_jsonl,
        )
        group_for_video = {
            sample.video: scenario_group(sample.video) for sample in samples
        }
        probabilities = grouped_oof_probabilities(
            samples,
            args.variant,
            group_for_video,
            int(args.inner_folds),
            20260829,
        )
        operating_point = choose_causal_operating_point(
            probabilities,
            samples,
            float(args.false_support_cap),
            require_proposal_consistency=False,
        )
        decay, support_threshold, contradiction_threshold, margin = operating_point
        filtered = causal_filter_probabilities(probabilities, samples, decay)
        predictions = [
            decide(
                row,
                support_threshold,
                contradiction_threshold,
                margin,
                sample.proposal_consistent,
                require_proposal_consistency=False,
            )
            for row, sample in zip(filtered, samples)
        ]
        metrics = summarize_predictions(samples, predictions)
        score = deployment_score(metrics, float(args.false_support_cap))
        eligible = refresh_rate <= float(args.max_refresh_rate) + 1e-12
        record = {
            "name": name,
            "path": str(path),
            "refresh_rate": refresh_rate,
            "eligible": eligible,
            "selection_score": score,
            "operating_point": {
                "causal_decay": decay,
                "support_threshold": support_threshold,
                "contradiction_threshold": contradiction_threshold,
                "posterior_margin": margin,
            },
            "grouped_oof_metrics": metrics,
            "refresh_metadata": metadata,
        }
        candidates.append(record)
        candidate_key = (score, -refresh_rate, name)
        if eligible and candidate_key > selected_key:
            selected_key = candidate_key
            selected_samples = samples
            selected_feature_groups = feature_groups
            selected_path = path
            selected_operating_point = operating_point

    if selected_samples is None or selected_path is None or selected_operating_point is None:
        raise RuntimeError("No candidate satisfies the configured refresh budget")

    labels = np.asarray(
        [LABEL_TO_INDEX[sample.truth] for sample in selected_samples],
        dtype=np.int64,
    )
    estimator = make_model(labels, args.variant)
    estimator.fit(
        np.stack(
            [feature_vector(sample, args.variant) for sample in selected_samples]
        ),
        labels,
        sample_weight=sample_weights(labels),
    )
    selected_payload = np.load(selected_path, allow_pickle=False)
    raw_appearance_dim = int(selected_payload["roi_embedding"].shape[1])
    appearance_projection = projection_matrix(raw_appearance_dim, 64)
    ordered_feature_names = ordered_features(selected_feature_groups, args.variant)
    decay, support_threshold, contradiction_threshold, margin = selected_operating_point
    selected_record = next(
        record for record in candidates if Path(record["path"]) == selected_path
    )
    artifact = {
        "format_version": 1,
        "estimator": estimator,
        "outcomes": tuple(OUTCOMES),
        "feature_names": tuple(ordered_feature_names),
        "appearance_projection": appearance_projection,
        "operating_point": {
            "causal_decay": float(decay),
            "support_threshold": float(support_threshold),
            "contradiction_threshold": float(contradiction_threshold),
            "posterior_margin": float(margin),
            "require_proposal_consistency": False,
        },
        "appearance": {
            "encoder": "dinov2_vitb14",
            "image_size": 224,
            "roi_padding": 0.15,
            "raw_dimension": raw_appearance_dim,
            "projected_dimension": int(appearance_projection.shape[1]),
            "refresh": selected_record["refresh_metadata"],
        },
        "training": {
            "variant": args.variant,
            "source": "assistant_replay_only",
            "samples": len(selected_samples),
            "robot_view_labels_used": False,
            "candidate_robot_images_used": False,
            "ground_truth_boxes_used": False,
            "future_frames_used": False,
            "reported_metrics_are_grouped_oof": True,
        },
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output_model, compress=3)
    manifest = {
        "selected_candidate": selected_record,
        "candidates": candidates,
        "artifact": {
            "path": str(args.output_model),
            "features": len(ordered_feature_names),
            "samples": len(selected_samples),
            "outcomes": list(OUTCOMES),
        },
        "protocol": {
            "selection": "scenario-grouped OOF on assistant sessions",
            "false_support_cap": float(args.false_support_cap),
            "max_refresh_rate": float(args.max_refresh_rate),
            "test_or_robot_labels_used_for_selection": False,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
