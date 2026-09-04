"""Train the sparse online step-proposal model from assistant replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for directory in (ROOT, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from evaluate_learned_evidence_step_proposal import (  # noqa: E402
    STEP_INDEX,
    STEPS,
    balanced_weights,
    base_feature_names,
    build_samples,
    feature_names,
    model,
)


def projection_matrix(raw_dim: int, projection_dim: int = 64) -> np.ndarray:
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
    parser.add_argument("--kb", type=Path, required=True)
    parser.add_argument("--embedding-npz", type=Path, required=True)
    parser.add_argument("--refresh-stride", type=int, default=4)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    samples, appearance_names = build_samples(
        args.summary_csv,
        args.timeline_csv,
        args.kb,
        args.embedding_npz,
        causal_appearance_history=False,
        appearance_cache_stride=1,
    )
    labels = np.asarray(
        [STEP_INDEX[sample.target] for sample in samples],
        dtype=np.int64,
    )
    estimator = model()
    estimator.fit(
        np.asarray([sample.features for sample in samples], dtype=np.float64),
        labels,
        sample_weight=balanced_weights(labels),
    )
    embedding_payload = np.load(args.embedding_npz, allow_pickle=False)
    raw_dimension = int(embedding_payload["roi_embedding"].shape[1])
    projection = projection_matrix(raw_dimension)
    names = (
        base_feature_names()
        + appearance_names
        + feature_names()[len(base_feature_names()) :]
    )
    artifact = {
        "format_version": 1,
        "estimator": estimator,
        "steps": tuple(STEPS),
        "feature_names": tuple(names),
        "appearance_projection": projection,
        "appearance": {
            "encoder": "dinov2_vitb14",
            "image_size": 224,
            "roi_padding": 0.15,
            "raw_dimension": raw_dimension,
            "projected_dimension": int(projection.shape[1]),
            "refresh_stride": max(1, int(args.refresh_stride)),
        },
        "training": {
            "source": "assistant_replay_only",
            "samples": len(samples),
            "robot_view_labels_used": False,
            "candidate_robot_images_used": False,
            "ground_truth_boxes_used": False,
            "future_frames_used": False,
            "appearance_history_used": False,
            "posterior_history_used": False,
        },
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output_model, compress=3)
    manifest = {
        "artifact": str(args.output_model),
        "samples": len(samples),
        "features": len(names),
        "refresh_stride": max(1, int(args.refresh_stride)),
        "protocol": artifact["training"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
