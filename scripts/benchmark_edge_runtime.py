"""Benchmark the deployed INSPECT pipeline with synchronized stage timings.

The benchmark runs the same detector, geometry, scene-graph, proposal, memory,
and verification path used by the assistant. Warm-up frames are excluded from
the report. No ground-truth labels are read and no policy outputs are changed.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspect_assist import load_runtime_package

load_runtime_package()

from inspect_runtime.components.logging import JsonlCsvLogger
from inspect_runtime.config import load_config
from inspect_runtime.runtime.pipeline import build_default_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Recorded RGB video used for replay.")
    parser.add_argument("--config", required=True, help="Deployment YAML configuration.")
    parser.add_argument("--kb", default=str(ROOT / "KB.json"), help="Skill knowledge base.")
    parser.add_argument("--weights", required=True, help="Detector checkpoint.")
    parser.add_argument("--device", default="0", help="Inference device passed to the pipeline.")
    parser.add_argument("--source-kind", choices=("video", "camera"), default="camera")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--geometry-resolution-level", type=int, default=None)
    parser.add_argument("--visual-change-threshold", type=float, default=None)
    parser.add_argument("--camera-detection-conf", type=float, default=None)
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "runtime" / "pipeline_latency.json"),
    )
    parser.add_argument(
        "--no-cuda-sync",
        action="store_true",
        help="Disable synchronization. Intended only for profiler-overhead checks.",
    )
    parser.add_argument(
        "--disable-routing",
        action="store_true",
        help="Run every perception stage on every frame for an always-on reference.",
    )
    return parser.parse_args()


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * min(1.0, max(0.0, q))
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: Iterable[float]) -> Dict[str, float]:
    clean = [float(value) for value in values]
    if not clean:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0, "max_ms": 0.0}
    return {
        "mean_ms": round(statistics.fmean(clean), 4),
        "p50_ms": round(statistics.median(clean), 4),
        "p90_ms": round(percentile(clean, 0.90), 4),
        "max_ms": round(max(clean), 4),
    }


def hardware_metadata() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
    try:
        import torch

        payload["torch"] = torch.__version__
        payload["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            payload.update(
                {
                    "cuda_device": torch.cuda.get_device_name(index),
                    "cuda_total_memory_gib": round(properties.total_memory / (1024**3), 3),
                }
            )
    except Exception as exc:
        payload["torch_error"] = type(exc).__name__
    return payload


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    output_path = Path(args.output)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Detector weights not found: {args.weights}")
    if args.samples <= 0 or args.warmup < 0 or args.frame_stride <= 0:
        raise ValueError("samples and frame-stride must be positive; warmup must be non-negative")

    config = load_config(args.config)
    config.edge_runtime.profiling_enabled = True
    config.edge_runtime.synchronize_cuda_for_timing = not bool(args.no_cuda_sync)
    config.edge_runtime.enabled = not bool(args.disable_routing)
    if args.geometry_resolution_level is not None:
        config.geometry.resolution_level = int(args.geometry_resolution_level)
    if args.visual_change_threshold is not None:
        config.edge_runtime.visual_change_threshold = float(args.visual_change_threshold)
    if args.camera_detection_conf is not None:
        config.camera.detection_conf = float(args.camera_detection_conf)
    config.video.write_annotated = False
    config.ui.enabled = False
    config.voice.enabled = False
    config.speech.enabled = False
    config.assistant.enabled = False
    runlog_root = output_path.parent / "runlogs"
    config.runlog.save_dir = str(runlog_root)

    pipeline = build_default_pipeline(
        config=config,
        kb_path=str(args.kb),
        yolo_weights=str(args.weights),
        device=str(args.device),
        state_path=None,
        interactive=False,
    )
    logger = JsonlCsvLogger(str(runlog_root), video_path)
    pipeline.prepare_run(logger, source_uri=str(video_path))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        pipeline.finalize_run(logger, source_uri=str(video_path))
        raise RuntimeError(f"Failed to open video: {video_path}")
    if args.start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_frame))

    required = int(args.warmup) + int(args.samples)
    collected: List[Dict[str, float]] = []
    warmup_profiles: List[Dict[str, float]] = []
    frame_indices: List[int] = []
    frame_records: List[Dict[str, Any]] = []
    prev_step: str | None = None
    decoded = 0
    processed = 0
    try:
        while processed < required:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if decoded % int(args.frame_stride) != 0:
                decoded += 1
                continue
            decoded += 1
            frame = pipeline.process_frame(
                frame_bgr=frame_bgr,
                frame_index=frame_index,
                prev_step_for_transition=prev_step,
                source_kind=str(args.source_kind),
            )
            profile = {str(key): float(value) for key, value in frame.runtime_ms.items()}
            logger.log_iteration(
                {
                    **pipeline.build_iteration_payload(processed + 1, frame, source_kind=str(args.source_kind)),
                    "benchmark_warmup": processed < int(args.warmup),
                }
            )
            if frame.claim_decision is not None:
                if frame.claim_decision.committed_step:
                    prev_step = frame.claim_decision.committed_step
            else:
                decision = pipeline.decision_step(frame)
                if decision not in {"", "HOLD", "INVALID"}:
                    prev_step = decision
            if processed < int(args.warmup):
                warmup_profiles.append(profile)
            else:
                collected.append(profile)
                frame_indices.append(frame_index)
                frame_records.append(
                    {
                        "frame_index": frame_index,
                        "fused_step": str(frame.fusion_result.step_id),
                        "decision_step": str(pipeline.decision_step(frame)),
                        "claim_id": str(frame.claim_decision.claim_id) if frame.claim_decision else "",
                        "claim_state": str(frame.claim_decision.state) if frame.claim_decision else "",
                        "claim_support": float(frame.claim_decision.support_score) if frame.claim_decision else 0.0,
                        "claim_contradiction": (
                            float(frame.claim_decision.contradiction_score) if frame.claim_decision else 0.0
                        ),
                        "claim_margin": (
                            float(frame.claim_decision.counterfactual_margin) if frame.claim_decision else 0.0
                        ),
                        "missing_roles": (
                            sorted(str(role) for role in frame.claim_decision.missing_roles)
                            if frame.claim_decision
                            else []
                        ),
                        "compute_plan": dict(frame.compute_plan),
                        "runtime_ms": profile,
                    }
                )
            processed += 1
    finally:
        capture.release()
        pipeline.finalize_run(logger, source_uri=str(video_path))

    if len(collected) < int(args.samples):
        raise RuntimeError(f"Video ended after {len(collected)} measured frames; requested {args.samples}")

    stage_names = sorted({name for profile in collected for name in profile})
    stage_summary = {
        stage: summarize(profile.get(stage, 0.0) for profile in collected)
        for stage in stage_names
    }
    total_values = [profile.get("total", 0.0) for profile in collected]
    total_summary = stage_summary.get("total", summarize(total_values))
    mean_total = float(total_summary["mean_ms"])
    target_ms = float(config.edge_runtime.target_perception_ms)
    invocation_keys = (
        "run_detection",
        "run_geometry",
        "run_segmentation",
        "run_retrieval",
        "run_memory_embedding",
    )
    invocation_rates = {
        key: round(
            sum(bool(record["compute_plan"].get(key, False)) for record in frame_records)
            / max(1, len(frame_records)),
            6,
        )
        for key in invocation_keys
    }
    payload = {
        "protocol": {
            "path": "deployed_step_pipeline",
            "routing_enabled": bool(config.edge_runtime.enabled),
            "source_kind": str(args.source_kind),
            "warmup_frames": int(args.warmup),
            "measured_frames": int(args.samples),
            "frame_stride": int(args.frame_stride),
            "frame_indices": frame_indices,
            "cuda_synchronized": not bool(args.no_cuda_sync),
            "ground_truth_used": False,
            "geometry_resolution_level": int(config.geometry.resolution_level),
            "visual_change_threshold": float(config.edge_runtime.visual_change_threshold),
            "camera_detection_conf": config.camera.detection_conf,
        },
        "hardware": hardware_metadata(),
        "runtime": {
            "target_perception_ms": target_ms,
            "deadline_miss_rate": round(
                sum(value > target_ms for value in total_values) / max(1, len(total_values)), 6
            ),
            "mean_throughput_fps": round(1000.0 / mean_total, 4) if mean_total > 0.0 else 0.0,
            "stages": stage_summary,
            "stage_invocation_rates": invocation_rates,
        },
        "artifacts": {
            "video_name": video_path.name,
            "config_name": Path(args.config).name,
            "weights_name": Path(args.weights).name,
            "runlog": str(logger.run_dir),
        },
        "frames": frame_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved benchmark: {output_path}")


if __name__ == "__main__":
    main()
