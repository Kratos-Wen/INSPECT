"""Command-line entry point for the modular step pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .config import apply_ablation_preset, load_config
from .inspect_system import SYSTEM_FULL_NAME, SYSTEM_NAME, build_inspect_artifacts
from .inspect_system.evidence_graph import load_evidence_graph
from .inspect_system.learned_verifier import train_calibrated_verifier
from .inspect_system.verifier import load_robot_observations
from .runtime.experiments import run_ablation_suite
from .runtime.live_assistant import build_live_assistant
from .runtime.pipeline import build_default_pipeline


def _default_yolo_weights() -> Optional[str]:
    candidate = Path(__file__).resolve().parent / "best.pt"
    return str(candidate) if candidate.exists() else None


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    default_weights = _default_yolo_weights()
    parser = argparse.ArgumentParser("MICA modular step pipeline")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--video", help="Path to the input video.")
    source_group.add_argument("--camera", type=int, default=None, help="Camera index for live assistant mode.")
    parser.add_argument(
        "--yolo-weights",
        required=default_weights is None,
        default=default_weights,
        help=(
            "Path to the YOLO weights file. "
            f"Defaults to the bundled weights at {default_weights}."
            if default_weights
            else "Path to the YOLO weights file."
        ),
    )
    parser.add_argument("--kb", required=True, help="Path to the knowledge-base JSON file.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the YAML config. Defaults to the package example config.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device, for example 'cpu' or '0' for the first CUDA device.",
    )
    parser.add_argument(
        "--state-path",
        default=None,
        help="Optional path for persisting the online fusion state.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable console feedback for online adaptation or manual review prompts.",
    )
    parser.add_argument(
        "--ablation-preset",
        default=None,
        choices=[
            "memory-off",
            "session-only",
            "long-term-only",
            "no-auto-capture",
            "gru",
            "gru-aux",
            "gru-agg",
            "gru-agg-offline-gate",
            "full-online-adapt",
        ],
        help="Optional ablation preset applied after loading the YAML config.",
    )
    parser.add_argument(
        "--gallery-root",
        default=None,
        help="Optional gallery root override. If omitted, the package auto-discovers common gallery folders.",
    )
    parser.add_argument(
        "--run-suite",
        action="store_true",
        help="Run an ablation suite and write CSV/Markdown summaries instead of a single run.",
    )
    parser.add_argument(
        "--suite-presets",
        default="custom,memory-off,session-only,long-term-only,no-auto-capture",
        help="Comma-separated presets used when --run-suite is enabled. Use 'corl-standard' to expand to the standard CoRL temporal ablations.",
    )
    parser.add_argument(
        "--suite-embeds",
        default=None,
        help="Optional comma-separated visual retrieval modes for --run-suite, for example 'hybrid-4,clip'.",
    )
    parser.add_argument(
        "--voice-mode",
        default=None,
        choices=["off", "manual", "always_on"],
        help="Optional voice mode override for live assistant mode.",
    )
    parser.add_argument(
        "--mic-device",
        type=int,
        default=None,
        help="Optional microphone device index override for live assistant mode.",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Disable the OpenCV live UI in camera mode.",
    )
    return parser


def build_inspect_arg_parser() -> argparse.ArgumentParser:
    """Create the INSPECT artifact-builder CLI parser."""

    parser = argparse.ArgumentParser(
        f"{SYSTEM_NAME}: {SYSTEM_FULL_NAME}",
        description=(
            "Build assistance-as-supervision artifacts from an existing MICA run directory. "
            "This does not rerun perception; it reuses iterations.jsonl, feedback.jsonl, "
            "review events, memory signals, and scene-graph evidence."
        ),
    )
    parser.add_argument("--run-dir", required=True, help="Existing run directory containing iterations.jsonl.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for INSPECT artifacts. Defaults to <run-dir>/inspect_artifacts.",
    )
    parser.add_argument(
        "--robot-observations",
        default=None,
        help="Optional JSONL/JSON robot-view observations using the INSPECT evidence vocabulary.",
    )
    parser.add_argument(
        "--view-candidates",
        default=None,
        help="Optional JSONL/JSON finite view candidates for active observation decisions.",
    )
    parser.add_argument(
        "--state-specs",
        default=None,
        help="Optional JSON/YAML StateSpec file that decomposes steps into precondition, interaction, transition, postcondition, negative, and admissibility evidence.",
    )
    parser.add_argument(
        "--verifier-model",
        default=None,
        help="Optional learned calibrated verifier model JSON. If supplied with --robot-observations, it replaces the rule-based verifier.",
    )
    parser.add_argument(
        "--min-auto-confidence",
        type=float,
        default=0.78,
        help="Minimum stable fused confidence for treating an uncorrected frame as verified trace supervision.",
    )
    parser.add_argument("--graph-min-support", type=float, default=0.15, help="Minimum evidence support kept in the graph.")
    parser.add_argument("--topk-per-state", type=int, default=24, help="Maximum verified_by evidence edges per state.")
    parser.add_argument("--verification-threshold", type=float, default=0.55, help="Robot verifier acceptance threshold.")
    parser.add_argument("--anomaly-threshold", type=float, default=0.30, help="Robot verifier low-confidence anomaly threshold.")
    return parser


def _run_inspect(argv: list[str]) -> None:
    parser = build_inspect_arg_parser()
    args = parser.parse_args(argv)
    summary = build_inspect_artifacts(
        run_dir=Path(args.run_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        robot_observations_path=Path(args.robot_observations) if args.robot_observations else None,
        view_candidates_path=Path(args.view_candidates) if args.view_candidates else None,
        state_specs_path=Path(args.state_specs) if args.state_specs else None,
        verifier_model_path=Path(args.verifier_model) if args.verifier_model else None,
        min_auto_confidence=float(args.min_auto_confidence),
        graph_min_support=float(args.graph_min_support),
        topk_per_state=int(args.topk_per_state),
        verification_threshold=float(args.verification_threshold),
        anomaly_threshold=float(args.anomaly_threshold),
    )
    print(f"{SYSTEM_NAME} artifacts written to {summary['output_dir']}")
    print(f"  traces: {summary['trace_path']}")
    print(f"  graph:  {summary['graph_path']}")
    if "verification_path" in summary:
        print(f"  robot verification: {summary['verification_path']}")
    if "active_observation_path" in summary:
        print(f"  active observation: {summary['active_observation_path']}")
    print(f"  summary: {summary['summary_path']}")


def build_inspect_train_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(f"{SYSTEM_NAME} learned verifier trainer")
    parser.add_argument("--graph", required=True, help="Path to procedural_evidence_graph.json.")
    parser.add_argument("--robot-observations", required=True, help="Labeled robot observation JSONL/JSON.")
    parser.add_argument("--model-out", required=True, help="Output JSON model path.")
    parser.add_argument("--threshold", type=float, default=0.55, help="Verifier decision threshold.")
    parser.add_argument("--epochs", type=int, default=600, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=0.08, help="Learning rate.")
    parser.add_argument("--l2", type=float, default=0.001, help="L2 regularization.")
    return parser


def _run_inspect_train(argv: list[str]) -> None:
    parser = build_inspect_train_arg_parser()
    args = parser.parse_args(argv)
    graph = load_evidence_graph(Path(args.graph))
    observations = load_robot_observations(Path(args.robot_observations))
    report = train_calibrated_verifier(
        graph=graph,
        observations=observations,
        model_path=Path(args.model_out),
        threshold=float(args.threshold),
        epochs=int(args.epochs),
        lr=float(args.lr),
        l2=float(args.l2),
    )
    print(f"{SYSTEM_NAME} learned verifier written to {args.model_out}")
    print(f"  examples: {report.num_examples}")
    print(f"  positives: {report.num_positive}")
    print(f"  negatives: {report.num_negative}")
    print(f"  loss: {report.loss:.4f}")
    print(f"  accuracy: {report.accuracy:.4f}")


def main(argv: Optional[list[str]] = None) -> None:
    """Run the modular step pipeline from the command line."""

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0].lower() == "inspect":
        _run_inspect(raw_argv[1:])
        return
    if raw_argv and raw_argv[0].lower() in {"inspect-train-verifier", "inspect-train"}:
        _run_inspect_train(raw_argv[1:])
        return

    parser = build_arg_parser()
    args = parser.parse_args(raw_argv)

    config = apply_ablation_preset(load_config(args.config), args.ablation_preset)
    if args.gallery_root:
        config.experts.gallery_root = str(args.gallery_root)
    if args.voice_mode:
        config.voice.mode = str(args.voice_mode)
        config.voice.enabled = str(args.voice_mode).lower() != "off"
    if args.mic_device is not None:
        config.voice.device = int(args.mic_device)
    if args.no_ui:
        config.ui.enabled = False

    if args.run_suite:
        if args.video is None:
            parser.error("--run-suite requires --video, not --camera.")
        presets = [item.strip() for item in str(args.suite_presets).split(",") if item.strip()]
        embed_modes = None
        if args.suite_embeds:
            embed_modes = [item.strip() for item in str(args.suite_embeds).split(",") if item.strip()]
        run_ablation_suite(
            base_config=config,
            video_path=Path(args.video),
            kb_path=args.kb,
            yolo_weights=str(args.yolo_weights),
            device=args.device,
            interactive=bool(args.interactive),
            gallery_root=args.gallery_root,
            presets=presets,
            embed_modes=embed_modes,
        )
        return

    state_path = Path(args.state_path) if args.state_path else None
    if args.camera is not None:
        assistant = build_live_assistant(
            config=config,
            kb_path=args.kb,
            yolo_weights=str(args.yolo_weights),
            device=args.device,
            state_path=state_path,
            interactive=bool(args.interactive),
            camera_index=int(args.camera),
        )
        assistant.run(int(args.camera))
        return

    pipeline = build_default_pipeline(
        config=config,
        kb_path=args.kb,
        yolo_weights=str(args.yolo_weights),
        device=args.device,
        state_path=state_path,
        interactive=bool(args.interactive),
    )
    pipeline.run(Path(args.video))
