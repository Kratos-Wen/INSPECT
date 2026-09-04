"""Command-line entry point for INSPECT."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from .components import TimelineFeedbackProvider
from .config import apply_memory_preset, load_config
from .inspect_system import SYSTEM_FULL_NAME, SYSTEM_NAME
from .inspect_system.active_observation import load_view_candidates
from .inspect_system.assistant_evidence import (
    EvidenceRequirementProfile,
    profile_from_assistant_evidence_events,
)
from .inspect_system.ego_evaluation import evaluate_ego_run
from .inspect_system.robot_adapter import (
    RecordingRobotAdapter,
    RobotProceduralDecisionLoop,
)
from .inspect_system.learned_view_lattice_policy import LearnedViewLatticePolicy
from .inspect_system.view_lattice_policy import (
    FixedViewLatticePolicy,
    write_lattice_decisions_jsonl,
)
from .inspect_system.workflow import build_inspect_artifacts
from .inspect_system.evidence_graph import load_evidence_graph
from .inspect_system.ego_evaluation import load_ego_state_segments
from .inspect_system.gallery_builder import build_gallery_from_ego_video
from .inspect_system.learned_verifier import (
    CalibratedEvidenceVerifier,
    train_calibrated_verifier,
)
from .inspect_system.robot_observation_builder import (
    build_robot_observations_from_frames,
)
from .inspect_system.verifier import (
    RobotProceduralStateVerifier,
    load_robot_observations,
)
from .runtime.live_assistant import build_live_assistant
from .runtime.pipeline import build_default_pipeline


def _default_yolo_weights() -> Optional[str]:
    candidate = Path(__file__).resolve().parent / "best.pt"
    return str(candidate) if candidate.exists() else None


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    default_weights = _default_yolo_weights()
    parser = argparse.ArgumentParser(
        "INSPECT trace engine and robot verification pipeline"
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--video", help="Path to the input video.")
    source_group.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera index for live trace-engine mode.",
    )
    parser.add_argument(
        "--yolo-weights",
        required=default_weights is None,
        default=default_weights,
        help=(
            "Path to the YOLO weights file. "
            "Defaults to bundled best.pt in the repository root."
            if default_weights
            else "Path to the YOLO weights file."
        ),
    )
    parser.add_argument(
        "--kb", required=True, help="Path to the knowledge-base JSON file."
    )
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
    memory_preset_choices = [
        "memory-off",
        "session-only",
        "long-term-only",
        "no-auto-capture",
    ]
    parser.add_argument(
        "--memory-preset",
        default=None,
        choices=memory_preset_choices,
        help="Optional memory subsystem preset applied after loading the YAML config.",
    )
    parser.add_argument(
        "--ablation-preset",
        dest="memory_preset",
        default=None,
        choices=memory_preset_choices,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gallery-root",
        default=None,
        help="Optional gallery root override. If omitted, the package auto-discovers common gallery folders.",
    )
    parser.add_argument(
        "--simulate-feedback-gt",
        default=None,
        help=(
            "Optional GT timeline TXT/JSON/CSV used to simulate human corrections for offline videos. "
            "Labels outside S1-S4, such as wrong, become invalid hard-negative feedback."
        ),
    )
    parser.add_argument(
        "--simulate-feedback-every-sec",
        type=float,
        default=0.5,
        help="Minimum interval between simulated feedback events when --simulate-feedback-gt is used.",
    )
    parser.add_argument(
        "--simulate-feedback-fps",
        type=float,
        default=None,
        help="Optional FPS override for time-based simulated feedback GT.",
    )
    parser.add_argument(
        "--voice-mode",
        default=None,
        choices=["off", "manual", "always_on"],
        help="Optional voice mode override for live trace-engine mode.",
    )
    parser.add_argument(
        "--mic-device",
        type=int,
        default=None,
        help="Optional microphone device index override for live trace-engine mode.",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Disable the OpenCV live UI in camera mode.",
    )
    parser.add_argument(
        "--eval-realtime",
        action="store_true",
        help="In live camera mode, display and write realtime FPS/latency metrics.",
    )
    return parser


def build_inspect_arg_parser() -> argparse.ArgumentParser:
    """Create the INSPECT artifact-builder CLI parser."""

    parser = argparse.ArgumentParser(
        f"{SYSTEM_NAME}: {SYSTEM_FULL_NAME}",
        description=(
            "Build assistance-as-supervision artifacts from an existing INSPECT trace run directory. "
            "This does not rerun perception; it reuses iterations.jsonl, feedback.jsonl, "
            "review events, memory signals, and scene-graph evidence."
        ),
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Existing run directory containing iterations.jsonl.",
    )
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
        help="Optional JSONL/JSON finite calibrated view candidates for fixed-lattice robot view decisions.",
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
    parser.add_argument(
        "--graph-min-support",
        type=float,
        default=0.15,
        help="Minimum evidence support kept in the graph.",
    )
    parser.add_argument(
        "--topk-per-state",
        type=int,
        default=24,
        help="Maximum verified_by evidence edges per state.",
    )
    parser.add_argument(
        "--verification-threshold",
        type=float,
        default=0.55,
        help="Robot verifier acceptance threshold.",
    )
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=0.30,
        help="Robot verifier low-confidence anomaly threshold.",
    )
    return parser


def _run_inspect(argv: list[str]) -> None:
    parser = build_inspect_arg_parser()
    args = parser.parse_args(argv)
    summary = build_inspect_artifacts(
        run_dir=Path(args.run_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        robot_observations_path=(
            Path(args.robot_observations) if args.robot_observations else None
        ),
        view_candidates_path=(
            Path(args.view_candidates) if args.view_candidates else None
        ),
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
        print(f"  fixed-lattice view decisions: {summary['active_observation_path']}")
    print(f"  summary: {summary['summary_path']}")


def build_inspect_train_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(f"{SYSTEM_NAME} learned verifier trainer")
    parser.add_argument(
        "--graph", required=True, help="Path to procedural_evidence_graph.json."
    )
    parser.add_argument(
        "--robot-observations",
        required=True,
        help="Labeled robot observation JSONL/JSON.",
    )
    parser.add_argument("--model-out", required=True, help="Output JSON model path.")
    parser.add_argument(
        "--threshold", type=float, default=0.55, help="Verifier decision threshold."
    )
    parser.add_argument("--epochs", type=int, default=600, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=0.08, help="Learning rate.")
    parser.add_argument("--l2", type=float, default=0.001, help="L2 regularization.")
    return parser


def build_robot_observation_arg_parser() -> argparse.ArgumentParser:
    """Create the robot-view observation builder parser."""

    default_weights = _default_yolo_weights()
    parser = argparse.ArgumentParser(
        f"{SYSTEM_NAME} robot observation builder",
        description=(
            "Convert robot or active-camera RGB frames into RobotObservation JSONL. "
            "This is the automatic robot-side grounding path used before state verification."
        ),
    )
    parser.add_argument(
        "--frames",
        required=True,
        help="Directory of robot-view images, one image path, or a JSON/JSONL manifest.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional JSON/JSONL manifest with image_path, view_id, prev_state, candidate_states, and labels.",
    )
    parser.add_argument(
        "--output", required=True, help="Output robot_observations.jsonl path."
    )
    parser.add_argument(
        "--yolo-weights",
        required=default_weights is None,
        default=default_weights,
        help="Path to YOLO weights. Defaults to bundled best.pt when present.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config path. Defaults to resources/config.example.yaml.",
    )
    parser.add_argument(
        "--device", default="cpu", help="Inference device, for example cpu or 0."
    )
    parser.add_argument(
        "--state-specs",
        default=None,
        help="Optional StateSpec file. If --candidate-states is omitted, all state_ids become candidates.",
    )
    parser.add_argument(
        "--candidate-states",
        default="",
        help="Comma-separated candidate procedural states for all frames unless the manifest overrides them.",
    )
    parser.add_argument(
        "--prev-state",
        default="",
        help="Default previous state unless the manifest overrides it.",
    )
    parser.add_argument(
        "--view-id",
        default="",
        help="Default robot view id unless the manifest overrides it.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for quick checks.",
    )
    parser.add_argument(
        "--detection-conf",
        type=float,
        default=None,
        help="Optional detector confidence override for robot frames.",
    )
    return parser


def build_decision_loop_replay_arg_parser() -> argparse.ArgumentParser:
    """Create the recorded robot decision-loop replay parser."""

    parser = argparse.ArgumentParser(
        f"{SYSTEM_NAME} recorded robot decision loop",
        description=(
            "Replay observe-verify-select-view robot decisions from precomputed "
            "RobotObservation JSONL and finite calibrated view candidates."
        ),
    )
    parser.add_argument(
        "--graph", required=True, help="Path to procedural_evidence_graph.json."
    )
    parser.add_argument(
        "--robot-observations",
        required=True,
        help="RobotObservation JSONL/JSON in observation order.",
    )
    parser.add_argument(
        "--view-candidates", required=True, help="Finite view candidates JSONL/JSON."
    )
    parser.add_argument(
        "--output", required=True, help="Output decision_loop_replay.json path."
    )
    parser.add_argument(
        "--verifier-model",
        default=None,
        help="Optional calibrated verifier model JSON.",
    )
    parser.add_argument(
        "--evidence-requirement-profile",
        default=None,
        help="Optional Assistant-use evidence requirement profile for fixed_lattice mode.",
    )
    _add_learned_view_policy_args(parser)
    parser.add_argument(
        "--max-observation-moves",
        type=int,
        default=2,
        help="Maximum fixed-lattice view changes.",
    )
    parser.add_argument(
        "--initial-view",
        default="",
        help="Initial view id passed to the robot adapter.",
    )
    parser.add_argument(
        "--online-view-state-output",
        default=None,
        help="Optional path for the adapted reveal model; session-local R is written as a sidecar.",
    )
    return parser


def build_evidence_requirement_profile_arg_parser() -> argparse.ArgumentParser:
    """Create parser for Assistant-use claim->evidence-view profiles."""

    parser = argparse.ArgumentParser(
        f"{SYSTEM_NAME} Assistant evidence requirement profile",
        description=(
            "Build a claim-conditioned evidence-view requirement profile from human Assistant-use "
            "events. These events do not contain robot V0-V5 labels; they only encode which "
            "evidence became necessary or visible during feedback/correction."
        ),
    )
    parser.add_argument(
        "--event-log",
        action="append",
        required=True,
        help="Assistant evidence events JSONL. Repeatable.",
    )
    parser.add_argument(
        "--output", required=True, help="Output evidence_requirement_profile.json."
    )
    return parser


def build_lattice_policy_arg_parser() -> argparse.ArgumentParser:
    """Create parser for fixed V0-V5 view-lattice policy."""

    parser = argparse.ArgumentParser(
        f"{SYSTEM_NAME} fixed view-lattice policy",
        description=(
            "Select robot-executable calibrated view IDs from finite view candidates. "
            "The policy uses Assistant-learned claim->evidence requirements plus robot-side "
            "view-candidate metadata; it does not use human-labeled robot view preferences."
        ),
    )
    parser.add_argument(
        "--graph", required=True, help="Path to procedural_evidence_graph.json."
    )
    parser.add_argument(
        "--robot-observations",
        required=True,
        help="RobotObservation JSONL/JSON in observation order.",
    )
    parser.add_argument(
        "--view-candidates",
        required=True,
        help="Finite V0-V5 view candidates JSONL/JSON.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output fixed_view_lattice_decisions.jsonl path.",
    )
    parser.add_argument(
        "--verifier-model",
        default=None,
        help="Optional calibrated verifier model JSON.",
    )
    parser.add_argument(
        "--evidence-requirement-profile",
        default=None,
        help="Optional Assistant-use claim->evidence-view requirement profile JSON.",
    )
    _add_learned_view_policy_args(parser)
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional observation limit."
    )
    parser.add_argument(
        "--include-verified",
        action="store_true",
        help="Also select views for verified states. By default only observe/uncertain states are scored.",
    )
    return parser


def _add_learned_view_policy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--view-policy-config",
        default=None,
        help="JSON bundle for the adopted ActiveSelector policy and online update settings.",
    )
    parser.add_argument(
        "--reveal-model",
        default=None,
        help=(
            "Assistant-trained relative reveal model. When provided, the runtime uses "
            "ActiveSelector instead of the heuristic fixed-view policy."
        ),
    )
    parser.add_argument(
        "--requirement-report",
        default=None,
        help="Projected assistant claim-role labels used to estimate R(c,q).",
    )
    parser.add_argument(
        "--requirement-calibration",
        default=None,
        help="Assistant-only calibration JSON for the R(c,q)/pi combination.",
    )
    parser.add_argument("--requirement-beta", type=float, default=2.0)
    parser.add_argument("--requirement-strength", type=float, default=1.0)
    parser.add_argument("--requirement-temperature", type=float, default=1.0)
    parser.add_argument("--view-motion-cost", type=float, default=0.05)
    parser.add_argument("--view-threshold", type=float, default=0.02)
    parser.add_argument(
        "--disable-online-view-update",
        action="store_true",
        help="Disable causal session-local R/pi updates after robot re-verification.",
    )
    parser.add_argument("--online-view-update-weight", type=float, default=1.0)
    parser.add_argument("--online-requirement-scale", type=float, default=1.0)
    parser.add_argument("--online-requirement-blend", type=float, default=1.0)
    parser.add_argument("--online-requirement-prior-strength", type=float, default=0.05)
    parser.add_argument("--online-view-gain-deadband", type=float, default=0.05)


def build_ego_evaluation_arg_parser() -> argparse.ArgumentParser:
    """Create the ego/video state timeline evaluation parser."""

    parser = argparse.ArgumentParser(
        f"{SYSTEM_NAME} ego-video state evaluation",
        description=(
            "Evaluate an offline INSPECT ego/video run against state timeline GT. "
            "The GT can be segment annotations or audio-derived transition events such as "
            "`timestamp,text=state two`."
        ),
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Existing run directory containing iterations.jsonl.",
    )
    parser.add_argument(
        "--gt",
        required=True,
        help="GT TXT/JSON/JSONL/CSV with state segments or transition events.",
    )
    parser.add_argument(
        "--output", required=True, help="Output evaluation_summary.json path."
    )
    parser.add_argument(
        "--aligned-output",
        default=None,
        help="Optional JSONL output with one row per labeled iteration.",
    )
    parser.add_argument(
        "--prediction-key",
        default="fused_step",
        help="Iteration field to evaluate, e.g. fused_step, temporal_step, state_step, retrieval_step.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional FPS override for time-based GT. Defaults to video FPS from run meta when available.",
    )
    parser.add_argument(
        "--transition-window-sec",
        type=float,
        default=3.0,
        help="Window used to match GT state transitions to predicted transitions.",
    )
    return parser


def build_gallery_arg_parser() -> argparse.ArgumentParser:
    """Create the annotated ego-video gallery builder parser."""

    parser = argparse.ArgumentParser(
        f"{SYSTEM_NAME} retrieval gallery builder",
        description=(
            "Build a step-retrieval gallery from an annotated ego video. Legal states become "
            "positive reference images; wrong/invalid intervals become hard negatives."
        ),
    )
    parser.add_argument("--video", required=True, help="Annotated ego video.")
    parser.add_argument(
        "--gt", required=True, help="GT TXT/JSON/JSONL/CSV timeline, e.g. 0~24 step1."
    )
    parser.add_argument("--output", required=True, help="Output gallery root.")
    parser.add_argument(
        "--steps", default="S1,S2,S3,S4", help="Comma-separated legal step IDs."
    )
    parser.add_argument(
        "--samples-per-segment",
        type=int,
        default=18,
        help="Frames sampled from each labeled segment.",
    )
    parser.add_argument(
        "--margin-sec",
        type=float,
        default=1.0,
        help="Seconds skipped near segment boundaries.",
    )
    parser.add_argument("--quality", type=int, default=92, help="JPEG quality.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing gallery images before sampling.",
    )
    return parser


def _csv_items(text: str) -> list[str]:
    return [item.strip().upper() for item in str(text or "").split(",") if item.strip()]


def _run_robot_observation_builder(argv: list[str]) -> None:
    parser = build_robot_observation_arg_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.detection_conf is not None:
        config.detection.conf = float(args.detection_conf)
    observations = build_robot_observations_from_frames(
        frames_path=Path(args.frames),
        output_path=Path(args.output),
        config=config,
        yolo_weights=str(args.yolo_weights),
        device=str(args.device),
        manifest_path=Path(args.manifest) if args.manifest else None,
        candidate_states=_csv_items(args.candidate_states),
        prev_state=str(args.prev_state or ""),
        view_id=str(args.view_id or ""),
        state_specs_path=Path(args.state_specs) if args.state_specs else None,
        max_frames=args.max_frames,
    )
    print(f"{SYSTEM_NAME} robot observations written to {args.output}")
    print(f"  observations: {len(observations)}")


def _build_robot_view_policy(args, candidates, profile):
    if getattr(args, "view_policy_config", None):
        config_path = Path(args.view_policy_config).resolve()
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if payload.get("uses_candidate_view_images") or payload.get(
            "uses_robot_utility_labels"
        ):
            raise RuntimeError("View-policy bundle declares forbidden supervision.")

        def bundled_path(name: str) -> Path | None:
            value = str(payload.get(name, "")).strip()
            if not value:
                return None
            path = Path(value)
            return path if path.is_absolute() else (config_path.parent / path).resolve()

        reveal_model = bundled_path("reveal_model")
        if reveal_model is None:
            raise ValueError("View-policy bundle requires reveal_model.")
        policy = LearnedViewLatticePolicy.from_paths(
            reveal_model=reveal_model,
            requirement_report=bundled_path("requirement_report"),
            requirement_calibration=bundled_path("requirement_calibration"),
            candidates=candidates,
            requirement_beta=float(payload.get("requirement_beta", 0.25)),
            requirement_strength=float(payload.get("requirement_strength", 1.0)),
            requirement_temperature=float(payload.get("requirement_temperature", 4.0)),
            lambda_cost=float(payload.get("lambda_cost", 0.05)),
            tau_view=float(payload.get("tau_view", 0.02)),
            online_update_enabled=bool(payload.get("online_update_enabled", True)),
            online_update_weight=float(payload.get("online_update_weight", 1.0)),
            online_requirement_scale=float(
                payload.get("online_requirement_scale", 1.0)
            ),
            online_requirement_blend=float(
                payload.get("online_requirement_blend", 1.0)
            ),
            online_requirement_prior_strength=float(
                payload.get("online_requirement_prior_strength", 0.001)
            ),
            online_gain_deadband=float(payload.get("online_gain_deadband", 0.05)),
        )
        destination_field = getattr(policy.selector.model, "session_ray_field", None)
        if destination_field is not None:
            destination_field.strength = float(
                payload.get("session_destination_strength", 1.0)
            )
        return policy
    if getattr(args, "reveal_model", None):
        return LearnedViewLatticePolicy.from_paths(
            reveal_model=Path(args.reveal_model),
            requirement_report=(
                Path(args.requirement_report) if args.requirement_report else None
            ),
            requirement_calibration=(
                Path(args.requirement_calibration)
                if args.requirement_calibration
                else None
            ),
            candidates=candidates,
            requirement_beta=float(args.requirement_beta),
            requirement_strength=float(args.requirement_strength),
            requirement_temperature=float(args.requirement_temperature),
            lambda_cost=float(args.view_motion_cost),
            tau_view=float(args.view_threshold),
            online_update_enabled=not bool(args.disable_online_view_update),
            online_update_weight=float(args.online_view_update_weight),
            online_requirement_scale=float(args.online_requirement_scale),
            online_requirement_blend=float(args.online_requirement_blend),
            online_requirement_prior_strength=float(
                args.online_requirement_prior_strength
            ),
            online_gain_deadband=float(args.online_view_gain_deadband),
        )
    return FixedViewLatticePolicy(candidates, requirement_profile=profile)


def _run_decision_loop_replay(argv: list[str]) -> None:
    parser = build_decision_loop_replay_arg_parser()
    args = parser.parse_args(argv)
    graph = load_evidence_graph(Path(args.graph))
    observations = load_robot_observations(Path(args.robot_observations))
    candidates = load_view_candidates(Path(args.view_candidates))
    verifier = (
        CalibratedEvidenceVerifier.load(Path(args.verifier_model))
        if args.verifier_model
        else None
    )
    robot = RecordingRobotAdapter(observations)
    profile = (
        EvidenceRequirementProfile.load(Path(args.evidence_requirement_profile))
        if args.evidence_requirement_profile
        else None
    )
    view_policy = _build_robot_view_policy(args, candidates, profile)
    loop = RobotProceduralDecisionLoop(
        graph=graph,
        robot=robot,
        verifier=verifier,
        view_candidates=candidates,
        view_lattice_policy=view_policy,
        max_observation_moves=int(args.max_observation_moves),
    )
    result = loop.step(initial_view=str(args.initial_view or ""))
    online_state = None
    if args.online_view_state_output and hasattr(view_policy, "save_online_state"):
        online_state = view_policy.save_online_state(
            Path(args.online_view_state_output)
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "result": result.to_dict(),
        "action_log": robot.action_log(),
        "num_observations": len(observations),
        "num_view_candidates": len(candidates),
        "online_view_state": online_state,
    }
    import json

    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"{SYSTEM_NAME} recorded decision loop written to {args.output}")
    print(f"  final_action: {result.recommended_action}")
    print(f"  predicted_state: {result.predicted_state}")


def _run_build_evidence_requirement_profile(argv: list[str]) -> None:
    parser = build_evidence_requirement_profile_arg_parser()
    args = parser.parse_args(argv)
    profile = profile_from_assistant_evidence_events(
        [Path(item) for item in args.event_log]
    )
    profile.save(Path(args.output))
    print(f"{SYSTEM_NAME} evidence requirement profile written to {args.output}")
    print(f"  records_seen: {profile.metadata.get('records_seen', 0)}")
    print(f"  records_used: {profile.metadata.get('records_used', 0)}")
    print(f"  claims: {len(profile.claim_view_weights)}")
    print("  contains_robot_view_labels: false")


def _run_lattice_policy(argv: list[str]) -> None:
    parser = build_lattice_policy_arg_parser()
    args = parser.parse_args(argv)
    graph = load_evidence_graph(Path(args.graph))
    observations = load_robot_observations(Path(args.robot_observations))
    if args.limit is not None:
        observations = observations[: max(0, int(args.limit))]
    candidates = load_view_candidates(Path(args.view_candidates))
    verifier = (
        CalibratedEvidenceVerifier.load(Path(args.verifier_model))
        if args.verifier_model
        else RobotProceduralStateVerifier(graph)
    )
    profile = (
        EvidenceRequirementProfile.load(Path(args.evidence_requirement_profile))
        if args.evidence_requirement_profile
        else None
    )
    policy = _build_robot_view_policy(args, candidates, profile)
    decisions = []
    selected_counts: Counter[str] = Counter()
    for observation in observations:
        result = verifier.verify(observation)
        if not bool(args.include_verified) and result.recommended_action != "observe":
            continue
        decision = policy.select_view(observation, result, graph)
        decisions.append(decision)
        if decision.selected_view:
            selected_counts[decision.selected_view] += 1
    write_lattice_decisions_jsonl(decisions, Path(args.output))
    print(f"{SYSTEM_NAME} fixed view-lattice decisions written to {args.output}")
    print(f"  observations: {len(observations)}")
    print(f"  decisions: {len(decisions)}")
    print(f"  selected views: {dict(selected_counts)}")


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


def _run_ego_evaluation(argv: list[str]) -> None:
    parser = build_ego_evaluation_arg_parser()
    args = parser.parse_args(argv)
    summary = evaluate_ego_run(
        run_dir=Path(args.run_dir),
        gt_path=Path(args.gt),
        output_path=Path(args.output),
        prediction_key=str(args.prediction_key),
        fps=args.fps,
        transition_window_sec=float(args.transition_window_sec),
        write_aligned_path=Path(args.aligned_output) if args.aligned_output else None,
    )
    print(f"{SYSTEM_NAME} ego evaluation written to {args.output}")
    print(f"  labeled iterations: {summary['num_labeled_iterations']}")
    print(f"  accuracy: {summary['accuracy']}")
    print(f"  macro_f1: {summary['macro_f1']}")


def _run_gallery_builder(argv: list[str]) -> None:
    parser = build_gallery_arg_parser()
    args = parser.parse_args(argv)
    summary = build_gallery_from_ego_video(
        video_path=Path(args.video),
        gt_path=Path(args.gt),
        output_dir=Path(args.output),
        steps=_csv_items(args.steps),
        samples_per_segment=int(args.samples_per_segment),
        margin_sec=float(args.margin_sec),
        image_quality=int(args.quality),
        overwrite=bool(args.overwrite),
    )
    print(f"{SYSTEM_NAME} retrieval gallery written to {args.output}")
    print(f"  positives: {summary.positives}")
    print(f"  wrong hard negatives: {summary.negatives}")
    print(f"  skipped: {summary.skipped}")


def _video_fps(video_path: str | Path, fallback: float = 30.0) -> float:
    try:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        capture.release()
        return fps if fps > 1e-6 else float(fallback)
    except Exception:
        return float(fallback)


def _build_simulated_feedback_provider(
    args: argparse.Namespace, config
) -> TimelineFeedbackProvider | None:
    gt_path = getattr(args, "simulate_feedback_gt", None)
    video_path = getattr(args, "video", None)
    if not gt_path or not video_path:
        return None
    fps = (
        float(args.simulate_feedback_fps)
        if args.simulate_feedback_fps is not None
        else _video_fps(video_path)
    )
    segments = load_ego_state_segments(Path(gt_path), fps=fps)
    return TimelineFeedbackProvider(
        segments=[segment.to_dict() for segment in segments],
        steps=config.experts.steps,
        min_interval_frames=int(
            round(max(0.05, float(args.simulate_feedback_every_sec)) * fps)
        ),
    )


def main(argv: Optional[list[str]] = None) -> None:
    """Run the modular step pipeline from the command line."""

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0].lower() == "inspect":
        _run_inspect(raw_argv[1:])
        return
    if raw_argv and raw_argv[0].lower() in {"inspect-train-verifier", "inspect-train"}:
        _run_inspect_train(raw_argv[1:])
        return
    if raw_argv and raw_argv[0].lower() in {
        "inspect-build-robot-observations",
        "inspect-build-observations",
        "build-robot-observations",
    }:
        _run_robot_observation_builder(raw_argv[1:])
        return
    if raw_argv and raw_argv[0].lower() in {
        "inspect-replay-loop",
        "inspect-replay-decision-loop",
        "replay-decision-loop",
    }:
        _run_decision_loop_replay(raw_argv[1:])
        return
    if raw_argv and raw_argv[0].lower() in {
        "inspect-build-evidence-requirement-profile",
        "build-evidence-requirement-profile",
        "build-assistant-evidence-profile",
    }:
        _run_build_evidence_requirement_profile(raw_argv[1:])
        return
    if raw_argv and raw_argv[0].lower() in {
        "inspect-lattice-view-policy",
        "inspect-fixed-view-policy",
        "fixed-view-policy",
    }:
        _run_lattice_policy(raw_argv[1:])
        return
    if raw_argv and raw_argv[0].lower() in {
        "inspect-evaluate-ego",
        "inspect-eval-ego",
        "evaluate-ego",
        "eval-ego",
    }:
        _run_ego_evaluation(raw_argv[1:])
        return
    if raw_argv and raw_argv[0].lower() in {
        "inspect-build-gallery",
        "build-gallery",
        "build-retrieval-gallery",
    }:
        _run_gallery_builder(raw_argv[1:])
        return

    parser = build_arg_parser()
    args = parser.parse_args(raw_argv)

    config = apply_memory_preset(load_config(args.config), args.memory_preset)
    if args.gallery_root:
        config.experts.gallery_root = str(args.gallery_root)
    if args.voice_mode:
        config.voice.mode = str(args.voice_mode)
        config.voice.enabled = str(args.voice_mode).lower() != "off"
    if args.mic_device is not None:
        config.voice.device = int(args.mic_device)
    if args.no_ui:
        config.ui.enabled = False
    if args.eval_realtime:
        config.runlog.eval_realtime = True
    simulated_feedback_provider = _build_simulated_feedback_provider(args, config)

    state_path = Path(args.state_path) if args.state_path else None
    if args.camera is not None:
        assistant = build_live_assistant(
            config=config,
            kb_path=args.kb,
            yolo_weights=str(args.yolo_weights),
            device=args.device,
            state_path=state_path,
            interactive=bool(args.interactive),
            feedback_provider_override=simulated_feedback_provider,
            camera_index=int(args.camera),
        )
        assistant.run(int(args.camera), eval_realtime=bool(args.eval_realtime))
        return

    pipeline = build_default_pipeline(
        config=config,
        kb_path=args.kb,
        yolo_weights=str(args.yolo_weights),
        device=args.device,
        state_path=state_path,
        interactive=bool(args.interactive),
        feedback_provider_override=simulated_feedback_provider,
    )
    pipeline.run(Path(args.video))
