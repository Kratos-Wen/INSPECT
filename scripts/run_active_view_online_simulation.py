"""Run active-view videos through the online Assistant with GT-simulated feedback.

This script is for validating that INSPECT can mine useful supervision from
normal online use. It does not use robot six-view images. It converts
procedural timeline rows into per-video simulated human feedback, runs the
Assistant once per video, builds online claim/evidence traces, mines
low-evidence -> high-evidence transitions, and reports step/active-view
statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


def normalize_key(value: object) -> str:
    text = str(value or "").lower().strip().replace("-", "_").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text).strip("_")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def active_view_gt_state(row: Mapping[str, str], include_unresolved: bool = True) -> str:
    outcome = normalize_key(row.get("outcome", ""))
    if outcome == "supported":
        return str(row.get("step_id", "")).strip().upper()
    if outcome == "contradicted":
        return "INVALID"
    if outcome == "unresolved" and include_unresolved:
        return "UNCERTAIN"
    return ""


def split_gt_by_video(timeline_csv: Path, output_dir: Path, include_unresolved: bool = True) -> Dict[str, Path]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in read_csv(timeline_csv):
        if normalize_key(row.get("skip", "")) in {"1", "true", "yes"}:
            continue
        state = active_view_gt_state(row, include_unresolved=include_unresolved)
        if not state:
            continue
        video = str(row.get("video", "")).strip()
        if not video:
            continue
        record = {
            "start_frame": row.get("start_frame", ""),
            "end_frame": row.get("end_frame", ""),
            "state": state,
            "source": "active_view_timeline_for_online_sim",
            "text": f"step={row.get('step_id','')};claim={row.get('claim_id','')};outcome={row.get('outcome','')};note={row.get('note','')}",
        }
        grouped.setdefault(Path(video).name, []).append(record)
    paths: Dict[str, Path] = {}
    for video_name, rows in grouped.items():
        rows = sorted(rows, key=lambda item: int(float(item.get("start_frame") or 0)))
        path = output_dir / f"{Path(video_name).stem}_gt.csv"
        write_csv(path, rows, ["start_frame", "end_frame", "state", "source", "text"])
        paths[video_name] = path
    return paths


def resolution_supervision_video_names(timeline_csv: Path) -> set[str]:
    """Return sessions with an observed supported/contradicted resolution."""

    return {
        Path(str(row.get("video", "")).strip()).name
        for row in read_csv(timeline_csv)
        if normalize_key(row.get("outcome", "")) in {"supported", "contradicted"}
        and str(row.get("video", "")).strip()
        and normalize_key(row.get("skip", "")) not in {"1", "true", "yes"}
    }


def write_fast_config(path: Path, runlog_dir: Path, stride: int, detail_level: str = "debug") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"""video:
  stride: {int(stride)}
  write_annotated: false
detection:
  tta: false
  use_builtin_tta: false
geometry:
  backend: gradient
segmentation:
  backend: none
  fail_on_unavailable: false
runlog:
  save_dir: "{str(runlog_dir).replace(chr(92), '/')}"
  detail_level: "{detail_level}"
  persist_temporal_tokens: true
review:
  enabled: true
  evidence_prompt_enabled: false
voice:
  enabled: false
speech:
  enabled: false
ui:
  enabled: false
"""
    path.write_text(payload, encoding="utf-8")


def write_full_moge_config(
    path: Path,
    runlog_dir: Path,
    stride: int,
    detail_level: str = "debug",
    detector_conf: float = 0.10,
    identity_commit_conf: float = 0.50,
    identity_commit_track_margin: float = 0.12,
    role_bridge_enabled: bool = False,
    role_bridge_max_gap: int = 120,
    detector_end2end: bool | None = None,
    evidence_scorer_path: str = "",
    counterfactual_margin: float = 0.0,
    admissibility_gate_enabled: bool = True,
    memory_gate_enabled: bool = True,
    specialized_counterfactual_enabled: bool = True,
    prerequisite_bootstrap_enabled: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"""video:
  stride: {int(stride)}
  write_annotated: false
detection:
  conf: {float(detector_conf):.6f}
  identity_commit_conf: {float(identity_commit_conf):.6f}
  identity_commit_track_margin: {float(identity_commit_track_margin):.6f}
  role_bridge_enabled: {str(bool(role_bridge_enabled)).lower()}
  role_bridge_max_gap: {int(role_bridge_max_gap)}
  end2end: {str(detector_end2end).lower() if detector_end2end is not None else 'null'}
  tta: false
  use_builtin_tta: false
geometry:
  backend: moge
  moge_model: Ruicheng/moge-2-vits-normal
  use_fp16: true
  resolution_level: 9
  apply_mask: true
temporal:
  backend: ultralytics_botsort
track_evidence:
  enabled: true
scene_evidence:
  enabled: true
scene_graph:
  enabled: true
interaction:
  enabled: true
segmentation:
  backend: none
  fail_on_unavailable: false
runlog:
  save_dir: "{str(runlog_dir).replace(chr(92), '/')}"
  detail_level: "{detail_level}"
  persist_temporal_tokens: true
review:
  enabled: true
  evidence_prompt_enabled: false
voice:
  enabled: false
speech:
  enabled: false
assistant:
  enabled: true
  llm_enabled: false
claim_verifier:
  enabled: {str(bool(evidence_scorer_path)).lower()}
  evidence_scorer_path: "{str(evidence_scorer_path).replace(chr(92), '/')}"
  support_threshold: 0.35
  contradiction_threshold: 0.35
  counterfactual_margin: {float(counterfactual_margin):.6f}
  admissibility_gate_enabled: {str(bool(admissibility_gate_enabled)).lower()}
  memory_gate_enabled: {str(bool(memory_gate_enabled)).lower()}
  specialized_counterfactual_enabled: {str(bool(specialized_counterfactual_enabled)).lower()}
  prerequisite_bootstrap_enabled: {str(bool(prerequisite_bootstrap_enabled)).lower()}
  prerequisite_confirmation_frames: 2
  ema_decay: 0.55
  require_step_match_for_support: true
memory:
  preset: session-only
  enabled: true
  session_enabled: true
  long_term_enabled: false
ui:
  enabled: false
"""
    path.write_text(payload, encoding="utf-8")


def run_command(command: List[str], cwd: Path, dry_run: bool = False) -> int:
    print(" ".join(command))
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=str(cwd))
    return int(completed.returncode)


def newest_run(runlog_dir: Path, before: set[str]) -> Optional[Path]:
    candidates = [path for path in runlog_dir.iterdir() if path.is_dir() and path.name not in before]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for line in handle if line.strip())


def resumable_run(runlog_dir: Path, output_dir: Path, video: Path) -> Optional[Path]:
    """Return the latest fully postprocessed run for ``video``."""

    report = output_dir / "evidence_trace_events" / f"{video.stem}_report.json"
    if not report.exists() or not runlog_dir.exists():
        return None
    candidates = sorted(
        (
            path
            for path in runlog_dir.glob(f"{video.stem}_*")
            if path.is_dir() and count_jsonl(path / "iterations.jsonl") > 0
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def run_pipeline_for_video(
    *,
    python: str,
    repo_root: Path,
    video: Path,
    gt: Path,
    kb: Path,
    config: Path,
    weights: Path,
    runlog_dir: Path,
    output_dir: Path,
    device: str,
    feedback_every_sec: float,
    feedback_mode: str,
    dry_run: bool,
    resume_run: Optional[Path] = None,
) -> Dict[str, Any]:
    run_dir = resume_run
    returncode = 0
    if run_dir is None:
        before = {path.name for path in runlog_dir.iterdir() if path.is_dir()} if runlog_dir.exists() else set()
        command = [
            python,
            "-B",
            "-m",
            "inspect_assist",
            "--video",
            str(video),
            "--kb",
            str(kb),
            "--config",
            str(config),
            "--yolo-weights",
            str(weights),
            "--device",
            str(device),
            "--memory-preset",
            "session-only",
        ]
        if feedback_mode == "simulated":
            command.extend(
                [
                    "--simulate-feedback-gt",
                    str(gt),
                    "--simulate-feedback-every-sec",
                    str(feedback_every_sec),
                ]
            )
        returncode = run_command(command, cwd=repo_root, dry_run=dry_run)
        run_dir = newest_run(runlog_dir, before) if returncode == 0 and not dry_run else None
    result: Dict[str, Any] = {
        "video": str(video),
        "gt": str(gt),
        "feedback_mode": feedback_mode,
        "returncode": returncode,
        "run_dir": str(run_dir) if run_dir else "",
    }
    if returncode != 0 or run_dir is None or dry_run:
        return result

    trace_path = output_dir / "online_claim_evidence_traces" / f"{video.stem}.jsonl"
    trace_report = output_dir / "online_claim_evidence_traces" / f"{video.stem}_report.json"
    mine_path = output_dir / "evidence_trace_events" / f"{video.stem}_events.jsonl"
    mine_report = output_dir / "evidence_trace_events" / f"{video.stem}_report.json"
    rel_dir = output_dir / "relative_action_events" / video.stem
    rel_report = output_dir / "relative_action_events" / f"{video.stem}_report.json"

    commands = [
        [
            python,
            "scripts/build_online_claim_evidence_trace_from_run.py",
            "--run-dir",
            str(run_dir),
            "--output",
            str(trace_path),
            "--report-json",
            str(trace_report),
        ],
        [
            python,
            "scripts/mine_active_view_transitions_from_evidence_trace.py",
            "--frame-samples",
            str(trace_path),
            "--output",
            str(mine_path),
            "--report-json",
            str(mine_report),
            "--max-pairs-per-group",
            "12",
        ],
        [
            python,
            "scripts/mine_inspect_active_view_transitions.py",
            "--assistant-events",
            str(mine_path),
            "--output-dir",
            str(rel_dir),
            "--report-json",
            str(rel_report),
        ],
    ]
    for command in commands:
        code = run_command(command, cwd=repo_root, dry_run=dry_run)
        if code != 0:
            result["postprocess_returncode"] = code
            break
    result.update(
        {
            "iterations": count_jsonl(run_dir / "iterations.jsonl"),
            "feedback": count_jsonl(run_dir / "feedback.jsonl"),
            "online_trace": count_jsonl(trace_path),
            "evidence_trace_events": count_jsonl(mine_path),
            "relative_action_events": sum(count_jsonl(path) for path in rel_dir.glob("*.jsonl")),
            "trace_report": str(trace_report),
            "mine_report": str(mine_report),
            "relative_report": str(rel_report),
        }
    )
    for prefix, report_path in (("trace", trace_report), ("mine", mine_report), ("relative", rel_report)):
        report = load_json(report_path)
        for key in ("by_world_outcome", "by_claim_id", "by_transfer_type", "by_action"):
            if key in report:
                result[f"{prefix}_{key}"] = report[key]
        for key in ("samples", "mined_events", "events", "transferable"):
            if key in report:
                result[f"{prefix}_{key}"] = report[key]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--timeline-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--kb", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", default="0")
    parser.add_argument("--detector-conf", type=float, default=0.10)
    parser.add_argument("--identity-commit-conf", type=float, default=0.50)
    parser.add_argument("--identity-commit-track-margin", type=float, default=0.12)
    parser.add_argument("--role-bridge", action="store_true", help="Causally bridge short role-level detector dropouts with local keypoint tracks.")
    parser.add_argument("--role-bridge-max-gap", type=int, default=12)
    parser.add_argument(
        "--detector-head",
        choices=("default", "end2end", "one2many"),
        default="default",
    )
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--feedback-every-sec", type=float, default=0.35)
    parser.add_argument(
        "--feedback-mode",
        choices=("none", "simulated"),
        default="simulated",
        help=(
            "Use 'none' for leakage-free automatic evaluation. Use "
            "'simulated' only when producing assistant-use trace supervision."
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--resolution-supervision-only",
        action="store_true",
        help=(
            "Replay only sessions whose human timeline contains a supported or "
            "contradicted outcome. This prevents unresolved-only sessions from "
            "becoming self-labeled reveal supervision."
        ),
    )
    parser.add_argument("--only", action="append", default=[], help="Run only videos whose name contains this substring; repeatable.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only runs with non-empty iterations and a completed event report.",
    )
    parser.add_argument("--full-moge", action="store_true", help="Use detector + MoGe + scene graph + interaction evidence instead of the fast gradient geometry config.")
    parser.add_argument(
        "--evidence-scorer-path",
        type=Path,
        help="Frozen claim scorer used by an explicit end-to-end verifier replay.",
    )
    parser.add_argument("--counterfactual-margin", type=float, default=0.0)
    parser.add_argument("--disable-admissibility-gate", action="store_true")
    parser.add_argument("--disable-memory-gate", action="store_true")
    parser.add_argument("--disable-specialized-counterfactual", action="store_true")
    parser.add_argument("--disable-prerequisite-bootstrap", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    gt_dir = output_root / "gt_per_video"
    runlog_dir = output_root / "runs_modular"
    config_path = output_root / "active_view_online_sim_config.yaml"
    if args.full_moge:
        detector_end2end = None
        if args.detector_head == "end2end":
            detector_end2end = True
        elif args.detector_head == "one2many":
            detector_end2end = False
        write_full_moge_config(
            config_path,
            runlog_dir=runlog_dir,
            stride=int(args.stride),
            detector_conf=float(args.detector_conf),
            identity_commit_conf=float(args.identity_commit_conf),
            identity_commit_track_margin=float(args.identity_commit_track_margin),
            role_bridge_enabled=bool(args.role_bridge),
            role_bridge_max_gap=int(args.role_bridge_max_gap),
            detector_end2end=detector_end2end,
            evidence_scorer_path=str(args.evidence_scorer_path or ""),
            counterfactual_margin=float(args.counterfactual_margin),
            admissibility_gate_enabled=not bool(args.disable_admissibility_gate),
            memory_gate_enabled=not bool(args.disable_memory_gate),
            specialized_counterfactual_enabled=not bool(
                args.disable_specialized_counterfactual
            ),
            prerequisite_bootstrap_enabled=not bool(
                args.disable_prerequisite_bootstrap
            ),
        )
    else:
        write_fast_config(config_path, runlog_dir=runlog_dir, stride=int(args.stride))
    gt_paths = split_gt_by_video(args.timeline_csv, gt_dir)
    videos = sorted(path for path in args.video_dir.glob("*.mp4") if path.name in gt_paths)
    if args.resolution_supervision_only:
        resolved_names = resolution_supervision_video_names(args.timeline_csv)
        videos = [path for path in videos if path.name in resolved_names]
    if args.only:
        needles = [normalize_key(item) for item in args.only]
        videos = [path for path in videos if any(needle in normalize_key(path.name) for needle in needles)]
    if args.limit and args.limit > 0:
        videos = videos[: int(args.limit)]
    summary_rows: List[Dict[str, Any]] = []
    for index, video in enumerate(videos, start=1):
        print(f"\n=== [{index}/{len(videos)}] {video.name} ===")
        existing = resumable_run(runlog_dir, output_root, video) if args.resume else None
        if existing is not None:
            print(f"Resuming completed run: {existing}")
        summary_rows.append(
            run_pipeline_for_video(
                python=args.python,
                repo_root=args.repo_root,
                video=video,
                gt=gt_paths[video.name],
                kb=args.kb,
                config=config_path,
                weights=args.yolo_weights,
                runlog_dir=runlog_dir,
                output_dir=output_root,
                device=str(args.device),
                feedback_every_sec=float(args.feedback_every_sec),
                feedback_mode=str(args.feedback_mode),
                dry_run=bool(args.dry_run),
                resume_run=existing,
            )
        )
    summary_path = output_root / "online_simulation_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    csv_path = output_root / "online_simulation_summary.csv"
    flat_fields = [
        "video",
        "returncode",
        "run_dir",
        "iterations",
        "feedback",
        "online_trace",
        "evidence_trace_events",
        "relative_action_events",
        "relative_transferable",
    ]
    write_csv(csv_path, summary_rows, flat_fields)
    aggregate: Dict[str, Any] = {
        "videos": len(summary_rows),
        "ok": sum(1 for row in summary_rows if int(row.get("returncode", -1)) == 0),
        "iterations": sum(int(row.get("iterations", 0) or 0) for row in summary_rows),
        "feedback": sum(int(row.get("feedback", 0) or 0) for row in summary_rows),
        "online_trace": sum(int(row.get("online_trace", 0) or 0) for row in summary_rows),
        "evidence_trace_events": sum(int(row.get("evidence_trace_events", 0) or 0) for row in summary_rows),
        "relative_action_events": sum(int(row.get("relative_action_events", 0) or 0) for row in summary_rows),
        "relative_transferable": sum(int(row.get("relative_transferable", 0) or 0) for row in summary_rows),
        "summary_json": str(summary_path),
        "summary_csv": str(csv_path),
        "config": str(config_path),
    }
    aggregate_path = output_root / "online_simulation_aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
