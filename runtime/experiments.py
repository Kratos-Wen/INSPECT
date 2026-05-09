"""Batch experiment helpers for ablation studies."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from ..config import AppConfig, apply_ablation_preset
from .pipeline import build_default_pipeline

CORL_STANDARD_PRESETS = ["gru", "gru-aux", "gru-agg", "gru-agg-offline-gate", "full-online-adapt"]


def _slugify(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value.lower()).strip("-") or "run"


def _summarize_run(run_dir: Path) -> dict[str, object]:
    iterations_path = run_dir / "iterations.jsonl"
    events_path = run_dir / "events.jsonl"
    feedback_path = run_dir / "feedback.jsonl"
    entries = []
    events = []
    feedback_entries = []
    if iterations_path.exists():
        entries = [json.loads(line) for line in iterations_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if events_path.exists():
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if feedback_path.exists():
        feedback_entries = [
            json.loads(line) for line in feedback_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    if not entries:
        return {
            "frames": 0,
            "final_step": "",
            "mean_fused_conf": 0.0,
            "stable_ratio": 0.0,
            "memory_active_ratio": 0.0,
            "review_count": 0,
            "review_intervention_count": 0,
            "feedback_count": 0,
        }
    frames = len(entries)
    mean_fused_conf = sum(float(entry.get("fused_conf", 0.0)) for entry in entries) / float(frames)
    stable_ratio = sum(1 for entry in entries if bool(entry.get("stable"))) / float(frames)
    memory_active_ratio = sum(1 for entry in entries if bool(entry.get("memory_active"))) / float(frames)
    final_step = str(entries[-1].get("fused_step", ""))
    review_events = [event for event in events if str(event.get("event_type", "")) == "review.decision"]
    review_intervention_count = sum(
        1
        for event in review_events
        if str(event.get("payload", {}).get("action", "")) in {"hold", "prefer_candidate", "request_human"}
    )
    return {
        "frames": frames,
        "final_step": final_step,
        "mean_fused_conf": mean_fused_conf,
        "stable_ratio": stable_ratio,
        "memory_active_ratio": memory_active_ratio,
        "review_count": len(review_events),
        "review_intervention_count": review_intervention_count,
        "feedback_count": len(feedback_entries),
    }


def run_ablation_suite(
    base_config: AppConfig,
    video_path: Path,
    kb_path: str,
    yolo_weights: str,
    device: str,
    interactive: bool = False,
    gallery_root: Optional[str] = None,
    presets: Optional[Iterable[str]] = None,
    embed_modes: Optional[Iterable[str]] = None,
) -> Path:
    """Run a batch of preset/embed combinations and write a CSV + markdown report."""

    requested_presets = list(presets or ["custom", "memory-off", "session-only", "long-term-only", "no-auto-capture"])
    suite_presets: List[str] = []
    for preset in requested_presets:
        normalized = str(preset).strip().lower()
        if normalized == "corl-standard":
            suite_presets.extend(CORL_STANDARD_PRESETS)
        elif normalized:
            suite_presets.append(str(preset).strip())
    suite_embed_modes = list(embed_modes or [base_config.experts.gallery_embed])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_root = Path(base_config.runlog.save_dir) / f"suite_{video_path.stem}_{timestamp}"
    runs_root = suite_root / "runs"
    states_root = suite_root / "states"
    runs_root.mkdir(parents=True, exist_ok=True)
    states_root.mkdir(parents=True, exist_ok=True)

    results: List[dict[str, object]] = []
    for preset in suite_presets:
        for embed_mode in suite_embed_modes:
            config = deepcopy(base_config)
            config.runlog.save_dir = str(runs_root)
            config.experts.gallery_embed = str(embed_mode)
            if gallery_root:
                config.experts.gallery_root = gallery_root
            config = apply_ablation_preset(config, None if preset == "custom" else preset)
            state_path = states_root / f"{_slugify(preset)}_{_slugify(embed_mode)}_fusion_state.json"
            try:
                pipeline = build_default_pipeline(
                    config=config,
                    kb_path=kb_path,
                    yolo_weights=yolo_weights,
                    device=device,
                    state_path=state_path,
                    interactive=interactive and preset == "custom",
                )
                run_dir = pipeline.run(video_path)
                summary = _summarize_run(run_dir)
                results.append(
                    {
                        "preset": preset,
                        "embed_mode": embed_mode,
                        "status": "ok",
                        "run_dir": str(run_dir),
                        **summary,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "preset": preset,
                        "embed_mode": embed_mode,
                        "status": "error",
                        "run_dir": "",
                        "frames": 0,
                        "final_step": "",
                        "mean_fused_conf": 0.0,
                        "stable_ratio": 0.0,
                        "memory_active_ratio": 0.0,
                        "review_count": 0,
                        "review_intervention_count": 0,
                        "feedback_count": 0,
                        "error": str(exc),
                    }
                )

    csv_path = suite_root / "ablation_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "preset",
                "embed_mode",
                "status",
                "frames",
                "final_step",
                "mean_fused_conf",
                "stable_ratio",
                "memory_active_ratio",
                "review_count",
                "review_intervention_count",
                "feedback_count",
                "run_dir",
                "error",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    markdown_path = suite_root / "ablation_results.md"
    lines = [
        "| preset | embed | status | frames | final step | mean fused conf | stable ratio | memory active ratio | reviews | interventions | feedback |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            "| {preset} | {embed_mode} | {status} | {frames} | {final_step} | {mean_fused_conf:.4f} | {stable_ratio:.4f} | {memory_active_ratio:.4f} | {review_count} | {review_intervention_count} | {feedback_count} |".format(
                preset=row.get("preset", ""),
                embed_mode=row.get("embed_mode", ""),
                status=row.get("status", ""),
                frames=int(row.get("frames", 0) or 0),
                final_step=row.get("final_step", ""),
                mean_fused_conf=float(row.get("mean_fused_conf", 0.0) or 0.0),
                stable_ratio=float(row.get("stable_ratio", 0.0) or 0.0),
                memory_active_ratio=float(row.get("memory_active_ratio", 0.0) or 0.0),
                review_count=int(row.get("review_count", 0) or 0),
                review_intervention_count=int(row.get("review_intervention_count", 0) or 0),
                feedback_count=int(row.get("feedback_count", 0) or 0),
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return suite_root
