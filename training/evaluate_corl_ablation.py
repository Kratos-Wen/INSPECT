"""Batch evaluation for the standard CoRL temporal/gating ablations."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from ..config import apply_ablation_preset, load_config
from .evaluate_context_gate import evaluate_context_gate_bundle

STANDARD_PRESETS = ["gru", "gru-aux", "gru-agg", "gru-agg-offline-gate", "full-online-adapt"]
PAPER_LABELS = {
    "gru": "GRU",
    "gru-aux": "GRU + Aux",
    "gru-agg": "GRU + Agg",
    "gru-agg-offline-gate": "GRU + Agg + Offline Gate",
    "full-online-adapt": "Full Online Adapt",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the standard CoRL ablation presets and write table files.")
    parser.add_argument("--dataset", required=True, help="Path to the exported .pt dataset.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the YAML config. Defaults to the package example config.",
    )
    parser.add_argument(
        "--presets",
        default=",".join(STANDARD_PRESETS),
        help="Comma-separated preset list. Defaults to the standard CoRL preset set.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to runs_modular/corl_eval_<timestamp>.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Evaluation batch size.")
    parser.add_argument("--device", default="cpu", help="Evaluation device.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    presets = [item.strip() for item in str(args.presets).split(",") if item.strip()]
    output_dir = _resolve_output_dir(raw_value=args.output_dir, save_dir=config.runlog.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    for preset in presets:
        try:
            applied = apply_ablation_preset(deepcopy(config), preset)
            result = evaluate_context_gate_bundle(
                dataset_path=args.dataset,
                temporal_checkpoint_path=applied.temporal_step.checkpoint_path,
                context_gate_checkpoint_path=(
                    applied.online_fusion.context_gate_checkpoint if applied.online_fusion.context_gate_enabled else None
                ),
                batch_size=int(args.batch_size),
                device=args.device,
            )
            results.append(
                {
                    "preset": preset,
                    "status": "ok",
                    "temporal_checkpoint": str(applied.temporal_step.checkpoint_path),
                    "context_gate_checkpoint": str(applied.online_fusion.context_gate_checkpoint),
                    "learned_token_aggregation": bool(applied.temporal_step.learned_token_aggregation),
                    "context_gate_enabled": bool(applied.online_fusion.context_gate_enabled),
                    "memory_enabled": bool(applied.memory.enabled),
                    "review_enabled": bool(applied.review.enabled),
                    "temporal_accuracy": float(result["temporal_accuracy"]),
                    "fused_accuracy": float(result["fused_accuracy"]),
                    "accuracy_gain": float(result["fused_accuracy"]) - float(result["temporal_accuracy"]),
                    "temporal_weighted_loss": float(result["temporal_weighted_loss"]),
                    "fused_weighted_loss": float(result["fused_weighted_loss"]),
                    "loss_gain": float(result["temporal_weighted_loss"]) - float(result["fused_weighted_loss"]),
                    "gate_state": float(result["mean_gates"]["state"]),
                    "gate_temporal": float(result["mean_gates"]["temporal"]),
                    "gate_retrieval": float(result["mean_gates"]["retrieval"]),
                    "gate_memory": float(result["mean_gates"]["memory"]),
                    "details": result,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "preset": preset,
                    "status": "error",
                    "temporal_checkpoint": "",
                    "context_gate_checkpoint": "",
                    "learned_token_aggregation": False,
                    "context_gate_enabled": False,
                    "memory_enabled": False,
                    "review_enabled": False,
                    "temporal_accuracy": 0.0,
                    "fused_accuracy": 0.0,
                    "accuracy_gain": 0.0,
                    "temporal_weighted_loss": 0.0,
                    "fused_weighted_loss": 0.0,
                    "loss_gain": 0.0,
                    "gate_state": 0.0,
                    "gate_temporal": 0.0,
                    "gate_retrieval": 0.0,
                    "gate_memory": 0.0,
                    "error": str(exc),
                }
            )

    _write_outputs(output_dir=output_dir, results=results)
    print(json.dumps({"output_dir": str(output_dir), "num_rows": len(results)}, indent=2))


def _resolve_output_dir(*, raw_value: str, save_dir: str) -> Path:
    value = str(raw_value or "").strip()
    if value:
        return Path(value)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(save_dir) / f"corl_eval_{timestamp}"


def _write_outputs(*, output_dir: Path, results: List[Dict[str, object]]) -> None:
    csv_path = output_dir / "corl_ablation_results.csv"
    fieldnames = [
        "preset",
        "status",
        "temporal_accuracy",
        "fused_accuracy",
        "accuracy_gain",
        "temporal_weighted_loss",
        "fused_weighted_loss",
        "loss_gain",
        "gate_state",
        "gate_temporal",
        "gate_retrieval",
        "gate_memory",
        "learned_token_aggregation",
        "context_gate_enabled",
        "memory_enabled",
        "review_enabled",
        "temporal_checkpoint",
        "context_gate_checkpoint",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    json_path = output_dir / "corl_ablation_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    main_csv_path = output_dir / "corl_main_table.csv"
    with main_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "temporal_accuracy", "fused_accuracy", "accuracy_gain", "fused_weighted_loss"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "method": PAPER_LABELS.get(str(row.get("preset", "")), str(row.get("preset", ""))),
                    "temporal_accuracy": row.get("temporal_accuracy", 0.0),
                    "fused_accuracy": row.get("fused_accuracy", 0.0),
                    "accuracy_gain": row.get("accuracy_gain", 0.0),
                    "fused_weighted_loss": row.get("fused_weighted_loss", 0.0),
                }
            )

    main_markdown_path = output_dir / "corl_main_table.md"
    main_lines = [
        "| method | temp acc | fused acc | gain | fused loss |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        main_lines.append(
            "| {label} | {temporal_accuracy:.4f} | {fused_accuracy:.4f} | {accuracy_gain:.4f} | {fused_weighted_loss:.4f} |".format(
                label=PAPER_LABELS.get(str(row.get("preset", "")), str(row.get("preset", ""))),
                temporal_accuracy=float(row.get("temporal_accuracy", 0.0) or 0.0),
                fused_accuracy=float(row.get("fused_accuracy", 0.0) or 0.0),
                accuracy_gain=float(row.get("accuracy_gain", 0.0) or 0.0),
                fused_weighted_loss=float(row.get("fused_weighted_loss", 0.0) or 0.0),
            )
        )
    main_markdown_path.write_text("\n".join(main_lines) + "\n", encoding="utf-8")

    ablation_csv_path = output_dir / "corl_ablation_table.csv"
    with ablation_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "preset",
                "status",
                "temporal_accuracy",
                "fused_accuracy",
                "accuracy_gain",
                "temporal_weighted_loss",
                "fused_weighted_loss",
                "gate_state",
                "gate_temporal",
                "gate_retrieval",
                "gate_memory",
                "learned_token_aggregation",
                "context_gate_enabled",
                "memory_enabled",
                "review_enabled",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "preset": row.get("preset", ""),
                    "status": row.get("status", ""),
                    "temporal_accuracy": row.get("temporal_accuracy", 0.0),
                    "fused_accuracy": row.get("fused_accuracy", 0.0),
                    "accuracy_gain": row.get("accuracy_gain", 0.0),
                    "temporal_weighted_loss": row.get("temporal_weighted_loss", 0.0),
                    "fused_weighted_loss": row.get("fused_weighted_loss", 0.0),
                    "gate_state": row.get("gate_state", 0.0),
                    "gate_temporal": row.get("gate_temporal", 0.0),
                    "gate_retrieval": row.get("gate_retrieval", 0.0),
                    "gate_memory": row.get("gate_memory", 0.0),
                    "learned_token_aggregation": row.get("learned_token_aggregation", False),
                    "context_gate_enabled": row.get("context_gate_enabled", False),
                    "memory_enabled": row.get("memory_enabled", False),
                    "review_enabled": row.get("review_enabled", False),
                }
            )

    ablation_markdown_path = output_dir / "corl_ablation_table.md"
    ablation_lines = [
        "| preset | status | temp acc | fused acc | gain | temp loss | fused loss | g_state | g_temp | g_ret | g_mem | agg | gate | memory | review |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in results:
        ablation_lines.append(
            "| {preset} | {status} | {temporal_accuracy:.4f} | {fused_accuracy:.4f} | {accuracy_gain:.4f} | {temporal_weighted_loss:.4f} | {fused_weighted_loss:.4f} | {gate_state:.4f} | {gate_temporal:.4f} | {gate_retrieval:.4f} | {gate_memory:.4f} | {learned_token_aggregation} | {context_gate_enabled} | {memory_enabled} | {review_enabled} |".format(
                preset=row.get("preset", ""),
                status=row.get("status", ""),
                temporal_accuracy=float(row.get("temporal_accuracy", 0.0) or 0.0),
                fused_accuracy=float(row.get("fused_accuracy", 0.0) or 0.0),
                accuracy_gain=float(row.get("accuracy_gain", 0.0) or 0.0),
                temporal_weighted_loss=float(row.get("temporal_weighted_loss", 0.0) or 0.0),
                fused_weighted_loss=float(row.get("fused_weighted_loss", 0.0) or 0.0),
                gate_state=float(row.get("gate_state", 0.0) or 0.0),
                gate_temporal=float(row.get("gate_temporal", 0.0) or 0.0),
                gate_retrieval=float(row.get("gate_retrieval", 0.0) or 0.0),
                gate_memory=float(row.get("gate_memory", 0.0) or 0.0),
                learned_token_aggregation=str(bool(row.get("learned_token_aggregation", False))).lower(),
                context_gate_enabled=str(bool(row.get("context_gate_enabled", False))).lower(),
                memory_enabled=str(bool(row.get("memory_enabled", False))).lower(),
                review_enabled=str(bool(row.get("review_enabled", False))).lower(),
            )
        )
    ablation_markdown_path.write_text("\n".join(ablation_lines) + "\n", encoding="utf-8")

    legacy_markdown_path = output_dir / "corl_ablation_results.md"
    legacy_markdown_path.write_text("\n".join(ablation_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
