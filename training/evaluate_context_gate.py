"""Evaluate temporal checkpoints with or without an offline context gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..components.online_fusion import AdaptiveExpertFusion
from .context_gate_utils import ContextGateModel, build_gate_inputs, build_temporal_model, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a temporal checkpoint with an optional context gate.")
    parser.add_argument("--dataset", required=True, help="Path to the exported .pt dataset.")
    parser.add_argument("--temporal-checkpoint", required=True, help="Path to the temporal checkpoint.")
    parser.add_argument(
        "--context-gate-checkpoint",
        default="",
        help="Optional path to the context gate JSON checkpoint. If omitted, evaluation is temporal-only.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Evaluation batch size.")
    parser.add_argument("--device", default="cpu", help="Evaluation device.")
    return parser


def evaluate_context_gate_bundle(
    *,
    dataset_path: str | Path,
    temporal_checkpoint_path: str | Path,
    context_gate_checkpoint_path: str | Path | None = None,
    batch_size: int = 128,
    device: str = "cpu",
) -> Dict[str, object]:
    dataset = torch.load(dataset_path, map_location="cpu")
    checkpoint = torch.load(temporal_checkpoint_path, map_location="cpu")
    gate_payload = None
    gate_path = str(context_gate_checkpoint_path or "").strip()
    if gate_path:
        gate_payload = json.loads(Path(gate_path).read_text(encoding="utf-8"))

    x = dataset["x"].float()
    y = dataset["y"].long()
    weights = dataset["weights"].float()
    steps: List[str] = list(dataset["steps"])
    component_names: List[str] = list(dataset.get("component_names", []))
    relation_types: List[str] = list(dataset.get("relation_types", []))
    resolved_device = resolve_device(device)

    temporal_model = build_temporal_model(dataset=dataset, checkpoint=checkpoint, device=resolved_device, trainable=False)
    gate_model = _load_gate_model(payload=gate_payload, device=resolved_device) if gate_payload is not None else None
    if gate_model is not None:
        gate_model.eval()

    loader = DataLoader(TensorDataset(x, y, weights), batch_size=max(1, int(batch_size)), shuffle=False)
    temporal_confusion = torch.zeros((len(steps), len(steps)), dtype=torch.int64)
    fused_confusion = torch.zeros((len(steps), len(steps)), dtype=torch.int64)
    total_weight = 0.0
    total_temporal_loss = 0.0
    total_fused_loss = 0.0
    total_temporal_correct = 0
    total_fused_correct = 0
    gate_sums = torch.zeros((4,), dtype=torch.float32)
    gate_batches = 0

    with torch.no_grad():
        for batch_x, batch_y, batch_w in loader:
            batch_x = batch_x.to(resolved_device)
            batch_y = batch_y.to(resolved_device)
            batch_w = batch_w.to(resolved_device)

            outputs, context_features, expert_scores = build_gate_inputs(
                temporal_model=temporal_model,
                batch_x=batch_x,
                relation_types=relation_types,
                component_names=component_names,
            )
            temporal_logits = outputs["step_logits"]
            if gate_model is None:
                fused_scores = temporal_logits
                gates = None
            else:
                fused_scores, gates = gate_model(context_features, expert_scores)

            temporal_terms = F.cross_entropy(temporal_logits, batch_y, reduction="none")
            fused_terms = F.cross_entropy(fused_scores, batch_y, reduction="none")
            temporal_predictions = torch.argmax(temporal_logits, dim=-1)
            fused_predictions = torch.argmax(fused_scores, dim=-1)

            total_temporal_loss += float((temporal_terms * batch_w).sum().item())
            total_fused_loss += float((fused_terms * batch_w).sum().item())
            total_weight += float(batch_w.sum().item())
            total_temporal_correct += int((temporal_predictions == batch_y).sum().item())
            total_fused_correct += int((fused_predictions == batch_y).sum().item())
            gate_batches += int(batch_y.numel())
            if gates is not None:
                gate_sums += gates.detach().cpu().sum(dim=0)
            else:
                gate_sums += torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32) * float(batch_y.numel())

            for truth, pred in zip(batch_y.detach().cpu(), temporal_predictions.detach().cpu()):
                temporal_confusion[int(truth), int(pred)] += 1
            for truth, pred in zip(batch_y.detach().cpu(), fused_predictions.detach().cpu()):
                fused_confusion[int(truth), int(pred)] += 1

    return {
        "num_samples": int(x.shape[0]),
        "temporal_weighted_loss": float(total_temporal_loss / max(1e-6, total_weight)),
        "fused_weighted_loss": float(total_fused_loss / max(1e-6, total_weight)),
        "temporal_accuracy": float(total_temporal_correct) / float(max(1, int(x.shape[0]))),
        "fused_accuracy": float(total_fused_correct) / float(max(1, int(x.shape[0]))),
        "steps": steps,
        "temporal_confusion": temporal_confusion.tolist(),
        "fused_confusion": fused_confusion.tolist(),
        "mean_gates": {
            name: float(gate_sums[index].item() / max(1, gate_batches))
            for index, name in enumerate(["state", "temporal", "retrieval", "memory"])
        },
        "context_gate_enabled": bool(gate_model is not None),
        "temporal_checkpoint": str(temporal_checkpoint_path),
        "context_gate_checkpoint": str(context_gate_checkpoint_path or ""),
        "temporal_checkpoint_metadata": checkpoint.get("metadata", {}),
        "context_gate_metadata": gate_payload.get("metadata", {}) if gate_payload is not None else {},
    }


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate_context_gate_bundle(
        dataset_path=args.dataset,
        temporal_checkpoint_path=args.temporal_checkpoint,
        context_gate_checkpoint_path=args.context_gate_checkpoint or None,
        batch_size=int(args.batch_size),
        device=args.device,
    )
    print(json.dumps(result, indent=2))


def _load_gate_model(*, payload: dict[str, object], device: torch.device) -> ContextGateModel:
    feature_names = list(payload.get("feature_names") or [])
    if feature_names and feature_names != list(AdaptiveExpertFusion.CONTEXT_FEATURE_NAMES):
        raise ValueError("Context gate feature names do not match the runtime fusion interface.")
    base_gates = payload.get("base_gates") or [0.5, 0.45, 0.5, 0.2]
    gate_model = ContextGateModel(
        num_features=len(AdaptiveExpertFusion.CONTEXT_FEATURE_NAMES),
        num_experts=4,
        context_gate_scale=float(payload.get("metadata", {}).get("context_gate_scale", 0.35)),
        init_base_gates=[float(value) for value in base_gates],
    ).to(device)
    with torch.no_grad():
        normalized = torch.tensor(base_gates, dtype=torch.float32, device=device)
        normalized = torch.clamp(normalized, min=1e-6)
        normalized = normalized / torch.clamp(normalized.sum(), min=1e-6)
        gate_model.base_gate_logits.copy_(torch.log(normalized))
        gate_model.linear.weight.copy_(torch.tensor(payload["context_gate_weights"], dtype=torch.float32, device=device))
        gate_model.linear.bias.copy_(torch.tensor(payload["context_gate_bias"], dtype=torch.float32, device=device))
    return gate_model


if __name__ == "__main__":
    main()
