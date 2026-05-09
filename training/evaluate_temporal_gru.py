"""Evaluate a trained temporal GRU checkpoint on an exported dataset."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .modeling import StreamingTemporalGRUModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a streaming temporal GRU checkpoint.")
    parser.add_argument("--dataset", required=True, help="Path to the exported .pt dataset.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained checkpoint.")
    parser.add_argument("--batch-size", type=int, default=128, help="Evaluation batch size.")
    parser.add_argument("--device", default="cpu", help="Evaluation device.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = torch.load(args.dataset, map_location="cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    x = dataset["x"].float()
    y = dataset["y"].long()
    weights = dataset["weights"].float()
    steps = list(dataset["steps"])
    component_names = list(dataset.get("component_names", []))
    relation_types = list(dataset.get("relation_types", []))
    flag_names = list(dataset.get("flag_names", []))
    aggregator_config = checkpoint.get("aggregator_config") or {}

    model = StreamingTemporalGRUModel(
        input_size=int(x.shape[-1]),
        hidden_size=int(checkpoint.get("hidden_size", len(steps))),
        num_steps=len(steps),
        num_components=len(component_names),
        num_relations=len(relation_types),
        num_scalars=8,
        use_learned_token_aggregation=bool(aggregator_config),
        token_hidden_size=int(aggregator_config.get("hidden_size", 24) or 24),
        token_output_size=int(aggregator_config.get("output_size", 48) or 48),
        num_flag_targets=len(flag_names),
    )
    model.gru.load_state_dict(checkpoint["gru_state_dict"])
    model.head.load_state_dict(checkpoint["head_state_dict"])
    if checkpoint.get("next_bootstrap_head_state_dict") is not None:
        model.next_bootstrap_head.load_state_dict(checkpoint["next_bootstrap_head_state_dict"])
    if checkpoint.get("next_relation_head_state_dict") is not None:
        model.next_relation_head.load_state_dict(checkpoint["next_relation_head_state_dict"])
    if checkpoint.get("next_flag_head_state_dict") is not None:
        model.next_flag_head.load_state_dict(checkpoint["next_flag_head_state_dict"])
    if model.token_aggregator is not None and checkpoint.get("aggregator_state_dict") is not None:
        model.token_aggregator.load_state_dict(checkpoint["aggregator_state_dict"])
    device = _resolve_device(args.device)
    model = model.to(device)
    model.eval()

    loader = DataLoader(TensorDataset(x, y, weights), batch_size=max(1, int(args.batch_size)), shuffle=False)
    total_loss = 0.0
    total_weight = 0.0
    total_correct = 0
    confusion = torch.zeros((len(steps), len(steps)), dtype=torch.int64)
    with torch.no_grad():
        for batch_x, batch_y, batch_w in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            logits = model(batch_x)
            losses = F.cross_entropy(logits, batch_y, reduction="none")
            predictions = torch.argmax(logits, dim=-1)
            total_loss += float((losses * batch_w).sum().item())
            total_weight += float(batch_w.sum().item())
            total_correct += int((predictions == batch_y).sum().item())
            for truth, pred in zip(batch_y.detach().cpu(), predictions.detach().cpu()):
                confusion[int(truth), int(pred)] += 1

    result = {
        "num_samples": int(x.shape[0]),
        "weighted_loss": float(total_loss / max(1e-6, total_weight)),
        "accuracy": float(total_correct) / float(max(1, int(x.shape[0]))),
        "steps": steps,
        "confusion": confusion.tolist(),
        "checkpoint_metadata": checkpoint.get("metadata", {}),
    }
    print(json.dumps(result, indent=2))


def _resolve_device(device: str) -> torch.device:
    normalized = str(device or "cpu").strip().lower()
    if normalized in {"", "cpu"}:
        return torch.device("cpu")
    if normalized.startswith("cuda") and torch.cuda.is_available():
        return torch.device(normalized)
    if normalized.isdigit() and torch.cuda.is_available():
        return torch.device(f"cuda:{normalized}")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
