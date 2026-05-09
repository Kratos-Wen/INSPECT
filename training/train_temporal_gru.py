"""Train a small temporal GRU checkpoint from exported sequence samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .modeling import StreamingTemporalGRUModel, bootstrap_initialize, checkpoint_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a streaming temporal GRU from exported MICA samples.")
    parser.add_argument("--dataset", required=True, help="Path to the exported .pt dataset.")
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer weight decay.")
    parser.add_argument("--hidden-size", type=int, default=0, help="Override GRU hidden size; defaults to num_steps.")
    parser.add_argument("--device", default="cpu", help="Training device, for example cpu or cuda:0.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction when multiple runs are available.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--bootstrap-init", action="store_true", help="Initialize the GRU close to the EMA baseline before training.")
    parser.add_argument("--learned-token-aggregation", action="store_true", help="Enable learned token aggregation before the GRU.")
    parser.add_argument("--token-hidden-size", type=int, default=24, help="Hidden size for the learned token aggregator.")
    parser.add_argument("--token-output-size", type=int, default=48, help="Output size for the learned token aggregator.")
    parser.add_argument("--aux-bootstrap-weight", type=float, default=0.20, help="Weight for next-bootstrap regression loss.")
    parser.add_argument("--aux-relation-weight", type=float, default=0.10, help="Weight for next-relation prediction loss.")
    parser.add_argument("--aux-flag-weight", type=float, default=0.06, help="Weight for next-flag prediction loss.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    dataset = torch.load(args.dataset, map_location="cpu")
    x = dataset["x"].float()
    y = dataset["y"].long()
    weights = dataset["weights"].float()
    aux_mask = dataset.get("aux_mask", torch.ones_like(weights)).float()
    aux_next_bootstrap = dataset.get("aux_next_bootstrap", torch.zeros((x.shape[0], len(dataset["steps"]))))
    aux_next_relations = dataset.get("aux_next_relations", torch.zeros((x.shape[0], len(dataset.get("relation_types", [])))))
    aux_next_flags = dataset.get("aux_next_flags", torch.zeros((x.shape[0], len(dataset.get("flag_names", [])))))
    steps: List[str] = list(dataset["steps"])
    component_names: List[str] = list(dataset["component_names"])
    relation_types: List[str] = list(dataset["relation_types"])
    flag_names: List[str] = list(dataset.get("flag_names", []))
    window_size = int(dataset["window_size"])
    run_names: List[str] = list(dataset.get("run_names", []))

    train_indices, val_indices = _split_indices(run_names, num_items=int(x.shape[0]), val_fraction=float(args.val_fraction))
    train_loader = _build_loader(
        x,
        y,
        weights,
        aux_mask,
        aux_next_bootstrap.float(),
        aux_next_relations.float(),
        aux_next_flags.float(),
        train_indices,
        batch_size=int(args.batch_size),
        shuffle=True,
    )
    val_loader = (
        _build_loader(
            x,
            y,
            weights,
            aux_mask,
            aux_next_bootstrap.float(),
            aux_next_relations.float(),
            aux_next_flags.float(),
            val_indices,
            batch_size=int(args.batch_size),
            shuffle=False,
        )
        if val_indices
        else None
    )

    device = _resolve_device(args.device)
    hidden_size = int(args.hidden_size) if int(args.hidden_size) > 0 else len(steps)
    model = StreamingTemporalGRUModel(
        input_size=int(x.shape[-1]),
        hidden_size=hidden_size,
        num_steps=len(steps),
        num_components=len(component_names),
        num_relations=len(relation_types),
        num_scalars=8,
        use_learned_token_aggregation=bool(args.learned_token_aggregation),
        token_hidden_size=int(args.token_hidden_size),
        token_output_size=int(args.token_output_size),
        num_flag_targets=len(flag_names),
    ).to(device)
    if bool(args.bootstrap_init):
        bootstrap_initialize(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    history: List[Dict[str, float]] = []
    best_state = None
    best_val = None
    for epoch in range(1, int(args.epochs) + 1):
        train_loss, train_acc, train_aux = _run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            training=True,
            aux_bootstrap_weight=float(args.aux_bootstrap_weight),
            aux_relation_weight=float(args.aux_relation_weight),
            aux_flag_weight=float(args.aux_flag_weight),
        )
        epoch_metrics: Dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            **{f"train_{key}": float(value) for key, value in train_aux.items()},
        }
        if val_loader is not None:
            val_loss, val_acc, val_aux = _run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                training=False,
                aux_bootstrap_weight=float(args.aux_bootstrap_weight),
                aux_relation_weight=float(args.aux_relation_weight),
                aux_flag_weight=float(args.aux_flag_weight),
            )
            epoch_metrics["val_loss"] = float(val_loss)
            epoch_metrics["val_acc"] = float(val_acc)
            epoch_metrics.update({f"val_{key}": float(value) for key, value in val_aux.items()})
            if best_val is None or val_acc >= best_val:
                best_val = float(val_acc)
                best_state = {
                    "gru_state_dict": {key: value.detach().cpu() for key, value in model.gru.state_dict().items()},
                    "head_state_dict": {key: value.detach().cpu() for key, value in model.head.state_dict().items()},
                    "next_bootstrap_head_state_dict": {
                        key: value.detach().cpu() for key, value in model.next_bootstrap_head.state_dict().items()
                    },
                    "next_relation_head_state_dict": {
                        key: value.detach().cpu() for key, value in model.next_relation_head.state_dict().items()
                    },
                    "next_flag_head_state_dict": {
                        key: value.detach().cpu() for key, value in model.next_flag_head.state_dict().items()
                    },
                    "aggregator_state_dict": (
                        {key: value.detach().cpu() for key, value in model.token_aggregator.state_dict().items()}
                        if model.token_aggregator is not None
                        else None
                    ),
                }
        history.append(epoch_metrics)

    if best_state is not None:
        model.gru.load_state_dict(best_state["gru_state_dict"])
        model.head.load_state_dict(best_state["head_state_dict"])
        model.next_bootstrap_head.load_state_dict(best_state["next_bootstrap_head_state_dict"])
        model.next_relation_head.load_state_dict(best_state["next_relation_head_state_dict"])
        model.next_flag_head.load_state_dict(best_state["next_flag_head_state_dict"])
        if model.token_aggregator is not None and best_state.get("aggregator_state_dict") is not None:
            model.token_aggregator.load_state_dict(best_state["aggregator_state_dict"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(
        model,
        steps=steps,
        component_names=component_names,
        relation_types=relation_types,
        flag_names=flag_names,
        window_size=window_size,
        metadata={
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "bootstrap_init": bool(args.bootstrap_init),
            "learned_token_aggregation": bool(args.learned_token_aggregation),
            "token_hidden_size": int(args.token_hidden_size),
            "token_output_size": int(args.token_output_size),
            "aux_bootstrap_weight": float(args.aux_bootstrap_weight),
            "aux_relation_weight": float(args.aux_relation_weight),
            "aux_flag_weight": float(args.aux_flag_weight),
            "history": history,
            "train_samples": int(len(train_indices)),
            "val_samples": int(len(val_indices)),
            "label_source_counts": dict(dataset.get("label_source_counts", {})),
        },
    )
    torch.save(payload, output_path)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "train_samples": len(train_indices),
                "val_samples": len(val_indices),
                "history_tail": history[-3:],
            },
            indent=2,
        )
    )


def _build_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor,
    aux_mask: torch.Tensor,
    aux_next_bootstrap: torch.Tensor,
    aux_next_relations: torch.Tensor,
    aux_next_flags: torch.Tensor,
    indices: List[int],
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    subset = TensorDataset(
        x[indices],
        y[indices],
        weights[indices],
        aux_mask[indices],
        aux_next_bootstrap[indices],
        aux_next_relations[indices],
        aux_next_flags[indices],
    )
    return DataLoader(subset, batch_size=max(1, int(batch_size)), shuffle=bool(shuffle))


def _split_indices(run_names: List[str], num_items: int, val_fraction: float) -> Tuple[List[int], List[int]]:
    if num_items <= 1:
        return list(range(num_items)), []
    if not run_names or len(run_names) != num_items:
        cutoff = max(1, int(round(num_items * (1.0 - max(0.0, min(0.9, val_fraction))))))
        return list(range(cutoff)), list(range(cutoff, num_items))

    unique_runs = []
    seen = set()
    for name in run_names:
        if name not in seen:
            unique_runs.append(name)
            seen.add(name)
    if len(unique_runs) <= 1:
        cutoff = max(1, int(round(num_items * (1.0 - max(0.0, min(0.9, val_fraction))))))
        return list(range(cutoff)), list(range(cutoff, num_items))

    val_run_count = max(1, int(round(len(unique_runs) * max(0.0, min(0.5, val_fraction)))))
    val_runs = set(unique_runs[-val_run_count:])
    train_indices = [index for index, run_name in enumerate(run_names) if run_name not in val_runs]
    val_indices = [index for index, run_name in enumerate(run_names) if run_name in val_runs]
    if not train_indices:
        train_indices = val_indices[:-1]
        val_indices = val_indices[-1:]
    return train_indices, val_indices


def _run_epoch(
    model: StreamingTemporalGRUModel,
    loader: DataLoader,
    *,
    optimizer,
    device: torch.device,
    training: bool,
    aux_bootstrap_weight: float,
    aux_relation_weight: float,
    aux_flag_weight: float,
) -> Tuple[float, float, Dict[str, float]]:
    if training:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_correct = 0
    total_weight = 0.0
    total_items = 0
    total_bootstrap = 0.0
    total_relation = 0.0
    total_flag = 0.0
    total_aux_weight = 0.0
    for batch in loader:
        (
            batch_x,
            batch_y,
            batch_w,
            batch_aux_mask,
            batch_next_bootstrap,
            batch_next_relations,
            batch_next_flags,
        ) = batch
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_w = batch_w.to(device)
        batch_aux_mask = batch_aux_mask.to(device)
        batch_next_bootstrap = batch_next_bootstrap.to(device)
        batch_next_relations = batch_next_relations.to(device)
        batch_next_flags = batch_next_flags.to(device)
        if training:
            optimizer.zero_grad()
        outputs = model(batch_x, return_aux=True)
        step_logits = outputs["step_logits"]
        losses = F.cross_entropy(step_logits, batch_y, reduction="none")
        loss = (losses * batch_w).sum() / torch.clamp(batch_w.sum(), min=1e-6)

        aux_weight_denom = torch.clamp((batch_w * batch_aux_mask).sum(), min=1e-6)
        bootstrap_loss = torch.tensor(0.0, device=device)
        if batch_next_bootstrap.shape[-1] > 0 and aux_bootstrap_weight > 0.0:
            bootstrap_terms = F.mse_loss(outputs["next_bootstrap"], batch_next_bootstrap, reduction="none").mean(dim=-1)
            bootstrap_loss = (bootstrap_terms * batch_w * batch_aux_mask).sum() / aux_weight_denom
            loss = loss + float(aux_bootstrap_weight) * bootstrap_loss

        relation_loss = torch.tensor(0.0, device=device)
        if batch_next_relations.shape[-1] > 0 and aux_relation_weight > 0.0:
            relation_terms = F.binary_cross_entropy_with_logits(
                outputs["next_relations"],
                batch_next_relations,
                reduction="none",
            ).mean(dim=-1)
            relation_loss = (relation_terms * batch_w * batch_aux_mask).sum() / aux_weight_denom
            loss = loss + float(aux_relation_weight) * relation_loss

        flag_loss = torch.tensor(0.0, device=device)
        if batch_next_flags.shape[-1] > 0 and aux_flag_weight > 0.0:
            flag_terms = F.binary_cross_entropy_with_logits(
                outputs["next_flags"],
                batch_next_flags,
                reduction="none",
            ).mean(dim=-1)
            flag_loss = (flag_terms * batch_w * batch_aux_mask).sum() / aux_weight_denom
            loss = loss + float(aux_flag_weight) * flag_loss

        if training:
            loss.backward()
            optimizer.step()

        predictions = torch.argmax(step_logits, dim=-1)
        total_loss += float((losses * batch_w).sum().item())
        total_correct += int((predictions == batch_y).sum().item())
        total_weight += float(batch_w.sum().item())
        total_items += int(batch_y.numel())
        total_bootstrap += float((bootstrap_loss * aux_weight_denom).item())
        total_relation += float((relation_loss * aux_weight_denom).item())
        total_flag += float((flag_loss * aux_weight_denom).item())
        total_aux_weight += float(aux_weight_denom.item())

    mean_loss = total_loss / max(1e-6, total_weight)
    accuracy = float(total_correct) / float(max(1, total_items))
    aux_metrics = {
        "bootstrap_loss": float(total_bootstrap / max(1e-6, total_aux_weight)),
        "relation_loss": float(total_relation / max(1e-6, total_aux_weight)),
        "flag_loss": float(total_flag / max(1e-6, total_aux_weight)),
    }
    return mean_loss, accuracy, aux_metrics


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
