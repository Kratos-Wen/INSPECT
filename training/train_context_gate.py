"""Train a graph-conditioned context gate with optional joint temporal optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..components.online_fusion import AdaptiveExpertFusion
from .context_gate_utils import (
    ContextGateModel,
    build_gate_inputs,
    build_temporal_checkpoint_payload,
    build_temporal_model,
    resolve_device,
    split_indices,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an offline graph-conditioned context gate with freeze/joint/alternating modes."
    )
    parser.add_argument("--dataset", required=True, help="Path to the exported .pt dataset.")
    parser.add_argument("--temporal-checkpoint", required=True, help="Path to the trained temporal GRU checkpoint.")
    parser.add_argument("--output", required=True, help="Output JSON checkpoint path for the context gate.")
    parser.add_argument(
        "--mode",
        default="freeze",
        choices=["freeze", "joint", "alternating"],
        help="freeze: gate only; joint: optimize temporal+gate together; alternating: alternate gate and temporal phases.",
    )
    parser.add_argument(
        "--temporal-output",
        default="",
        help="Optional output checkpoint path for the updated temporal model. Auto-generated for non-freeze modes.",
    )
    parser.add_argument("--epochs", type=int, default=16, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the context gate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer weight decay for the context gate.")
    parser.add_argument("--temporal-lr", type=float, default=5e-4, help="Learning rate for the temporal model.")
    parser.add_argument(
        "--temporal-weight-decay",
        type=float,
        default=1e-4,
        help="Optimizer weight decay for the temporal model.",
    )
    parser.add_argument(
        "--temporal-ce-weight",
        type=float,
        default=0.35,
        help="Extra weight on the temporal step loss when the temporal model is trainable.",
    )
    parser.add_argument("--aux-bootstrap-weight", type=float, default=0.20, help="Weight for next-bootstrap regression loss.")
    parser.add_argument("--aux-relation-weight", type=float, default=0.10, help="Weight for next-relation prediction loss.")
    parser.add_argument("--aux-flag-weight", type=float, default=0.06, help="Weight for next-flag prediction loss.")
    parser.add_argument(
        "--alternating-phase-length",
        type=int,
        default=1,
        help="Number of epochs per phase when --mode alternating is used.",
    )
    parser.add_argument(
        "--warmup-gate-epochs",
        type=int,
        default=2,
        help="Initial epochs that train only the context gate before enabling temporal updates.",
    )
    parser.add_argument(
        "--warmup-temporal-epochs",
        type=int,
        default=1,
        help="Optional epochs that train only the temporal model after gate warmup.",
    )
    parser.add_argument(
        "--temporal-loss-ramp-epochs",
        type=int,
        default=3,
        help="Number of temporal-training epochs used to ramp temporal and auxiliary losses to full strength.",
    )
    parser.add_argument("--device", default="cpu", help="Training device, for example cpu or cuda:0.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction when multiple runs are available.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--context-gate-scale", type=float, default=0.35, help="Scaling factor used before softmax gating.")
    parser.add_argument(
        "--selection-metric",
        default="fused_delta",
        choices=["fused_delta", "fused_acc"],
        help="Best-checkpoint selection rule on the validation split.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(int(args.seed))

    dataset = torch.load(args.dataset, map_location="cpu")
    checkpoint = torch.load(args.temporal_checkpoint, map_location="cpu")
    x = dataset["x"].float()
    y = dataset["y"].long()
    weights = dataset["weights"].float()
    aux_mask = dataset.get("aux_mask", torch.ones_like(weights)).float()
    aux_next_bootstrap = dataset.get("aux_next_bootstrap", torch.zeros((x.shape[0], len(dataset["steps"])))).float()
    aux_next_relations = dataset.get(
        "aux_next_relations", torch.zeros((x.shape[0], len(dataset.get("relation_types", []))))
    ).float()
    aux_next_flags = dataset.get("aux_next_flags", torch.zeros((x.shape[0], len(dataset.get("flag_names", []))))).float()
    component_names: List[str] = list(dataset.get("component_names", []))
    relation_types: List[str] = list(dataset.get("relation_types", []))
    run_names: List[str] = list(dataset.get("run_names", []))

    train_indices, val_indices = split_indices(
        run_names,
        num_items=int(x.shape[0]),
        val_fraction=float(args.val_fraction),
    )
    train_loader = _build_loader(
        x=x,
        y=y,
        weights=weights,
        aux_mask=aux_mask,
        aux_next_bootstrap=aux_next_bootstrap,
        aux_next_relations=aux_next_relations,
        aux_next_flags=aux_next_flags,
        indices=train_indices,
        batch_size=int(args.batch_size),
        shuffle=True,
    )
    val_loader = (
        _build_loader(
            x=x,
            y=y,
            weights=weights,
            aux_mask=aux_mask,
            aux_next_bootstrap=aux_next_bootstrap,
            aux_next_relations=aux_next_relations,
            aux_next_flags=aux_next_flags,
            indices=val_indices,
            batch_size=int(args.batch_size),
            shuffle=False,
        )
        if val_indices
        else None
    )

    device = resolve_device(args.device)
    mode = str(args.mode).strip().lower()
    temporal_model = build_temporal_model(
        dataset=dataset,
        checkpoint=checkpoint,
        device=device,
        trainable=mode != "freeze",
    )
    gate_model = ContextGateModel(
        num_features=len(AdaptiveExpertFusion.CONTEXT_FEATURE_NAMES),
        num_experts=4,
        context_gate_scale=float(args.context_gate_scale),
        init_base_gates=[0.5, 0.45, 0.5, 0.2],
    ).to(device)
    optimizer = _build_optimizer(
        gate_model=gate_model,
        temporal_model=temporal_model,
        mode=mode,
        gate_lr=float(args.lr),
        gate_weight_decay=float(args.weight_decay),
        temporal_lr=float(args.temporal_lr),
        temporal_weight_decay=float(args.temporal_weight_decay),
    )

    history: List[Dict[str, float]] = []
    best_state: Optional[Dict[str, object]] = None
    best_score: Optional[Tuple[float, float, float]] = None
    best_epoch: int = 0

    for epoch in range(1, int(args.epochs) + 1):
        gate_trainable, temporal_trainable, phase_name, phase_temporal_epoch = _phase_for_epoch(
            epoch=epoch,
            mode=mode,
            alternating_phase_length=max(1, int(args.alternating_phase_length)),
            warmup_gate_epochs=max(0, int(args.warmup_gate_epochs)),
            warmup_temporal_epochs=max(0, int(args.warmup_temporal_epochs)),
        )
        temporal_scale = _temporal_ramp_scale(
            phase_temporal_epoch=phase_temporal_epoch,
            ramp_epochs=max(0, int(args.temporal_loss_ramp_epochs)),
        )
        train_metrics = _run_epoch(
            temporal_model=temporal_model,
            gate_model=gate_model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            training=True,
            gate_trainable=gate_trainable,
            temporal_trainable=temporal_trainable,
            relation_types=relation_types,
            component_names=component_names,
            temporal_ce_weight=float(args.temporal_ce_weight),
            aux_bootstrap_weight=float(args.aux_bootstrap_weight),
            aux_relation_weight=float(args.aux_relation_weight),
            aux_flag_weight=float(args.aux_flag_weight),
            temporal_loss_scale=float(temporal_scale),
        )
        epoch_metrics: Dict[str, float] = {
            "epoch": float(epoch),
            "phase": phase_name,
            "temporal_loss_scale": float(temporal_scale),
        }
        epoch_metrics.update({f"train_{key}": float(value) for key, value in train_metrics.items()})

        if val_loader is not None:
            val_metrics = _run_epoch(
                temporal_model=temporal_model,
                gate_model=gate_model,
                loader=val_loader,
                optimizer=None,
                device=device,
                training=False,
                gate_trainable=False,
                temporal_trainable=False,
                relation_types=relation_types,
                component_names=component_names,
                temporal_ce_weight=float(args.temporal_ce_weight),
                aux_bootstrap_weight=float(args.aux_bootstrap_weight),
                aux_relation_weight=float(args.aux_relation_weight),
                aux_flag_weight=float(args.aux_flag_weight),
                temporal_loss_scale=float(temporal_scale),
            )
            epoch_metrics.update({f"val_{key}": float(value) for key, value in val_metrics.items()})
            candidate_score = _selection_score(
                metrics=val_metrics,
                selection_metric=str(args.selection_metric).strip().lower(),
            )
            if best_score is None or candidate_score > best_score:
                best_score = candidate_score
                best_epoch = int(epoch)
                best_state = _capture_best_state(
                    gate_model=gate_model,
                    temporal_model=temporal_model,
                )
        history.append(epoch_metrics)

    if best_state is not None:
        _restore_best_state(
            gate_model=gate_model,
            temporal_model=temporal_model,
            best_state=best_state,
        )

    gate_output_path = Path(args.output)
    gate_output_path.parent.mkdir(parents=True, exist_ok=True)
    gate_payload = gate_model.export_payload(
        metadata={
            "mode": mode,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "gate_lr": float(args.lr),
            "gate_weight_decay": float(args.weight_decay),
            "temporal_lr": float(args.temporal_lr),
            "temporal_weight_decay": float(args.temporal_weight_decay),
            "temporal_ce_weight": float(args.temporal_ce_weight),
            "aux_bootstrap_weight": float(args.aux_bootstrap_weight),
            "aux_relation_weight": float(args.aux_relation_weight),
            "aux_flag_weight": float(args.aux_flag_weight),
            "alternating_phase_length": int(args.alternating_phase_length),
            "warmup_gate_epochs": int(args.warmup_gate_epochs),
            "warmup_temporal_epochs": int(args.warmup_temporal_epochs),
            "temporal_loss_ramp_epochs": int(args.temporal_loss_ramp_epochs),
            "seed": int(args.seed),
            "context_gate_scale": float(args.context_gate_scale),
            "selection_metric": str(args.selection_metric),
            "best_epoch": int(best_epoch),
            "best_score": list(best_score) if best_score is not None else [],
            "temporal_checkpoint": str(args.temporal_checkpoint),
            "train_samples": int(len(train_indices)),
            "val_samples": int(len(val_indices)),
            "history": history,
            "label_source_counts": dict(dataset.get("label_source_counts", {})),
        }
    )
    gate_output_path.write_text(json.dumps(gate_payload, indent=2), encoding="utf-8")

    temporal_output_path: Optional[Path] = None
    if mode != "freeze":
        temporal_output_path = _resolve_temporal_output_path(args.temporal_output, gate_output_path)
        temporal_output_path.parent.mkdir(parents=True, exist_ok=True)
        temporal_payload = build_temporal_checkpoint_payload(
            temporal_model=temporal_model,
            dataset=dataset,
            metadata={
                "source_temporal_checkpoint": str(args.temporal_checkpoint),
                "context_gate_checkpoint": str(gate_output_path),
                "mode": mode,
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "gate_lr": float(args.lr),
                "temporal_lr": float(args.temporal_lr),
                "temporal_ce_weight": float(args.temporal_ce_weight),
                "aux_bootstrap_weight": float(args.aux_bootstrap_weight),
                "aux_relation_weight": float(args.aux_relation_weight),
                "aux_flag_weight": float(args.aux_flag_weight),
                "alternating_phase_length": int(args.alternating_phase_length),
                "warmup_gate_epochs": int(args.warmup_gate_epochs),
                "warmup_temporal_epochs": int(args.warmup_temporal_epochs),
                "temporal_loss_ramp_epochs": int(args.temporal_loss_ramp_epochs),
                "selection_metric": str(args.selection_metric),
                "best_epoch": int(best_epoch),
                "best_score": list(best_score) if best_score is not None else [],
                "history": history,
                "train_samples": int(len(train_indices)),
                "val_samples": int(len(val_indices)),
                "label_source_counts": dict(dataset.get("label_source_counts", {})),
            },
        )
        torch.save(temporal_payload, temporal_output_path)

    print(
        json.dumps(
            {
                "mode": mode,
                "gate_output_path": str(gate_output_path),
                "temporal_output_path": str(temporal_output_path) if temporal_output_path is not None else "",
                "train_samples": len(train_indices),
                "val_samples": len(val_indices),
                "best_epoch": best_epoch,
                "best_score": list(best_score) if best_score is not None else [],
                "history_tail": history[-3:],
            },
            indent=2,
        )
    )


def _build_loader(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor,
    aux_mask: torch.Tensor,
    aux_next_bootstrap: torch.Tensor,
    aux_next_relations: torch.Tensor,
    aux_next_flags: torch.Tensor,
    indices: List[int],
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


def _build_optimizer(
    *,
    gate_model: ContextGateModel,
    temporal_model,
    mode: str,
    gate_lr: float,
    gate_weight_decay: float,
    temporal_lr: float,
    temporal_weight_decay: float,
) -> torch.optim.Optimizer:
    param_groups = [
        {
            "params": list(gate_model.parameters()),
            "lr": float(gate_lr),
            "weight_decay": float(gate_weight_decay),
        }
    ]
    if str(mode).strip().lower() != "freeze":
        param_groups.append(
            {
                "params": list(temporal_model.parameters()),
                "lr": float(temporal_lr),
                "weight_decay": float(temporal_weight_decay),
            }
        )
    return torch.optim.AdamW(param_groups)


def _phase_for_epoch(
    *,
    epoch: int,
    mode: str,
    alternating_phase_length: int,
    warmup_gate_epochs: int,
    warmup_temporal_epochs: int,
) -> Tuple[bool, bool, str, int]:
    normalized = str(mode).strip().lower()
    if normalized == "freeze":
        return True, False, "gate", 0
    epoch_index = max(1, int(epoch))
    if epoch_index <= max(0, int(warmup_gate_epochs)):
        return True, False, "warmup_gate", 0
    epoch_after_gate = epoch_index - max(0, int(warmup_gate_epochs))
    if epoch_after_gate <= max(0, int(warmup_temporal_epochs)):
        return False, True, "warmup_temporal", epoch_after_gate
    epoch_after_warmup = epoch_after_gate - max(0, int(warmup_temporal_epochs))
    if normalized == "joint":
        return True, True, "joint", epoch_after_warmup
    block = ((epoch_after_warmup - 1) // max(1, int(alternating_phase_length))) % 2
    if block == 0:
        return True, False, "gate", 0
    alternating_temporal_epoch = ((epoch_after_warmup - 1) // max(1, int(alternating_phase_length))) + 1
    return False, True, "temporal", alternating_temporal_epoch


def _temporal_ramp_scale(*, phase_temporal_epoch: int, ramp_epochs: int) -> float:
    if phase_temporal_epoch <= 0:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return float(min(1.0, float(phase_temporal_epoch) / float(max(1, int(ramp_epochs)))))


def _set_module_trainable(module: torch.nn.Module, *, trainable: bool, training: bool) -> None:
    module.train(mode=bool(training and trainable))
    if not (training and trainable):
        module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(bool(training and trainable))


def _run_epoch(
    *,
    temporal_model,
    gate_model: ContextGateModel,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    training: bool,
    gate_trainable: bool,
    temporal_trainable: bool,
    relation_types: List[str],
    component_names: List[str],
    temporal_ce_weight: float,
    aux_bootstrap_weight: float,
    aux_relation_weight: float,
    aux_flag_weight: float,
    temporal_loss_scale: float,
) -> Dict[str, float]:
    _set_module_trainable(gate_model, trainable=gate_trainable, training=training)
    _set_module_trainable(temporal_model, trainable=temporal_trainable, training=training)

    total_fused_loss = 0.0
    total_temporal_loss = 0.0
    total_correct = 0
    total_temporal_correct = 0
    total_weight = 0.0
    total_items = 0
    total_bootstrap = 0.0
    total_relation = 0.0
    total_flag = 0.0
    total_aux_weight = 0.0
    total_gate_vector = torch.zeros((gate_model.num_experts,), dtype=torch.float32)

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

        if training and optimizer is not None:
            optimizer.zero_grad()

        outputs, context_features, expert_scores = build_gate_inputs(
            temporal_model=temporal_model,
            batch_x=batch_x,
            relation_types=relation_types,
            component_names=component_names,
        )
        temporal_logits = outputs["step_logits"]
        fused_scores, gates = gate_model(context_features, expert_scores)

        fused_terms = F.cross_entropy(fused_scores, batch_y, reduction="none")
        temporal_terms = F.cross_entropy(temporal_logits, batch_y, reduction="none")
        denom = torch.clamp(batch_w.sum(), min=1e-6)
        fused_loss = (fused_terms * batch_w).sum() / denom
        temporal_loss = (temporal_terms * batch_w).sum() / denom
        total_loss = fused_loss

        aux_weight_denom = torch.clamp((batch_w * batch_aux_mask).sum(), min=1e-6)
        bootstrap_loss = torch.tensor(0.0, device=device)
        relation_loss = torch.tensor(0.0, device=device)
        flag_loss = torch.tensor(0.0, device=device)
        effective_temporal_ce_weight = float(temporal_ce_weight) * float(max(0.0, temporal_loss_scale))
        effective_aux_bootstrap_weight = float(aux_bootstrap_weight) * float(max(0.0, temporal_loss_scale))
        effective_aux_relation_weight = float(aux_relation_weight) * float(max(0.0, temporal_loss_scale))
        effective_aux_flag_weight = float(aux_flag_weight) * float(max(0.0, temporal_loss_scale))
        if temporal_trainable:
            total_loss = total_loss + effective_temporal_ce_weight * temporal_loss
            if batch_next_bootstrap.shape[-1] > 0 and effective_aux_bootstrap_weight > 0.0:
                bootstrap_terms = F.mse_loss(outputs["next_bootstrap"], batch_next_bootstrap, reduction="none").mean(dim=-1)
                bootstrap_loss = (bootstrap_terms * batch_w * batch_aux_mask).sum() / aux_weight_denom
                total_loss = total_loss + effective_aux_bootstrap_weight * bootstrap_loss
            if batch_next_relations.shape[-1] > 0 and effective_aux_relation_weight > 0.0:
                relation_terms = F.binary_cross_entropy_with_logits(
                    outputs["next_relations"], batch_next_relations, reduction="none"
                ).mean(dim=-1)
                relation_loss = (relation_terms * batch_w * batch_aux_mask).sum() / aux_weight_denom
                total_loss = total_loss + effective_aux_relation_weight * relation_loss
            if batch_next_flags.shape[-1] > 0 and effective_aux_flag_weight > 0.0:
                flag_terms = F.binary_cross_entropy_with_logits(
                    outputs["next_flags"], batch_next_flags, reduction="none"
                ).mean(dim=-1)
                flag_loss = (flag_terms * batch_w * batch_aux_mask).sum() / aux_weight_denom
                total_loss = total_loss + effective_aux_flag_weight * flag_loss

        if training and optimizer is not None:
            total_loss.backward()
            optimizer.step()

        fused_predictions = torch.argmax(fused_scores, dim=-1)
        temporal_predictions = torch.argmax(temporal_logits, dim=-1)
        total_fused_loss += float((fused_terms * batch_w).sum().item())
        total_temporal_loss += float((temporal_terms * batch_w).sum().item())
        total_correct += int((fused_predictions == batch_y).sum().item())
        total_temporal_correct += int((temporal_predictions == batch_y).sum().item())
        total_weight += float(batch_w.sum().item())
        total_items += int(batch_y.numel())
        total_bootstrap += float((bootstrap_loss * aux_weight_denom).item())
        total_relation += float((relation_loss * aux_weight_denom).item())
        total_flag += float((flag_loss * aux_weight_denom).item())
        total_aux_weight += float(aux_weight_denom.item())
        total_gate_vector += gates.detach().cpu().sum(dim=0)

    average_gate = total_gate_vector / max(1, total_items)
    return {
        "fused_loss": float(total_fused_loss / max(1e-6, total_weight)),
        "temporal_loss": float(total_temporal_loss / max(1e-6, total_weight)),
        "fused_acc": float(total_correct) / float(max(1, total_items)),
        "temporal_acc": float(total_temporal_correct) / float(max(1, total_items)),
        "fused_delta": float(total_correct - total_temporal_correct) / float(max(1, total_items)),
        "bootstrap_loss": float(total_bootstrap / max(1e-6, total_aux_weight)),
        "relation_loss": float(total_relation / max(1e-6, total_aux_weight)),
        "flag_loss": float(total_flag / max(1e-6, total_aux_weight)),
        "gate_state": float(average_gate[0].item()) if average_gate.numel() > 0 else 0.0,
        "gate_temporal": float(average_gate[1].item()) if average_gate.numel() > 1 else 0.0,
        "gate_retrieval": float(average_gate[2].item()) if average_gate.numel() > 2 else 0.0,
        "gate_memory": float(average_gate[3].item()) if average_gate.numel() > 3 else 0.0,
        "temporal_loss_scale": float(temporal_loss_scale),
    }


def _selection_score(*, metrics: Dict[str, float], selection_metric: str) -> Tuple[float, float, float]:
    normalized = str(selection_metric or "fused_delta").strip().lower()
    if normalized == "fused_acc":
        primary = float(metrics.get("fused_acc", 0.0))
    else:
        primary = float(metrics.get("fused_delta", 0.0))
    secondary = float(metrics.get("fused_acc", 0.0))
    tertiary = -float(metrics.get("fused_loss", 0.0))
    return primary, secondary, tertiary


def _capture_best_state(*, gate_model: ContextGateModel, temporal_model) -> Dict[str, object]:
    return {
        "gate_base_gate_logits": gate_model.base_gate_logits.detach().cpu(),
        "gate_linear_weight": gate_model.linear.weight.detach().cpu(),
        "gate_linear_bias": gate_model.linear.bias.detach().cpu(),
        "temporal_state_dict": {
            key: value.detach().cpu() for key, value in temporal_model.state_dict().items()
        },
    }


def _restore_best_state(*, gate_model: ContextGateModel, temporal_model, best_state: Dict[str, object]) -> None:
    with torch.no_grad():
        gate_model.base_gate_logits.copy_(best_state["gate_base_gate_logits"])
        gate_model.linear.weight.copy_(best_state["gate_linear_weight"])
        gate_model.linear.bias.copy_(best_state["gate_linear_bias"])
    temporal_model.load_state_dict(best_state["temporal_state_dict"])


def _resolve_temporal_output_path(raw_value: str, gate_output_path: Path) -> Path:
    value = str(raw_value or "").strip()
    if value:
        return Path(value)
    return gate_output_path.with_name(f"{gate_output_path.stem}_temporal.ckpt")


if __name__ == "__main__":
    main()
