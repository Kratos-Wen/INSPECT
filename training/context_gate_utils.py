"""Shared utilities for offline graph-conditioned context gating."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..components.online_fusion import AdaptiveExpertFusion
from .modeling import StreamingTemporalGRUModel, checkpoint_payload


class ContextGateModel(nn.Module):
    """Lightweight offline-trainable context gate over fixed expert scores."""

    def __init__(
        self,
        *,
        num_features: int,
        num_experts: int,
        context_gate_scale: float,
        init_base_gates: List[float],
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.num_experts = int(num_experts)
        self.context_gate_scale = float(context_gate_scale)
        init_tensor = torch.tensor(init_base_gates, dtype=torch.float32)
        init_tensor = torch.clamp(init_tensor, min=1e-4)
        init_tensor = init_tensor / init_tensor.sum()
        self.base_gate_logits = nn.Parameter(torch.log(init_tensor))
        self.linear = nn.Linear(self.num_features, self.num_experts)

    def forward(self, context_features: torch.Tensor, expert_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_logits = self.base_gate_logits.unsqueeze(0) + self.context_gate_scale * self.linear(context_features)
        gates = torch.softmax(gate_logits, dim=-1)
        fused_scores = torch.sum(gates.unsqueeze(-1) * expert_scores, dim=1)
        return fused_scores, gates

    def export_payload(self, *, metadata: Dict[str, object]) -> Dict[str, object]:
        base_gates = torch.softmax(self.base_gate_logits.detach().cpu(), dim=-1).tolist()
        return {
            "expert_names": ["state", "temporal", "retrieval", "memory"],
            "feature_names": list(AdaptiveExpertFusion.CONTEXT_FEATURE_NAMES),
            "base_gates": [float(value) for value in base_gates],
            "context_gate_weights": self.linear.weight.detach().cpu().tolist(),
            "context_gate_bias": self.linear.bias.detach().cpu().tolist(),
            "metadata": dict(metadata),
        }


def build_temporal_model(
    *,
    dataset: Dict[str, object],
    checkpoint: Dict[str, object],
    device: torch.device,
    trainable: bool,
) -> StreamingTemporalGRUModel:
    steps = list(dataset["steps"])
    component_names = list(dataset.get("component_names", []))
    relation_types = list(dataset.get("relation_types", []))
    flag_names = list(dataset.get("flag_names", []))
    aggregator_config = checkpoint.get("aggregator_config") or {}
    model = StreamingTemporalGRUModel(
        input_size=int(dataset["x"].shape[-1]),
        hidden_size=int(checkpoint.get("hidden_size", len(steps))),
        num_steps=len(steps),
        num_components=len(component_names),
        num_relations=len(relation_types),
        num_scalars=8,
        use_learned_token_aggregation=bool(aggregator_config),
        token_hidden_size=int(aggregator_config.get("hidden_size", 24) or 24),
        token_output_size=int(aggregator_config.get("output_size", 48) or 48),
        num_flag_targets=len(flag_names),
    ).to(device)
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
    model.train(mode=trainable)
    for parameter in model.parameters():
        parameter.requires_grad_(bool(trainable))
    return model


def build_gate_inputs(
    *,
    temporal_model: StreamingTemporalGRUModel,
    batch_x: torch.Tensor,
    relation_types: List[str],
    component_names: List[str],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    outputs = temporal_model(batch_x, return_aux=True)
    temporal_scores = normalize_scores(outputs["step_logits"])
    last_frame = batch_x[:, -1, :]
    num_steps = temporal_scores.shape[-1]
    state_scores, retrieval_scores, memory_scores, combined_scores, prev_scores, visible, relevant, relations, scalars = split_last_frame(
        last_frame,
        num_steps=num_steps,
        num_components=len(component_names),
        num_relations=len(relation_types),
    )
    expert_scores = torch.stack([state_scores, temporal_scores, retrieval_scores, memory_scores], dim=1)
    context_features = build_context_features(
        combined_scores=combined_scores,
        prev_scores=prev_scores,
        visible=visible,
        relevant=relevant,
        relations=relations,
        scalars=scalars,
        state_scores=state_scores,
        temporal_scores=temporal_scores,
        retrieval_scores=retrieval_scores,
        memory_scores=memory_scores,
        relation_index={name: index for index, name in enumerate(relation_types)},
    )
    return outputs, context_features, expert_scores


def split_indices(run_names: List[str], num_items: int, val_fraction: float) -> Tuple[List[int], List[int]]:
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


def build_temporal_checkpoint_payload(
    *,
    temporal_model: StreamingTemporalGRUModel,
    dataset: Dict[str, object],
    metadata: Dict[str, object],
) -> Dict[str, object]:
    return checkpoint_payload(
        temporal_model,
        steps=list(dataset["steps"]),
        component_names=list(dataset.get("component_names", [])),
        relation_types=list(dataset.get("relation_types", [])),
        flag_names=list(dataset.get("flag_names", [])),
        window_size=int(dataset["window_size"]),
        metadata=metadata,
    )


def split_last_frame(
    frame: torch.Tensor,
    *,
    num_steps: int,
    num_components: int,
    num_relations: int,
) -> Tuple[torch.Tensor, ...]:
    cursor = 0
    combined = frame[:, cursor : cursor + num_steps]
    cursor += num_steps
    state = frame[:, cursor : cursor + num_steps]
    cursor += num_steps
    retrieval = frame[:, cursor : cursor + num_steps]
    cursor += num_steps
    memory = frame[:, cursor : cursor + num_steps]
    cursor += num_steps
    prev = frame[:, cursor : cursor + num_steps]
    cursor += num_steps
    visible = frame[:, cursor : cursor + num_components]
    cursor += num_components
    relevant = frame[:, cursor : cursor + num_components]
    cursor += num_components
    relations = frame[:, cursor : cursor + num_relations]
    cursor += num_relations
    scalars = frame[:, cursor:]
    return state, retrieval, memory, combined, prev, visible, relevant, relations, scalars


def normalize_scores(logits: torch.Tensor) -> torch.Tensor:
    values = logits - logits.min(dim=-1, keepdim=True).values
    denom = torch.clamp(values.max(dim=-1, keepdim=True).values, min=1e-6)
    return values / denom


def build_context_features(
    *,
    combined_scores: torch.Tensor,
    prev_scores: torch.Tensor,
    visible: torch.Tensor,
    relevant: torch.Tensor,
    relations: torch.Tensor,
    scalars: torch.Tensor,
    state_scores: torch.Tensor,
    temporal_scores: torch.Tensor,
    retrieval_scores: torch.Tensor,
    memory_scores: torch.Tensor,
    relation_index: Dict[str, int],
) -> torch.Tensor:
    def rel(name: str) -> torch.Tensor:
        index = relation_index.get(name)
        if index is None:
            return torch.zeros((relations.shape[0],), dtype=relations.dtype, device=relations.device)
        return relations[:, index]

    has_visual = scalars[:, 4]
    memory_active = scalars[:, 3]
    review_hold = scalars[:, 5]
    review_request_human = scalars[:, 6]
    visible_density = torch.clamp(visible.sum(dim=-1) / 4.0, min=0.0, max=1.0)
    relevant_density = torch.clamp(relevant.sum(dim=-1) / 4.0, min=0.0, max=1.0)
    contact_density = torch.clamp(rel("contacting") + rel("supported_by"), min=0.0, max=1.0)
    support_density = torch.clamp(rel("supporting") + rel("supported_by"), min=0.0, max=1.0)
    directional_density = torch.clamp(
        rel("in_front_of") + rel("behind") + rel("left_of") + rel("right_of") + rel("above") + rel("below"),
        min=0.0,
        max=1.0,
    )
    overlap_density = torch.clamp(rel("overlapping"), min=0.0, max=1.0)
    top2 = torch.topk(combined_scores, k=min(2, combined_scores.shape[-1]), dim=-1).values
    if top2.shape[-1] == 1:
        bootstrap_margin = top2[:, 0]
    else:
        bootstrap_margin = top2[:, 0] - top2[:, 1]

    expert_tops = torch.stack(
        [
            torch.argmax(state_scores, dim=-1),
            torch.argmax(temporal_scores, dim=-1),
            torch.argmax(retrieval_scores, dim=-1),
            torch.argmax(memory_scores, dim=-1),
        ],
        dim=-1,
    )
    unique_support = F.one_hot(expert_tops, num_classes=combined_scores.shape[-1]).amax(dim=1).sum(dim=-1).float()
    expert_disagreement = torch.clamp(unique_support - 1.0, min=0.0) / float(expert_tops.shape[-1])

    prev_valid = prev_scores.max(dim=-1).values > 0.0
    prev_top = torch.argmax(prev_scores, dim=-1)
    combined_top = torch.argmax(combined_scores, dim=-1)
    transition_active = torch.where(prev_valid & (prev_top != combined_top), torch.ones_like(has_visual), torch.zeros_like(has_visual))

    return torch.stack(
        [
            has_visual,
            memory_active,
            review_hold,
            review_request_human,
            visible_density,
            relevant_density,
            contact_density,
            support_density,
            directional_density,
            overlap_density,
            torch.clamp(bootstrap_margin, min=0.0, max=1.0),
            torch.clamp(expert_disagreement, min=0.0, max=1.0),
            transition_active,
        ],
        dim=-1,
    )


def resolve_device(device: str) -> torch.device:
    normalized = str(device or "cpu").strip().lower()
    if normalized in {"", "cpu"}:
        return torch.device("cpu")
    if normalized.startswith("cuda") and torch.cuda.is_available():
        return torch.device(normalized)
    if normalized.isdigit() and torch.cuda.is_available():
        return torch.device(f"cuda:{normalized}")
    return torch.device("cpu")
