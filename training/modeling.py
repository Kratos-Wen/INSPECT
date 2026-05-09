"""Model definitions shared by temporal training and evaluation scripts."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn

from ..components.temporal_layers import LearnedTokenAggregator


class StreamingTemporalGRUModel(nn.Module):
    """A small GRUCell-based classifier compatible with the runtime GRU expert."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_steps: int,
        num_components: int = 0,
        num_relations: int = 0,
        num_scalars: int = 8,
        use_learned_token_aggregation: bool = False,
        token_hidden_size: int = 24,
        token_output_size: int = 48,
        num_flag_targets: int = 0,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_steps = int(num_steps)
        self.num_components = int(num_components)
        self.num_relations = int(num_relations)
        self.num_scalars = int(num_scalars)
        self.num_flag_targets = int(num_flag_targets)
        self.use_learned_token_aggregation = bool(use_learned_token_aggregation)
        self.token_aggregator: Optional[LearnedTokenAggregator] = None
        recurrent_input_size = self.input_size
        if self.use_learned_token_aggregation:
            self.token_aggregator = LearnedTokenAggregator(
                num_steps=self.num_steps,
                num_components=self.num_components,
                num_relations=self.num_relations,
                num_scalars=self.num_scalars,
                hidden_size=max(4, int(token_hidden_size)),
                output_size=max(4, int(token_output_size)),
            )
            recurrent_input_size = int(self.token_aggregator.output_size)
        self.recurrent_input_size = int(recurrent_input_size)
        self.gru = nn.GRUCell(self.recurrent_input_size, self.hidden_size)
        self.head = nn.Linear(self.hidden_size, self.num_steps)
        self.next_bootstrap_head = nn.Linear(self.hidden_size, self.num_steps)
        self.next_relation_head = nn.Linear(self.hidden_size, self.num_relations)
        self.next_flag_head = nn.Linear(self.hidden_size, self.num_flag_targets)

    def forward(self, sequence: torch.Tensor, return_aux: bool = False):
        """Run a batch of sequences through the streaming GRU."""

        batch_size = int(sequence.shape[0])
        hidden = torch.zeros((batch_size, self.hidden_size), dtype=sequence.dtype, device=sequence.device)
        for index in range(int(sequence.shape[1])):
            step_input = sequence[:, index, :]
            if self.token_aggregator is not None:
                step_input = self.token_aggregator(step_input)
            hidden = self.gru(step_input, hidden)
        step_logits = self.head(hidden)
        if not return_aux:
            return step_logits
        return {
            "step_logits": step_logits,
            "next_bootstrap": self.next_bootstrap_head(hidden),
            "next_relations": self.next_relation_head(hidden),
            "next_flags": self.next_flag_head(hidden),
        }


def bootstrap_initialize(model: StreamingTemporalGRUModel, ema_alpha: float = 0.55) -> None:
    """Initialize the model close to a stable EMA-style baseline."""

    for parameter in model.gru.parameters():
        nn.init.zeros_(parameter)
    for parameter in model.head.parameters():
        nn.init.zeros_(parameter)
    for parameter in model.next_bootstrap_head.parameters():
        nn.init.zeros_(parameter)
    for parameter in model.next_relation_head.parameters():
        nn.init.zeros_(parameter)
    for parameter in model.next_flag_head.parameters():
        nn.init.zeros_(parameter)

    hidden = model.hidden_size
    steps = model.num_steps
    alpha = max(1e-3, min(0.999, float(ema_alpha)))
    keep = max(1e-3, min(0.999, 1.0 - alpha))

    with torch.no_grad():
        model.gru.bias_ih[:hidden].fill_(4.0)
        update_bias = float(math.log(keep / max(1e-6, 1.0 - keep)))
        model.gru.bias_ih[hidden : 2 * hidden].fill_(update_bias)
        rows = min(hidden, steps)
        for index in range(rows):
            model.gru.weight_ih[2 * hidden + index, index] = 1.0
            model.head.weight[index, index] = 1.0
            model.next_bootstrap_head.weight[index, index] = 1.0


def checkpoint_payload(
    model: StreamingTemporalGRUModel,
    *,
    steps,
    component_names,
    relation_types,
    flag_names,
    window_size: int,
    metadata: Dict[str, object],
) -> Dict[str, object]:
    """Build a runtime-compatible checkpoint payload."""

    return {
        "gru_state_dict": model.gru.state_dict(),
        "head_state_dict": model.head.state_dict(),
        "next_bootstrap_head_state_dict": model.next_bootstrap_head.state_dict(),
        "next_relation_head_state_dict": model.next_relation_head.state_dict(),
        "next_flag_head_state_dict": model.next_flag_head.state_dict(),
        "aggregator_state_dict": model.token_aggregator.state_dict() if model.token_aggregator is not None else None,
        "aggregator_config": model.token_aggregator.config_dict() if model.token_aggregator is not None else None,
        "steps": list(steps),
        "component_names": list(component_names),
        "relation_types": list(relation_types),
        "flag_names": list(flag_names),
        "window_size": int(window_size),
        "hidden_size": int(model.hidden_size),
        "input_size": int(model.input_size),
        "recurrent_input_size": int(model.recurrent_input_size),
        "metadata": dict(metadata),
    }
