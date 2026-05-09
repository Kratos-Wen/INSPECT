"""Shared lightweight temporal layers used by runtime and training."""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn


class LearnedTokenAggregator(nn.Module):
    """Compress structured frame vectors into compact learned token embeddings."""

    def __init__(
        self,
        *,
        num_steps: int,
        num_components: int,
        num_relations: int,
        num_scalars: int,
        hidden_size: int = 24,
        output_size: int = 48,
    ) -> None:
        super().__init__()
        self.num_steps = int(num_steps)
        self.num_components = int(num_components)
        self.num_relations = int(num_relations)
        self.num_scalars = int(num_scalars)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)

        self.step_encoder = nn.Sequential(
            nn.Linear(5, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.component_encoder = nn.Sequential(
            nn.Linear(2, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.relation_encoder = nn.Sequential(
            nn.Linear(max(1, self.num_relations), self.hidden_size),
            nn.SiLU(),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(max(1, self.num_scalars), self.hidden_size),
            nn.SiLU(),
        )
        self.step_query = nn.Parameter(torch.zeros(self.hidden_size))
        self.component_query = nn.Parameter(torch.zeros(self.hidden_size))
        self.output = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.output_size),
            nn.SiLU(),
        )
        self.reset_parameters()

    def forward(self, frame_vector: torch.Tensor) -> torch.Tensor:
        """Aggregate one or more raw frame vectors into compact learned tokens."""

        original_shape = frame_vector.shape[:-1]
        vector = frame_vector.reshape(-1, frame_vector.shape[-1])
        grouped = self._split_groups(vector)

        step_tokens = self.step_encoder(grouped["step_bundle"])
        step_pooled = self._attention_pool(step_tokens, self.step_query)

        if self.num_components > 0:
            component_tokens = self.component_encoder(grouped["component_bundle"])
            component_pooled = self._attention_pool(component_tokens, self.component_query)
        else:
            component_pooled = torch.zeros(
                (vector.shape[0], self.hidden_size),
                dtype=vector.dtype,
                device=vector.device,
            )

        relation_input = grouped["relation_vec"]
        if relation_input.shape[-1] == 0:
            relation_input = torch.zeros((vector.shape[0], 1), dtype=vector.dtype, device=vector.device)
        relation_pooled = self.relation_encoder(relation_input)

        scalar_input = grouped["scalar_vec"]
        if scalar_input.shape[-1] == 0:
            scalar_input = torch.zeros((vector.shape[0], 1), dtype=vector.dtype, device=vector.device)
        scalar_pooled = self.scalar_encoder(scalar_input)

        merged = torch.cat([step_pooled, component_pooled, relation_pooled, scalar_pooled], dim=-1)
        encoded = self.output(merged)
        return encoded.reshape(*original_shape, self.output_size)

    def config_dict(self) -> Dict[str, int]:
        return {
            "num_steps": self.num_steps,
            "num_components": self.num_components,
            "num_relations": self.num_relations,
            "num_scalars": self.num_scalars,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
        }

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.step_query, mean=0.0, std=0.02)
        nn.init.normal_(self.component_query, mean=0.0, std=0.02)

    def _split_groups(self, frame_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        cursor = 0
        combined = frame_vector[:, cursor : cursor + self.num_steps]
        cursor += self.num_steps
        state = frame_vector[:, cursor : cursor + self.num_steps]
        cursor += self.num_steps
        retrieval = frame_vector[:, cursor : cursor + self.num_steps]
        cursor += self.num_steps
        memory = frame_vector[:, cursor : cursor + self.num_steps]
        cursor += self.num_steps
        prev = frame_vector[:, cursor : cursor + self.num_steps]
        cursor += self.num_steps

        visible = frame_vector[:, cursor : cursor + self.num_components]
        cursor += self.num_components
        relevant = frame_vector[:, cursor : cursor + self.num_components]
        cursor += self.num_components

        relation_vec = frame_vector[:, cursor : cursor + self.num_relations]
        cursor += self.num_relations
        scalar_vec = frame_vector[:, cursor : cursor + self.num_scalars]

        step_bundle = torch.stack([combined, state, retrieval, memory, prev], dim=-1)
        component_bundle = (
            torch.stack([visible, relevant], dim=-1)
            if self.num_components > 0
            else frame_vector.new_zeros((frame_vector.shape[0], 0, 2))
        )
        return {
            "step_bundle": step_bundle,
            "component_bundle": component_bundle,
            "relation_vec": relation_vec,
            "scalar_vec": scalar_vec,
        }

    def _attention_pool(self, tokens: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] == 0:
            return torch.zeros(
                (tokens.shape[0], tokens.shape[-1]),
                dtype=tokens.dtype,
                device=tokens.device,
            )
        scale = math.sqrt(max(1, tokens.shape[-1]))
        scores = torch.einsum("bnh,h->bn", tokens, query) / scale
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bn,bnh->bh", weights, tokens)
