"""Streaming temporal step experts built on compact evidence tokens."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from ..core_types import EvidenceToken, StepPrediction
from .step_graph import CompiledStepGraphPrior
from .temporal_layers import LearnedTokenAggregator
from .timeline_store import EvidenceTimelineStore


class TemporalEvidenceVectorizer:
    """Vectorize evidence tokens for lightweight streaming temporal models."""

    DEFAULT_RELATIONS: Sequence[str] = (
        "contacting",
        "supporting",
        "supported_by",
        "in_front_of",
        "behind",
        "left_of",
        "right_of",
        "above",
        "below",
        "overlapping",
    )
    FLAG_NAMES: Sequence[str] = (
        "memory_active",
        "has_visual_evidence",
        "review_hold",
        "review_request_human",
        "review_prefer_candidate",
    )

    def __init__(
        self,
        steps: Iterable[str],
        component_names: Iterable[str],
        relation_types: Optional[Iterable[str]] = None,
        state_score_weight: float = 0.60,
        retrieval_score_weight: float = 0.25,
        memory_score_weight: float = 0.15,
    ) -> None:
        self.steps = [str(step).strip().upper() for step in steps]
        self.component_names = [str(name).strip().lower() for name in component_names]
        relation_iter = relation_types if relation_types is not None else self.DEFAULT_RELATIONS
        self.relation_types = [str(name).strip().lower() for name in relation_iter]
        self.num_steps = len(self.steps)
        self.num_components = len(self.component_names)
        self.num_relations = len(self.relation_types)
        self.num_scalars = 8
        self.state_score_weight = float(state_score_weight)
        self.retrieval_score_weight = float(retrieval_score_weight)
        self.memory_score_weight = float(memory_score_weight)
        self.dim = (
            self.num_steps
            + self.num_steps * 4
            + self.num_components * 2
            + self.num_relations
            + self.num_scalars
        )

    def vectorize(self, token: EvidenceToken) -> np.ndarray:
        """Convert one evidence token into a fixed-size float32 feature vector."""

        features: List[float] = []
        combined = self.bootstrap_scores(token)
        features.extend(float(combined.get(step_id, 0.0)) for step_id in self.steps)
        features.extend(float(token.state_scores.get(step_id, 0.0)) for step_id in self.steps)
        features.extend(float(token.retrieval_scores.get(step_id, 0.0)) for step_id in self.steps)
        features.extend(float(token.memory_scores.get(step_id, 0.0)) for step_id in self.steps)
        prev_step = str(token.prev_step or "").strip().upper()
        features.extend(1.0 if prev_step == step_id else 0.0 for step_id in self.steps)
        for component_name in self.component_names:
            features.append(self._scaled_count(token.visible_counts.get(component_name, 0)))
        for component_name in self.component_names:
            features.append(self._scaled_count(token.relevant_counts.get(component_name, 0)))
        for relation_type in self.relation_types:
            features.append(self._scaled_count(token.relation_counts.get(relation_type, 0)))
        features.extend(
            [
                float(token.state_confidence),
                float(token.retrieval_confidence),
                float(token.memory_confidence),
                1.0 if token.memory_active else 0.0,
                1.0 if token.has_visual_evidence else 0.0,
                1.0 if str(token.review_action).strip().lower() == "hold" else 0.0,
                1.0 if str(token.review_action).strip().lower() == "request_human" else 0.0,
                1.0 if str(token.review_action).strip().lower() == "prefer_candidate" else 0.0,
            ]
        )
        return np.asarray(features, dtype=np.float32)

    def bootstrap_scores(self, token: EvidenceToken) -> Dict[str, float]:
        """Return the deterministic per-step frame scores used to bootstrap temporal models."""

        combined: Dict[str, float] = {}
        total_weight = max(
            1e-6,
            self.state_score_weight + self.retrieval_score_weight + self.memory_score_weight,
        )
        for step_id in self.steps:
            state = float(token.state_scores.get(step_id, 0.0))
            retrieval = float(token.retrieval_scores.get(step_id, 0.0))
            memory = float(token.memory_scores.get(step_id, 0.0))
            combined[step_id] = (
                self.state_score_weight * state
                + self.retrieval_score_weight * retrieval
                + self.memory_score_weight * memory
            ) / total_weight
        return combined

    def bootstrap_array(self, token: EvidenceToken) -> np.ndarray:
        return np.asarray([float(self.bootstrap_scores(token).get(step_id, 0.0)) for step_id in self.steps], dtype=np.float32)

    def relation_presence_vector(self, token: EvidenceToken) -> np.ndarray:
        return np.asarray(
            [1.0 if int(token.relation_counts.get(relation_type, 0)) > 0 else 0.0 for relation_type in self.relation_types],
            dtype=np.float32,
        )

    def flag_vector(self, token: EvidenceToken) -> np.ndarray:
        review_action = str(token.review_action).strip().lower()
        return np.asarray(
            [
                1.0 if token.memory_active else 0.0,
                1.0 if token.has_visual_evidence else 0.0,
                1.0 if review_action == "hold" else 0.0,
                1.0 if review_action == "request_human" else 0.0,
                1.0 if review_action == "prefer_candidate" else 0.0,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _scaled_count(value: object) -> float:
        count = max(0.0, float(value or 0.0))
        return float(min(1.0, math.log1p(count) / math.log(4.0)))


class _BaseTemporalStepExpert:
    """Common utilities shared by streaming temporal experts."""

    def __init__(
        self,
        steps: List[str],
        timeline: EvidenceTimelineStore,
        graph_prior: CompiledStepGraphPrior,
        enabled: bool = True,
    ) -> None:
        self.steps = [str(step).strip().upper() for step in steps]
        self.timeline = timeline
        self.graph_prior = graph_prior
        self.enabled = bool(enabled)

    def _empty_prediction(self, reason: str, prev_step: Optional[str] = None) -> StepPrediction:
        return StepPrediction(
            step_id=str(prev_step or (self.steps[0] if self.steps else "")).strip().upper(),
            confidence=0.0,
            scores={step_id: 0.0 for step_id in self.steps},
            extras={"active": False, "reason": reason},
        )

    def _finalize_scores(
        self,
        raw_scores: Dict[str, float],
        token: EvidenceToken,
        extras: Dict[str, object],
    ) -> StepPrediction:
        adjusted, graph_bias = self.graph_prior.apply(
            raw_scores,
            token=token,
            timeline=self.timeline,
            prev_step=token.prev_step,
        )
        normalized = self._normalize_scores(adjusted)
        top_step = max(normalized, key=normalized.get) if normalized else (self.steps[0] if self.steps else "")
        confidence = self._confidence(normalized)
        runner_up = self._runner_up(normalized, top_step)
        return StepPrediction(
            step_id=top_step,
            confidence=confidence,
            scores=normalized,
            extras={
                "active": True,
                "runner_up": runner_up,
                "graph_bias": graph_bias,
                "history_size": len(self.timeline),
                **extras,
            },
        )

    @staticmethod
    def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        values = np.array([float(scores[key]) for key in scores], dtype=np.float32)
        values = values - values.min()
        maximum = float(values.max()) if values.size > 0 else 0.0
        if maximum > 1e-6:
            values = values / maximum
        return {step_id: float(values[index]) for index, step_id in enumerate(scores)}

    @staticmethod
    def _confidence(scores: Dict[str, float]) -> float:
        if not scores:
            return 0.0
        ordered = sorted((float(value) for value in scores.values()), reverse=True)
        top = ordered[0]
        second = ordered[1] if len(ordered) > 1 else 0.0
        margin = max(0.0, top - second)
        return float(min(1.0, 0.55 * top + 0.45 * margin))

    @staticmethod
    def _runner_up(scores: Dict[str, float], top_step: str) -> Optional[str]:
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        for step_id, _ in ordered:
            if step_id != top_step:
                return step_id
        return None


class CausalTemporalStepExpert(_BaseTemporalStepExpert):
    """A lightweight causal temporal expert using EMA smoothing plus graph priors."""

    def __init__(
        self,
        steps: List[str],
        timeline: EvidenceTimelineStore,
        graph_prior: CompiledStepGraphPrior,
        vectorizer: TemporalEvidenceVectorizer,
        ema_alpha: float = 0.55,
        enabled: bool = True,
    ) -> None:
        super().__init__(steps=steps, timeline=timeline, graph_prior=graph_prior, enabled=enabled)
        self.vectorizer = vectorizer
        self.ema_alpha = float(ema_alpha)
        self._ema_scores = {step_id: 0.0 for step_id in self.steps}

    def predict(self, payload: object) -> StepPrediction:
        token = payload if isinstance(payload, EvidenceToken) else None
        if token is None or not self.enabled:
            return self._empty_prediction(reason="disabled")
        if not token.has_visual_evidence:
            return self._empty_prediction(reason="no_visual_evidence", prev_step=token.prev_step)

        self.timeline.append(token)
        current_scores = self.vectorizer.bootstrap_scores(token)
        smoothed = {}
        for step_id in self.steps:
            previous = float(self._ema_scores.get(step_id, 0.0))
            current = float(current_scores.get(step_id, 0.0))
            score = self.ema_alpha * current + (1.0 - self.ema_alpha) * previous
            self._ema_scores[step_id] = score
            smoothed[step_id] = score
        return self._finalize_scores(
            smoothed,
            token=token,
            extras={
                "reason": "ema_graph",
                "frame_scores": current_scores,
                "ema_scores": dict(smoothed),
            },
        )

    def reset(self) -> None:
        self.timeline.clear()
        self._ema_scores = {step_id: 0.0 for step_id in self.steps}


class StreamingGRUTemporalStepExpert(_BaseTemporalStepExpert):
    """A streaming GRU expert with hidden-state caching and optional checkpoint loading."""

    def __init__(
        self,
        steps: List[str],
        timeline: EvidenceTimelineStore,
        graph_prior: CompiledStepGraphPrior,
        vectorizer: TemporalEvidenceVectorizer,
        ema_alpha: float = 0.55,
        hidden_size: int = 0,
        device: str = "cpu",
        checkpoint_path: str = "",
        learned_token_aggregation: bool = False,
        token_hidden_size: int = 24,
        token_output_size: int = 48,
        enabled: bool = True,
    ) -> None:
        super().__init__(steps=steps, timeline=timeline, graph_prior=graph_prior, enabled=enabled)
        self.vectorizer = vectorizer
        self.device = self._resolve_device(device)
        checkpoint_config = self._peek_checkpoint_aggregator(Path(checkpoint_path)) if checkpoint_path else None
        if checkpoint_config is not None:
            learned_token_aggregation = True
            token_hidden_size = int(checkpoint_config.get("hidden_size", token_hidden_size))
            token_output_size = int(checkpoint_config.get("output_size", token_output_size))

        self.token_aggregator: Optional[LearnedTokenAggregator] = None
        if learned_token_aggregation:
            self.token_aggregator = LearnedTokenAggregator(
                num_steps=self.vectorizer.num_steps,
                num_components=self.vectorizer.num_components,
                num_relations=self.vectorizer.num_relations,
                num_scalars=self.vectorizer.num_scalars,
                hidden_size=max(4, int(token_hidden_size)),
                output_size=max(4, int(token_output_size)),
            ).to(self.device)

        self.hidden_size = int(hidden_size) if int(hidden_size) > 0 else len(self.steps)
        self.input_size = int(self.token_aggregator.output_size if self.token_aggregator is not None else self.vectorizer.dim)
        self.ema_alpha = float(ema_alpha)
        self.gru = nn.GRUCell(self.input_size, self.hidden_size).to(self.device)
        self.head = nn.Linear(self.hidden_size, len(self.steps)).to(self.device)
        self.hidden: Optional[torch.Tensor] = None
        self.trained = False
        if checkpoint_path:
            self.trained = self._load_checkpoint(Path(checkpoint_path))
        if not self.trained:
            self._bootstrap_initialize()
        self.gru.eval()
        self.head.eval()
        if self.token_aggregator is not None:
            self.token_aggregator.eval()

    def predict(self, payload: object) -> StepPrediction:
        token = payload if isinstance(payload, EvidenceToken) else None
        if token is None or not self.enabled:
            return self._empty_prediction(reason="disabled")
        if not token.has_visual_evidence:
            return self._empty_prediction(reason="no_visual_evidence", prev_step=token.prev_step)

        self.timeline.append(token)
        frame_scores = self.vectorizer.bootstrap_scores(token)
        feature_vector = self.vectorizer.vectorize(token)
        input_tensor = self._encode_vector(feature_vector)
        if self.hidden is None:
            self.hidden = torch.zeros((1, self.hidden_size), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            self.hidden = self.gru(input_tensor, self.hidden)
            logits = self.head(self.hidden).squeeze(0).detach().cpu().numpy().astype(np.float32)

        raw_scores = {step_id: float(logits[index]) for index, step_id in enumerate(self.steps)}
        return self._finalize_scores(
            raw_scores,
            token=token,
            extras={
                "reason": "gru_stream",
                "frame_scores": frame_scores,
                "hidden_norm": float(torch.norm(self.hidden).item()),
                "device": str(self.device),
                "trained": bool(self.trained),
                "learned_token_aggregation": bool(self.token_aggregator is not None),
            },
        )

    def reset(self) -> None:
        self.timeline.clear()
        self.hidden = None

    def _bootstrap_initialize(self) -> None:
        """Initialize the GRU/head pair to behave like a stable EMA over bootstrap frame scores."""

        for parameter in self.gru.parameters():
            nn.init.zeros_(parameter)
        for parameter in self.head.parameters():
            nn.init.zeros_(parameter)

        hidden = self.hidden_size
        steps = len(self.steps)
        alpha = max(1e-3, min(0.999, self.ema_alpha))
        keep = max(1e-3, min(0.999, 1.0 - alpha))

        with torch.no_grad():
            self.gru.bias_ih[:hidden].fill_(4.0)
            self.gru.bias_hh[:hidden].fill_(0.0)
            update_bias = float(math.log(keep / max(1e-6, 1.0 - keep)))
            self.gru.bias_ih[hidden : 2 * hidden].fill_(update_bias)
            self.gru.bias_hh[hidden : 2 * hidden].fill_(0.0)
            rows = min(hidden, steps)
            for index in range(rows):
                self.gru.weight_ih[2 * hidden + index, index] = 1.0
                self.head.weight[index, index] = 1.0

    def _encode_vector(self, feature_vector: np.ndarray) -> torch.Tensor:
        input_tensor = torch.from_numpy(feature_vector).to(self.device).unsqueeze(0)
        if self.token_aggregator is None:
            return input_tensor
        with torch.no_grad():
            return self.token_aggregator(input_tensor)

    def _load_checkpoint(self, path: Path) -> bool:
        if not path.exists():
            return False
        payload = torch.load(path, map_location=self.device)
        if isinstance(payload, dict) and self.token_aggregator is not None and payload.get("aggregator_state_dict"):
            self.token_aggregator.load_state_dict(payload["aggregator_state_dict"])
        if isinstance(payload, dict) and "gru_state_dict" in payload and "head_state_dict" in payload:
            self.gru.load_state_dict(payload["gru_state_dict"])
            self.head.load_state_dict(payload["head_state_dict"])
            return True
        if isinstance(payload, dict) and "model_state_dict" in payload:
            state_dict = dict(payload["model_state_dict"])
            gru_state = {key.replace("gru.", "", 1): value for key, value in state_dict.items() if key.startswith("gru.")}
            head_state = {key.replace("head.", "", 1): value for key, value in state_dict.items() if key.startswith("head.")}
            if gru_state and head_state:
                self.gru.load_state_dict(gru_state)
                self.head.load_state_dict(head_state)
                return True
        return False

    @staticmethod
    def _peek_checkpoint_aggregator(path: Path) -> Optional[Dict[str, int]]:
        if not path.exists():
            return None
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            return None
        config = payload.get("aggregator_config")
        if not isinstance(config, dict):
            return None
        return {str(key): int(value) for key, value in config.items() if isinstance(value, (int, float))}

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        normalized = str(device or "cpu").strip().lower()
        if normalized in {"", "auto"}:
            return torch.device("cpu")
        if normalized == "cpu":
            return torch.device("cpu")
        if normalized.isdigit() and torch.cuda.is_available():
            return torch.device(f"cuda:{normalized}")
        if normalized.startswith("cuda") and torch.cuda.is_available():
            return torch.device(normalized)
        return torch.device("cpu")
