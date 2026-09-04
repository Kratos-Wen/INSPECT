"""Bounded timeline storage for streaming temporal evidence."""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

from ..core_types import EvidenceToken


class EvidenceTimelineStore:
    """Maintain a small bounded history of evidence tokens for causal temporal reasoning."""

    def __init__(self, maxlen: int = 12) -> None:
        self.maxlen = max(1, int(maxlen))
        self._tokens: Deque[EvidenceToken] = deque(maxlen=self.maxlen)

    def append(self, token: EvidenceToken) -> None:
        self._tokens.append(token)

    def clear(self) -> None:
        self._tokens.clear()

    def recent(self) -> List[EvidenceToken]:
        return list(self._tokens)

    def __len__(self) -> int:
        return len(self._tokens)

    def last(self) -> Optional[EvidenceToken]:
        return self._tokens[-1] if self._tokens else None

    def step_support(self, step_id: str) -> float:
        """Return average temporal support for a step across the recent window."""

        normalized = str(step_id).strip().upper()
        if not normalized or not self._tokens:
            return 0.0
        total = 0.0
        weight_sum = 0.0
        for index, token in enumerate(self._tokens, start=1):
            weight = float(index) / float(len(self._tokens))
            score = max(
                float(token.state_scores.get(normalized, 0.0)),
                float(token.retrieval_scores.get(normalized, 0.0)),
                float(token.memory_scores.get(normalized, 0.0)),
            )
            total += weight * score
            weight_sum += weight
        return total / max(1e-6, weight_sum)

    def consecutive_prev_step(self, step_id: str) -> int:
        """Count how many recent tokens report the given step as the previous confirmed step."""

        normalized = str(step_id).strip().upper()
        if not normalized:
            return 0
        count = 0
        for token in reversed(self._tokens):
            if str(token.prev_step or "").strip().upper() != normalized:
                break
            count += 1
        return count

    def mean_object_count(self, component_name: str) -> float:
        """Return the mean visible count for a component over the stored window."""

        normalized = str(component_name).strip().lower()
        if not normalized or not self._tokens:
            return 0.0
        total = sum(float(token.visible_counts.get(normalized, 0)) for token in self._tokens)
        return total / float(len(self._tokens))

    def mean_relation_count(self, predicate: str) -> float:
        """Return the mean count of one relation predicate across the stored window."""

        normalized = str(predicate).strip().lower()
        if not normalized or not self._tokens:
            return 0.0
        total = sum(float(token.relation_counts.get(normalized, 0)) for token in self._tokens)
        return total / float(len(self._tokens))

    def latest_review_action(self) -> str:
        token = self.last()
        return str(token.review_action).strip().lower() if token is not None else ""
