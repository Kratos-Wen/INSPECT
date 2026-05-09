"""Similarity-based retrieval over structured episodic memories."""

from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Set

import numpy as np

from .types import MemoryMatch, MemoryObservation


def _cosine_similarity(query: np.ndarray, vector: List[float]) -> float:
    candidate = np.asarray(vector, dtype=np.float32)
    denom = float(np.linalg.norm(query) * np.linalg.norm(candidate))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(query, candidate) / denom)


def _token_overlap(query_tokens: Set[str], candidate_tokens: Iterable[str]) -> float:
    candidate = set(candidate_tokens)
    if not query_tokens and not candidate:
        return 0.0
    return float(len(query_tokens & candidate)) / float(max(1, len(query_tokens | candidate)))


class EventMemoryRetriever:
    """Retrieve similar events across session and long-term stores."""

    def __init__(
        self,
        topk: int = 6,
        max_per_source: int = 3,
        vector_weight: float = 0.72,
        token_weight: float = 0.20,
        prev_step_bonus: float = 0.08,
        recency_half_life_sec: float = 1800.0,
        source_weights: Dict[str, float] | None = None,
    ) -> None:
        self.topk = max(1, int(topk))
        self.max_per_source = max(1, int(max_per_source))
        self.vector_weight = float(vector_weight)
        self.token_weight = float(token_weight)
        self.prev_step_bonus = float(prev_step_bonus)
        self.recency_half_life_sec = float(recency_half_life_sec)
        self.source_weights = {
            "session": 1.0,
            "long_term": 0.85,
            **(source_weights or {}),
        }

    def search(
        self,
        observation: MemoryObservation,
        query_vector: np.ndarray,
        query_tokens: List[str],
        stores: Dict[str, object],
    ) -> List[MemoryMatch]:
        """Retrieve diversified matches across the provided stores."""

        now = time.time()
        token_set = set(query_tokens)
        per_store: Dict[str, List[MemoryMatch]] = {}

        for store_name, store in stores.items():
            if store is None:
                continue
            matches: List[MemoryMatch] = []
            for record in getattr(store, "records")():
                vector_score = max(0.0, _cosine_similarity(query_vector, record.vector))
                token_score = _token_overlap(token_set, record.tokens)
                prev_bonus = self.prev_step_bonus if observation.prev_step and record.prev_step == observation.prev_step else 0.0
                recency_weight = 1.0
                if store_name != "session" and self.recency_half_life_sec > 0.0:
                    age = max(0.0, now - float(record.timestamp))
                    recency_weight = math.exp(-math.log(2.0) * age / self.recency_half_life_sec)
                total = max(0.0, self.vector_weight * vector_score + self.token_weight * token_score + prev_bonus)
                total *= float(self.source_weights.get(store_name, 1.0)) * max(0.05, float(record.trust)) * recency_weight
                if total <= 0.0:
                    continue
                matches.append(
                    MemoryMatch(
                        record=record,
                        store_name=store_name,
                        total_score=float(total),
                        vector_score=float(vector_score),
                        token_score=float(token_score),
                        prev_step_bonus=float(prev_bonus),
                        recency_weight=float(recency_weight),
                    )
                )
            matches.sort(key=lambda item: item.total_score, reverse=True)
            per_store[store_name] = matches[: max(self.max_per_source * 2, self.topk)]

        return self._diversify(per_store)

    def _diversify(self, per_store: Dict[str, List[MemoryMatch]]) -> List[MemoryMatch]:
        diversified: List[MemoryMatch] = []
        active = {name: list(matches) for name, matches in per_store.items() if matches}
        while len(diversified) < self.topk and active:
            ordered_sources = sorted(
                active,
                key=lambda name: active[name][0].total_score if active[name] else 0.0,
                reverse=True,
            )
            emitted = False
            for store_name in ordered_sources:
                matches = active.get(store_name, [])
                if not matches:
                    continue
                diversified.append(matches.pop(0))
                emitted = True
                if len(diversified) >= self.topk:
                    break
                if len([item for item in diversified if item.store_name == store_name]) >= self.max_per_source:
                    active.pop(store_name, None)
                elif not matches:
                    active.pop(store_name, None)
            if not emitted:
                break
        return diversified[: self.topk]
