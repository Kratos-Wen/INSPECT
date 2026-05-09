"""Build a dense step prior from retrieved memory matches."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np

from ..types import StepPrediction
from .types import MemoryMatch


class WeightedMemoryPrior:
    """Aggregate retrieved memories into a step prior."""

    def __init__(
        self,
        steps: List[str],
        corrected_source_gain: float = 1.15,
        accepted_source_gain: float = 1.0,
        auto_source_gain: float = 0.85,
    ) -> None:
        self.steps = [str(step).strip().upper() for step in steps]
        self.corrected_source_gain = float(corrected_source_gain)
        self.accepted_source_gain = float(accepted_source_gain)
        self.auto_source_gain = float(auto_source_gain)

    def build(
        self,
        matches: List[MemoryMatch],
        fallback_step: str,
        reason: str,
    ) -> StepPrediction:
        """Convert retrieved events into a step-wise score vector."""

        raw_scores: Dict[str, float] = {step_id: 0.0 for step_id in self.steps}
        if not matches:
            return StepPrediction(
                step_id=fallback_step,
                confidence=0.0,
                scores=raw_scores,
                extras={"active": False, "reason": reason, "matches": []},
            )

        for match in matches:
            step_id = str(match.record.step_id).strip().upper()
            if step_id not in raw_scores:
                continue
            raw_scores[step_id] += float(match.total_score) * self._source_gain(match.record.source)

        high = max(raw_scores.values()) if raw_scores else 0.0
        normalized = {step_id: (float(value / high) if high > 0.0 else 0.0) for step_id, value in raw_scores.items()}
        top_step = max(normalized, key=normalized.get) if normalized else fallback_step

        ordered = sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)
        logits = np.array([score for _, score in ordered], dtype=np.float32)
        if logits.size > 0:
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs = probs / max(1e-6, float(probs.sum()))
            confidence = float(probs[0])
        else:
            confidence = 0.0
        source_counts = Counter(match.store_name for match in matches)

        return StepPrediction(
            step_id=top_step,
            confidence=confidence,
            scores=normalized,
            extras={
                "active": True,
                "reason": reason,
                "source_counts": dict(source_counts),
                "matches": [
                    {
                        "record_id": match.record.record_id,
                        "store": match.store_name,
                        "step_id": match.record.step_id,
                        "source": match.record.source,
                        "score": match.total_score,
                        "vector_score": match.vector_score,
                        "token_score": match.token_score,
                    }
                    for match in matches
                ],
            },
        )

    def _source_gain(self, source: str) -> float:
        normalized = str(source).strip().lower()
        if "correct" in normalized:
            return self.corrected_source_gain
        if "accept" in normalized:
            return self.accepted_source_gain
        return self.auto_source_gain
