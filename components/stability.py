"""Stability policies for clean online feedback events."""

from __future__ import annotations

from collections import Counter
from typing import List, Tuple

from ..types import Detection


def detection_signature(detections: List[Detection]) -> Tuple[Tuple[str, int], ...]:
    """Return a canonical signature from fused detections."""

    counts = Counter(detection.name for detection in detections)
    return tuple(sorted(counts.items()))


class DualStabilityTracker:
    """Require both stable detections and stable step predictions."""

    def __init__(self, stable_n: int = 5, require_same_step: int = 2) -> None:
        self.stable_n = max(1, int(stable_n))
        self.require_same_step = max(1, int(require_same_step))
        self.prev_signature: Tuple[Tuple[str, int], ...] | None = None
        self.prev_step: str | None = None
        self.signature_streak = 0
        self.step_streak = 0

    def update(self, detections: List[Detection], fused_step: str) -> Tuple[bool, Tuple[Tuple[str, int], ...]]:
        """Update the stability counters and report whether feedback is allowed."""

        signature = detection_signature(detections)
        if signature and signature == self.prev_signature:
            self.signature_streak += 1
        else:
            self.signature_streak = 1 if signature else 0

        step_id = str(fused_step).strip().upper()
        if step_id and step_id == self.prev_step:
            self.step_streak += 1
        else:
            self.step_streak = 1 if step_id else 0

        self.prev_signature = signature
        self.prev_step = step_id
        is_stable = self.signature_streak >= self.stable_n and self.step_streak >= self.require_same_step
        return is_stable, signature
