"""File-backed operational state tracking."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from ..types import FeedbackEvent
from .types import OpsSnapshot


class OpsStateTracker:
    """Maintain a compact Doing/Review/Blocked/Next view of the runtime."""

    def __init__(self, enabled: bool = True, persist_history: bool = True) -> None:
        self.enabled = bool(enabled)
        self.persist_history = bool(persist_history)
        self.counts = {
            "reviews_requested": 0,
            "review_approve": 0,
            "review_hold": 0,
            "review_request_human": 0,
            "review_prefer_candidate": 0,
            "feedback_accept": 0,
            "feedback_corrected": 0,
            "reviewer_feedback": 0,
        }
        self.snapshot_path: Optional[Path] = None
        self._history_handle = None
        self._last_signature: tuple[str, int, str, str] | None = None

    def attach_run(self, run_dir: Path) -> None:
        """Bind the tracker to a concrete run directory."""

        self.close()
        if not self.enabled:
            return
        run_dir = Path(run_dir)
        self.snapshot_path = run_dir / "ops_state.json"
        if self.persist_history:
            self._history_handle = (run_dir / "ops_state_history.jsonl").open("w", encoding="utf-8")

    def record_review(self, action: str) -> None:
        """Increment review counters."""

        if not self.enabled:
            return
        self.counts["reviews_requested"] += 1
        key = f"review_{str(action).strip().lower()}"
        if key in self.counts:
            self.counts[key] += 1

    def record_feedback(self, feedback: FeedbackEvent) -> None:
        """Increment feedback counters."""

        if not self.enabled:
            return
        if feedback.accepted:
            self.counts["feedback_accept"] += 1
        else:
            self.counts["feedback_corrected"] += 1
        if str(feedback.source).strip().lower() == "reviewer":
            self.counts["reviewer_feedback"] += 1

    def update(
        self,
        status: str,
        frame_index: int,
        step_id: str,
        reason: str,
        review_action: str = "",
        extras: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> None:
        """Persist the latest compact ops-state snapshot."""

        if not self.enabled or self.snapshot_path is None:
            return
        snapshot = OpsSnapshot(
            status=str(status),
            frame_index=int(frame_index),
            step_id=str(step_id),
            reason=str(reason),
            review_action=str(review_action),
            counts=dict(self.counts),
            extras=dict(extras or {}),
        )
        signature = (snapshot.status, snapshot.frame_index, snapshot.step_id, snapshot.reason)
        if not force and signature == self._last_signature:
            return
        self.snapshot_path.write_text(json.dumps(asdict(snapshot), indent=2), encoding="utf-8")
        if self._history_handle is not None:
            self._history_handle.write(json.dumps(asdict(snapshot)) + "\n")
            self._history_handle.flush()
        self._last_signature = signature

    def close(self) -> None:
        """Flush and close the state history."""

        if self._history_handle is not None:
            self._history_handle.close()
            self._history_handle = None
