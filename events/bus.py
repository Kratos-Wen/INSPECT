"""JSONL-backed sparse event bus."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .types import PipelineEvent


class JsonlEventBus:
    """Persist sparse runtime events without introducing message-bus overhead."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._handle = None
        self._path: Optional[Path] = None

    def attach_run(self, run_dir: Path) -> None:
        """Bind the event bus to one concrete run directory."""

        self.close()
        if not self.enabled:
            return
        self._path = Path(run_dir) / "events.jsonl"
        self._handle = self._path.open("w", encoding="utf-8")

    def emit(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        frame_index: Optional[int] = None,
    ) -> None:
        """Write one sparse event."""

        if not self.enabled or self._handle is None:
            return
        record = PipelineEvent(
            event_type=str(event_type),
            timestamp=time.time(),
            frame_index=int(frame_index) if frame_index is not None else None,
            payload=dict(payload or {}),
        )
        self._handle.write(json.dumps(asdict(record)) + "\n")
        self._handle.flush()

    def close(self) -> None:
        """Close the active event stream, if any."""

        if self._handle is not None:
            self._handle.close()
            self._handle = None
