"""Structured logging for the modular step pipeline."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict


class JsonlCsvLogger:
    """Write JSONL diagnostics and a compact CSV summary."""

    def __init__(self, save_root: str, video_path: Path) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{video_path.stem}_{timestamp}"
        self._run_dir = Path(save_root) / run_name
        self._run_dir.mkdir(parents=True, exist_ok=True)

        self.iterations_file = (self._run_dir / "iterations.jsonl").open("w", encoding="utf-8")
        self.feedback_file = (self._run_dir / "feedback.jsonl").open("w", encoding="utf-8")
        self.summary_csv = (self._run_dir / "summary.csv").open("w", encoding="utf-8", newline="")
        self.summary_writer = csv.writer(self.summary_csv)
        self.summary_writer.writerow(
            [
                "iter",
                "frame_index",
                "num_raw",
                "num_fused",
                "num_relevant",
                "stable",
                "state_step",
                "state_conf",
                "retrieval_step",
                "retrieval_conf",
                "memory_step",
                "memory_conf",
                "fused_step",
                "fused_conf",
                "gate_state",
                "gate_retrieval",
                "gate_memory",
            ]
        )

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def log_meta(self, payload: Dict[str, object]) -> None:
        """Persist run-level metadata."""

        (self._run_dir / "meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def log_iteration(self, payload: Dict[str, object]) -> None:
        """Write one iteration record to JSONL and CSV."""

        self.iterations_file.write(json.dumps(payload) + "\n")
        self.iterations_file.flush()
        self.summary_writer.writerow(
            [
                payload.get("iter"),
                payload.get("frame_index"),
                payload.get("num_raw"),
                payload.get("num_fused"),
                payload.get("num_relevant"),
                int(bool(payload.get("stable"))),
                payload.get("state_step"),
                f"{float(payload.get('state_conf', 0.0)):.4f}",
                payload.get("retrieval_step"),
                f"{float(payload.get('retrieval_conf', 0.0)):.4f}",
                payload.get("memory_step"),
                f"{float(payload.get('memory_conf', 0.0)):.4f}",
                payload.get("fused_step"),
                f"{float(payload.get('fused_conf', 0.0)):.4f}",
                f"{float(payload.get('gate_state', 0.0)):.4f}",
                f"{float(payload.get('gate_retrieval', 0.0)):.4f}",
                f"{float(payload.get('gate_memory', 0.0)):.4f}",
            ]
        )
        self.summary_csv.flush()

    def log_feedback(self, payload: Dict[str, object]) -> None:
        """Write one feedback event."""

        self.feedback_file.write(json.dumps(payload) + "\n")
        self.feedback_file.flush()

    def close(self) -> None:
        """Close open file handles."""

        self.iterations_file.close()
        self.feedback_file.close()
        self.summary_csv.close()
