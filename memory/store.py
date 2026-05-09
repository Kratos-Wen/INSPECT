"""Persistent and in-memory stores for episodic memory records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from .types import MemoryRecord


def _cosine_similarity(first: List[float], second: List[float]) -> float:
    a = np.asarray(first, dtype=np.float32)
    b = np.asarray(second, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _serialize_record(record: MemoryRecord) -> dict[str, object]:
    return {
        "record_id": record.record_id,
        "run_id": record.run_id,
        "frame_index": record.frame_index,
        "timestamp": record.timestamp,
        "step_id": record.step_id,
        "prev_step": record.prev_step,
        "source": record.source,
        "trust": record.trust,
        "accepted": record.accepted,
        "note": record.note,
        "signature": [list(item) for item in record.signature],
        "relevant_signature": [list(item) for item in record.relevant_signature],
        "relation_signature": [list(item) for item in record.relation_signature],
        "geometry_stats": record.geometry_stats,
        "scene_graph_stats": record.scene_graph_stats,
        "expert_steps": record.expert_steps,
        "expert_confidences": record.expert_confidences,
        "expert_scores": record.expert_scores,
        "num_detections": record.num_detections,
        "num_relevant": record.num_relevant,
        "has_visual_evidence": record.has_visual_evidence,
        "visual_embedding": record.visual_embedding,
        "vector": record.vector,
        "tokens": record.tokens,
    }


def _deserialize_record(payload: dict[str, object]) -> MemoryRecord:
    return MemoryRecord(
        record_id=str(payload.get("record_id", "")),
        run_id=str(payload.get("run_id", "")),
        frame_index=int(payload.get("frame_index", 0)),
        timestamp=float(payload.get("timestamp", 0.0)),
        step_id=str(payload.get("step_id", "")),
        prev_step=str(payload.get("prev_step")) if payload.get("prev_step") is not None else None,
        source=str(payload.get("source", "")),
        trust=float(payload.get("trust", 0.0)),
        accepted=bool(payload.get("accepted", False)),
        note=str(payload.get("note", "")),
        signature=tuple((str(name), int(count)) for name, count in payload.get("signature", [])),
        relevant_signature=tuple((str(name), int(count)) for name, count in payload.get("relevant_signature", [])),
        relation_signature=tuple((str(name), int(count)) for name, count in payload.get("relation_signature", [])),
        geometry_stats={str(key): float(value) for key, value in (payload.get("geometry_stats", {}) or {}).items()},
        scene_graph_stats={str(key): float(value) for key, value in (payload.get("scene_graph_stats", {}) or {}).items()},
        expert_steps={str(key): str(value) for key, value in (payload.get("expert_steps", {}) or {}).items()},
        expert_confidences={
            str(key): float(value) for key, value in (payload.get("expert_confidences", {}) or {}).items()
        },
        expert_scores={
            str(expert): {str(step_id): float(score) for step_id, score in (scores or {}).items()}
            for expert, scores in (payload.get("expert_scores", {}) or {}).items()
        },
        num_detections=int(payload.get("num_detections", 0)),
        num_relevant=int(payload.get("num_relevant", 0)),
        has_visual_evidence=bool(payload.get("has_visual_evidence", False)),
        visual_embedding=[float(value) for value in (payload.get("visual_embedding", []) or [])],
        vector=[float(value) for value in (payload.get("vector", []) or [])],
        tokens=[str(value) for value in (payload.get("tokens", []) or [])],
    )


class InMemoryEventStore:
    """Simple in-memory store used for fast per-run memories."""

    def __init__(self, dedup_similarity: float = 0.97) -> None:
        self.dedup_similarity = float(dedup_similarity)
        self._records: List[MemoryRecord] = []

    def records(self) -> List[MemoryRecord]:
        return list(self._records)

    def append(self, record: MemoryRecord) -> bool:
        if self._is_duplicate(record):
            return False
        self._records.append(record)
        return True

    def count(self) -> int:
        return len(self._records)

    def close(self) -> None:
        return None

    def _is_duplicate(self, record: MemoryRecord) -> bool:
        for candidate in reversed(self._records[-256:]):
            if candidate.step_id != record.step_id:
                continue
            if candidate.prev_step != record.prev_step:
                continue
            if candidate.source != record.source:
                continue
            if _cosine_similarity(candidate.vector, record.vector) >= self.dedup_similarity:
                return True
        return False


class JsonlEventStore(InMemoryEventStore):
    """JSONL-backed store that keeps records in memory and on disk."""

    def __init__(self, path: Path, dedup_similarity: float = 0.97) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(dedup_similarity=dedup_similarity)
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    self._records.append(_deserialize_record(json.loads(line)))
        self._handle = self.path.open("a", encoding="utf-8")

    def append(self, record: MemoryRecord) -> bool:
        if not super().append(record):
            return False
        self._handle.write(json.dumps(_serialize_record(record)) + "\n")
        self._handle.flush()
        return True

    def close(self) -> None:
        self._handle.close()
