"""Knowledge-base helpers for the modular step pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from ..types import Detection


class KnowledgeBase:
    """Knowledge-base wrapper with canonicalization helpers."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self.payload = payload
        self.component_records = self._build_component_records()
        self.alias_map = self._build_alias_map()

    @classmethod
    def from_path(cls, path: str | Path) -> "KnowledgeBase":
        """Load a JSON knowledge base from disk."""

        kb_path = Path(path)
        if not kb_path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {kb_path}")
        payload = json.loads(kb_path.read_text(encoding="utf-8"))
        return cls(payload)

    def workflow_steps(self, fallback_steps: Sequence[str]) -> List[str]:
        """Return workflow step ids or the provided fallback list."""

        workflow = self.payload.get("workflow", []) or []
        steps = [str(item.get("id", "")).strip().upper() for item in workflow if item.get("id")]
        return steps or [str(step).strip().upper() for step in fallback_steps]

    def requirements_for(self, step_id: str) -> Dict[str, Any]:
        """Return the requirement object for a given workflow step."""

        target = str(step_id).strip().upper()
        for item in self.payload.get("workflow", []) or []:
            if str(item.get("id", "")).strip().upper() == target:
                return item.get("requires", {}) or {}
        return {}

    def component_names(self) -> List[str]:
        """Return canonical component names sorted for feature encoding."""

        names = set(self.component_records.keys()) or set(self.alias_map.values())
        return sorted(name for name in names if name)

    def component_record(self, name: str) -> Dict[str, Any]:
        """Return one component record for the requested canonical or alias name."""

        canonical = self.alias_map.get(str(name).strip().lower(), str(name).strip().lower())
        return dict(self.component_records.get(canonical, {}))

    def component_display_name(self, name: str) -> str:
        """Return the human-readable display name for a component."""

        canonical = self.alias_map.get(str(name).strip().lower(), str(name).strip().lower())
        record = self.component_records.get(canonical, {})
        return str(record.get("name", canonical)).strip() or canonical

    def next_step(self, step_id: str) -> str:
        """Return the next workflow step, or the current one if it is terminal."""

        steps = self.workflow_steps([])
        current = str(step_id).strip().upper()
        if current not in steps:
            return current
        index = steps.index(current)
        return steps[min(len(steps) - 1, index + 1)] if steps else current

    def workflow_entry(self, step_id: str) -> Dict[str, Any]:
        """Return the full workflow entry for a step identifier."""

        target = str(step_id).strip().upper()
        for item in self.payload.get("workflow", []) or []:
            if str(item.get("id", "")).strip().upper() == target:
                return dict(item)
        return {}

    def canonicalize(self, detections: Iterable[Detection]) -> List[Detection]:
        """Map aliases to canonical names for each detection."""

        normalized: List[Detection] = []
        for detection in detections:
            name = self.alias_map.get(detection.name.lower(), detection.name.lower())
            normalized.append(
                Detection(
                    name=name,
                    xyxy=detection.xyxy,
                    confidence=detection.confidence,
                    meta=dict(detection.meta),
                )
            )
        return normalized

    def _build_alias_map(self) -> Dict[str, str]:
        aliases = self.payload.get("aliases", {}) or {}
        mapping: Dict[str, str] = {name: name for name in self.component_records.keys()}
        for canonical, values in aliases.items():
            canonical_name = str(canonical).strip().lower()
            mapping[canonical_name] = canonical_name
            for value in values or []:
                mapping[str(value).strip().lower()] = canonical_name
        return mapping

    def _build_component_records(self) -> Dict[str, Dict[str, Any]]:
        records: Dict[str, Dict[str, Any]] = {}
        for component in self.payload.get("components", []) or []:
            name = str(component.get("name", "")).strip().lower()
            if name:
                records[name] = dict(component)

        if records:
            return records

        for key, value in self.payload.items():
            if str(key).strip().lower() in {"workflow", "components", "aliases"}:
                continue
            if isinstance(value, dict):
                canonical = str(key).strip().lower()
                record = dict(value)
                record.setdefault("name", str(key).strip())
                records[canonical] = record
        return records
