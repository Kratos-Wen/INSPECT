"""StateSpec loading and evidence decomposition for INSPECT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from .types import StateSpec


def _load_structured(path: Path) -> object:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return {}
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional runtime package
            raise RuntimeError("YAML StateSpec files require PyYAML. Use JSON or install PyYAML.") from exc
        return yaml.safe_load(text)
    return json.loads(text)


def load_state_specs(path: Path) -> Dict[str, StateSpec]:
    """Load StateSpec records keyed by nominal step and state id."""

    raw = _load_structured(Path(path))
    if isinstance(raw, dict) and "states" in raw:
        items = raw.get("states") or []
    elif isinstance(raw, dict):
        items = []
        for key, value in raw.items():
            if isinstance(value, dict):
                payload = dict(value)
                payload.setdefault("state_id", key)
                items.append(payload)
    elif isinstance(raw, list):
        items = raw
    else:
        items = []

    specs: Dict[str, StateSpec] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        spec = StateSpec.from_dict(dict(item))
        if not spec.state_id:
            continue
        specs[spec.state_id.upper()] = spec
        if spec.nominal_step:
            specs[spec.nominal_step.upper()] = spec
    return specs


def state_spec_for(
    specs: Optional[Dict[str, StateSpec]],
    predicted_state: str,
    verified_state: str = "",
) -> Optional[StateSpec]:
    if not specs:
        return None
    for key in (verified_state, predicted_state):
        if key and key.upper() in specs:
            return specs[key.upper()]
    return None


def _spec_set(values: Iterable[str]) -> Set[str]:
    return {str(item).strip() for item in values if str(item).strip()}


def _looks_like_interaction(key: str) -> bool:
    return (
        key.startswith("hand:")
        or key.startswith("contact:")
        or key.startswith("tool_contact:")
        or key.startswith("interaction:")
        or "hand_object" in key
        or "contact_phase" in key
    )


def _looks_like_transition(key: str) -> bool:
    return key.startswith("transition:") or key.startswith("memory_match:") or key.startswith("temporal:")


def _looks_like_postcondition(key: str) -> bool:
    return key.startswith("relation:")


def _looks_like_precondition(key: str) -> bool:
    return key.startswith("object:") or key.startswith("focus_object:") or key.startswith("visual:")


def decompose_evidence(
    observed_evidence: Iterable[str],
    missing_evidence: Iterable[str],
    failure_cues: Iterable[str],
    spec: Optional[StateSpec] = None,
) -> Dict[str, List[str]]:
    """Split raw evidence keys into verification-level evidence groups."""

    observed = [str(item) for item in observed_evidence]
    missing = [str(item) for item in missing_evidence]
    failures = [str(item) for item in failure_cues]
    preconditions: Set[str] = set()
    interactions: Set[str] = set()
    transitions: Set[str] = set()
    postconditions: Set[str] = set()
    negatives: Set[str] = set()
    admissibility: Set[str] = set()

    spec_pre = _spec_set(spec.preconditions if spec else [])
    spec_inter = _spec_set(spec.interaction_evidence if spec else [])
    spec_trans = _spec_set(spec.transition_evidence if spec else [])
    spec_post = _spec_set(spec.postcondition_evidence if spec else [])
    spec_neg = _spec_set(spec.negative_evidence if spec else [])
    spec_adm = _spec_set(spec.admissibility_evidence if spec else [])

    for key in observed:
        if key in spec_pre:
            preconditions.add(key)
        if key in spec_inter:
            interactions.add(key)
        if key in spec_trans:
            transitions.add(key)
        if key in spec_post:
            postconditions.add(key)
        if key in spec_neg:
            negatives.add(key)
        if key in spec_adm:
            admissibility.add(key)
        if not spec:
            if _looks_like_precondition(key):
                preconditions.add(key)
            elif _looks_like_interaction(key):
                interactions.add(key)
            elif _looks_like_transition(key):
                transitions.add(key)
            elif _looks_like_postcondition(key):
                postconditions.add(key)

    for key in failures:
        if f"failure:{key}" in spec_neg:
            negatives.add(f"failure:{key}")

    for key in missing:
        if key in spec_post:
            postconditions.discard(key)
        if key in spec_neg:
            negatives.add(key)

    return {
        "precondition_evidence": sorted(preconditions),
        "interaction_evidence": sorted(interactions),
        "transition_evidence": sorted(transitions),
        "postcondition_evidence": sorted(postconditions),
        "negative_evidence": sorted(negatives),
        "admissibility_evidence": sorted(admissibility),
    }
