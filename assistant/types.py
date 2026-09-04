"""Typed payloads for the contextual live assistant."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class AssistantSnapshot:
    """Current perception state exposed to the contextual assistant."""

    frame_index: int
    step_id: str
    step_confidence: float
    runner_up: Optional[str]
    visible_objects: List[str]
    relevant_objects: List[str]
    object_counts: Dict[str, int]
    scene_relations: List[str]
    memory_step: str
    memory_confidence: float
    relation_facts: List[Dict[str, str]] = field(default_factory=list)
    recent_steps: List["StepTimelineEvent"] = field(default_factory=list)
    recent_feedback: List["FeedbackTimelineEvent"] = field(default_factory=list)
    memory_matches: List["MemoryMatchSummary"] = field(default_factory=list)
    memory_reason: str = ""
    memory_recalled: bool = False
    review_action: str = ""
    review_reason: str = ""
    proposed_step: str = ""
    active_claim: str = ""
    claim_state: str = "insufficient"
    claim_support: float = 0.0
    claim_contradiction: float = 0.0
    claim_margin: float = 0.0
    claim_admissible: bool = True
    missing_evidence_roles: List[str] = field(default_factory=list)
    product_family: str = ""
    acquisition_mode: str = "monitor"
    external_observation_recommended: bool = False


@dataclass(frozen=True)
class StepTimelineEvent:
    """One recent confirmed or accepted step transition."""

    frame_index: int
    step_id: str
    confidence: float
    source: str
    stable: bool = False


@dataclass(frozen=True)
class FeedbackTimelineEvent:
    """One recent explicit or weak supervision event."""

    frame_index: int
    label: str
    source: str
    accepted: bool = False
    note: str = ""


@dataclass(frozen=True)
class MemoryMatchSummary:
    """Compact summary of one recalled memory match."""

    step_id: str
    store: str
    source: str
    score: float


@dataclass(frozen=True)
class ParsedAssistantQuery:
    """Intent and slot candidates extracted from a natural-language query."""

    text: str
    intent: str
    tokens: List[str] = field(default_factory=list)
    target_component_hint: str = ""
    target_step_hint: str = ""
    relation_type: str = ""
    asks_reason: bool = False
    asks_next: bool = False


@dataclass(frozen=True)
class ResolvedAssistantQuery:
    """A query after resolving components and applying lightweight dialogue context."""

    text: str
    intent: str
    tokens: List[str] = field(default_factory=list)
    target_component: str = ""
    target_step: str = ""
    relation_type: str = ""
    asks_reason: bool = False
    clarification: str = ""
    can_answer: bool = True
    confidence: float = 0.0
    used_history: bool = False


@dataclass(frozen=True)
class AssistantEvidence:
    """Grounded evidence selected for one resolved query."""

    intent: str
    facts: Dict[str, object] = field(default_factory=dict)
    grounded: bool = True
    note: str = ""


@dataclass(frozen=True)
class PolicyDecision:
    """High-level answer policy for a grounded query."""

    action: str
    message: str = ""
    style: str = "concise"


@dataclass(frozen=True)
class AssistantReply:
    """One assistant answer to a live voice or text query."""

    query: str
    answer: str
    route: str
    status: str = "answer"
    evidence: Dict[str, object] = field(default_factory=dict)
