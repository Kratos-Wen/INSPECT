"""Compile lightweight step-graph priors from the workflow JSON."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..types import EvidenceToken
from .kb import KnowledgeBase
from .timeline_store import EvidenceTimelineStore


@dataclass(frozen=True)
class StepVariantRule:
    """Compiled one-variant rule set for a workflow step."""

    all_of: Dict[str, int]
    any_of: Dict[str, int]
    forbid: Tuple[str, ...]
    expected_relations: Tuple[Tuple[str, str, str], ...] = ()
    forbidden_relations: Tuple[Tuple[str, str, str], ...] = ()


class CompiledStepGraphPrior:
    """Apply lightweight step-graph biases to causal temporal step scores."""

    def __init__(
        self,
        kb: KnowledgeBase,
        steps: Sequence[str],
        transitions: Optional[Dict[str, List[str]]] = None,
        transition_penalty: float = 0.18,
        requirement_bonus: float = 0.10,
        requirement_penalty: float = 0.12,
        forbid_penalty: float = 0.16,
        relation_bonus: float = 0.04,
        relation_penalty: float = 0.06,
    ) -> None:
        self.kb = kb
        self.steps = [str(step).strip().upper() for step in steps]
        self.transitions = {
            str(key).strip().upper(): [str(value).strip().upper() for value in values]
            for key, values in (transitions or {}).items()
        }
        self.transition_penalty = float(transition_penalty)
        self.requirement_bonus = float(requirement_bonus)
        self.requirement_penalty = float(requirement_penalty)
        self.forbid_penalty = float(forbid_penalty)
        self.relation_bonus = float(relation_bonus)
        self.relation_penalty = float(relation_penalty)
        self.rules = {step_id: self._compile_step(step_id) for step_id in self.steps}

    def bias_for(
        self,
        step_id: str,
        token: EvidenceToken,
        timeline: EvidenceTimelineStore,
        prev_step: Optional[str] = None,
    ) -> float:
        """Return a small bias for one candidate step given current evidence and recent history."""

        normalized = str(step_id).strip().upper()
        bias = 0.0
        previous = str(prev_step or token.prev_step or "").strip().upper()
        if previous and self.transitions:
            allowed = set(self.transitions.get(previous, []))
            if allowed and normalized not in allowed:
                bias -= self.transition_penalty

        variants = self.rules.get(normalized, [])
        if variants:
            bias += max(self._variant_bias(rule, token) for rule in variants)

        # Small persistence bonus for temporally consistent support.
        bias += 0.05 * float(timeline.step_support(normalized))

        # Be conservative after an explicit review hold/request_human.
        review_action = timeline.latest_review_action()
        if review_action in {"hold", "request_human"} and normalized != previous:
            bias -= 0.04
        return bias

    def apply(
        self,
        scores: Dict[str, float],
        token: EvidenceToken,
        timeline: EvidenceTimelineStore,
        prev_step: Optional[str] = None,
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        """Apply graph biases to all step scores and return adjusted scores plus raw bias terms."""

        adjusted = {str(step_id).strip().upper(): float(value) for step_id, value in scores.items()}
        bias_terms: Dict[str, float] = {}
        for step_id in self.steps:
            bias = self.bias_for(step_id, token=token, timeline=timeline, prev_step=prev_step)
            adjusted[step_id] = float(adjusted.get(step_id, 0.0) + bias)
            bias_terms[step_id] = float(bias)
        return adjusted, bias_terms

    def _compile_step(self, step_id: str) -> List[StepVariantRule]:
        workflow_entry = self.kb.workflow_entry(step_id)
        requires = dict(workflow_entry.get("requires") or {})
        variants = list(requires.get("variants") or [])
        compiled: List[StepVariantRule] = []
        for variant in variants:
            compiled.append(
                StepVariantRule(
                    all_of={str(key).strip().lower(): int(value) for key, value in dict(variant.get("all_of") or {}).items()},
                    any_of={str(key).strip().lower(): int(value) for key, value in dict(variant.get("any_of") or {}).items()},
                    forbid=tuple(str(item).strip().lower() for item in list(variant.get("forbid") or []) if str(item).strip()),
                    expected_relations=self._compile_relations(variant.get("expected_relations") or workflow_entry.get("expected_relations") or []),
                    forbidden_relations=self._compile_relations(variant.get("forbidden_relations") or workflow_entry.get("forbidden_relations") or []),
                )
            )
        return compiled

    def _variant_bias(self, rule: StepVariantRule, token: EvidenceToken) -> float:
        visible = token.visible_counts
        relation_facts = set(token.relation_facts)

        matched = 0
        missing = 0
        for name, count in rule.all_of.items():
            if int(visible.get(name, 0)) >= int(count):
                matched += 1
            else:
                missing += 1
        total_all = max(1, len(rule.all_of))
        matched_ratio = float(matched) / float(total_all)
        missing_ratio = float(missing) / float(total_all)

        any_ratio = 0.0
        if rule.any_of:
            any_hits = sum(1 for name, count in rule.any_of.items() if int(visible.get(name, 0)) >= int(count))
            any_ratio = float(any_hits) / float(max(1, len(rule.any_of)))

        forbid_hits = sum(1 for name in rule.forbid if int(visible.get(name, 0)) > 0)
        forbid_ratio = float(forbid_hits) / float(max(1, len(rule.forbid))) if rule.forbid else 0.0

        expected_hits = sum(1 for item in rule.expected_relations if item in relation_facts)
        expected_ratio = float(expected_hits) / float(max(1, len(rule.expected_relations))) if rule.expected_relations else 0.0

        forbidden_hits = sum(1 for item in rule.forbidden_relations if item in relation_facts)
        forbidden_ratio = (
            float(forbidden_hits) / float(max(1, len(rule.forbidden_relations)))
            if rule.forbidden_relations
            else 0.0
        )

        return (
            self.requirement_bonus * matched_ratio
            + 0.5 * self.requirement_bonus * any_ratio
            - self.requirement_penalty * missing_ratio
            - self.forbid_penalty * forbid_ratio
            + self.relation_bonus * expected_ratio
            - self.relation_penalty * forbidden_ratio
        )

    @staticmethod
    def _compile_relations(values: Iterable[object]) -> Tuple[Tuple[str, str, str], ...]:
        compiled: List[Tuple[str, str, str]] = []
        for item in values:
            if isinstance(item, dict):
                subject = str(item.get("subject", "")).strip().lower()
                predicate = str(item.get("predicate", "")).strip().lower()
                obj = str(item.get("object", "")).strip().lower()
                if subject and predicate and obj:
                    compiled.append((subject, predicate, obj))
                continue
            if isinstance(item, (list, tuple)) and len(item) == 3:
                subject, predicate, obj = item
                compiled.append((str(subject).strip().lower(), str(predicate).strip().lower(), str(obj).strip().lower()))
        return tuple(compiled)
