"""Grounded scene-aware assistant for live Q&A without heavy latency."""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Iterable, List, Optional

from ..components.kb import KnowledgeBase
from .evidence import select_evidence
from .grounded_llm import GroundedLLMResponder
from .parser import parse_query
from .policy import decide_policy
from .responder import render_answer
from .resolver import DialogueState, resolve_query, update_dialogue_state
from .types import AssistantReply, AssistantSnapshot


class ContextualAssistant:
    """Answer natural-language questions from current live context with grounded constraints."""

    def __init__(
        self,
        kb: KnowledgeBase,
        enabled: bool = True,
        history_limit: int = 12,
        relation_limit: int = 3,
        overlay_chars: int = 120,
        llm: GroundedLLMResponder | None = None,
        llm_intents: Optional[Iterable[str]] = None,
        llm_async_refine: bool = False,
    ) -> None:
        self.kb = kb
        self.enabled = bool(enabled)
        self.history_limit = int(history_limit)
        self.relation_limit = int(relation_limit)
        self.overlay_chars = int(overlay_chars)
        self.llm = llm
        self.llm_async_refine = bool(llm_async_refine)
        self._llm_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inspect-llm") if self.llm_async_refine else None
        self._pending_refinements: List[tuple[Future, AssistantReply]] = []
        self.llm_intents = (
            None
            if llm_intents is None
            else {str(intent).strip().lower() for intent in llm_intents if str(intent).strip()}
        )
        self.history: List[AssistantReply] = []
        self.dialogue_state = DialogueState()

    def answer(self, query: str, snapshot: AssistantSnapshot) -> AssistantReply:
        """Answer one query using a parse-resolve-evidence-policy-response pipeline."""

        started = time.perf_counter()
        runtime_ms: dict[str, float] = {}
        llm_profile: dict[str, object] = {}
        llm_invoked = False
        normalized = str(query).strip()
        stage_started = time.perf_counter()
        parsed = parse_query(normalized)
        resolved = resolve_query(parsed, snapshot, self.kb, self.dialogue_state)
        runtime_ms["parse_resolve"] = (time.perf_counter() - stage_started) * 1000.0
        stage_started = time.perf_counter()
        evidence = select_evidence(resolved, snapshot, self.kb)
        decision = decide_policy(resolved, evidence, snapshot, enabled=self.enabled)
        runtime_ms["evidence_policy"] = (time.perf_counter() - stage_started) * 1000.0
        if decision.action == "answer":
            stage_started = time.perf_counter()
            answer = render_answer(resolved, evidence, snapshot)
            runtime_ms["structured_response"] = (time.perf_counter() - stage_started) * 1000.0
            llm_allowed = bool(
                self.llm is not None
                and (self.llm_intents is None or resolved.intent in self.llm_intents)
                and evidence.grounded
            )
            if llm_allowed and self.llm is not None:
                llm_invoked = True
                if self._llm_executor is None:
                    stage_started = time.perf_counter()
                    answer = self.llm.answer(
                        resolved=resolved,
                        evidence=evidence,
                        snapshot=snapshot,
                        fallback_answer=answer,
                        dialogue_history=self.history,
                    )
                    runtime_ms["llm"] = (time.perf_counter() - stage_started) * 1000.0
                    llm_profile = self.llm.last_profile
                else:
                    history = list(self.history)
                    future = self._llm_executor.submit(
                        self._run_llm_refinement,
                        resolved,
                        evidence,
                        snapshot,
                        answer,
                        history,
                    )
                    runtime_ms["llm_enqueue"] = 0.0
        else:
            answer = decision.message
        runtime_ms["total"] = (time.perf_counter() - started) * 1000.0
        runtime_ms = {key: round(value, 4) for key, value in runtime_ms.items()}

        reply = AssistantReply(
            query=normalized,
            answer=answer,
            route=resolved.intent,
            status=decision.action,
            evidence={
                "intent": resolved.intent,
                "target_component": resolved.target_component,
                "target_step": resolved.target_step,
                "relation_type": resolved.relation_type,
                "used_history": resolved.used_history,
                "policy": decision.action,
                "grounded": evidence.grounded,
                "facts": dict(evidence.facts),
                "llm_status": self.llm.status if llm_invoked and self.llm is not None else "not_invoked",
                "llm_route": (
                    "deferred"
                    if llm_invoked and self._llm_executor is not None
                    else "invoked"
                    if llm_invoked
                    else "unavailable"
                    if self.llm is None
                    else "ungrounded"
                    if not evidence.grounded
                    else "structured_intent"
                ),
                "runtime_ms": runtime_ms,
                "llm_profile": llm_profile,
            },
        )
        self.history.append(reply)
        if llm_invoked and self._llm_executor is not None:
            self._pending_refinements.append((future, reply))
        if len(self.history) > self.history_limit:
            self.history.pop(0)
        if decision.action == "answer":
            update_dialogue_state(self.dialogue_state, resolved)
        return reply

    def _run_llm_refinement(self, resolved, evidence, snapshot, fallback_answer, dialogue_history):
        started = time.perf_counter()
        assert self.llm is not None
        answer = self.llm.answer(
            resolved=resolved,
            evidence=evidence,
            snapshot=snapshot,
            fallback_answer=fallback_answer,
            dialogue_history=dialogue_history,
        )
        return answer, self.llm.last_profile, (time.perf_counter() - started) * 1000.0

    def poll_refinement(self) -> Optional[AssistantReply]:
        """Return one completed optional LLM refinement without blocking perception."""

        for index, (future, base_reply) in enumerate(self._pending_refinements):
            if not future.done():
                continue
            self._pending_refinements.pop(index)
            try:
                answer, profile, latency_ms = future.result()
            except Exception:
                return None
            evidence = dict(base_reply.evidence)
            runtime = dict(evidence.get("runtime_ms", {}))
            runtime["async_llm"] = round(float(latency_ms), 4)
            evidence.update(
                {
                    "llm_status": self.llm.status if self.llm is not None else "unavailable",
                    "llm_route": "async_refined",
                    "runtime_ms": runtime,
                    "llm_profile": dict(profile),
                }
            )
            refined = AssistantReply(
                query=base_reply.query,
                answer=answer,
                route=base_reply.route,
                status=base_reply.status,
                evidence=evidence,
            )
            for history_index in range(len(self.history) - 1, -1, -1):
                if self.history[history_index] is base_reply:
                    self.history[history_index] = refined
                    break
            return refined
        return None

    def latest_overlay_text(self) -> str:
        """Return the most recent answer shortened for the UI overlay."""

        if not self.history:
            return ""
        return self.history[-1].answer[: self.overlay_chars]

    def interrupt(self) -> None:
        """Interrupt any optional LLM generation in progress."""

        if self.llm is not None:
            self.llm.interrupt()
        for future, _reply in self._pending_refinements:
            future.cancel()
        self._pending_refinements = [item for item in self._pending_refinements if item[0].running()]
