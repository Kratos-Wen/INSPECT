"""Optional local LLM surface layer for grounded assistant answers."""

from __future__ import annotations

import json
import queue
import re
import threading
import textwrap
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import AssistantEvidence, AssistantReply, AssistantSnapshot, ResolvedAssistantQuery


class GroundedLLMResponder:
    """Generate concise answers from structured INSPECT evidence.

    The LLM is deliberately downstream of the verifier. It receives the current
    evidence snapshot and a deterministic fallback answer, then rewrites or
    explains that answer. It must not create new procedural decisions.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        provider: str,
        model_id: str,
        model_path: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        top_p: float = 0.9,
        timeout_sec: float = 6.0,
        history_turns: int = 4,
        max_context_chars: int = 5200,
        answer_word_limit: int = 55,
        streaming: bool = True,
        load_on_start: bool = False,
        fallback_to_template: bool = True,
        include_fallback_in_prompt: bool = True,
        grounding_guard_enabled: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.provider = str(provider or "none").strip().lower()
        self.model_id = str(model_id or "").strip()
        self.model_path = str(model_path or "").strip()
        self.device_map = str(device_map or "auto").strip()
        self.torch_dtype = str(torch_dtype or "auto").strip().lower()
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.timeout_sec = float(timeout_sec)
        self.history_turns = max(0, int(history_turns))
        self.max_context_chars = max(1200, int(max_context_chars))
        self.answer_word_limit = max(12, int(answer_word_limit))
        self.streaming = bool(streaming)
        self.fallback_to_template = bool(fallback_to_template)
        self.include_fallback_in_prompt = bool(include_fallback_in_prompt)
        self.grounding_guard_enabled = bool(grounding_guard_enabled)
        self.trust_remote_code = bool(trust_remote_code)
        self._tokenizer = None
        self._model = None
        self._load_error = ""
        self._last_profile: Dict[str, Any] = {}
        self._stop_event = threading.Event()
        if self.enabled and load_on_start:
            self._ensure_loaded()

    @property
    def status(self) -> str:
        if not self.enabled:
            return "disabled"
        if self._model is not None:
            return f"ready[{self.provider}]"
        if self._load_error:
            return f"fallback[{self._load_error[:80]}]"
        return f"lazy[{self.provider}]"

    @property
    def last_profile(self) -> Dict[str, Any]:
        return dict(self._last_profile)

    def answer(
        self,
        *,
        resolved: ResolvedAssistantQuery,
        evidence: AssistantEvidence,
        snapshot: AssistantSnapshot,
        fallback_answer: str,
        dialogue_history: Optional[List[AssistantReply]] = None,
    ) -> str:
        """Return an LLM answer, or the deterministic fallback if unavailable."""

        started = time.perf_counter()
        self._last_profile = {
            "provider": self.provider,
            "model_id": self.model_id,
            "guard_rejected": False,
            "guard_reason": "",
        }
        self._stop_event.clear()
        if not self.enabled or self.provider in {"none", "off", "template"}:
            self._last_profile["total_ms"] = round((time.perf_counter() - started) * 1000.0, 4)
            return fallback_answer
        if self.provider != "transformers":
            self._load_error = f"unsupported provider {self.provider}"
            self._last_profile["total_ms"] = round((time.perf_counter() - started) * 1000.0, 4)
            return fallback_answer
        try:
            load_started = time.perf_counter()
            self._ensure_loaded()
            self._last_profile["load_ms"] = round((time.perf_counter() - load_started) * 1000.0, 4)
            prompt_started = time.perf_counter()
            prompt = self._build_prompt(
                resolved=resolved,
                evidence=evidence,
                snapshot=snapshot,
                fallback_answer=fallback_answer,
                dialogue_history=dialogue_history or [],
            )
            self._last_profile["prompt_ms"] = round((time.perf_counter() - prompt_started) * 1000.0, 4)
            generation_started = time.perf_counter()
            generated = self._generate(prompt)
            self._last_profile["generation_ms"] = round((time.perf_counter() - generation_started) * 1000.0, 4)
            answer = _postprocess_answer(generated, fallback_answer, self.answer_word_limit)
            guard_started = time.perf_counter()
            violation = (
                _grounding_violation(answer, resolved=resolved, evidence=evidence, snapshot=snapshot)
                if self.grounding_guard_enabled
                else ""
            )
            self._last_profile["guard_ms"] = round((time.perf_counter() - guard_started) * 1000.0, 4)
            if violation:
                self._last_profile["guard_rejected"] = True
                self._last_profile["guard_reason"] = violation
                answer = fallback_answer
            self._last_profile["total_ms"] = round((time.perf_counter() - started) * 1000.0, 4)
            return answer
        except Exception as exc:  # pragma: no cover - runtime fallback path
            self._load_error = f"{type(exc).__name__}: {exc}"
            self._last_profile["error"] = self._load_error
            self._last_profile["total_ms"] = round((time.perf_counter() - started) * 1000.0, 4)
            if self.fallback_to_template:
                return fallback_answer
            raise

    def interrupt(self) -> None:
        """Request that the current generation stop as soon as possible."""

        self._stop_event.set()

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        source = _resolve_model_source(self.model_path, self.model_id)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                source,
                trust_remote_code=self.trust_remote_code,
            )
        except Exception:
            # Older `tokenizers` builds may fail to parse Qwen3 tokenizer.json.
            # The slow Qwen tokenizer is slower to initialize but works from the
            # same local files and keeps the runtime fully offline.
            self._tokenizer = AutoTokenizer.from_pretrained(
                source,
                trust_remote_code=self.trust_remote_code,
                use_fast=False,
            )
        kwargs: Dict[str, Any] = {
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.torch_dtype and self.torch_dtype != "auto":
            import torch

            dtype_map = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            kwargs["torch_dtype"] = dtype_map.get(self.torch_dtype, "auto")
        else:
            kwargs["torch_dtype"] = "auto"
        self._model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
        self._model.eval()

    def _generate(self, prompt: str) -> str:
        assert self._tokenizer is not None
        assert self._model is not None
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the grounded dialogue layer of INSPECT. "
                    "Use only the provided structured evidence. "
                    "Do not override verifier decisions. "
                    "If evidence is insufficient, say what is missing. "
                    "Answer concisely in the user's language. "
                    "Copy part and step names exactly from the evidence. "
                    "Use prior dialogue only to resolve follow-up references."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self._tokenizer([text], return_tensors="pt")
        device = getattr(self._model, "device", None)
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}
        do_sample = self.temperature > 0
        stop_at = time.monotonic() + self.timeout_sec

        class _StopOnEventOrTimeout(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs) -> bool:  # type: ignore[override]
                return self_event.is_set() or time.monotonic() >= stop_at

        self_event = self._stop_event
        generation_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "temperature": self.temperature if do_sample else None,
            "top_p": self.top_p if do_sample else None,
            "pad_token_id": self._tokenizer.eos_token_id,
            "stopping_criteria": StoppingCriteriaList([_StopOnEventOrTimeout()]),
        }
        if not self.streaming:
            outputs = self._model.generate(**generation_kwargs)
            input_len = inputs["input_ids"].shape[-1]
            new_tokens = outputs[0][input_len:]
            return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=0.2,
        )
        generation_kwargs["streamer"] = streamer
        def _run_generate() -> None:
            try:
                self._model.generate(**generation_kwargs)
            except Exception:
                self._stop_event.set()
                try:
                    streamer.on_finalized_text("", stream_end=True)
                except Exception:
                    pass

        thread = threading.Thread(target=_run_generate, daemon=True)
        thread.start()
        chunks: List[str] = []
        while thread.is_alive() or not self._stop_event.is_set():
            if self._stop_event.is_set() or time.monotonic() >= stop_at:
                self._stop_event.set()
                break
            try:
                chunk = next(streamer)
            except StopIteration:
                break
            except queue.Empty:
                continue
            chunks.append(str(chunk))
        thread.join(timeout=0.2)
        return "".join(chunks)

    def _build_prompt(
        self,
        *,
        resolved: ResolvedAssistantQuery,
        evidence: AssistantEvidence,
        snapshot: AssistantSnapshot,
        fallback_answer: str,
        dialogue_history: List[AssistantReply],
    ) -> str:
        compact_snapshot = _compact_snapshot(snapshot)
        compact_evidence = _compact(evidence)
        compact_history = _compact_history(dialogue_history[-self.history_turns :])
        payload = {
            "query": resolved.text,
            "intent": resolved.intent,
            "resolved_slots": {
                "target_component": resolved.target_component,
                "target_step": resolved.target_step,
                "relation_type": resolved.relation_type,
                "asks_reason": resolved.asks_reason,
            },
            "selected_evidence": compact_evidence,
            "current_state": compact_snapshot,
            "dialogue_history": compact_history,
        }
        if self.include_fallback_in_prompt:
            payload["deterministic_answer"] = fallback_answer
        body = json.dumps(payload, ensure_ascii=False, indent=2)
        if len(body) > self.max_context_chars:
            body = body[: self.max_context_chars] + "\n...<truncated>"
        return textwrap.dedent(
            f"""
            Answer the user using only the structured evidence below. Keep the
            verifier decision and evidence boundary implied by the selected
            evidence.

            {body}

            Constraints:
            - Never invent unseen parts, missing evidence, or step outcomes.
            - If the verifier is unresolved/cannot tell, do not claim success.
            - Do not mention implementation details; just answer the user.
            - For component details, safety, and troubleshooting, use selected_evidence even when the part is not currently visible.
            - Keep the response under {self.answer_word_limit} words unless the user explicitly asks for detail.
            - For follow-up questions, use dialogue_history only to resolve references such as it/that/again.
            - Copy part names and step IDs exactly as written in current_state.
            - If a referenced part is not in current_state or selected_evidence, say you cannot confirm it.
            - Prefer direct guidance for an apprentice worker.
            """
        ).strip()


def _resolve_model_source(model_path: str, model_id: str) -> str:
    if model_path:
        path = Path(model_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        if path.exists():
            return str(path)
    if model_id:
        return model_id
    raise ValueError("No LLM model_path or model_id configured.")


def _compact(obj: Any) -> Any:
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _compact(v) for k, v in obj.items() if _keep_value(v)}
    if isinstance(obj, (list, tuple)):
        return [_compact(item) for item in obj[:16] if _keep_value(item)]
    return obj


def _compact_snapshot(snapshot: AssistantSnapshot) -> Dict[str, Any]:
    """Build the bounded LLM context from online INSPECT state."""

    return {
        "frame_index": snapshot.frame_index,
        "current_step": {
            "step_id": snapshot.step_id,
            "confidence": round(float(snapshot.step_confidence), 3),
            "runner_up": snapshot.runner_up,
        },
        "visible_objects": snapshot.visible_objects[:8],
        "relevant_objects": snapshot.relevant_objects[:6],
        "object_counts": snapshot.object_counts,
        "relations": snapshot.scene_relations[:8],
        "relation_facts": snapshot.relation_facts[:8],
        "memory": {
            "step": snapshot.memory_step,
            "confidence": round(float(snapshot.memory_confidence), 3),
            "recalled": snapshot.memory_recalled,
            "reason": snapshot.memory_reason,
            "matches": _compact(snapshot.memory_matches[:4]),
        },
        "review": {
            "action": snapshot.review_action,
            "reason": snapshot.review_reason,
        },
        "recent_steps": _compact(snapshot.recent_steps[-5:]),
        "recent_feedback": _compact(snapshot.recent_feedback[-5:]),
    }


def _compact_history(history: List[AssistantReply]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for item in history:
        items.append(
            {
                "user": item.query,
                "assistant": item.answer,
                "intent": item.route,
                "status": item.status,
                "target_component": (item.evidence or {}).get("target_component", ""),
                "target_step": (item.evidence or {}).get("target_step", ""),
            }
        )
    return items


def _postprocess_answer(generated: str, fallback: str, word_limit: int) -> str:
    text = str(generated or "").strip()
    if not text:
        return fallback
    for marker in ["<|im_end|>", "<|endoftext|>", "SYSTEM:", "USER:", "ASSISTANT:"]:
        text = text.replace(marker, "")
    lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
    if not lines:
        return fallback
    text = " ".join(lines[:3]).strip()
    words = text.split()
    if len(words) > word_limit:
        text = " ".join(words[:word_limit]).rstrip(".,;:") + "."
    return text


_STEP_PATTERN = re.compile(r"\bS\d+\b", re.IGNORECASE)
_TYPE_PATTERN = re.compile(r"\btype[\s_-]*(\d+)(?=$|[\s_-])", re.IGNORECASE)


def _steps_in(value: Any) -> set[str]:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        found: set[str] = set()
        for item in value.values():
            found.update(_steps_in(item))
        return found
    if isinstance(value, (list, tuple, set)):
        found = set()
        for item in value:
            found.update(_steps_in(item))
        return found
    return {match.upper() for match in _STEP_PATTERN.findall(str(value or ""))}


def _types_in(value: Any) -> set[str]:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        found: set[str] = set()
        for item in value.values():
            found.update(_types_in(item))
        return found
    if isinstance(value, (list, tuple, set)):
        found = set()
        for item in value:
            found.update(_types_in(item))
        return found
    return {match for match in _TYPE_PATTERN.findall(str(value or ""))}


def _grounding_violation(
    answer: str,
    *,
    resolved: ResolvedAssistantQuery,
    evidence: AssistantEvidence,
    snapshot: AssistantSnapshot,
) -> str:
    """Reject generated identifiers that are not licensed by selected evidence."""

    mentioned_steps = _steps_in(answer)
    allowed_steps = _steps_in(evidence)
    allowed_steps.update(_steps_in(resolved.target_step))
    allowed_steps.update(_steps_in(snapshot.step_id))
    if resolved.intent in {"history_step", "memory_context"}:
        allowed_steps.update(_steps_in(snapshot.recent_steps))
        allowed_steps.update(_steps_in(snapshot.memory_step))

    if mentioned_steps - allowed_steps:
        return "unsupported_step_reference"
    facts = evidence.facts
    if resolved.intent == "current_step":
        expected = _steps_in(facts.get("step_id", snapshot.step_id))
        if expected and not expected.issubset(mentioned_steps):
            return "current_step_not_preserved"
    if resolved.intent == "next_step":
        expected = _steps_in(facts.get("next_step", ""))
        if expected and not expected.issubset(mentioned_steps):
            return "next_step_not_preserved"

    mentioned_types = _types_in(answer)
    allowed_types = _types_in(evidence)
    if mentioned_types - allowed_types:
        return "unsupported_component_reference"
    return ""


def _keep_value(value: Any) -> bool:
    if value is None:
        return False
    if value == "":
        return False
    if value == [] or value == {}:
        return False
    return True
