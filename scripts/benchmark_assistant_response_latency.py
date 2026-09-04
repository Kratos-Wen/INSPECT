"""Benchmark post-ASR first response and optional async LLM refinement.

This benchmark starts after a user utterance has been transcribed. It uses
frozen online-replay snapshots and never changes verifier state. The primary
latency is time to the grounded structured response; optional LLM refinement
is measured separately because it is outside the interaction-critical path.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspect_assist import load_runtime_package

load_runtime_package()

from inspect_runtime.assistant.engine import ContextualAssistant
from inspect_runtime.assistant.grounded_llm import GroundedLLMResponder
from inspect_runtime.components.kb import KnowledgeBase
from scripts.evaluate_assistant_qa_diagnostics import choose_records_all, read_csv, snapshot_from_record


def _percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * min(1.0, max(0.0, q))
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(values: Iterable[float]) -> Dict[str, float]:
    items = [float(value) for value in values]
    return {
        "mean_ms": round(statistics.fmean(items), 4) if items else 0.0,
        "p50_ms": round(_percentile(items, 0.5), 4),
        "p90_ms": round(_percentile(items, 0.9), 4),
        "max_ms": round(max(items), 4) if items else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--kb", type=Path, default=ROOT / "components.json")
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data" / "assistant_qa" / "single_turn_questions.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshots", type=int, default=3)
    parser.add_argument("--questions-per-snapshot", type=int, default=4)
    parser.add_argument("--with-local-llm", action="store_true")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-path", default=os.environ.get("INSPECT_QWEN3_MODEL_PATH", ""))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--refinement-timeout-sec", type=float, default=3.0)
    args = parser.parse_args()

    records = choose_records_all(args.summary_csv, 1)[: max(1, int(args.snapshots))]
    questions = read_csv(args.questions)[: max(1, int(args.questions_per_snapshot))]
    if not records or not questions:
        raise RuntimeError("The replay summary and question set must both be non-empty.")
    kb = KnowledgeBase.from_path(args.kb)
    llm = None
    if args.with_local_llm:
        llm = GroundedLLMResponder(
            enabled=True,
            provider="transformers",
            model_id=str(args.model_id),
            model_path=str(args.model_path),
            device_map=str(args.device_map),
            torch_dtype=str(args.torch_dtype),
            max_new_tokens=48,
            temperature=0.0,
            timeout_sec=2.0,
            history_turns=3,
            max_context_chars=2400,
            answer_word_limit=40,
            streaming=True,
            load_on_start=True,
            fallback_to_template=True,
            include_fallback_in_prompt=True,
            grounding_guard_enabled=True,
        )
    assistant = ContextualAssistant(
        kb=kb,
        llm=llm,
        llm_async_refine=bool(llm),
    )
    rows: List[Dict[str, Any]] = []
    for record in records:
        snapshot = snapshot_from_record(record)
        for question in questions:
            started = time.perf_counter()
            reply = assistant.answer(str(question["question"]), snapshot)
            first_ms = (time.perf_counter() - started) * 1000.0
            refinement_ms = 0.0
            refinement = None
            if llm is not None and reply.evidence.get("llm_route") == "deferred":
                deadline = time.monotonic() + float(args.refinement_timeout_sec)
                while time.monotonic() < deadline:
                    refinement = assistant.poll_refinement()
                    if refinement is not None:
                        refinement_ms = (time.perf_counter() - started) * 1000.0
                        break
                    time.sleep(0.005)
            profile = dict((refinement or reply).evidence.get("llm_profile") or {})
            rows.append(
                {
                    "frame_index": int(snapshot.frame_index),
                    "question": str(question["question"]),
                    "intent": str(reply.route),
                    "status": str(reply.status),
                    "first_response_ms": round(first_ms, 4),
                    "refinement_ready_ms": round(refinement_ms, 4),
                    "llm_requested": bool(reply.evidence.get("llm_route") == "deferred"),
                    "llm_completed": bool(refinement is not None),
                    "guard_rejected": bool(profile.get("guard_rejected", False)),
                    "refinement_changed_text": bool(refinement and refinement.answer != reply.answer),
                }
            )
    first = [row["first_response_ms"] for row in rows]
    refined = [row["refinement_ready_ms"] for row in rows if row["llm_completed"]]
    requested = [row for row in rows if row["llm_requested"]]
    payload = {
        "protocol": {
            "latency_origin": "recognized text available (post-ASR)",
            "first_response": "grounded structured response",
            "optional_refinement": "asynchronous local LLM outside verifier path",
            "queries": len(rows),
            "snapshots": len(records),
            "local_llm": bool(llm),
            "model_id": str(args.model_id) if llm else "none",
            "ground_truth_used": False,
        },
        "latency": {
            "first_response": _summary(first),
            "async_refinement_ready": _summary(refined),
        },
        "routing": {
            "llm_request_rate": len(requested) / max(1, len(rows)),
            "llm_completion_rate": sum(row["llm_completed"] for row in requested) / max(1, len(requested)),
            "grounding_guard_reject_rate": sum(row["guard_rejected"] for row in requested) / max(1, len(requested)),
            "refinement_changed_text_rate": sum(row["refinement_changed_text"] for row in requested) / max(1, len(requested)),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"protocol": payload["protocol"], "latency": payload["latency"], "routing": payload["routing"]}, indent=2))


if __name__ == "__main__":
    main()
