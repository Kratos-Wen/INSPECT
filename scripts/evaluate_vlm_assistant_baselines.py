"""Evaluate VLM baselines for INSPECT assistant verification.

This script runs two non-oracle prompt protocols:

``direct``
    Current RGB frame plus a user question containing the active claim.

``procedure_context``
    Current RGB frame plus the active claim, a compact procedure description,
    and non-oracle procedural context.  It does not include detector boxes,
    scene graphs, verifier scores, candidate robot views, or ground-truth
    outcomes.

Cloud API keys are read only from environment variables.  They are never
accepted as command-line arguments.  The local Qwen backend loads one frozen
checkpoint once per run and uses the same prompts and images as the APIs.
"""

from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_procedural_claim_triage_calibrated import (  # noqa: E402
    OUTCOMES,
    Record,
    build_records,
    video_key,
)
from scripts.evaluate_vlm_procedural_verifier_gemini import extract_frame  # noqa: E402


_LOCAL_QWEN_RUNTIME: Dict[str, Any] = {}
PROMPT_PROTOCOL_VERSION = "triage_v2"


DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["supported", "contradicted", "unresolved"]},
        "missing_evidence": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "object_presence",
                    "identity",
                    "slot_relation",
                    "seating_boundary",
                    "alignment",
                    "occlusion",
                    "other",
                ],
            },
        },
        "visible_evidence": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["decision", "missing_evidence", "visible_evidence", "reason"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = (
    "You are a non-oracle vision-language baseline for industrial assembly verification. "
    "Use only the provided image and prompt context. Perform a symmetric three-way decision: "
    "choose supported when visible evidence establishes the claim; choose contradicted when "
    "visible evidence establishes an incompatible state, such as a wrong part, a part outside "
    "the slot, a clear seating gap, or clear misalignment; otherwise choose unresolved. "
    "Judge visible identity and geometry directly and do not require motion, touch, or hidden "
    "mechanical fit when the claimed visual relation is already clear. Do not infer hidden state. "
    "Return exactly one compact JSON object that follows the requested schema. "
    "Do not include markdown, code fences, prefixes, or explanatory prose outside JSON. "
    "Keep the reason under 12 words. "
    "If the image or context is insufficient, choose unresolved."
)


STEP_DESCRIPTIONS = {
    "S1": "Verify that the target gearbox housing is present and usable.",
    "S2": "Insert the required gear into the housing slot.",
    "S3": "Verify that the inserted gear is correct and seated in the housing.",
    "S4": "Place and seat the gearbox cover on the housing.",
}


CLAIM_DESCRIPTIONS = {
    "cover_seated": "the gearbox cover is fully seated on the housing",
    "smallgear_inserted": "the small gear is inserted in the housing slot",
    "biggear_inserted": "the large gear is inserted in the housing slot",
    "gear_inserted": "the gear is inserted in the housing slot",
    "gear_identity": "the visible gear identity is the required one for the assembly",
}


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


ROW_FIELDS = [
    "video",
    "frame",
    "step_id",
    "claim_id",
    "truth",
    "image_path",
    "provider",
    "model",
    "protocol",
    "prompt_version",
    "prompt_sha256",
    "prompt",
    "raw_response",
    "pred",
    "parse_error",
    "latency_sec",
    "cache_hit",
]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def prompt_sha256(prompt: str) -> str:
    payload = f"{PROMPT_PROTOCOL_VERSION}\n{SYSTEM_PROMPT}\n{prompt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def row_key(
    provider: str,
    model: str,
    protocol: str,
    record: Record,
    prompt: str,
) -> Tuple[str, str, str, str, str, str, str]:
    return (
        str(provider),
        str(model),
        str(protocol),
        str(record.video),
        str(record.frame),
        str(record.claim_id),
        prompt_sha256(prompt),
    )


def cache_key(row: Mapping[str, Any]) -> Tuple[str, str, str, str, str, str, str]:
    return (
        str(row.get("provider", "")),
        str(row.get("model", "")),
        str(row.get("protocol", "")),
        str(row.get("video", "")),
        str(row.get("frame", "")),
        str(row.get("claim_id", "")),
        str(row.get("prompt_sha256", "")),
    )


def load_response_cache(
    path: Path,
    reuse_invalid: bool,
) -> Dict[Tuple[str, str, str, str, str, str, str], Dict[str, Any]]:
    cache: Dict[Tuple[str, str, str, str, str, str, str], Dict[str, Any]] = {}
    for row in read_csv_rows(path):
        pred = str(row.get("pred", "")).strip()
        if not pred:
            continue
        if pred == "invalid_response" and not reuse_invalid:
            continue
        raw = str(row.get("raw_response", ""))
        if pred != "invalid_response" and not raw:
            continue
        cache[cache_key(row)] = row
    return cache


def retry_delay_seconds(error_text: str) -> float | None:
    text = str(error_text or "")
    patterns = [
        r"retryDelay['\"]?\s*:\s*['\"]?([0-9.]+)s",
        r"Please retry in ([0-9.]+)s",
        r"retry after ([0-9.]+) seconds",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return max(0.0, float(match.group(1)))
            except ValueError:
                return None
    if "429" in text or "rate limit" in text.lower() or "RESOURCE_EXHAUSTED" in text:
        return 15.0
    return None


def data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def claim_text(claim_id: str) -> str:
    key = str(claim_id or "").strip().lower()
    return CLAIM_DESCRIPTIONS.get(key, key.replace("_", " "))


def procedure_context(record: Record) -> str:
    step = str(record.step_id or "").strip().upper()
    ordered = ["S1", "S2", "S3", "S4"]
    previous = [sid for sid in ordered if sid < step]
    previous_text = ", ".join(f"{sid}: {STEP_DESCRIPTIONS[sid]}" for sid in previous) or "No previous step context."
    current = STEP_DESCRIPTIONS.get(step, "Unknown procedural step.")
    return (
        f"Procedure order: S1 -> S2 -> S3 -> S4.\n"
        f"Prior procedural context: {previous_text}\n"
        f"Current step {step}: {current}"
    )


def make_prompt(record: Record, protocol: str) -> str:
    user_question = f"Can the current image verify this active claim: {claim_text(record.claim_id)}?"
    if protocol == "direct":
        return (
            f"User question: {user_question}\n"
            "Decide from the current RGB image only. Do not use procedure history or detector outputs.\n"
            "Return only compact JSON with decision, missing_evidence, visible_evidence, and reason."
        )
    if protocol == "procedure_context":
        return (
            f"User question: {user_question}\n"
            f"Active claim id: {record.claim_id}\n"
            f"Target step: {record.step_id}\n"
            f"{procedure_context(record)}\n"
            "Do not use detector boxes, scene graphs, verifier scores, candidate-view images, or ground-truth labels.\n"
            "Return only compact JSON with decision, missing_evidence, visible_evidence, and reason."
        )
    raise ValueError(f"Unknown protocol: {protocol}")


def parse_json_response(text: str) -> Tuple[str, Dict[str, Any], str]:
    raw = str(text or "").strip()
    if not raw:
        return "invalid_response", {}, "empty response"
    candidates = [raw]
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match and match.group(0) != raw:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        decision = str(obj.get("decision", "")).strip().lower()
        if decision in OUTCOMES:
            return decision, obj, ""
        return "invalid_response", obj, f"invalid decision {decision!r}"
    return "invalid_response", {}, "not valid JSON"


def summarize(records: Sequence[Record], preds: Sequence[str]) -> Dict[str, Any]:
    totals = Counter()
    confusion = Counter()
    for rec, pred in zip(records, preds):
        truth = rec.truth
        if truth not in OUTCOMES:
            continue
        totals["total"] += 1
        totals[f"truth_{truth}"] += 1
        totals[f"pred_{pred}"] += 1
        if pred in OUTCOMES:
            confusion[(truth, pred)] += 1
        if pred == truth:
            totals["correct"] += 1

    def safe(num: float, den: float) -> float | None:
        return None if den <= 0 else num / den

    def f1(tp: int, fp: int, fn: int) -> float:
        precision = safe(tp, tp + fp) or 0.0
        recall = safe(tp, tp + fn) or 0.0
        return 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)

    class_metrics: Dict[str, Dict[str, Any]] = {}
    f1_values: List[float] = []
    for label in OUTCOMES:
        tp = confusion[(label, label)]
        fp = sum(confusion[(truth, label)] for truth in OUTCOMES if truth != label)
        fn = totals[f"truth_{label}"] - tp
        label_f1 = f1(tp, fp, fn)
        f1_values.append(label_f1)
        class_metrics[label] = {
            "total": totals[f"truth_{label}"],
            "correct": tp,
            "recall": safe(tp, totals[f"truth_{label}"]),
            "precision": safe(tp, tp + fp),
            "f1": label_f1,
        }

    invalid = totals["pred_invalid_response"]
    return {
        "total": totals["total"],
        "valid_response_rate": safe(totals["total"] - invalid, totals["total"]),
        "invalid_response_rate": safe(invalid, totals["total"]),
        "triage_accuracy": safe(totals["correct"], totals["total"]),
        "triage_macro_f1": safe(sum(f1_values), len(f1_values)),
        "support_accuracy": safe(confusion[("supported", "supported")], totals["truth_supported"]),
        "contradiction_recall": safe(confusion[("contradicted", "contradicted")], totals["truth_contradicted"]),
        "unresolved_coverage": safe(confusion[("unresolved", "unresolved")], totals["truth_unresolved"]),
        "false_accept_rate_on_non_supported": safe(
            confusion[("contradicted", "supported")] + confusion[("unresolved", "supported")],
            totals["truth_contradicted"] + totals["truth_unresolved"],
        ),
        "safe_non_accept_rate_on_contradicted": safe(
            totals["truth_contradicted"] - confusion[("contradicted", "supported")],
            totals["truth_contradicted"],
        ),
        "pred_counts": {label: totals[f"pred_{label}"] for label in [*OUTCOMES, "invalid_response"]},
        "confusion": {f"{truth}->{pred}": confusion[(truth, pred)] for truth in OUTCOMES for pred in OUTCOMES},
        "class_metrics": class_metrics,
    }


def balanced_sample_records(records: Sequence[Record], per_class: int, seed: int) -> List[Record]:
    """Deterministically sample a balanced, claim-diverse subset for paid VLM calls.

    The full replay set contains many adjacent frames from the same video.  For API
    baselines we therefore evaluate a small balanced subset instead of calling a
    paid VLM on every frame.  Sampling is stratified by the triage label and
    round-robin over claim ids to avoid spending the budget on one claim type.
    """
    if per_class <= 0:
        return list(records)
    rng = random.Random(seed)
    by_truth_claim: Dict[str, Dict[str, List[Record]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec.truth in OUTCOMES:
            by_truth_claim[rec.truth][rec.claim_id].append(rec)
    sampled: List[Record] = []
    for truth in OUTCOMES:
        claim_groups = {
            claim: sorted(items, key=lambda r: (video_key(r.video), r.frame, r.claim_id))
            for claim, items in by_truth_claim.get(truth, {}).items()
            if items
        }
        for items in claim_groups.values():
            rng.shuffle(items)
        claims = sorted(claim_groups)
        truth_rows: List[Record] = []
        while claims and len(truth_rows) < per_class:
            next_claims = []
            for claim in claims:
                items = claim_groups[claim]
                if items and len(truth_rows) < per_class:
                    truth_rows.append(items.pop())
                if items:
                    next_claims.append(claim)
            claims = next_claims
        sampled.extend(truth_rows)
    sampled.sort(key=lambda rec: (rec.truth, video_key(rec.video), rec.frame, rec.claim_id))
    return sampled


def evaluation_key(video: str, frame: int | str, truth: str) -> Tuple[str, str, str]:
    return (PureWindowsPath(str(video)).name, str(frame), str(truth))


def load_eligible_keys(path: Path) -> set[Tuple[str, str, str]]:
    rows = read_csv_rows(path)
    required = {"video", "frame", "truth"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"Eligibility CSV must contain {sorted(required)}")
    return {evaluation_key(row["video"], row["frame"], row["truth"]) for row in rows}


def call_gemini(*, model: str, image_path: Path, prompt: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    gemini_schema = copy.deepcopy(DECISION_SCHEMA)
    gemini_schema.pop("additionalProperties", None)
    config_kwargs: Dict[str, Any] = {
        "system_instruction": SYSTEM_PROMPT,
        "temperature": 0.0,
        "max_output_tokens": 512,
        "response_mime_type": "application/json",
    }
    try:
        config_kwargs["response_schema"] = gemini_schema
        config = types.GenerateContentConfig(**config_kwargs)
    except Exception:
        config_kwargs.pop("response_schema", None)
        config = types.GenerateContentConfig(**config_kwargs)
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_path.read_bytes(), mime_type="image/jpeg"),
            prompt,
        ],
        config=config,
    )
    return str(getattr(response, "text", "") or "")


def call_openai(*, model: str, image_path: Path, prompt: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    schema = {
        "type": "json_schema",
        "name": "inspect_vlm_decision",
        "schema": DECISION_SCHEMA,
        "strict": True,
    }
    image_url = data_url(image_path)
    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": image_url},
                        {"type": "input_text", "text": prompt},
                    ],
                },
            ],
            text={"format": schema},
            temperature=0,
            max_output_tokens=512,
        )
        return str(getattr(response, "output_text", "") or "")
    except Exception:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=512,
        )
        return str(response.choices[0].message.content or "")


def call_qwen_local(*, model: str, image_path: Path, prompt: str) -> str:
    """Run a frozen Qwen3-VL checkpoint without detector or verifier inputs."""
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    runtime_key = str(Path(model).resolve()) if Path(model).exists() else str(model)
    if _LOCAL_QWEN_RUNTIME.get("key") != runtime_key:
        if not torch.cuda.is_available():
            raise RuntimeError("The local Qwen3-VL baseline requires CUDA.")
        processor = AutoProcessor.from_pretrained(model)
        vlm = AutoModelForImageTextToText.from_pretrained(
            model,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        vlm.eval()
        _LOCAL_QWEN_RUNTIME.clear()
        _LOCAL_QWEN_RUNTIME.update({"key": runtime_key, "processor": processor, "model": vlm})

    processor = _LOCAL_QWEN_RUNTIME["processor"]
    vlm = _LOCAL_QWEN_RUNTIME["model"]
    with Image.open(image_path) as source:
        image = source.convert("RGB").copy()
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "resized_height": 448,
                    "resized_width": 448,
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(vlm.device)
    with torch.inference_mode():
        generated = vlm.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=192,
            use_cache=True,
        )
    prompt_length = inputs["input_ids"].shape[-1]
    response_ids = generated[:, prompt_length:]
    return str(
        processor.batch_decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    )


def call_provider(provider: str, model: str, image_path: Path, prompt: str) -> str:
    if provider == "gemini":
        return call_gemini(model=model, image_path=image_path, prompt=prompt)
    if provider == "openai":
        return call_openai(model=model, image_path=image_path, prompt=prompt)
    if provider == "qwen_local":
        return call_qwen_local(model=model, image_path=image_path, prompt=prompt)
    raise ValueError(f"Unknown provider: {provider}")


def default_model(provider: str) -> str:
    defaults = {
        "gemini": "gemini-3.5-flash",
        "openai": "gpt-4.1",
        "qwen_local": "Qwen/Qwen3-VL-4B-Instruct",
    }
    return defaults[provider]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("gemini", "openai", "qwen_local"), required=True)
    parser.add_argument("--protocol", choices=("direct", "procedure_context"), required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--timeline-csv", type=Path, required=True)
    parser.add_argument("--evidence-scorer", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--frame-cache", type=Path, default=Path("outputs/vlm_assistant_frames"))
    parser.add_argument(
        "--eligible-records-csv",
        type=Path,
        default=None,
        help="Restrict sampling to video/frame/truth keys evaluated by the comparison method.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--balanced-per-class",
        type=int,
        default=0,
        help="Cost-capped balanced sample size per triage class. 0 evaluates all records.",
    )
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse valid rows already present in --output-csv.")
    parser.add_argument("--reuse-invalid", action="store_true", help="Also reuse invalid_response rows when --resume is set.")
    parser.add_argument("--request-interval-sec", type=float, default=0.0, help="Sleep between API requests to respect rate limits.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retry API calls after rate-limit or transient failures.")
    args = parser.parse_args()

    model = str(args.model or default_model(args.provider))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    cache_jsonl = args.output_json.with_suffix(".cache.jsonl")
    response_cache = load_response_cache(args.output_csv, bool(args.reuse_invalid)) if args.resume else {}
    records = sorted(
        build_records(args.summary_csv, args.timeline_csv, args.evidence_scorer),
        key=lambda rec: (video_key(rec.video), rec.frame, rec.claim_id),
    )
    full_record_count = len(records)
    eligible_record_count = full_record_count
    eligibility_sha256 = None
    if args.eligible_records_csv is not None:
        eligible_keys = load_eligible_keys(args.eligible_records_csv)
        records = [
            record
            for record in records
            if evaluation_key(record.video, record.frame, record.truth) in eligible_keys
        ]
        eligible_record_count = len(records)
        eligibility_sha256 = file_sha256(args.eligible_records_csv)
    if int(args.balanced_per_class) > 0:
        records = balanced_sample_records(records, int(args.balanced_per_class), int(args.sample_seed))
    rows: List[Dict[str, Any]] = []
    eval_records: List[Record] = []
    preds: List[str] = []
    errors = 0
    cache_hits = 0
    api_calls = 0
    for idx, record in enumerate(records):
        if args.limit and idx >= int(args.limit):
            break
        prompt = make_prompt(record, args.protocol)
        key = row_key(args.provider, model, args.protocol, record, prompt)
        cached = response_cache.get(key)
        if cached is not None and not args.dry_run:
            cached_row = dict(cached)
            cached_row["cache_hit"] = "1"
            rows.append(cached_row)
            eval_records.append(record)
            preds.append(str(cached_row.get("pred", "") or "invalid_response"))
            cache_hits += 1
            write_csv(args.output_csv, rows, ROW_FIELDS)
            write_jsonl(cache_jsonl, rows)
            continue

        frame_path = None if args.dry_run else extract_frame(record.video, record.frame, args.frame_cache)
        raw = ""
        pred = ""
        parse_error = ""
        latency = 0.0
        if args.dry_run:
            pred = ""
        elif frame_path is None:
            pred = "invalid_response"
            parse_error = "frame extraction failed"
            errors += 1
        else:
            start = time.perf_counter()
            last_error = ""
            for attempt in range(int(args.max_retries) + 1):
                try:
                    if api_calls > 0 and float(args.request_interval_sec) > 0:
                        time.sleep(float(args.request_interval_sec))
                    api_calls += 1
                    raw = call_provider(args.provider, model, frame_path, prompt)
                    latency = time.perf_counter() - start
                    pred, _obj, parse_error = parse_json_response(raw)
                    last_error = ""
                    break
                except Exception as exc:
                    last_error = f"api_error: {type(exc).__name__}: {exc}"
                    delay = retry_delay_seconds(last_error)
                    if attempt < int(args.max_retries) and delay is not None:
                        time.sleep(delay + 1.0)
                        continue
                    latency = time.perf_counter() - start
                    raw = ""
                    pred = "invalid_response"
                    parse_error = last_error
                    errors += 1
                    break
            eval_records.append(record)
            preds.append(pred)
        rows.append(
            {
                "video": record.video,
                "frame": record.frame,
                "step_id": record.step_id,
                "claim_id": record.claim_id,
                "truth": record.truth,
                "image_path": str(frame_path or ""),
                "provider": args.provider,
                "model": model,
                "protocol": args.protocol,
                "prompt_version": PROMPT_PROTOCOL_VERSION,
                "prompt_sha256": prompt_sha256(prompt),
                "prompt": prompt,
                "raw_response": raw,
                "pred": pred,
                "parse_error": parse_error,
                "latency_sec": f"{latency:.4f}",
                "cache_hit": "0",
            }
        )
        write_csv(args.output_csv, rows, ROW_FIELDS)
        write_jsonl(cache_jsonl, rows)
        if len(rows) % 5 == 0 or len(rows) == len(records):
            print(f"completed={len(rows)}/{len(records)}", flush=True)

    metrics: Dict[str, Any] = {
        "provider": args.provider,
        "model": model,
        "protocol": args.protocol,
        "dry_run": bool(args.dry_run),
        "rows": len(rows),
        "full_records_available": full_record_count,
        "eligible_records_available": eligible_record_count,
        "eligibility_filter_sha256": eligibility_sha256,
        "balanced_sampling_after_eligibility_filter": args.eligible_records_csv is not None,
        "balanced_per_class": int(args.balanced_per_class),
        "sample_seed": int(args.sample_seed),
        "api_errors": errors,
        "api_calls": api_calls,
        "cache_hits": cache_hits,
        "non_oracle": True,
        "prompt_protocol": args.protocol,
        "prompt_protocol_version": PROMPT_PROTOCOL_VERSION,
        "inputs": (
            "current RGB frame + user question"
            if args.protocol == "direct"
            else "current RGB frame + active claim + procedure context"
        ),
        "withheld_inputs": [
            "detector boxes",
            "scene graph",
            "verifier scores",
            "candidate robot views",
            "ground-truth outcomes",
        ],
    }
    if args.provider == "qwen_local":
        metrics["local_inference"] = {
            "frozen_checkpoint": True,
            "image_size": [448, 448],
            "do_sample": False,
            "max_new_tokens": 192,
        }
    if preds:
        metrics.update(summarize(eval_records, preds))
    args.output_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    write_csv(
        args.output_csv,
        rows,
        ROW_FIELDS,
    )
    write_jsonl(cache_jsonl, rows)
    metrics["cache_jsonl"] = str(cache_jsonl)
    args.output_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
