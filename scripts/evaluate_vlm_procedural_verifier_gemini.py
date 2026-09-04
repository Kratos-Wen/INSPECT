"""Evaluate a Gemini VLM as a procedural claim verifier.

The script can run in two phases:

1. ``--dry-run`` writes the prompt table without calling the API.
2. Without ``--dry-run``, it calls Gemini and evaluates the returned
   supported / contradicted / unresolved labels.

Set ``GEMINI_API_KEY`` only in the current process/session when running the
API call.  The key is intentionally not accepted as a command-line argument so
it cannot leak through shell history or process listings.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_procedural_claim_triage_calibrated import (  # noqa: E402
    OUTCOMES,
    Record,
    build_records,
    summarize_predictions,
    video_key,
)


SYSTEM_INSTRUCTION = """You are evaluating an industrial assembly image.
Answer only with one JSON object:
{"decision": "supported" | "contradicted" | "unresolved", "reason": "..."}
Supported means the visual evidence and history justify the claim.
Contradicted means the claim is visually false or procedurally inadmissible.
Unresolved means the current image is insufficient or ambiguous.
Do not guess when the evidence is missing."""


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                yield item


def read_summary_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def build_iteration_index(summary_csv: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    index: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in read_summary_rows(summary_csv):
        if str(row.get("returncode", "")).strip() not in {"", "0"}:
            continue
        video = str(row.get("video", ""))
        run_dir = Path(str(row.get("run_dir", "")))
        for item in read_jsonl(run_dir / "iterations.jsonl") or []:
            frame = int(item.get("frame_index", item.get("frame", -1)))
            index[(video, frame)] = item
    return index


def extract_frame(video: str, frame_index: int, output_dir: Path) -> Optional[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{Path(video).stem}_frame_{frame_index:06d}.jpg"
    if out.exists():
        return out
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    cv2.imwrite(str(out), frame)
    return out


def compact_history(record: Record, previous_by_video: Dict[str, List[Record]], max_items: int) -> str:
    prev = previous_by_video[record.video][-max_items:]
    if not prev:
        return "No prior verified history is available."
    pieces = []
    for item in prev:
        pieces.append(f"frame {item.frame}: claim={item.claim_id}, annotated_state={item.truth}, step={item.step_id}")
    return "\n".join(pieces)


def compact_evidence(item: Dict[str, Any]) -> str:
    token = item.get("evidence_token") or {}
    visible = token.get("visible_counts", {})
    relations = token.get("relation_facts", [])
    tracks = token.get("track_counts", {})
    return json.dumps(
        {
            "visible_counts": visible,
            "track_counts": tracks,
            "relations": relations[:8] if isinstance(relations, list) else relations,
            "fused_step": item.get("fused_step"),
            "decision_step": item.get("decision_step"),
        },
        ensure_ascii=False,
    )


def make_prompt(record: Record, item: Dict[str, Any], history: str) -> str:
    return (
        f"Active claim: {record.claim_id}\n"
        f"Target procedural step: {record.step_id}\n"
        f"Current assistant step proposal: {record.fused_step}\n"
        f"Compact evidence extracted by the system:\n{compact_evidence(item)}\n"
        f"Bounded procedure history:\n{history}\n"
        "Classify the active claim from the image and context."
    )


def parse_decision(text: str) -> str:
    lower = text.strip().lower()
    try:
        obj = json.loads(text)
        decision = str(obj.get("decision", "")).strip().lower()
        if decision in OUTCOMES:
            return decision
    except Exception:
        pass
    match = re.search(r"\b(supported|contradicted|unresolved)\b", lower)
    if match:
        return match.group(1)
    return "unresolved"


def image_part(path: Path) -> Dict[str, Any]:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"mime_type": "image/jpeg", "data": data}


def call_gemini(*, model: str, api_key: str, image_path: Path, prompt: str) -> str:
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:  # pragma: no cover - depends on optional package.
        raise RuntimeError("Install google-genai to run Gemini baseline: python -m pip install google-genai") from exc

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_path.read_bytes(), mime_type="image/jpeg"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.0,
            max_output_tokens=128,
            response_mime_type="application/json",
        ),
    )
    return str(getattr(response, "text", "") or "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--timeline-csv", type=Path, required=True)
    parser.add_argument("--evidence-scorer", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--frame-cache", type=Path, default=Path("outputs/vlm_gemini_frames"))
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--history-items", type=int, default=6)
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "")
    records = build_records(args.summary_csv, args.timeline_csv, args.evidence_scorer)
    iteration_index = build_iteration_index(args.summary_csv)
    previous_by_video: Dict[str, List[Record]] = defaultdict(list)
    rows: List[Dict[str, Any]] = []
    eval_records: List[Record] = []
    preds: List[str] = []

    for idx, record in enumerate(sorted(records, key=lambda rec: (video_key(rec.video), rec.frame, rec.claim_id))):
        if args.limit and idx >= int(args.limit):
            break
        item = iteration_index.get((record.video, record.frame), {})
        frame_path = extract_frame(record.video, record.frame, args.frame_cache)
        history = compact_history(record, previous_by_video, int(args.history_items))
        prompt = make_prompt(record, item, history)
        raw = ""
        pred = ""
        if not args.dry_run:
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Set it only in the current shell/session; "
                    "do not pass API keys on the command line."
                )
            if frame_path is None:
                raw = '{"decision":"unresolved","reason":"frame extraction failed"}'
            else:
                raw = call_gemini(model=str(args.model), api_key=api_key, image_path=frame_path, prompt=prompt)
            pred = parse_decision(raw)
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
                "model": args.model,
                "prompt": prompt,
                "raw_response": raw,
                "pred": pred,
            }
        )
        previous_by_video[record.video].append(record)

    metrics: Dict[str, Any] = {
        "model": args.model,
        "dry_run": bool(args.dry_run),
        "rows": len(rows),
        "prompt_protocol": "image + active claim + bounded procedure history + compact system evidence",
    }
    if preds:
        metrics.update(summarize_predictions(eval_records, preds))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    write_csv(args.output_csv, rows)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
