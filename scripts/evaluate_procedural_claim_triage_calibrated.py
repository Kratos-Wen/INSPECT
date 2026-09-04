"""Evaluate calibrated procedural claim triage for INSPECT.

This script is intentionally separate from
``evaluate_procedural_claim_triage.py`` so the previous fixed-threshold setup
is preserved.  It tests a single verifier decision rule with three pieces that
belong together:

  1. claim-specific threshold calibration,
  2. causal evidence accumulation over past frames only,
  3. explicit supported / contradicted / unresolved triage.

Thresholds are calibrated in leave-one-video-out mode by default: each test
video is evaluated with thresholds selected from all other videos.  The
evidence scorer itself is not retrained here; this isolates whether the final
triage rule is the bottleneck.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspect_system.evidence_scorer import (  # noqa: E402
    PrototypeEvidenceScorer,
    extract_relation_features,
    normalize_claim,
)


OUTCOMES = ("supported", "contradicted", "unresolved")
VALID_STATES = {"S1", "S2", "S3", "S4"}


@dataclass
class Record:
    video: str
    frame: int
    step_id: str
    claim_id: str
    truth: str
    fused_step: str
    step_match: bool
    support_raw: float
    contradiction_raw: float
    visibility_raw: float
    support: float = 0.0
    contradiction: float = 0.0


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
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


def video_key(value: object) -> str:
    return Path(str(value or "")).name


def state_id(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in {"STEP1", "STEP2", "STEP3", "STEP4"}:
        return "S" + text[-1]
    return text


def group_timeline(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in read_csv(path):
        if str(row.get("skip", "")).strip().lower() in {"1", "true", "yes"}:
            continue
        try:
            start = int(float(row.get("start_frame") or 0))
            end = int(float(row.get("end_frame") or start))
        except ValueError:
            continue
        outcome = str(row.get("outcome", "")).strip().lower()
        if outcome not in OUTCOMES:
            continue
        item = dict(row)
        item["start_frame_i"] = start
        item["end_frame_i"] = end
        item["outcome_norm"] = outcome
        grouped.setdefault(video_key(row.get("video")), []).append(item)
    for rows in grouped.values():
        rows.sort(key=lambda item: (int(item["start_frame_i"]), int(item["end_frame_i"])))
    return grouped


def timeline_at(rows: List[Dict[str, Any]], frame: int) -> Optional[Dict[str, Any]]:
    for row in rows:
        if int(row["start_frame_i"]) <= frame <= int(row["end_frame_i"]):
            return row
    return None


def build_records(
    summary_csv: Path,
    timeline_csv: Path,
    scorer_path: Path,
    proposal_top_k: int = 1,
) -> List[Record]:
    scorer = PrototypeEvidenceScorer.load(scorer_path)
    timelines = group_timeline(timeline_csv)
    records: List[Record] = []
    for run in read_csv(summary_csv):
        if str(run.get("returncode", "")).strip() not in {"", "0"}:
            continue
        video = str(run.get("video", ""))
        timeline = timelines.get(video_key(video), [])
        if not timeline:
            continue
        run_dir = Path(str(run.get("run_dir", "")))
        for item in iter_jsonl(run_dir / "iterations.jsonl"):
            frame = int(item.get("frame_index", item.get("frame", -1)))
            trow = timeline_at(timeline, frame)
            if not trow:
                continue
            step = state_id(trow.get("step_id", ""))
            claim = normalize_claim(trow.get("claim_id", ""), step)
            detections = list(item.get("fused_detections") or item.get("raw_detections") or [])
            features = extract_relation_features(
                detections,
                claim_id=claim,
                step_id=step,
                product=trow.get("assembly_set", ""),
                image_shape=(1080, 1920),
            )
            scores = scorer.score_features(features, claim_id=claim, step_id=step)
            fused_step = state_id(item.get("fused_step") or item.get("decision_step"))
            proposal_scores = {
                state_id(key): float(value)
                for key, value in dict(item.get("fusion_scores") or {}).items()
                if state_id(key) in VALID_STATES
            }
            if proposal_scores:
                proposal_steps = [
                    key
                    for key, _ in sorted(
                        proposal_scores.items(),
                        key=lambda pair: (-pair[1], pair[0]),
                    )
                ]
            else:
                proposal_steps = [fused_step]
                runner_up = state_id(item.get("fused_runner_up"))
                if runner_up in VALID_STATES and runner_up not in proposal_steps:
                    proposal_steps.append(runner_up)
            proposal_steps = proposal_steps[: max(1, int(proposal_top_k))]
            records.append(
                Record(
                    video=video,
                    frame=frame,
                    step_id=step,
                    claim_id=claim,
                    truth=str(trow["outcome_norm"]),
                    fused_step=fused_step,
                    step_match=bool(step in VALID_STATES and step in proposal_steps),
                    support_raw=float(scores.get("support_score", 0.0)),
                    contradiction_raw=float(scores.get("contradiction_score", 0.0)),
                    visibility_raw=float(scores.get("visibility_score", 0.0)),
                )
            )
    return records


def apply_causal_ema(records: List[Record], decay: float) -> List[Record]:
    """Fill support/contradiction with causal EMA, reset by video and claim."""
    state: Dict[Tuple[str, str], Tuple[float, float, bool]] = {}
    for rec in sorted(records, key=lambda r: (video_key(r.video), r.frame, r.claim_id)):
        key = (rec.video, rec.claim_id)
        prev_s, prev_c, seen = state.get(key, (0.0, 0.0, False))
        if not seen:
            rec.support = rec.support_raw
            rec.contradiction = rec.contradiction_raw
        else:
            rec.support = decay * prev_s + (1.0 - decay) * rec.support_raw
            rec.contradiction = decay * prev_c + (1.0 - decay) * rec.contradiction_raw
        state[key] = (rec.support, rec.contradiction, True)
    return records


def predict(rec: Record, thresholds: Mapping[str, float], require_step_match_for_support: bool) -> str:
    ts = float(thresholds.get("support_threshold", 0.64))
    tc = float(thresholds.get("contradiction_threshold", 0.64))
    margin = float(thresholds.get("margin_threshold", 0.05))
    if rec.contradiction >= tc and rec.contradiction >= rec.support + margin:
        return "contradicted"
    if rec.support >= ts and rec.support >= rec.contradiction + margin:
        if require_step_match_for_support and not rec.step_match:
            return "unresolved"
        return "supported"
    return "unresolved"


def safe_rate(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return num / den


def f1(tp: int, fp: int, fn: int) -> float:
    precision = safe_rate(tp, tp + fp) or 0.0
    recall = safe_rate(tp, tp + fn) or 0.0
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def summarize_predictions(records: Sequence[Record], preds: Sequence[str]) -> Dict[str, Any]:
    totals = Counter()
    confusion = Counter()
    for rec, pred in zip(records, preds):
        truth = rec.truth
        if truth not in OUTCOMES or pred not in OUTCOMES:
            continue
        totals["total"] += 1
        totals[f"truth_{truth}"] += 1
        totals[f"pred_{pred}"] += 1
        if truth == pred:
            totals["correct"] += 1
        confusion[(truth, pred)] += 1
    class_metrics: Dict[str, Dict[str, Any]] = {}
    f1_values: List[float] = []
    for label in OUTCOMES:
        tp = confusion[(label, label)]
        fp = sum(confusion[(truth, label)] for truth in OUTCOMES if truth != label)
        fn = sum(confusion[(label, pred)] for pred in OUTCOMES if pred != label)
        label_f1 = f1(tp, fp, fn)
        f1_values.append(label_f1)
        class_metrics[label] = {
            "total": totals[f"truth_{label}"],
            "correct": tp,
            "recall": safe_rate(tp, totals[f"truth_{label}"]),
            "precision": safe_rate(tp, tp + fp),
            "f1": label_f1,
        }
    return {
        "triage_accuracy": safe_rate(totals["correct"], totals["total"]),
        "triage_macro_f1": safe_rate(sum(f1_values), len(f1_values)),
        "support_accuracy": safe_rate(confusion[("supported", "supported")], totals["truth_supported"]),
        "contradiction_recall": safe_rate(confusion[("contradicted", "contradicted")], totals["truth_contradicted"]),
        "unresolved_coverage": safe_rate(confusion[("unresolved", "unresolved")], totals["truth_unresolved"]),
        "false_accept_rate_on_non_supported": safe_rate(
            confusion[("contradicted", "supported")] + confusion[("unresolved", "supported")],
            totals["truth_contradicted"] + totals["truth_unresolved"],
        ),
        "safe_non_accept_rate_on_contradicted": safe_rate(
            totals["truth_contradicted"] - confusion[("contradicted", "supported")],
            totals["truth_contradicted"],
        ),
        "confusion": {f"{truth}->{pred}": confusion[(truth, pred)] for truth in OUTCOMES for pred in OUTCOMES},
        "class_metrics": class_metrics,
        "pred_counts": {label: totals[f"pred_{label}"] for label in OUTCOMES},
        "total_iterations": totals["total"],
    }


def objective(metrics: Mapping[str, Any], max_false_accept: float) -> float:
    macro = float(metrics.get("triage_macro_f1") or 0.0)
    false_accept = float(metrics.get("false_accept_rate_on_non_supported") or 0.0)
    penalty = max(0.0, false_accept - max_false_accept) * 3.0
    # Maintain the paper priority: avoid false support, but do not reward a
    # degenerate all-unresolved policy.
    support = float(metrics.get("support_accuracy") or 0.0)
    contradiction = float(metrics.get("contradiction_recall") or 0.0)
    return macro + 0.08 * support + 0.08 * contradiction - penalty


def calibrate_thresholds(
    records: Sequence[Record],
    *,
    require_step_match_for_support: bool,
    max_false_accept: float,
    support_grid: Sequence[float],
    contradiction_grid: Sequence[float],
    margin_grid: Sequence[float],
) -> Dict[str, float]:
    if not records:
        return {"support_threshold": 0.64, "contradiction_threshold": 0.64, "margin_threshold": 0.05}
    best_score = -1e9
    best = {"support_threshold": 0.64, "contradiction_threshold": 0.64, "margin_threshold": 0.05}
    for ts in support_grid:
        for tc in contradiction_grid:
            for margin in margin_grid:
                thresholds = {
                    "support_threshold": float(ts),
                    "contradiction_threshold": float(tc),
                    "margin_threshold": float(margin),
                }
                preds = [predict(rec, thresholds, require_step_match_for_support) for rec in records]
                metrics = summarize_predictions(records, preds)
                score = objective(metrics, max_false_accept=max_false_accept)
                if score > best_score:
                    best_score = score
                    best = thresholds
    return best


def select_thresholds_for_video(
    all_records: Sequence[Record],
    test_video: str,
    *,
    mode: str,
    require_step_match_for_support: bool,
    max_false_accept: float,
    support_grid: Sequence[float],
    contradiction_grid: Sequence[float],
    margin_grid: Sequence[float],
) -> Dict[str, Dict[str, float]]:
    if mode == "global":
        train_records = [rec for rec in all_records if rec.video != test_video]
        return {
            "__global__": calibrate_thresholds(
                train_records,
                require_step_match_for_support=require_step_match_for_support,
                max_false_accept=max_false_accept,
                support_grid=support_grid,
                contradiction_grid=contradiction_grid,
                margin_grid=margin_grid,
            )
        }
    train_records_by_claim: Dict[str, List[Record]] = defaultdict(list)
    for rec in all_records:
        if rec.video == test_video:
            continue
        train_records_by_claim[rec.claim_id].append(rec)
    thresholds: Dict[str, Dict[str, float]] = {}
    global_threshold = calibrate_thresholds(
        [rec for rec in all_records if rec.video != test_video],
        require_step_match_for_support=require_step_match_for_support,
        max_false_accept=max_false_accept,
        support_grid=support_grid,
        contradiction_grid=contradiction_grid,
        margin_grid=margin_grid,
    )
    thresholds["__global__"] = global_threshold
    for claim, claim_records in train_records_by_claim.items():
        # Require at least two classes for claim-specific calibration.
        if len({rec.truth for rec in claim_records}) < 2:
            continue
        thresholds[claim] = calibrate_thresholds(
            claim_records,
            require_step_match_for_support=require_step_match_for_support,
            max_false_accept=max_false_accept,
            support_grid=support_grid,
            contradiction_grid=contradiction_grid,
            margin_grid=margin_grid,
        )
    return thresholds


def parse_grid(text: str) -> List[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--timeline-csv", type=Path, required=True)
    parser.add_argument("--evidence-scorer", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--threshold-mode", choices=["claim", "global"], default="claim")
    parser.add_argument("--ema-decay", type=float, default=0.55)
    parser.add_argument("--max-false-accept", type=float, default=0.10)
    parser.add_argument("--require-step-match-for-support", action="store_true")
    parser.add_argument(
        "--proposal-top-k",
        type=int,
        choices=(1, 2),
        default=1,
        help="Number of online proposal hypotheses expanded into claims.",
    )
    parser.add_argument("--support-grid", default="0.35,0.40,0.45,0.50,0.55,0.60,0.64,0.68,0.72")
    parser.add_argument("--contradiction-grid", default="0.35,0.40,0.45,0.50,0.55,0.60,0.64,0.68,0.72")
    parser.add_argument("--margin-grid", default="0.00,0.03,0.05,0.08,0.10,0.15")
    args = parser.parse_args()

    records = build_records(
        args.summary_csv,
        args.timeline_csv,
        args.evidence_scorer,
        proposal_top_k=int(args.proposal_top_k),
    )
    apply_causal_ema(records, decay=float(args.ema_decay))
    support_grid = parse_grid(args.support_grid)
    contradiction_grid = parse_grid(args.contradiction_grid)
    margin_grid = parse_grid(args.margin_grid)

    all_preds: List[str] = []
    detail_rows: List[Dict[str, Any]] = []
    thresholds_by_video: Dict[str, Dict[str, Dict[str, float]]] = {}
    by_video: Dict[str, List[Record]] = defaultdict(list)
    for rec in records:
        by_video[rec.video].append(rec)
    for video, video_records in sorted(by_video.items(), key=lambda item: video_key(item[0])):
        thresholds_map = select_thresholds_for_video(
            records,
            video,
            mode=args.threshold_mode,
            require_step_match_for_support=bool(args.require_step_match_for_support),
            max_false_accept=float(args.max_false_accept),
            support_grid=support_grid,
            contradiction_grid=contradiction_grid,
            margin_grid=margin_grid,
        )
        thresholds_by_video[video] = thresholds_map
        for rec in sorted(video_records, key=lambda item: item.frame):
            thresholds = thresholds_map.get(rec.claim_id, thresholds_map["__global__"])
            pred = predict(rec, thresholds, bool(args.require_step_match_for_support))
            all_preds.append(pred)
            detail_rows.append(
                {
                    "video": rec.video,
                    "frame": rec.frame,
                    "step_id": rec.step_id,
                    "claim_id": rec.claim_id,
                    "truth": rec.truth,
                    "pred": pred,
                    "correct": int(pred == rec.truth),
                    "fused_step": rec.fused_step,
                    "step_match": int(rec.step_match),
                    "support_raw": f"{rec.support_raw:.6f}",
                    "contradiction_raw": f"{rec.contradiction_raw:.6f}",
                    "support_ema": f"{rec.support:.6f}",
                    "contradiction_ema": f"{rec.contradiction:.6f}",
                    "support_threshold": f"{thresholds['support_threshold']:.4f}",
                    "contradiction_threshold": f"{thresholds['contradiction_threshold']:.4f}",
                    "margin_threshold": f"{thresholds['margin_threshold']:.4f}",
                }
            )

    # detail_rows is in video order.  Reconstruct the corresponding record order.
    ordered_records: List[Record] = []
    for video, video_records in sorted(by_video.items(), key=lambda item: video_key(item[0])):
        ordered_records.extend(sorted(video_records, key=lambda item: item.frame))
    metrics = summarize_predictions(ordered_records, all_preds)
    metrics.update(
        {
            "decision_rule": "leave-one-video calibrated claim triage with causal EMA",
            "threshold_mode": args.threshold_mode,
            "ema_decay": args.ema_decay,
            "max_false_accept": args.max_false_accept,
            "require_step_match_for_support": bool(args.require_step_match_for_support),
            "proposal_top_k": int(args.proposal_top_k),
            "thresholds_by_video": thresholds_by_video,
            "evidence_scorer": str(args.evidence_scorer),
            "summary_csv": str(args.summary_csv),
            "timeline_csv": str(args.timeline_csv),
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    write_csv(
        args.output_csv,
        detail_rows,
        [
            "video",
            "frame",
            "step_id",
            "claim_id",
            "truth",
            "pred",
            "correct",
            "fused_step",
            "step_match",
            "support_raw",
            "contradiction_raw",
            "support_ema",
            "contradiction_ema",
            "support_threshold",
            "contradiction_threshold",
            "margin_threshold",
        ],
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
