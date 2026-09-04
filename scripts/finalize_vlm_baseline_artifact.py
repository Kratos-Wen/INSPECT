"""Validate and anonymize cached local-VLM baseline artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path, PureWindowsPath
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def basename(value: str) -> str:
    text = str(value or "")
    return PureWindowsPath(text).name if text else ""


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def finalize_artifact(
    summary_path: Path,
    rows_path: Path,
    checkpoint_manifest_path: Path,
) -> tuple[dict[str, Any], list[dict[str, str]], list[str]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    with rows_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]

    if summary.get("provider") != "qwen_local":
        raise ValueError("Expected a qwen_local result")
    if summary.get("prompt_protocol_version") != "triage_v2":
        raise ValueError("Expected the symmetric triage_v2 prompt")
    if not summary.get("non_oracle"):
        raise ValueError("Result is not marked non-oracle")
    if int(summary.get("api_errors", -1)) != 0:
        raise ValueError("Result contains inference errors")
    if int(summary.get("rows", -1)) != len(rows):
        raise ValueError("Summary and row counts disagree")
    if len(rows) != 60:
        raise ValueError("Expected the frozen balanced 60-sample protocol")
    if not summary.get("balanced_sampling_after_eligibility_filter"):
        raise ValueError("VLM sample is not restricted to the comparison-method protocol")
    if int(summary.get("eligible_records_available", -1)) != 1315:
        raise ValueError("Expected the 1,315-frame comparable INSPECT pool")

    public_model_id = str(checkpoint["model_id"])
    checkpoint_digest = str(checkpoint["aggregate_sha256"])
    for row in rows:
        if row.get("provider") != "qwen_local":
            raise ValueError("Mixed providers in VLM rows")
        if row.get("prompt_version") != "triage_v2":
            raise ValueError("Mixed prompt versions in VLM rows")
        row["video"] = basename(row.get("video", ""))
        row["image_path"] = basename(row.get("image_path", ""))
        row["model"] = public_model_id
        row["checkpoint_sha256"] = checkpoint_digest

    if "checkpoint_sha256" not in fieldnames:
        model_index = fieldnames.index("model") + 1
        fieldnames.insert(model_index, "checkpoint_sha256")
    latencies = [float(row["latency_sec"]) for row in rows if row.get("latency_sec")]
    summary["model"] = public_model_id
    summary["checkpoint"] = {
        "aggregate_sha256": checkpoint_digest,
        "manifest_sha256": sha256(checkpoint_manifest_path),
        "total_bytes": checkpoint["total_bytes"],
    }
    summary["latency_sec"] = {
        "mean": statistics.fmean(latencies),
        "median": statistics.median(latencies),
        "p95": percentile(latencies, 0.95),
        "includes_first_load": True,
    }
    summary["source_paths_anonymized"] = True
    summary["source_artifact_sha256"] = {
        "summary": sha256(summary_path),
        "rows": sha256(rows_path),
    }
    summary.pop("cache_jsonl", None)
    return summary, rows, fieldnames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    summary, rows, fieldnames = finalize_artifact(
        args.summary,
        args.rows,
        args.checkpoint_manifest,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(args.output_json)
    print(args.output_csv)


if __name__ == "__main__":
    main()
