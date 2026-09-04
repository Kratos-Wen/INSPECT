import csv
import json
from pathlib import Path

from scripts.finalize_vlm_baseline_artifact import finalize_artifact


def test_finalized_vlm_artifact_removes_local_paths(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    rows_path = tmp_path / "rows.csv"
    manifest_path = tmp_path / "checkpoint.json"
    summary_path.write_text(
        json.dumps(
            {
                "provider": "qwen_local",
                "prompt_protocol_version": "triage_v2",
                "non_oracle": True,
                "api_errors": 0,
                "rows": 60,
                "balanced_sampling_after_eligibility_filter": True,
                "eligible_records_available": 1315,
                "model": "local_inputs/model",
                "cache_jsonl": "local_inputs/cache.jsonl",
            }
        ),
        encoding="utf-8",
    )
    fieldnames = [
        "video",
        "image_path",
        "provider",
        "model",
        "prompt_version",
        "latency_sec",
    ]
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(60):
            writer.writerow(
                {
                    "video": f"local_inputs/video_{index}.mp4",
                    "image_path": f"local_inputs/frame_{index}.jpg",
                    "provider": "qwen_local",
                    "model": "local_inputs/model",
                    "prompt_version": "triage_v2",
                    "latency_sec": "1.0",
                }
            )
    manifest_path.write_text(
        json.dumps(
            {
                "model_id": "Qwen/Qwen3-VL-4B-Instruct",
                "aggregate_sha256": "a" * 64,
                "total_bytes": 123,
            }
        ),
        encoding="utf-8",
    )

    summary, rows, fields = finalize_artifact(summary_path, rows_path, manifest_path)

    assert summary["model"] == "Qwen/Qwen3-VL-4B-Instruct"
    assert summary["source_paths_anonymized"] is True
    assert "cache_jsonl" not in summary
    assert "checkpoint_sha256" in fields
    assert rows[0]["video"] == "video_0.mp4"
    assert rows[0]["image_path"] == "frame_0.jpg"
    assert "local_inputs" not in json.dumps([summary, rows])
