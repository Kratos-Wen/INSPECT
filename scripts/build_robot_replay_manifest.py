from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


_PREV_STATE_BY_STEP = {
    "step1": "S1",
    "step2": "S1",
    "step3": "S2",
    "step4": "S3",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def build_replay_records(
    view_manifest: Path,
    trial_gt: Path,
) -> list[dict[str, Any]]:
    """Join RGB paths with trial labels without copying perception outputs."""

    views = _read_csv(view_manifest)
    gt_by_trial = {
        str(row.get("trial_id", "")).strip(): row
        for row in _read_csv(trial_gt)
        if str(row.get("trial_id", "")).strip()
    }
    records: list[dict[str, Any]] = []
    for view in views:
        trial_id = str(view.get("trial_id", "")).strip()
        if trial_id not in gt_by_trial:
            raise ValueError(f"Missing trial GT for {trial_id!r}")
        gt = dict(gt_by_trial[trial_id])
        step = str(gt.get("target_step", "")).strip().lower()
        if step not in _PREV_STATE_BY_STEP:
            raise ValueError(f"Unsupported target step {step!r} for {trial_id}")
        target_state = f"S{step.removeprefix('step')}"
        view_id = str(view.get("view_id", "")).strip().upper()
        image_path = str(view.get("image_path", "")).strip()
        if not view_id or not image_path:
            raise ValueError(f"Incomplete view-manifest row for {trial_id}: {view}")
        metadata = {
            **gt,
            "target_state": target_state,
            "trial_id": trial_id,
            "episode_id": trial_id,
            "view_id": view_id,
            "manifest_source": "robot_view_rgb_and_trial_gt",
            "contains_view_utility": False,
        }
        records.append(
            {
                "image_path": image_path,
                "observation_id": f"{trial_id}_{view_id}",
                "episode_id": trial_id,
                "view_id": view_id,
                "reset_tracker": True,
                "prev_state": _PREV_STATE_BY_STEP[step],
                "candidate_states": [target_state],
                "metadata": metadata,
            }
        )
    records.sort(key=lambda item: (item["episode_id"], item["view_id"]))
    return records


def write_jsonl(records: Iterable[dict[str, Any]], output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a label-preserving robot replay manifest from RGB and trial GT CSV files."
    )
    parser.add_argument("--view-manifest", type=Path, required=True)
    parser.add_argument("--trial-gt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_replay_records(args.view_manifest, args.trial_gt)
    write_jsonl(records, args.output)
    print(json.dumps({"records": len(records), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
