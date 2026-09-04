"""Measure robot inspection as assistant reveal-event supervision increases."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspect_system.active_view.evidence_transport import (  # noqa: E402
    HierarchicalEvidenceTransportModel,
    action_family,
)
from inspect_system.active_view.trace_event_miner import MinedEvent  # noqa: E402


def stable_hash(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def coverage_order(events: list[MinedEvent], seed: int) -> list[MinedEvent]:
    """Create a nested order using assistant metadata only.

    The greedy criterion first covers unseen claim, action-family, role, and
    source-video values. Robot observations and utility labels are never read.
    """

    remaining = list(events)
    ordered: list[MinedEvent] = []
    seen_claims: set[str] = set()
    seen_actions: set[str] = set()
    seen_roles: set[str] = set()
    seen_videos: set[str] = set()
    while remaining:
        def key(event: MinedEvent) -> tuple[int, int, int, int, int]:
            video = Path(event.video).name.lower()
            return (
                int(event.claim_id not in seen_claims),
                int(action_family(event.relative_action) not in seen_actions),
                int(event.evidence_role not in seen_roles),
                int(video not in seen_videos),
                -stable_hash(seed, event.event_id),
            )

        selected = max(remaining, key=key)
        remaining.remove(selected)
        ordered.append(selected)
        seen_claims.add(selected.claim_id)
        seen_actions.add(action_family(selected.relative_action))
        seen_roles.add(selected.evidence_role)
        seen_videos.add(Path(selected.video).name.lower())
    return ordered


def mean_std(values: Iterable[float]) -> dict[str, float]:
    rows = list(values)
    return {
        "mean": statistics.fmean(rows),
        "std": statistics.pstdev(rows) if len(rows) > 1 else 0.0,
        "min": min(rows),
        "max": max(rows),
    }


def evaluator_command(
    args: argparse.Namespace,
    model_path: Path,
    output_json: Path,
    output_rows: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(args.evaluator),
        "--observations", str(args.observations),
        "--trial-gt", str(args.trial_gt),
        "--evidence-scorer", str(args.evidence_scorer),
        "--reveal-model", str(model_path),
        "--requirement-report", str(args.requirement_report),
        "--requirement-calibration", str(args.requirement_calibration),
        "--current-geometry", str(args.current_geometry),
        "--evaluation-utility-csv", str(args.evaluation_utility_csv),
        "--output-json", str(output_json),
        "--output-rows-json", str(output_rows),
        "--lambda-cost", str(args.lambda_cost),
        "--tau-view", str(args.tau_view),
        "--commit-identity-confidence", str(args.commit_identity_confidence),
        "--commit-identity-margin", str(args.commit_identity_margin),
        "--commit-alternative-identity-confidence",
        str(args.commit_alternative_identity_confidence),
        "--commit-counterfactual-margin", str(args.commit_counterfactual_margin),
        "--commit-relation-threshold", str(args.commit_relation_threshold),
        "--only-variant", "INSPECT",
    ]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--trial-gt", type=Path, required=True)
    parser.add_argument("--evidence-scorer", type=Path, required=True)
    parser.add_argument("--requirement-report", type=Path, required=True)
    parser.add_argument("--requirement-calibration", type=Path, required=True)
    parser.add_argument("--current-geometry", type=Path, required=True)
    parser.add_argument("--evaluation-utility-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--budget", type=int, action="append", default=[])
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--base-concentration", type=float, default=4.0)
    parser.add_argument("--hierarchy-strength", type=float, default=4.0)
    parser.add_argument("--risk-beta", type=float, default=0.0)
    parser.add_argument("--gain-power", type=float, default=1.0)
    parser.add_argument("--confidence-scale", type=float, default=0.0)
    parser.add_argument("--lambda-cost", type=float, default=0.05)
    parser.add_argument("--tau-view", type=float, default=0.02)
    parser.add_argument("--commit-identity-confidence", type=float, default=0.25)
    parser.add_argument("--commit-identity-margin", type=float, default=-0.08)
    parser.add_argument(
        "--commit-alternative-identity-confidence", type=float, default=0.65
    )
    parser.add_argument("--commit-counterfactual-margin", type=float, default=0.0)
    parser.add_argument("--commit-relation-threshold", type=float, default=0.18)
    args = parser.parse_args()

    training = json.loads(args.training_report.read_text(encoding="utf-8"))
    events = [MinedEvent(**row) for row in training.get("trainable_events") or []]
    if not events:
        raise ValueError("Training report contains no trainable events")
    budgets = sorted(set(args.budget or [0, 2, 4, 8, 13, 18, len(events)]))
    if budgets[0] < 0 or budgets[-1] > len(events):
        raise ValueError(f"Budgets must lie in [0, {len(events)}]")
    seeds = sorted(set(args.seed or [0, 1, 2, 3, 4]))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    cache: dict[tuple[int, tuple[str, ...]], dict[str, Any]] = {}
    for budget in budgets:
        active_seeds = [seeds[0]] if budget in {0, len(events)} else seeds
        for seed in active_seeds:
            selected = coverage_order(events, seed)[:budget]
            event_ids = tuple(sorted(event.event_id for event in selected))
            cache_key = (budget, event_ids)
            if cache_key in cache:
                run = dict(cache[cache_key])
                run["seed"] = seed
                runs.append(run)
                continue

            stem = f"events_{budget:02d}_seed_{seed:02d}"
            model_path = args.output_dir / f"{stem}.json"
            result_path = args.output_dir / f"{stem}_eval.json"
            rows_path = args.output_dir / f"{stem}_rows.json"
            model = HierarchicalEvidenceTransportModel(
                base_concentration=args.base_concentration,
                hierarchy_strength=args.hierarchy_strength,
                risk_beta=args.risk_beta,
                gain_power=args.gain_power,
                confidence_scale=args.confidence_scale,
            ).fit_events(selected)
            model.save(model_path)
            completed = subprocess.run(
                evaluator_command(args, model_path, result_path, rows_path),
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Evaluation failed for {stem}:\n{completed.stderr}\n{completed.stdout}"
                )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            metrics = dict(result["results"]["INSPECT"])
            run = {
                "budget": budget,
                "seed": seed,
                "training_videos": len({Path(event.video).name for event in selected}),
                "selected_event_ids": list(event_ids),
                "selected_utility": float(metrics["selected_utility"]),
                "gain": float(metrics["gain"]),
                "resolve_at_1": float(metrics["resolve_at_1"]),
                "regret": float(metrics["regret"]),
                "moved": float(metrics["moved"]),
                "no_improvement": float(metrics["no_improvement"]),
            }
            cache[cache_key] = dict(run)
            runs.append(run)

    curve = []
    for budget in budgets:
        subset = [row for row in runs if row["budget"] == budget]
        curve.append(
            {
                "budget": budget,
                "runs": len(subset),
                "training_videos": sorted({row["training_videos"] for row in subset}),
                "selected_utility": mean_std(row["selected_utility"] for row in subset),
                "gain": mean_std(row["gain"] for row in subset),
                "resolve_at_1": mean_std(row["resolve_at_1"] for row in subset),
                "regret": mean_std(row["regret"] for row in subset),
                "no_improvement": mean_std(row["no_improvement"] for row in subset),
            }
        )

    payload = {
        "protocol": {
            "curve_scope": "relative reveal policy pi; requirement model and verifier frozen",
            "event_subset_selection": (
                "nested deterministic coverage over assistant claim, action family, "
                "evidence role, and source video; robot labels are not consulted"
            ),
            "robot_view_labels_used_for_training_or_subset_selection": False,
            "candidate_view_images_used": False,
            "training_events_available": len(events),
            "training_videos_available": len({Path(event.video).name for event in events}),
            "budgets": budgets,
            "seeds": seeds,
        },
        "curve": curve,
        "runs": runs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"protocol": payload["protocol"], "curve": curve}, indent=2))


if __name__ == "__main__":
    main()
