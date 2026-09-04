"""Nested held-out evaluation of continual object-centric evidence atoms."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspect_system.active_view.evidence_affordance_memory import (
    EvidenceAffordanceContext,
)
from inspect_system.active_view.object_centric_evidence_memory import (
    ObjectCentricEvidenceMemory,
    ObjectCentricKernelConfig,
    ObjectCentricMotion,
)
from inspect_system.active_view.ontology import normalize_key
from scripts.evaluate_evidence_affordance_memory import (
    evaluate_baseline,
    evaluate_memory,
    metrics,
    row_result,
    select_config,
    transition_groups,
)


def kernel_configs() -> tuple[ObjectCentricKernelConfig, ...]:
    configs = []
    bandwidths = (
        (0.25, 0.35, 0.35, 0.45),
        (0.45, 0.55, 0.45, 0.65),
        (0.75, 0.85, 0.70, 0.90),
    )
    contexts = ((0.35, 0.08), (0.75, 0.30), (1.0, 1.0))
    for mode in (
        "parallax",
        "axis_invariant",
        "direction",
        "direction_radius_roll",
        "camera_pose",
        "surface_hybrid",
        "local_surface_motion",
        "relation_surface_motion",
        "spherical_bearing",
    ):
        for azimuth, elevation, radius, rotation in bandwidths:
            for semantic, unrelated in contexts:
                for strength in (1.0, 2.5, 6.0):
                    for merge in (0.0, 0.65):
                        for session_boost in (1.5, 4.0, 12.0):
                            configs.append(
                                ObjectCentricKernelConfig(
                                    azimuth_bandwidth=azimuth,
                                    elevation_bandwidth=elevation,
                                    radius_bandwidth=radius,
                                    rotation_bandwidth=rotation,
                                    geometry_mode=mode,
                                    semantic_backoff=semantic,
                                    unrelated_backoff=unrelated,
                                    reference_mismatch=0.15,
                                    kernel_prior_strength=strength,
                                    session_boost=session_boost,
                                    merge_radius=merge,
                                )
                            )
    # Geometry remains opt-in when inner training folds do not support transfer.
    configs.append(ObjectCentricKernelConfig(kernel_prior_strength=1e6))
    return tuple(configs)


KERNEL_CONFIGS = kernel_configs()


def _key(value: object, fallback: str = "generic") -> str:
    return normalize_key(value) or fallback


@dataclass(frozen=True)
class GeometryRecord:
    event_id: str
    video: str
    transition_key: str
    action: str
    role: str
    counterfactual: str
    claim: str
    motion_source: str
    transferable: bool
    target: float
    weight: float
    motion: ObjectCentricMotion

    @property
    def evidence_context(self) -> EvidenceAffordanceContext:
        return EvidenceAffordanceContext(
            action=self.action,
            role=self.role,
            counterfactual=self.counterfactual,
            claim=self.claim,
        )


def transform_signed_gain(
    value: float,
    deadband: float,
    mode: str = "symmetric_deadband",
) -> float:
    gain = float(value)
    threshold = max(0.0, float(deadband))
    if mode == "minimum_improvement":
        return gain - threshold
    if mode != "symmetric_deadband":
        raise ValueError(f"Unknown signed-gain target mode: {mode}")
    magnitude = abs(gain)
    if magnitude <= threshold:
        return 0.0
    return math.copysign(magnitude - threshold, gain)


def read_records(
    path: Path,
    gain_deadband: float,
    gain_target_mode: str = "symmetric_deadband",
    target_field: str = "signed_counterfactual_margin_gain",
    balance_unit: str = "transition",
) -> list[GeometryRecord]:
    if balance_unit not in {"transition", "episode"}:
        raise ValueError(f"Unknown balance unit: {balance_unit}")
    payloads = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    usable = [
        item
        for item in payloads
        if item.get("status") in {"valid", "low_quality"}
        and isinstance(item.get("descriptor"), Mapping)
        and item.get(target_field) not in {None, ""}
    ]
    cached: list[tuple[Mapping[str, Any], str, str, str, float]] = []
    balance_mass: Counter[str] = Counter()
    for item in usable:
        video = Path(str(item.get("video", "unknown"))).name
        claim = _key(item.get("claim_id"))
        transition = "|".join(
            (
                video,
                str(item.get("before_frame", "")),
                str(item.get("after_frame", "")),
                claim,
            )
        )
        balance_key = (
            transition
            if balance_unit == "transition"
            else "|".join((video, str(item.get("after_frame", "")), claim))
        )
        raw_weight = (
            max(0.0, float(item.get("transfer_weight", 1.0) or 0.0))
            * max(0.0, float(item.get("label_confidence", 1.0) or 0.0))
            * max(0.0, float(item.get("evidence_stability", 1.0) or 0.0))
            * max(0.0, float(item.get("evidence_importance", 1.0) or 0.0))
        )
        balance_mass[balance_key] += (
            1.0 if balance_unit == "transition" else raw_weight
        )
        cached.append((item, video, transition, balance_key, raw_weight))

    records = []
    for item, video, transition, balance_key, raw_weight in cached:
        descriptor = item["descriptor"]
        records.append(
            GeometryRecord(
                event_id=str(item.get("event_id", transition)),
                video=video,
                transition_key=transition,
                action=_key(item.get("relative_action"), "unknown"),
                role=_key(
                    item.get("normalized_evidence_role") or item.get("evidence_role")
                ),
                counterfactual=_key(item.get("counterfactual_family")),
                claim=_key(item.get("claim_id")),
                motion_source=_key(descriptor.get("reference_type"), "unknown"),
                transferable=bool(item.get("transferable", False)),
                target=transform_signed_gain(
                    float(item[target_field]),
                    gain_deadband,
                    gain_target_mode,
                ),
                weight=raw_weight / max(1e-12, float(balance_mass[balance_key])),
                motion=ObjectCentricMotion.from_mapping(descriptor),
            )
        )
    return records


def fit_kernel(
    records: Sequence[GeometryRecord],
    discrete_config: Any,
    kernel_config: ObjectCentricKernelConfig,
) -> ObjectCentricEvidenceMemory:
    memory = ObjectCentricEvidenceMemory(discrete_config, kernel_config)
    return memory.fit(
        (row.evidence_context, row.motion, row.target, row.weight) for row in records
    )


def evaluate_kernel(
    train: Sequence[GeometryRecord],
    test: Sequence[GeometryRecord],
    discrete_config: Any,
    kernel_config: ObjectCentricKernelConfig,
    online: bool,
) -> list[dict[str, Any]]:
    memory = fit_kernel(train, discrete_config, kernel_config)
    results = []
    session_id = test[0].video if test and online else None
    config = {
        "discrete": asdict(discrete_config),
        "kernel": asdict(kernel_config),
    }
    for index, group in enumerate(transition_groups(test)):
        for record in group:
            prediction = memory.predict(
                record.evidence_context, record.motion, session_id=session_id
            )
            row = row_result(
                record,
                prediction.helpful_probability,
                prediction.expected_gain,
                prediction.effective_support,
                prediction.scope,
                transition_index=index,
            )
            row["config"] = config
            results.append(row)
        if online:
            for record in group:
                memory.update(
                    record.evidence_context,
                    record.motion,
                    record.target,
                    record.weight,
                    session_id=session_id,
                    update_shared=True,
                )
    return results


def inner_kernel_score(
    records: Sequence[GeometryRecord],
    discrete_config: Any,
    kernel_config: ObjectCentricKernelConfig,
    online: bool,
    objective: str = "brier",
) -> float:
    predictions = []
    for video in sorted({record.video for record in records}):
        train = [record for record in records if record.video != video]
        test = [record for record in records if record.video == video]
        if train and test:
            predictions.extend(
                evaluate_kernel(
                    train,
                    test,
                    discrete_config,
                    kernel_config,
                    online=online,
                )
            )
    if objective not in {"brier", "gain_mae"}:
        raise ValueError(f"Unknown model-selection objective: {objective}")
    return metrics(predictions).get(objective, math.inf)


def select_kernel_config(
    records: Sequence[GeometryRecord],
    discrete_config: Any,
    online: bool,
    allowed_geometry_modes: set[str] | None = None,
    objective: str = "brier",
) -> ObjectCentricKernelConfig:
    configs = KERNEL_CONFIGS
    if allowed_geometry_modes is not None:
        configs = tuple(
            config
            for config in configs
            if config.geometry_mode in allowed_geometry_modes
            or config.kernel_prior_strength >= 1e5
        )
        if not configs:
            raise ValueError("No kernel configurations match the allowed modes.")
    if not online:
        unique: dict[tuple[tuple[str, Any], ...], ObjectCentricKernelConfig] = {}
        for config in configs:
            signature = tuple(
                sorted(
                    (key, value)
                    for key, value in asdict(config).items()
                    if key != "session_boost"
                )
            )
            unique.setdefault(signature, config)
        configs = tuple(unique.values())
    scored = [
        (
            inner_kernel_score(
                records,
                discrete_config,
                config,
                online,
                objective,
            ),
            index,
            config,
        )
        for index, config in enumerate(configs)
    ]
    return min(scored, key=lambda item: (item[0], item[1]))[2]


def nested_leave_one_video_out(
    records: Sequence[GeometryRecord],
    *,
    objective: str = "brier",
    allowed_geometry_modes: set[str] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {
        "Current Transfer Gate": [],
        "Fixed Action Atoms": [],
        "Online Hierarchical Atoms": [],
        "Static Object-Centric Field": [],
        "Matched Frozen Object-Centric Field": [],
        "Online Object-Centric Field": [],
    }
    selected = []
    for video in sorted({record.video for record in records}):
        train = [record for record in records if record.video != video]
        test = [record for record in records if record.video == video]
        discrete = select_config(train, online=True, objective=objective)
        kernel = select_kernel_config(
            train,
            discrete,
            online=True,
            allowed_geometry_modes=allowed_geometry_modes,
            objective=objective,
        )
        static_discrete = select_config(
            train,
            online=False,
            objective=objective,
        )
        static_kernel = select_kernel_config(
            train,
            static_discrete,
            online=False,
            allowed_geometry_modes=allowed_geometry_modes,
            objective=objective,
        )
        output["Current Transfer Gate"].extend(evaluate_baseline(train, test, "gate"))
        output["Fixed Action Atoms"].extend(evaluate_baseline(train, test, "action"))
        output["Online Hierarchical Atoms"].extend(
            evaluate_memory(train, test, discrete, online=True)
        )
        output["Static Object-Centric Field"].extend(
            evaluate_kernel(
                train, test, static_discrete, static_kernel, online=False
            )
        )
        output["Matched Frozen Object-Centric Field"].extend(
            evaluate_kernel(train, test, discrete, kernel, online=False)
        )
        output["Online Object-Centric Field"].extend(
            evaluate_kernel(train, test, discrete, kernel, online=True)
        )
        selected.append(
            {
                "test_video": video,
                "discrete": asdict(discrete),
                "kernel": asdict(kernel),
                "static_discrete": asdict(static_discrete),
                "static_kernel": asdict(static_kernel),
            }
        )
    return output, selected


def subset_metrics(
    predictions: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        name: {
            "cold_start": metrics(
                [row for row in rows if int(row["transition_index"]) == 0]
            ),
            "after_feedback": metrics(
                [row for row in rows if int(row["transition_index"]) > 0]
            ),
        }
        for name, rows in predictions.items()
    }


def ranking_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    positive = [row for row in rows if int(row["label"]) == 1]
    negative = [row for row in rows if int(row["label"]) == 0]
    if not positive or not negative:
        return {"roc_auc": float("nan"), "average_precision": float("nan")}

    concordant = 0.0
    pair_weight = 0.0
    for left in positive:
        for right in negative:
            weight = float(left["weight"]) * float(right["weight"])
            pair_weight += weight
            if float(left["probability"]) > float(right["probability"]):
                concordant += weight
            elif float(left["probability"]) == float(right["probability"]):
                concordant += 0.5 * weight

    total_positive = sum(float(row["weight"]) for row in positive)
    true_positive = 0.0
    inspected = 0.0
    average_precision = 0.0
    for row in sorted(rows, key=lambda item: float(item["probability"]), reverse=True):
        weight = float(row["weight"])
        inspected += weight
        if int(row["label"]) == 1:
            true_positive += weight
            average_precision += (weight / total_positive) * (
                true_positive / inspected
            )
    return {
        "roc_auc": concordant / pair_weight,
        "average_precision": average_precision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--predictions-json", type=Path, required=True)
    parser.add_argument("--gain-deadband", type=float, default=0.05)
    parser.add_argument(
        "--gain-target-mode",
        choices=("symmetric_deadband", "minimum_improvement"),
        default="symmetric_deadband",
    )
    parser.add_argument(
        "--target-field",
        choices=(
            "signed_counterfactual_margin_gain",
            "resolution_decidability_gain",
        ),
        default="signed_counterfactual_margin_gain",
    )
    parser.add_argument(
        "--balance-unit",
        choices=("transition", "episode"),
        default="transition",
    )
    parser.add_argument("--transferable-only", action="store_true")
    parser.add_argument(
        "--transport-compatible-only",
        action="store_true",
        help="Restrict nested selection to cross-embodiment coordinate bases.",
    )
    args = parser.parse_args()

    records = read_records(
        args.geometry_jsonl,
        args.gain_deadband,
        args.gain_target_mode,
        args.target_field,
        args.balance_unit,
    )
    if args.transferable_only:
        records = [row for row in records if row.transferable]
    objective = (
        "gain_mae"
        if args.target_field == "resolution_decidability_gain"
        else "brier"
    )
    allowed_modes = (
        {
            "local_surface_motion",
            "relation_surface_motion",
            "parallax",
            "spherical_bearing",
        }
        if args.transport_compatible_only
        else None
    )
    predictions, selected = nested_leave_one_video_out(
        records,
        objective=objective,
        allowed_geometry_modes=allowed_modes,
    )
    summary = {
        "protocol": {
            "events": len(records),
            "effective_transitions": float(sum(row.weight for row in records)),
            "videos": len({row.video for row in records}),
            "split": "nested leave-one-video-out with causal prequential updates",
            "hyperparameter_selection": "inner leave-one-video-out on training videos only",
            "selection_objective": objective,
            "target_field": args.target_field,
            "balance_unit": args.balance_unit,
            "online_update_timing": "predict transition, then update from its resolved outcome",
            "net_helpful_gain": (
                "signed margin gain after a zero-symmetric frozen deadband"
                if args.gain_target_mode == "symmetric_deadband"
                else "signed margin gain minus a frozen minimum-improvement threshold"
            ),
            "gain_deadband": args.gain_deadband,
            "gain_target_mode": args.gain_target_mode,
            "geometry": "CoTracker3 correspondences plus endpoint MoGe-2 object/scene transport",
            "duplicate_transition_weighting": (
                "unit mass per resolution episode"
                if args.balance_unit == "episode"
                else "inverse multiplicity"
            ),
            "transport_compatible_only": bool(
                args.transport_compatible_only
            ),
            "transferable_only": bool(args.transferable_only),
            "candidate_view_images_used": False,
            "robot_view_labels_used": False,
            "ground_truth_boxes_used": False,
        },
        "results": {
            name: {**metrics(rows), **ranking_metrics(rows)}
            for name, rows in predictions.items()
        },
        "subsets": subset_metrics(predictions),
        "selected_configs": selected,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.predictions_json.write_text(
        json.dumps(predictions, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
