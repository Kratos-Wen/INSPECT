"""Train a candidate-level reveal calibrator from assistant geometry only."""

from __future__ import annotations

import argparse
import json
import numpy as np
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspect_system.active_view.object_centric_evidence_memory import (
    ObjectCentricEvidenceMemory,
)
from inspect_system.active_view.object_centric_reveal import (
    ObjectCentricCalibratedRevealModel,
)
from inspect_system.active_view.reveal_model import PriorTableRevealModel
from scripts.evaluate_continual_object_centric_memory import (
    read_records,
    select_kernel_config,
)
from scripts.evaluate_evidence_affordance_memory import select_config


TRANSPORT_COMPATIBLE_MODES = {
    "local_surface_motion",
    "relation_surface_motion",
    "parallax",
    "spherical_bearing",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-policy", type=Path, required=True)
    parser.add_argument("--geometry-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--gain-deadband", type=float, default=0.05)
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
    parser.add_argument(
        "--gain-target-mode",
        choices=("symmetric_deadband", "minimum_improvement"),
        default="symmetric_deadband",
    )
    parser.add_argument("--calibration-strength", type=float, default=1.0)
    parser.add_argument("--support-strength", type=float, default=2.0)
    parser.add_argument(
        "--geometry-scope",
        choices=("all_candidates", "top_action_family"),
        default="all_candidates",
        help=(
            "Apply the continuous geometry residual globally or only to lift "
            "reflection-equivalent actions inside the most likely coarse family."
        ),
    )
    parser.add_argument(
        "--transferable-only",
        action="store_true",
        help="Train only on events retained by the causal transfer gate.",
    )
    parser.add_argument(
        "--equal-transition-weight",
        action="store_true",
        help=(
            "Ignore transfer confidence and retain every geometry record with "
            "equal mass per source transition. This is an ungated diagnostic."
        ),
    )
    parser.add_argument(
        "--family-conditioned-bases",
        action="store_true",
        help="Select one assistant-only observation basis per counterfactual family.",
    )
    parser.add_argument(
        "--transport-compatible-bases",
        action="store_true",
        help=(
            "Restrict basis selection to descriptors with the same coordinate "
            "semantics for assistant motion and robot lattice displacements."
        ),
    )
    parser.add_argument(
        "--allowed-geometry-mode",
        action="append",
        choices=tuple(sorted(TRANSPORT_COMPATIBLE_MODES)),
        default=[],
        help=(
            "Optionally restrict assistant-only basis selection to one or more "
            "predeclared cross-embodiment coordinate systems."
        ),
    )
    parser.add_argument(
        '--geometry-action-conditioning',
        choices=('full', 'identifiable'),
        default='full',
        help=(
            'Limit unsigned geometry residuals to identifiable motion '
            'magnitude, preserving the base policy directional order.'
        ),
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
    if args.equal_transition_weight:
        if args.transferable_only:
            raise ValueError(
                "--equal-transition-weight and --transferable-only are mutually exclusive."
            )
        if args.balance_unit != "transition":
            raise ValueError(
                "--equal-transition-weight currently requires --balance-unit transition."
            )
        transition_sizes = Counter(row.transition_key for row in records)
        records = [
            replace(
                row,
                weight=1.0 / max(1, transition_sizes[row.transition_key]),
            )
            for row in records
        ]
    if not records:
        raise ValueError("No geometry records remain after filtering.")
    selection_objective = (
        "gain_mae"
        if args.target_field == "resolution_decidability_gain"
        else "brier"
    )
    discrete = select_config(
        records,
        online=False,
        objective=selection_objective,
    )
    requested_modes = set(args.allowed_geometry_mode)
    allowed_modes = (
        requested_modes
        if requested_modes
        else (
            TRANSPORT_COMPATIBLE_MODES
            if args.transport_compatible_bases
            else None
        )
    )
    kernel = select_kernel_config(
        records,
        discrete,
        online=False,
        allowed_geometry_modes=allowed_modes,
        objective=selection_objective,
    )
    memory = ObjectCentricEvidenceMemory(discrete, kernel).fit(
        (
            row.evidence_context,
            row.motion,
            row.target,
            row.weight,
        )
        for row in records
    )
    family_memories: dict[str, ObjectCentricEvidenceMemory] = {}
    family_configs: dict[str, dict[str, object]] = {}
    if args.family_conditioned_bases:
        for family in sorted({row.counterfactual for row in records}):
            family_records = [
                row for row in records if row.counterfactual == family
            ]
            family_videos = {row.video for row in family_records}
            if len(family_records) < 8 or len(family_videos) < 3:
                continue
            family_discrete = select_config(
                family_records,
                online=False,
                objective=selection_objective,
            )
            family_kernel = select_kernel_config(
                family_records,
                family_discrete,
                online=False,
                allowed_geometry_modes=allowed_modes,
                objective=selection_objective,
            )
            if (
                allowed_modes is not None
                and family_kernel.geometry_mode not in allowed_modes
            ):
                # Cross-validation returns a disabled high-prior fallback when
                # this family has no transport-compatible coordinate basis.
                # Omitting it lets runtime use the validated global memory
                # instead of replacing that memory with an identity factor of
                # one.
                family_configs[family] = {
                    "events": len(family_records),
                    "videos": len(family_videos),
                    "status": "global_transport_fallback",
                    "rejected_geometry_mode": family_kernel.geometry_mode,
                }
                continue
            family_memory = ObjectCentricEvidenceMemory(
                family_discrete,
                family_kernel,
            ).fit(
                (
                    row.evidence_context,
                    row.motion,
                    row.target,
                    row.weight,
                )
                for row in family_records
            )
            family_memories[family] = family_memory
            family_configs[family] = {
                "events": len(family_records),
                "videos": len(family_videos),
                "discrete": asdict(family_discrete),
                "kernel": asdict(family_kernel),
            }
    targets = np.asarray([row.target for row in records], dtype=np.float64)
    weights = np.asarray([row.weight for row in records], dtype=np.float64)
    gain_scale = max(
        0.05,
        float(np.average(np.abs(targets), weights=weights)),
    )
    model = ObjectCentricCalibratedRevealModel(
        base_model=PriorTableRevealModel.load(args.base_policy),
        memory=memory,
        family_memories=family_memories,
        calibration_strength=args.calibration_strength,
        geometry_action_conditioning=args.geometry_action_conditioning,
        support_strength=args.support_strength,
        gain_scale=gain_scale,
        geometry_scope=args.geometry_scope,
    )
    model.save(args.output)

    report = {
        'geometry_action_conditioning': args.geometry_action_conditioning,
        "output": str(args.output),
        "base_policy": str(args.base_policy),
        "geometry_jsonl": str(args.geometry_jsonl),
        "events": len(records),
        "effective_transitions": sum(row.weight for row in records),
        "videos": len({row.video for row in records}),
        "references": dict(Counter(row.motion.reference_type for row in records)),
        "gain_deadband": args.gain_deadband,
        "gain_target_mode": args.gain_target_mode,
        "target_field": args.target_field,
        "balance_unit": args.balance_unit,
        "discrete_config": asdict(discrete),
        "kernel_config": asdict(kernel),
        "family_conditioned_bases": bool(args.family_conditioned_bases),
        "transport_compatible_bases": bool(args.transport_compatible_bases),
        "allowed_geometry_modes": sorted(allowed_modes or []),
        "family_configs": family_configs,
        "calibration_strength": args.calibration_strength,
        "support_strength": args.support_strength,
        "geometry_scope": args.geometry_scope,
        "gain_scale": gain_scale,
        "selection_protocol": "assistant-only leave-one-video-out",
        "selection_objective": selection_objective,
        "causal_transfer_gate_required": bool(args.transferable_only),
        "transferable_only": bool(args.transferable_only),
        "equal_transition_weight": bool(args.equal_transition_weight),
        "uses_robot_view_training": False,
        "candidate_view_images_used": False,
        "ground_truth_boxes_used": False,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
