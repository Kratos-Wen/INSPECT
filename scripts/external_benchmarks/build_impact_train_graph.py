"""Build an IMPACT procedure graph from one official data split.

The wrapper calls the graph-mining implementation released with IMPACT while
restricting its input recordings to the requested split. This avoids using
validation or test executions to construct procedural prerequisites.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("impact_graph_miner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import IMPACT graph miner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_split(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impact-repo", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--min-cooccur-frac", type=float, default=0.25)
    parser.add_argument("--min-support-frac", type=float, default=0.20)
    parser.add_argument("--min-confidence", type=float, default=0.90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    task_root = args.impact_repo / "tasks" / "PSR" / "gemini_3_1_pro"
    miner = load_module(task_root / "learn_procedure_graph.py")
    alias_map = json.loads(
        (task_root / "configs" / "component_alias.json").read_text(encoding="utf-8")
    )
    split_file = (
        args.impact_repo
        / "dataset"
        / "ASR"
        / "splits"
        / f"{args.split}.split1.bundle"
    )
    annotation_dir = args.impact_repo / "dataset" / "ASR" / "annotations"
    stems = read_split(split_file)
    event_types = {"install_ok", "remove_ok", "recover_ok"}
    sequences = []
    for stem in stems:
        payload = json.loads(
            (annotation_dir / f"{stem}_asr.json").read_text(encoding="utf-8")
        )
        sequences.append(
            miner.extract_event_sequence(payload, alias_map, event_types)
        )

    minimum_cooccurrence = max(
        3, int(np.ceil(args.min_cooccur_frac * len(stems)))
    )
    minimum_support = max(
        3, int(np.ceil(args.min_support_frac * len(stems)))
    )
    kept, nodes = miner.learn_pairwise_constraints(
        sequences,
        minimum_cooccurrence,
        minimum_support,
        args.min_confidence,
    )
    edges, _ = miner.resolve_conflicts_and_make_dag(kept, nodes)
    graph = {
        "meta": {
            "source": "IMPACT official graph-mining implementation",
            "split": args.split,
            "num_videos": len(stems),
            "min_cooccur_frac": args.min_cooccur_frac,
            "min_support_frac": args.min_support_frac,
            "min_cooccur": minimum_cooccurrence,
            "min_support": minimum_support,
            "min_confidence": args.min_confidence,
            "event_types": sorted(event_types),
            "alias_map_used": True,
        },
        "nodes": sorted(nodes),
        "edges": [[source, target] for source, target in edges],
        "prereq": miner.edges_to_prereq(edges, nodes),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "split": args.split,
                "recordings": len(stems),
                "nodes": len(nodes),
                "edges": len(edges),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
