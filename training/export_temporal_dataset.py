"""Export temporal GRU training samples from existing run logs."""

from __future__ import annotations

import argparse
import json

from .data import build_temporal_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export temporal training samples from MICA run logs.")
    parser.add_argument("--runs-root", required=True, help="Root directory containing run folders, for example runs_modular.")
    parser.add_argument("--kb", required=True, help="Path to the knowledge-base JSON file.")
    parser.add_argument("--output", required=True, help="Output .pt dataset path.")
    parser.add_argument("--window-size", type=int, default=12, help="Sequence length for temporal training samples.")
    parser.add_argument("--min-stable-confidence", type=float, default=0.72, help="Minimum fused confidence for stable pseudo-labels.")
    parser.add_argument("--min-auto-confidence", type=float, default=0.80, help="Confidence threshold for stronger stable_auto labels.")
    parser.add_argument("--min-visual-confidence", type=float, default=0.28, help="Fallback confidence threshold for low-weight visual pseudo-labels.")
    parser.add_argument("--min-memory-confidence", type=float, default=0.45, help="Minimum memory confidence for memory-consensus labels.")
    parser.add_argument("--min-segment-length", type=int, default=3, help="Minimum contiguous visual-signature segment length for segment labels.")
    parser.add_argument("--min-segment-vote-ratio", type=float, default=0.58, help="Minimum dominant vote ratio for one segment label.")
    parser.add_argument("--min-segment-margin", type=float, default=0.10, help="Minimum relative margin between the top two segment candidates.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional maximum number of run directories to export.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_temporal_dataset(
        runs_root=args.runs_root,
        kb_path=args.kb,
        output_path=args.output,
        window_size=args.window_size,
        min_stable_confidence=args.min_stable_confidence,
        min_auto_confidence=args.min_auto_confidence,
        min_visual_confidence=args.min_visual_confidence,
        min_memory_confidence=args.min_memory_confidence,
        min_segment_length=args.min_segment_length,
        min_segment_vote_ratio=args.min_segment_vote_ratio,
        min_segment_margin=args.min_segment_margin,
        max_runs=args.max_runs,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
