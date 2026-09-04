"""Download the local Qwen3 model used by INSPECT's grounded dialogue layer."""

from __future__ import annotations

import argparse
from pathlib import Path


MODEL_PRESETS = {
    "0.6b": ("Qwen/Qwen3-0.6B", "models/qwen3-0.6b"),
    "1.7b": ("Qwen/Qwen3-1.7B", "models/qwen3-1.7b"),
    "4b": ("Qwen/Qwen3-4B", "models/qwen3-4b"),
    "8b": ("Qwen/Qwen3-8B", "models/qwen3-8b"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size",
        choices=sorted(MODEL_PRESETS),
        default="1.7b",
        help="Qwen3 size preset. Select the deployed model only after matched quality-latency evaluation.",
    )
    parser.add_argument("--model-id", default="", help="Override HuggingFace model id.")
    parser.add_argument("--output-dir", default="", help="Override local output directory.")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--include-full-repo",
        action="store_true",
        help="Download all files. By default only model/tokenizer/config files are fetched.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset_model_id, preset_output = MODEL_PRESETS[args.size]
    model_id = args.model_id.strip() or preset_model_id
    output_dir = Path(args.output_dir.strip() or preset_output)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[1] / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    allow_patterns = None
    if not args.include_full_repo:
        allow_patterns = [
            "*.json",
            "*.safetensors",
            "*.model",
            "*.txt",
            "*.py",
            "tokenizer*",
            "merges.txt",
            "vocab.json",
        ]
    print(f"Downloading {model_id} -> {output_dir}")
    snapshot_download(
        repo_id=model_id,
        revision=args.revision,
        local_dir=str(output_dir),
        allow_patterns=allow_patterns,
    )
    print(f"Downloaded {model_id} to {output_dir}")


if __name__ == "__main__":
    main()
