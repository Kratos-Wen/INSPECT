"""Run the released VG-AVS policy on current RGB with its native action format.

This records predictions, not six-view benchmark scores. VG-AVS actions are
planar heading/distance/final-yaw triples, whereas INSPECT uses a 3D lattice.
Any action-space adapter must be specified separately before scoring.
"""

from __future__ import annotations

import argparse
import os
import hashlib
import importlib.util
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Frozen evaluation artifacts from the paper runs, released separately from the
# code. Point INSPECT_ARTIFACTS_ROOT at your local copy (default: artifacts/).
# Missing inputs raise FileNotFoundError with the exact expected path.
ARTIFACTS = Path(os.environ.get("INSPECT_ARTIFACTS_ROOT", "artifacts"))

REVISION = "6481445d8b08091b9817ad56d3c102cc70408d7d"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_native_action(text):
    pattern = r"<head>\s*(-?\d+)\s*</head>\s*<fwd>\s*(\d+)\s*</fwd>\s*<view>\s*(-?\d+)\s*</view>"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    if "<head>" in text[matches[-1].end():]:
        return None
    head, forward, view = map(int, matches[-1].groups())
    if not (-180 < head <= 180 and -180 < view <= 180):
        return None
    return {"heading_degrees": head, "forward_centimeters": forward, "view_degrees": view}


def official_prompts(repository):
    path = repository / "src/open-r1-multimodal/src/open_r1/utils/prompt_templates.py"
    spec = importlib.util.spec_from_file_location("official_vgavs_prompts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ACTION_PROMPT_TEMPLATE, module.GRPO_FORMAT_PROMPT, sha256(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/vgavs/sft_grpo"),
                        help="Local path to the official VG-AVS sft_grpo checkpoint.")
    parser.add_argument("--repository", type=Path, default=Path("external/VG-AVS"),
                        help="Checkout of the official VG-AVS repository (for prompt templates).")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/official_vgavs")
    parser.add_argument("--limit", type=int, default=132)
    parser.add_argument("--precision", choices=["nf4", "bf16"], default="nf4")
    args = parser.parse_args()
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    torch.manual_seed(0)
    action_prompt, format_prompt, prompt_hash = official_prompts(args.repository)
    observations_path = ARTIFACTS / "robot/robot_observations_final_pi_moge_360.jsonl"
    trial_path = ARTIFACTS / "robot/strict_final_policy_component_ablation_rows_132.json"
    observations = {}
    for line in observations_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        metadata = item["metadata"]
        observations[(metadata["trial_id"], item["view_id"])] = {
            "rgb_path": metadata["rgb_path"], "claim_text": metadata["claim_text"],
        }
    trials = json.loads(trial_path.read_text(encoding="utf-8"))["INSPECT"]
    keys = sorted({(t["trial_id"], t["current_view"]) for t in trials})[:args.limit]
    assert len(keys) == min(args.limit, 132)
    protocol = {
        "model_id": "daehyeonchoi/VGAVS-model", "revision": REVISION,
        "checkpoint": "sft_grpo", "precision": args.precision,
        "official_prompt_sha256": prompt_hash,
        "max_new_tokens": 256, "do_sample": False,
        "input": "current RGB and active claim text only",
        "candidate_images_used": False, "outcome_labels_used": False,
        "action_mapping": None, "benchmark_scores": None,
        "scope": "Native-action checkpoint inference; not an official six-view reproduction.",
    }
    protocol_key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    cache_path = args.output / "predictions.jsonl"
    cached = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert row["protocol_key"] == protocol_key, "Different protocol in output cache"
            item = observations[(row["trial_id"], row["current_view"])]
            assert row["image_sha256"] == sha256(item["rgb_path"]), "Changed RGB in output cache"
            cached[(row["trial_id"], row["current_view"])] = row
    pending = [key for key in keys if key not in cached]
    if pending:
        kwargs = {"torch_dtype": torch.bfloat16, "device_map": {"": 0},
                  "attn_implementation": "sdpa", "local_files_only": True}
        if args.precision == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, **kwargs).eval()
        for key in pending:
            item = observations[key]
            image_path = Path(item["rgb_path"])
            question = "Is the following claim true? " + item["claim_text"]
            prompt = action_prompt.format(question=question) + format_prompt
            image = Image.open(image_path).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": prompt},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt",
                               padding=True, truncation=True, max_length=4096).to("cuda:0")
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=256, do_sample=False, use_cache=True,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            new_tokens = generated[:, inputs["input_ids"].shape[1]:]
            answer = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
            row = {
                "trial_id": key[0], "current_view": key[1], "protocol_key": protocol_key,
                "image_sha256": sha256(image_path), "prompt": prompt, "raw_output": answer,
                "action": parse_native_action(answer), "generation_seconds": elapsed,
                "generated_tokens": int(new_tokens.shape[1]),
            }
            with cache_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            cached[key] = row
            print(f"{len(cached)}/132 {key} action={row['action']} seconds={elapsed:.2f}", flush=True)
    rows = [cached[key] for key in keys]
    summary = {
        "protocol": protocol, "completed": len(rows),
        "valid_native_actions": sum(r["action"] is not None for r in rows),
        "mean_generation_seconds": sum(r["generation_seconds"] for r in rows) / max(len(rows), 1),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "source_observations_sha256": sha256(observations_path),
        "trial_manifest_sha256": sha256(trial_path),
        "script_sha256": sha256(__file__),
        "model_files": {
            path.name: {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(args.model.glob("*.safetensors"))
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
