"""Quick local evaluation using IFEval (subset) for fast iteration.

Usage:
    python3 /app/evaluate_local.py                 # eval adapter at /app/output
    python3 /app/evaluate_local.py --limit 20      # fewer samples for speed
    python3 /app/evaluate_local.py --base-only      # eval base model without adapter
"""

import argparse
import json
import os
import subprocess
import sys


def run_ifeval(model_path, adapter_path=None, limit=50, output_dir="/tmp/local_ifeval"):
    model_args = f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True,gpu_memory_utilization=0.8,max_model_len=4096"
    if adapter_path:
        model_args += f",enable_lora=True,max_lora_rank=64,lora_local_path={adapter_path}"
    cmd = [
        "lm_eval",
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", "ifeval",
        "--batch_size", "auto",
        "--output_path", output_dir,
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("lm_eval failed!")
        return None

    # Parse results
    import glob
    # lm_eval creates subdirectories like output_dir/model_name/results_*.json
    results_files = glob.glob(os.path.join(output_dir, "**", "results*.json"), recursive=True)
    if not results_files:
        print("No results.json found")
        return None

    with open(results_files[0]) as f:
        results = json.load(f)

    ifeval = results.get("results", {}).get("ifeval", {})
    return {
        "prompt_level_strict": ifeval.get("prompt_level_strict_acc,none", 0.0),
        "prompt_level_loose": ifeval.get("prompt_level_loose_acc,none", 0.0),
        "inst_level_strict": ifeval.get("inst_level_strict_acc,none", 0.0),
        "inst_level_loose": ifeval.get("inst_level_loose_acc,none", 0.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", default="/app/output")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--base-only", action="store_true")
    args = parser.parse_args()

    model_path = "/models/Qwen2.5-3B-Instruct"

    if args.base_only:
        print("=== Evaluating base model (no adapter) ===")
        scores = run_ifeval(model_path, limit=args.limit, output_dir="/tmp/base_ifeval")
    else:
        print(f"=== Evaluating adapter at {args.adapter_path} ===")
        scores = run_ifeval(model_path, args.adapter_path, args.limit, "/tmp/adapter_ifeval")

    if scores:
        print("\n" + "=" * 50)
        print("IFEval Results (approximate, subset evaluation):")
        print("=" * 50)
        for k, v in scores.items():
            print(f"  {k}: {v:.4f}")
        print(f"\nNote: This uses --limit {args.limit}. Full eval uses all ~500 prompts.")
    else:
        print("Evaluation failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
