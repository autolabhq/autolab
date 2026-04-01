#!/bin/bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-/logs/verifier}"
mkdir -p "$LOG_DIR"

fail_reward() {
    local reason="$1"
    echo "0.0" > "$LOG_DIR/reward.txt"
    printf '{"reward": 0.0, "reason": "%s"}\n' "$reason" > "$LOG_DIR/reward.json"
    cat "$LOG_DIR/reward.json"
    exit 0
}

check_hash() {
    local file="$1"
    local expected_file="$2"
    local actual expected
    actual="$(sha256sum "$file" | awk '{print $1}')"
    expected="$(awk '{print $1}' "$expected_file")"
    if [ "$actual" != "$expected" ]; then
        echo "Protected file modified: $file"
        fail_reward "protected_file_modified"
    fi
}

# === Guard: check protected files ===
echo "=== Checking protected files ==="
check_hash /app/train_config.yaml /orig/train_config.yaml.sha256
check_hash /app/convert_to_sharegpt.py /orig/convert_to_sharegpt.py.sha256
check_hash /app/train.sh /orig/train.sh.sha256

if [ ! -f /app/select_data.py ]; then
    fail_reward "no_select_data_py"
fi

# === Phase 1: Train with agent's data selection ===
echo "=== PHASE 1: Training with agent's data selection ==="
bash /app/train.sh 2>&1 | tee "$LOG_DIR/agent_train.log"

if [ ! -f /app/output/adapter_config.json ]; then
    fail_reward "agent_training_failed"
fi

# Move agent adapter aside
cp -r /app/output /tmp/agent_output

# === Phase 1.5: Train baseline adapter at runtime if needed ===
if [ ! -f /orig/baseline_output/adapter_config.json ]; then
    echo "=== PHASE 1.5: Training baseline model on GPU runtime ==="
    rm -rf /app/output /app/data/selected_train*.json
    cp /orig/select_data.py /app/select_data.py
    bash /app/train.sh 2>&1 | tee "$LOG_DIR/baseline_train.log"

    if [ ! -f /app/output/adapter_config.json ]; then
        fail_reward "baseline_training_failed"
    fi

    cp -r /app/output /tmp/baseline_output
    rm -rf /app/output /app/data/selected_train*.json
else
    cp -r /orig/baseline_output /tmp/baseline_output
fi

# === Phase 2: Evaluate agent adapter on IFEval ===
echo "=== PHASE 2: Evaluating agent model on IFEval ==="
lm_eval --model vllm \
    --model_args "pretrained=/models/Qwen2.5-3B-Instruct,lora_local_path=/tmp/agent_output,dtype=bfloat16,trust_remote_code=True,gpu_memory_utilization=0.8,max_model_len=4096,enable_lora=True,max_lora_rank=64" \
    --tasks ifeval \
    --batch_size auto \
    --output_path /tmp/agent_ifeval \
    2>&1 | tee "$LOG_DIR/agent_eval.log"

# === Phase 3: Evaluate baseline adapter on IFEval ===
echo "=== PHASE 3: Evaluating baseline model on IFEval ==="
lm_eval --model vllm \
    --model_args "pretrained=/models/Qwen2.5-3B-Instruct,lora_local_path=/tmp/baseline_output,dtype=bfloat16,trust_remote_code=True,gpu_memory_utilization=0.8,max_model_len=4096,enable_lora=True,max_lora_rank=64" \
    --tasks ifeval \
    --batch_size auto \
    --output_path /tmp/baseline_ifeval \
    2>&1 | tee "$LOG_DIR/baseline_eval.log"

# === Phase 4: Compute reward (multiplier scoring) ===
python3 - <<'PY'
import json
import glob
import os
from pathlib import Path

LOG_DIR = os.environ.get("LOG_DIR", "/logs/verifier")
REFERENCE = 0.42


def parse_ifeval_scores(output_path):
    results_files = glob.glob(os.path.join(output_path, "**", "results*.json"), recursive=True)
    if not results_files:
        return {"prompt_level_strict": 0.0, "inst_level_strict": 0.0}
    with open(results_files[0]) as f:
        results = json.load(f)
    r = results.get("results", {}).get("ifeval", {})
    return {
        "prompt_level_strict": r.get("prompt_level_strict_acc,none", 0.0),
        "inst_level_strict": r.get("inst_level_strict_acc,none", 0.0),
    }


agent_scores = parse_ifeval_scores("/tmp/agent_ifeval")
baseline_scores = parse_ifeval_scores("/tmp/baseline_ifeval")

agent_acc = agent_scores["prompt_level_strict"]
baseline_acc = baseline_scores["prompt_level_strict"]

# Multiplier reward
if agent_acc <= 0:
    reward = 0.0
else:
    reward = agent_acc / REFERENCE

payload = {
    "reward": round(reward, 4),
    "metric": round(agent_acc, 4),
    "metric_name": "ifeval_prompt_strict",
    "agent_prompt_strict": round(agent_acc, 4),
    "agent_inst_strict": round(agent_scores["inst_level_strict"], 4),
    "baseline_prompt_strict": round(baseline_acc, 4),
    "baseline_inst_strict": round(baseline_scores["inst_level_strict"], 4),
    "improvement": round(agent_acc - baseline_acc, 4),
}

Path(f"{LOG_DIR}/reward.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
Path(f"{LOG_DIR}/reward.txt").write_text(f"{payload['reward']}\n")
print(json.dumps(payload, indent=2))
PY
