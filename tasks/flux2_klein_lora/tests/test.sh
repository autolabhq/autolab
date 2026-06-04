#!/bin/bash
set -uo pipefail

###############################################################################
# Verifier for musubi-tuner__flux2_klein_lora
#
#   1. Read baseline_score + reference_score from task.toml (single source of
#      truth — CLAUDE.md #3). tests/task.toml is a byte-identical mirror,
#      copied into /tests/ by the verifier. Drift check below.
#   2. Integrity: SHA256-pin frozen files so agent can't tamper with scoring.
#   3. Loose /app/ whitelist — agents must not leave backup/helper files.
#   4. Require agent's LoRA file exists and has lora keys.
#   5. Evaluate the LoRA with /orig/app/evaluate.py (frozen copy).
#   6. Evaluate baseline (no_lora) with /orig/app/evaluate.py.
#   7. Compute multiplier reward: score / reference_score (no cap), gated by >baseline.
#
# Always exits 0 — failures surface via reward.json.
###############################################################################

LOG_DIR="${LOG_DIR:-/logs/verifier}"
mkdir -p "$LOG_DIR"

fail_reward() {
    local reason="$1"
    printf '0.0\n' > "$LOG_DIR/reward.txt"
    printf '{"reward": 0.0, "reason": "%s"}\n' "$reason" > "$LOG_DIR/reward.json"
    cat "$LOG_DIR/reward.json"
    exit 0
}

echo "============================================"
echo "  FLUX.2 klein LoRA Quality Verifier"
echo "============================================"

# ── 0. Read reference/baseline scores from task.toml ─────────────────────────
#      tests/task.toml is a byte-identical mirror of the root task.toml.
#      Harbor mounts tests/ into /tests/. We prefer the local mirror (for
#      local-run under $SCRIPT_DIR), then /tests/task.toml (in-container).
#      If the ROOT task.toml is also visible, cmp it against the mirror to
#      catch drift (so the mirror never silently falls behind).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_TOML=""
for cand in "$SCRIPT_DIR/task.toml" /tests/task.toml /task.toml /app/task.toml; do
    if [ -f "$cand" ]; then
        TASK_TOML="$cand"
        break
    fi
done
if [ -z "$TASK_TOML" ]; then
    fail_reward "task_toml_not_found"
fi

ROOT_TASK_TOML="$SCRIPT_DIR/../task.toml"
if [ -f "$ROOT_TASK_TOML" ] && [ "$ROOT_TASK_TOML" != "$TASK_TOML" ]; then
    if ! cmp -s "$ROOT_TASK_TOML" "$TASK_TOML"; then
        fail_reward "task_toml_mirror_drift"
    fi
fi

SCORES=$(python3 - "$TASK_TOML" <<'PYEOF'
# Container Python may be 3.10 — regex-parse both numbers.
import re, sys
with open(sys.argv[1]) as f:
    txt = f.read()
bm = re.search(r"\[perf_opt\.baseline\][^\[]*?\bscore\s*=\s*([0-9.eE+\-]+)", txt)
rm = re.search(r"\[perf_opt\.reference\][^\[]*?\bscore\s*=\s*([0-9.eE+\-]+)", txt)
if not bm:
    sys.exit("missing perf_opt.baseline.score")
if not rm:
    sys.exit("missing perf_opt.reference.score")
print(f"{float(bm.group(1))} {float(rm.group(1))}")
PYEOF
) || fail_reward "task_toml_parse_error"
BASELINE_SCORE=$(echo "$SCORES" | awk '{print $1}')
REFERENCE_SCORE=$(echo "$SCORES" | awk '{print $2}')

echo "  baseline=$BASELINE_SCORE  reference=$REFERENCE_SCORE"

# ── 1. SHA256 pin frozen files vs /orig/app ──────────────────────────────────
#      /orig/app is a build-time copy of /app. Agent modifies /app, but these
#      load-bearing files must stay byte-identical to the pristine copy.
check_hash() {
    local path="$1"
    local orig="$2"
    if [ ! -f "$path" ] || [ ! -f "$orig" ]; then
        fail_reward "missing_protected_file:$(basename "$path")"
    fi
    local a b
    a=$(sha256sum "$path" | awk '{print $1}')
    b=$(sha256sum "$orig" | awk '{print $1}')
    if [ "$a" != "$b" ]; then
        fail_reward "sha_mismatch:$(basename "$path")"
    fi
}

check_hash /app/evaluate.py      /orig/app/evaluate.py
check_hash /app/eval_prompts.txt /orig/app/eval_prompts.txt

# ── 2. /app/ whitelist (top-level entries only) ──────────────────────────────
#      Agent may edit train.sh / dataset.toml and write into output/.
#      evaluate.py / eval_prompts.txt are hash-pinned above.
ALLOWED_TOP=(
    "train.sh"
    "dataset.toml"
    "evaluate.py"
    "eval_prompts.txt"
    "output"
)
while IFS= read -r entry; do
    base="$(basename "$entry")"
    match=0
    for allowed in "${ALLOWED_TOP[@]}"; do
        if [ "$base" = "$allowed" ]; then match=1; break; fi
    done
    # Tolerate bytecode / editor scratch that musubi-tuner may emit,
    # and log files that training scripts write to /app/ top-level.
    case "$base" in
        __pycache__|.ipynb_checkpoints|*.pyc) match=1 ;;
        *.log) match=1 ;;
    esac
    if [ "$match" -ne 1 ]; then
        fail_reward "disallowed_app_entry:$base"
    fi
done < <(find /app -mindepth 1 -maxdepth 1)

# ── 3. Check agent's trained LoRA ────────────────────────────────────────────
echo ""
echo "[Step 1/4] Checking LoRA output..."
LORA_PATH="/app/output/lora.safetensors"
if [ ! -f "$LORA_PATH" ]; then
    echo "ERROR: No LoRA file found at $LORA_PATH"
    fail_reward "no_lora_file"
fi

python3 - "$LORA_PATH" <<'PYEOF' 2>&1 | tee -a "$LOG_DIR/lora_check.log"
import sys
from safetensors.torch import load_file
weights = load_file(sys.argv[1])
assert len(weights) > 0, "Empty LoRA file"
has_lora_keys = any("lora" in k.lower() for k in weights.keys())
assert has_lora_keys, f"No LoRA keys found in {list(weights.keys())[:5]}"
print(f"LoRA file valid: {len(weights)} tensors")
PYEOF
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    fail_reward "invalid_lora"
fi

# ── 4. Evaluate agent's LoRA (frozen evaluate.py copy) ───────────────────────
echo ""
echo "[Step 2/4] Evaluating agent's LoRA..."
python3 /orig/app/evaluate.py \
    --lora_path "$LORA_PATH" \
    --output_json "$LOG_DIR/optimized_score.json" \
    --save_images_dir "$LOG_DIR/generated_images_optimized" \
    2>&1 | tee "$LOG_DIR/eval_optimized.log"

if [ ! -f "$LOG_DIR/optimized_score.json" ]; then
    fail_reward "optimized_eval_failed"
fi
OPTIMIZED=$(python3 -c "import json; print(json.load(open('$LOG_DIR/optimized_score.json'))['score'])") \
    || fail_reward "optimized_score_parse_error"
echo "  Agent score: $OPTIMIZED"

# ── 5. Evaluate baseline (no LoRA) ───────────────────────────────────────────
echo ""
echo "[Step 3/4] Evaluating baseline (no LoRA)..."
python3 /orig/app/evaluate.py \
    --no_lora \
    --output_json "$LOG_DIR/baseline_score.json" \
    --save_images_dir "$LOG_DIR/generated_images_baseline" \
    2>&1 | tee "$LOG_DIR/eval_baseline.log"

if [ ! -f "$LOG_DIR/baseline_score.json" ]; then
    fail_reward "baseline_eval_failed"
fi
BASELINE_MEASURED=$(python3 -c "import json; print(json.load(open('$LOG_DIR/baseline_score.json'))['score'])") \
    || fail_reward "baseline_score_parse_error"
echo "  Baseline measured: $BASELINE_MEASURED  (task.toml baseline: $BASELINE_SCORE)"

echo ""
echo "[Step 4/4] Computing reward..."
REWARD=$(python3 - <<PYEOF
optimized = float("$OPTIMIZED")
baseline = float("$BASELINE_MEASURED")
reference = float("$REFERENCE_SCORE")
if optimized <= baseline or reference <= baseline:
    print(0.0)
else:
    r = (optimized - baseline) / (reference - baseline)
    print(round(max(0.0, min(1.0, r)), 4))
PYEOF
)

echo ""
echo "============================================"
echo "  RESULTS"
echo "============================================"
echo "  Baseline (task.toml):  $BASELINE_SCORE"
echo "  Baseline (measured):   $BASELINE_MEASURED"
echo "  Optimized:             $OPTIMIZED"
echo "  Reference:             $REFERENCE_SCORE"
echo "  Reward:                $REWARD"
echo "============================================"

python3 - <<PYEOF
import json
result = {
    "reward": float("$REWARD"),
    "metric": float("$OPTIMIZED"),
    "optimized_score": float("$OPTIMIZED"),
    "baseline_score_measured": float("$BASELINE_MEASURED"),
    "baseline_score_task_toml": float("$BASELINE_SCORE"),
    "reference_score": float("$REFERENCE_SCORE"),
}
with open("$LOG_DIR/reward.json", "w") as f:
    json.dump(result, f, indent=2)
PYEOF
echo "$REWARD" > "$LOG_DIR/reward.txt"
