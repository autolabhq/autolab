#!/usr/bin/env bash
# tests/test.sh — verifier for toy_isa_opt
#
# Writes reward JSON to /logs/verifier/reward.json
# DO NOT MODIFY THIS FILE.

set -euo pipefail

REWARD_FILE="/logs/verifier/reward.json"
mkdir -p /logs/verifier

# ── 0. Build file integrity check ────────────────────────────────────────
ORIG_HASH=$(cat /orig/build_file.sha256 2>/dev/null || echo "")
CURR_HASH=$(sha256sum /app/Makefile 2>/dev/null | cut -d' ' -f1 || echo "missing")
if [ -n "$ORIG_HASH" ] && [ "$ORIG_HASH" != "$CURR_HASH" ]; then
    echo "Build file was modified — scoring 0" >&2
    echo '{"reward": 0.0, "reason": "build_file_modified"}' > "/logs/verifier/reward.json"
    echo "0.0" > "/logs/verifier/reward.txt"
    exit 0
fi

write_reward() {
    local reward=$1 correct=$2 metric=$3 speedup=$4
    echo "{\"reward\": $reward, \"correctness\": $correct, \"metric\": $metric, \"speedup\": $speedup}" \
        > "$REWARD_FILE"
}

# ── Step 1: Build ──────────────────────────────────────────────────────────────
echo "=== Building ==="
if ! make -C /app 2>&1; then
    echo "Build failed."
    write_reward 0.0 false null null
    echo "0.0" > /logs/verifier/reward.txt
    exit 0
fi

# ── Step 2: Run with default seed (seed=0) for metric ─────────────────────────
echo "=== Running simulator ==="
OUTPUT=$(/app/solve 0 2>&1) || true
echo "Output: $OUTPUT"

VERIFY=$(echo "$OUTPUT" | sed -n 's/.*verify=\([^ ]*\).*/\1/p')
CYCLES=$(echo "$OUTPUT" | sed -n 's/.*cycles=\([0-9]*\).*/\1/p')

if [ -z "$VERIFY" ] || [ -z "$CYCLES" ]; then
    echo "Could not parse simulator output."
    write_reward 0.0 false null null
    echo "0.0" > /logs/verifier/reward.txt
    exit 0
fi

if [ "$VERIFY" != "ok" ]; then
    echo "Correctness FAILED: verify=$VERIFY"
    write_reward 0.0 false "$CYCLES" null
    echo "0.0" > /logs/verifier/reward.txt
    exit 0
fi

echo "Correctness OK. Cycles: $CYCLES"

# ── Step 2b: Run with extra seeds to reject hardcoded answers ─────────────────
for SEED in 42 137 999; do
    SOUT=$(/app/solve $SEED 2>&1) || true
    echo "seed=$SEED: $SOUT"
    SVERIFY=$(echo "$SOUT" | sed -n 's/.*verify=\([^ ]*\).*/\1/p')
    if [ "$SVERIFY" != "ok" ]; then
        echo "Correctness FAILED with seed=$SEED"
        write_reward 0.0 false "$CYCLES" null
        echo "0.0" > /logs/verifier/reward.txt
        exit 0
    fi
done

# ── Step 3: Reward computation ─────────────────────────────────────────────────
BASELINE=9220
BEST_KNOWN=1545

REWARD=$(python3 - "$CYCLES" "$BASELINE" "$BEST_KNOWN" <<'PYEOF'
import sys

cycles = int(sys.argv[1])
baseline = int(sys.argv[2])
best_known = int(sys.argv[3])

if cycles >= baseline:
    reward = 0.0
elif cycles <= best_known:
    reward = 1.0
else:
    reward = (baseline - cycles) / (baseline - best_known)

print(f"{round(reward, 4)}")
PYEOF
)

echo "cycles=$CYCLES reward=$REWARD"
write_reward "$REWARD" true "$CYCLES" null
echo "$REWARD" > /logs/verifier/reward.txt
