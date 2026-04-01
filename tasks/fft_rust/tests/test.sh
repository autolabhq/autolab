#!/usr/bin/env bash
# tests/test.sh — verifier for fft_rust
#
# Writes reward JSON to /logs/verifier/reward.json:
#   {"reward": <0.0–1.0>, "correctness": <bool>, "metric": <seconds>, "speedup": <float>}
#
# DO NOT MODIFY THIS FILE.

set -euo pipefail

REWARD_FILE="/logs/verifier/reward.json"
mkdir -p /logs/verifier

# ── 0. Build file integrity check ────────────────────────────────────────
ORIG_HASH=$(cat /orig/build_file.sha256 2>/dev/null || echo "")
CURR_HASH=$(sha256sum /app/Cargo.toml 2>/dev/null | cut -d' ' -f1 || echo "missing")
if [ -n "$ORIG_HASH" ] && [ "$ORIG_HASH" != "$CURR_HASH" ]; then
    echo "Build file was modified — scoring 0" >&2
    echo '{"reward": 0.0, "reason": "build_file_modified"}' > /logs/verifier/reward.json
    echo "0.0" > /logs/verifier/reward.txt
    exit 0
fi

write_reward() {
    local reward="$1" correctness="$2" metric="$3" speedup="$4"
    printf '{"reward": %s, "correctness": %s, "metric": %s, "speedup": %s}\n' \
        "$reward" "$correctness" "$metric" "$speedup" > "$REWARD_FILE"
}

# ── Step 1: Build ─────────────────────────────────────────────────────────────
echo "=== Building ==="
if ! cargo build --release --manifest-path=/app/Cargo.toml 2>&1; then
    echo "Build FAILED — score 0."
    write_reward 0.0 false null null
    echo "0.0" > /logs/verifier/reward.txt
    exit 0
fi

# ── Step 2: Correctness (small inputs, run by a Python script) ────────────────
echo ""
echo "=== Correctness check ==="
if ! python3 /tests/verify_correctness.py 2>&1; then
    echo "Correctness FAILED — score 0."
    write_reward 0.0 false null null
    echo "0.0" > /logs/verifier/reward.txt
    exit 0
fi

# ── Step 3: Benchmark ─────────────────────────────────────────────────────────
echo ""
echo "=== Benchmarking (warm-up + 3 timed runs, median reported) ==="
OUTPUT=$(/app/target/release/solve 2>/dev/null)
echo "$OUTPUT"

VERIFY_TAG=$(echo "$OUTPUT" | sed 's/.*verify=\([^ ]*\).*/\1/')
if [ "$VERIFY_TAG" != "ok" ]; then
    echo "Wrong answer on benchmark input."
    write_reward 0.0 false null null
    echo "0.0" > /logs/verifier/reward.txt
    exit 0
fi

SCORE=$(echo "$OUTPUT" | sed 's/.*time=\([0-9.]*\).*/\1/')
echo "Median time: ${SCORE}s"

# ── Step 4: Reward ────────────────────────────────────────────────────────────
BASELINE=10.0
REFERENCE=0.001

REWARD_AND_SPEEDUP=$(python3 - "$SCORE" "$BASELINE" "$REFERENCE" <<'PYEOF'
import sys, math

score      = float(sys.argv[1])
baseline   = float(sys.argv[2])
reference  = float(sys.argv[3])

speedup     = baseline / score
ref_speedup = baseline / reference      # 10 000×

if speedup <= 1.0:
    reward = 0.0
else:
    reward = min(1.0, 0.5 * math.log(speedup) / math.log(ref_speedup))

reward    = round(max(0.0, reward), 4)
speedup_r = round(speedup, 2)
print(f"{reward} {speedup_r}")
PYEOF
)

REWARD_VAL=$(echo "$REWARD_AND_SPEEDUP" | cut -d' ' -f1)
SPEEDUP_VAL=$(echo "$REWARD_AND_SPEEDUP" | cut -d' ' -f2)

echo "Speedup: ${SPEEDUP_VAL}×  |  Reward: ${REWARD_VAL}"

write_reward "$REWARD_VAL" true "$SCORE" "$SPEEDUP_VAL"
echo "$REWARD_VAL" > /logs/verifier/reward.txt
echo "Done. Results written to $REWARD_FILE"
