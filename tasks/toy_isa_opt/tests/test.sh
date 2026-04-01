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

# ── Step 2: Run and parse output ───────────────────────────────────────────────
echo "=== Running simulator ==="
OUTPUT=$(/app/solve 2>&1) || true
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

# ── Step 3: Reward computation ─────────────────────────────────────────────────
BASELINE=9220
REFERENCE=1500

RESULT=$(python3 - "$CYCLES" "$BASELINE" "$REFERENCE" <<'PYEOF'
import sys, math

cycles     = int(sys.argv[1])
baseline   = int(sys.argv[2])
reference  = int(sys.argv[3])

if cycles <= 0:
    print("0.0 0.0")
    sys.exit(0)

speedup = baseline / cycles
ref_speedup  = baseline / reference   # 6.1333...

if speedup <= 1.0:
    reward = 0.0
else:
    reward = min(1.0, 0.5 * math.log(speedup) / math.log(ref_speedup))

reward = max(0.0, round(reward, 4))
speedup = round(speedup, 4)
print(f"{reward} {speedup}")
PYEOF
)

REWARD=$(echo "$RESULT" | cut -d' ' -f1)
SPEEDUP=$(echo "$RESULT" | cut -d' ' -f2)

echo "Speedup: ${SPEEDUP}x  |  Reward: ${REWARD}"
write_reward "$REWARD" true "$CYCLES" "$SPEEDUP"
echo "$REWARD" > /logs/verifier/reward.txt
echo "Done. Written to $REWARD_FILE"
