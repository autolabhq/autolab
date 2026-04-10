#!/usr/bin/env bash
# tests/test.sh — verifier for vliw_scheduler
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
    echo "{\"reward\": $1, \"correctness\": $2, \"metric\": $3, \"speedup\": $4}" \
        > "$REWARD_FILE"
    python3 -c "
with open('/logs/verifier/reward.txt', 'w') as f:
    f.write(str($1))
"
}

# ── Step 1: Build ─────────────────────────────────────────────────────────────
echo "=== Building ==="
if ! make -C /app 2>&1; then
    echo "Build failed."
    write_reward 0.0 false null null
    exit 0
fi

# ── Step 2: Correctness + cycle count ────────────────────────────────────────
echo "=== Running ==="
OUTPUT=$(/app/solve 2>/dev/null)
echo "$OUTPUT"

CYCLES=$(echo "$OUTPUT" | sed 's/.*cycles=\([0-9]*\).*/\1/')
RESULT=$(echo "$OUTPUT" | sed 's/.*result=\([^ ]*\).*/\1/')

if [ "$RESULT" != "ok" ]; then
    echo "Correctness FAILED: $OUTPUT"
    write_reward 0.0 false null null
    exit 0
fi

echo "Cycles: ${CYCLES}  result=ok"

# ── Step 3: Reward ────────────────────────────────────────────────────────────
BASELINE=4080
BEST_KNOWN=1200

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
