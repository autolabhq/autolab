#!/usr/bin/env bash

set -euo pipefail

REWARD_FILE="/logs/verifier/reward.json"
mkdir -p /logs/verifier

# ── 0. Build file integrity check ────────────────────────────────────────
ORIG_HASH=$(cat /orig/build_file.sha256 2>/dev/null || echo "")
CURR_HASH=$(sha256sum /app/go.mod 2>/dev/null | cut -d' ' -f1 || echo "missing")
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

echo "=== Building ==="
if ! (cd /app && go build -o /app/solve . 2>&1); then
    write_reward 0.0 false null null
    echo "0.0" > "/logs/verifier/reward.txt"
    exit 0
fi

echo ""
echo "=== Correctness check ==="
/app/solve -mode verify -input /tests/verify_input.kv > /tmp/agent_output.txt 2>/dev/null
if ! diff -q /tests/reference_output.txt /tmp/agent_output.txt > /dev/null 2>&1; then
    diff /tests/reference_output.txt /tmp/agent_output.txt || true
    write_reward 0.0 false null null
    echo "0.0" > "/logs/verifier/reward.txt"
    exit 0
fi
echo "Correctness check PASSED."

echo ""
echo "=== Benchmarking ==="
OUTPUT=$(/app/solve -mode benchmark -input /tests/benchmark_input.kv -runs 5 2>/dev/null)
echo "$OUTPUT"

SCORE=$(echo "$OUTPUT" | sed 's/.*time=\([0-9.]*\).*/\1/')
BASELINE=9.5
REFERENCE=1.1

REWARD_AND_SPEEDUP=$(python3 - "$SCORE" "$BASELINE" "$REFERENCE" <<'PYEOF'
import sys, math

score = float(sys.argv[1])
baseline = float(sys.argv[2])
reference = float(sys.argv[3])

speedup = baseline / score
ref_speedup = baseline / reference

if speedup <= 1.0:
    reward = 0.0
else:
    reward = min(1.0, 0.5 * math.log(speedup) / math.log(ref_speedup))

print(f"{round(reward, 4)} {round(speedup, 2)}")
PYEOF
)

REWARD_VAL=$(echo "$REWARD_AND_SPEEDUP" | cut -d' ' -f1)
SPEEDUP_VAL=$(echo "$REWARD_AND_SPEEDUP" | cut -d' ' -f2)

write_reward "$REWARD_VAL" true "$SCORE" "$SPEEDUP_VAL"
echo "$REWARD_VAL" > "/logs/verifier/reward.txt"
echo "Done. Results written to $REWARD_FILE"
