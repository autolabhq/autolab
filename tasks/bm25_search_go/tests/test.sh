#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-/logs/verifier}"
if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
    LOG_DIR="$(pwd)/.verifier_logs"
    mkdir -p "$LOG_DIR"
fi

# ── 0. Build file integrity check ────────────────────────────────────────
ORIG_HASH=$(cat /orig/build_file.sha256 2>/dev/null || echo "")
CURR_HASH=$(sha256sum /app/go.mod 2>/dev/null | cut -d' ' -f1 || echo "missing")
if [ -n "$ORIG_HASH" ] && [ "$ORIG_HASH" != "$CURR_HASH" ]; then
    echo "Build file was modified — scoring 0" >&2
    echo '{"reward": 0.0, "reason": "build_file_modified"}' > "$LOG_DIR/reward.json"
    echo "0.0" > "$LOG_DIR/reward.txt"
    exit 0
fi

if [ -d /app ]; then
    cd /app
fi

write_reward() {
    printf '{"reward": %s, "correctness": %s, "metric": %s, "speedup": %s}\n' \
        "$1" "$2" "$3" "$4" > "$LOG_DIR/reward.json"
}

if ! go build -o ./solve .; then
    write_reward 0.0 false null null
    exit 0
fi

VERIFY_OUTPUT=$(./solve -mode verify 2>&1)
echo "Verify output: $VERIFY_OUTPUT"
if echo "$VERIFY_OUTPUT" | grep -q "result=wrong"; then
    write_reward 0.0 false null null
    exit 0
fi

OUTPUT=$(./solve -mode benchmark -runs 5 2>&1)
echo "Program output: $OUTPUT"
if echo "$OUTPUT" | grep -q "result=wrong"; then
    write_reward 0.0 false null null
    exit 0
fi

TIME=$(echo "$OUTPUT" | sed 's/.*time=\([0-9.]*\).*/\1/')
BASELINE=2.1
REFERENCE=0.03

RESULT=$(python3 - "$TIME" "$BASELINE" "$REFERENCE" <<'PYEOF'
import math, sys
score = float(sys.argv[1])
baseline = float(sys.argv[2])
reference = float(sys.argv[3])
speedup = baseline / score if score > 0 else 0.0
ref_speedup = baseline / reference
reward = 0.0 if speedup <= 1.0 else min(1.0, 0.5 * math.log(speedup) / math.log(ref_speedup))
print(f"{round(max(0.0, reward), 4)} {round(speedup, 2)}")
PYEOF
)

REWARD=$(echo "$RESULT" | cut -d' ' -f1)
SPEEDUP=$(echo "$RESULT" | cut -d' ' -f2)

echo "Speedup: ${SPEEDUP}x | Reward: ${REWARD}"
write_reward "$REWARD" true "$TIME" "$SPEEDUP"
echo "$REWARD" > "$LOG_DIR/reward.txt"
