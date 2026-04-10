#!/usr/bin/env bash
set -euo pipefail

APPDIR="${APPDIR:-/app}"
LOGDIR="${LOGDIR:-/logs/verifier}"
REWARD_FILE="$LOGDIR/reward.json"
mkdir -p "$LOGDIR"

write_reward() {
    printf '{"reward": %s, "correctness": %s, "metric": %s, "speedup": %s}\n' \
        "$1" "$2" "$3" "$4" > "$REWARD_FILE"
}

echo "=== Verify circuit ==="
OUTPUT=$(python3 "$APPDIR/main.py" 2>&1 | tee /dev/stderr | tail -1)
GATES=$(echo "$OUTPUT" | sed -n 's/.*gates=\([0-9]*\).*/\1/p')
VERIFY=$(echo "$OUTPUT" | sed -n 's/.*verify=\([^ ]*\).*/\1/p')

if [ "$VERIFY" != "ok" ]; then
    write_reward 0.0 false null null
    echo "0.0" > "$LOGDIR/reward.txt"
    exit 0
fi

BASELINE=128
BEST_KNOWN=53

REWARD_VAL=$(python3 - "$GATES" "$BASELINE" "$BEST_KNOWN" <<'PYEOF'
import sys

count = int(sys.argv[1])
baseline = int(sys.argv[2])
optimal = int(sys.argv[3])

if count >= baseline:
    reward = 0.0
elif count <= optimal:
    reward = 1.0
else:
    reward = (baseline - count) / (baseline - optimal)

print(f"{reward:.4f}")
PYEOF
)

echo "gates=$GATES reward=$REWARD_VAL"
write_reward "$REWARD_VAL" true "$GATES" null
echo "$REWARD_VAL" > "$LOGDIR/reward.txt"
