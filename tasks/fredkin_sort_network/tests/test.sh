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
REFERENCE=88

RESULT=$(python3 - "$GATES" "$BASELINE" "$REFERENCE" <<'PYEOF'
import math
import sys

score = float(sys.argv[1])
baseline = float(sys.argv[2])
reference = float(sys.argv[3])
speedup = baseline / score if score > 0 else 0.0
ref_speedup = baseline / reference if reference > 0 else 1.0
reward = 0.0 if speedup <= 1.01 or ref_speedup <= 1.0 else min(1.0, 0.5 * math.log(speedup) / math.log(ref_speedup))
print(f"{round(max(0.0, reward), 4)} {round(speedup, 2)}")
PYEOF
)

REWARD_VAL=$(echo "$RESULT" | cut -d' ' -f1)
SPEEDUP_VAL=$(echo "$RESULT" | cut -d' ' -f2)

echo "Speedup: ${SPEEDUP_VAL}x  |  Reward: ${REWARD_VAL}"
write_reward "$REWARD_VAL" true "$GATES" "$SPEEDUP_VAL"
echo "$REWARD_VAL" > "$LOGDIR/reward.txt"
