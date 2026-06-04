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

# ── Step 0: Integrity check on main.py ──
if [ -f /orig/protected_files.sha256 ]; then
    if ! sha256sum -c /orig/protected_files.sha256 >/dev/null 2>&1; then
        echo "Protected file(s) modified — scoring 0" >&2
        write_reward 0.0 false null null
        echo "0.0" > "$LOGDIR/reward.txt"
        exit 0
    fi
fi

echo "=== Run splay tree ==="
OUTPUT=$(python3 "$APPDIR/main.py" 2>&1 | tee /dev/stderr | tail -1)
ROTATIONS=$(echo "$OUTPUT" | sed -n 's/.*rotations=\([0-9]*\).*/\1/p')
VERIFY=$(echo "$OUTPUT" | sed -n 's/.*verify=\([^ ]*\).*/\1/p')

if [ "$VERIFY" != "ok" ]; then
    write_reward 0.0 false null null
    echo "0.0" > "$LOGDIR/reward.txt"
    exit 0
fi

# direction = higher: more rotations = better.
# Log "stretch" reward, matching adaptive_compression: reference is the
# 0.5-anchor (a strong known adversarial pattern), and 1.0 is reserved for
# agents that materially exceed it (e.g., hill-climbed bit-reversal variants
# or unknown SOTA constructions). Theoretical maximum rotations isn't known
# in closed form, so capping at the reference would give no headroom.
BASELINE=48656
REFERENCE=67008

RESULT=$(python3 - "$ROTATIONS" "$BASELINE" "$REFERENCE" <<'PYEOF'
import math, sys

score = float(sys.argv[1])
baseline = float(sys.argv[2])
reference = float(sys.argv[3])

# Minimum-improvement gate: must beat baseline by >=1% to score.
gate = baseline * 1.01
if score <= gate or reference <= baseline:
    reward = 0.0
else:
    # log stretch: score=reference -> 0.5; score=ref^2/baseline -> 1.0.
    r = 0.5 * math.log(score / baseline) / math.log(reference / baseline)
    reward = max(0.0, min(1.0, r))

reward = round(reward, 4)
improvement = score / baseline if baseline > 0 else 0.0
print(f"{reward} {round(improvement, 4)}")
PYEOF
)

REWARD_VAL=$(echo "$RESULT" | cut -d' ' -f1)
IMPROVEMENT_VAL=$(echo "$RESULT" | cut -d' ' -f2)

echo "Rotations: ${ROTATIONS}  |  Improvement: ${IMPROVEMENT_VAL}x  |  Reward: ${REWARD_VAL}"
write_reward "$REWARD_VAL" true "$ROTATIONS" "$IMPROVEMENT_VAL"
echo "$REWARD_VAL" > "$LOGDIR/reward.txt"
