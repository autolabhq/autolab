#!/usr/bin/env bash
# tests/test.sh — verifier for bvh_raytracer
#
# Writes JSON reward to /logs/verifier/reward.json:
#   {"reward": <0.0–1.0>, "correctness": <bool>, "metric": <seconds>, "speedup": <float>}
#
# DO NOT MODIFY THIS FILE.

set -euo pipefail

REWARD_FILE="/logs/verifier/reward.json"
mkdir -p /logs/verifier

# ── 0. Build file integrity check ────────────────────────────────────────
ORIG_HASH=$(cat /orig/build_file.sha256 2>/dev/null || echo "")
CURR_HASH=$(sha256sum /app/Makefile 2>/dev/null | cut -d' ' -f1 || echo "missing")
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

# ─── Step 1: Build ────────────────────────────────────────────────────────────
echo "=== Building ==="
if ! make -C /app 2>&1; then
    echo "Build FAILED."
    write_reward 0.0 false null null
    echo "0.0" > "/logs/verifier/reward.txt"
    exit 0
fi

# ─── Step 2: Correctness check ───────────────────────────────────────────────
echo "=== Correctness check ==="
AGENT_CS=$(/app/solve --verify 2>/dev/null)
REF_CS=$(cat /app/reference_checksum.txt)

if [ "$AGENT_CS" != "$REF_CS" ]; then
    echo "Correctness FAILED."
    echo "  Expected: $REF_CS"
    echo "  Got:      $AGENT_CS"
    write_reward 0.0 false null null
    echo "0.0" > "/logs/verifier/reward.txt"
    exit 0
fi
echo "Correctness PASSED ($AGENT_CS)"

# ─── Step 3: Benchmark (median of 5 renders) ─────────────────────────────────
echo "=== Benchmark (5 × 638×638 renders, 4096 triangles) ==="

SCORE=$(/app/solve 2>&1 | tee /dev/stderr | tail -1)
echo "Median time: ${SCORE}s"

# ─── Step 4: Compute reward ───────────────────────────────────────────────────
BASELINE=3.8
REFERENCE=0.030

RESULT=$(python3 - "$SCORE" "$BASELINE" "$REFERENCE" <<'PYEOF'
import sys, math

score      = float(sys.argv[1])
baseline   = float(sys.argv[2])
reference  = float(sys.argv[3])

speedup     = baseline / score
ref_speedup = baseline / reference

if speedup <= 1.0:
    reward = 0.0
else:
    reward = min(1.0, 0.5 * math.log(speedup) / math.log(ref_speedup))

reward = round(max(0.0, reward), 4)
print(f"{reward} {round(speedup, 2)}")
PYEOF
)

REWARD_VAL=$(echo "$RESULT" | cut -d' ' -f1)
SPEEDUP_VAL=$(echo "$RESULT" | cut -d' ' -f2)

echo "Speedup: ${SPEEDUP_VAL}×  |  Reward: ${REWARD_VAL}"
write_reward "$REWARD_VAL" true "$SCORE" "$SPEEDUP_VAL"
echo "$REWARD_VAL" > "/logs/verifier/reward.txt"
echo "Done. Results written to $REWARD_FILE"
