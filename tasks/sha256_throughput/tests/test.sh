#!/usr/bin/env bash
# tests/test.sh — verifier for sha256_throughput
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
}

# ── Step 1: Build ──────────────────────────────────────────────────────────────
echo "=== Building ==="
if ! make -C /app 2>&1; then
    echo "Build failed."
    write_reward 0.0 false null null
    echo "0.0" > "/logs/verifier/reward.txt"
    exit 0
fi

# ── Step 2: Correctness ────────────────────────────────────────────────────────
echo "=== Correctness tests ==="
if ! python3 /tests/verify_correctness.py 2>&1; then
    echo "Correctness FAILED."
    write_reward 0.0 false null null
    echo "0.0" > "/logs/verifier/reward.txt"
    exit 0
fi

# ── Step 3: Benchmark — 3 invocations, each with N_RUNS=3 median ──────────────
echo "=== Benchmarking (N_RUNS=3, 3 invocations for stability) ==="
TIMES=()
for trial in 1 2 3; do
    OUTPUT=$(/app/solve 2>/dev/null)
    T=$(echo      "$OUTPUT" | sed 's/.*time=\([0-9.]*\).*/\1/')
    RESULT=$(echo "$OUTPUT" | sed 's/.*result=\([^ ]*\).*/\1/')

    if [ "$RESULT" != "ok" ]; then
        echo "WRONG ANSWER on trial $trial: $OUTPUT"
        write_reward 0.0 false null null
        echo "0.0" > "/logs/verifier/reward.txt"
        exit 0
    fi
    echo "  Trial $trial: ${T}s  result=ok"
    TIMES+=("$T")
done

MEDIAN=$(python3 - "${TIMES[@]}" <<'PYEOF'
import sys
t = sorted(float(x) for x in sys.argv[1:])
n = len(t)
print(f"{(t[n//2] if n%2 else (t[n//2-1]+t[n//2])/2):.6f}")
PYEOF
)
echo "Median time: ${MEDIAN}s"

# ── Step 4: Reward ─────────────────────────────────────────────────────────────
BASELINE=2.5
REFERENCE=0.15

RESULT=$(python3 - "$MEDIAN" "$BASELINE" "$REFERENCE" <<'PYEOF'
import sys, math
score      = float(sys.argv[1])
baseline   = float(sys.argv[2])
reference  = float(sys.argv[3])
speedup    = baseline / score
ref_speedup = baseline / reference
reward     = 0.0 if speedup <= 1.0 else min(1.0, 0.5 * math.log(speedup) / math.log(ref_speedup))
print(f"{round(max(0.0, reward), 4)} {round(speedup, 2)}")
PYEOF
)

REWARD=$(echo "$RESULT" | cut -d' ' -f1)
SPEEDUP=$(echo "$RESULT" | cut -d' ' -f2)

echo "Speedup: ${SPEEDUP}x  |  Reward: ${REWARD}"
write_reward "$REWARD" true "$MEDIAN" "$SPEEDUP"
python3 -c "
reward = $REWARD
with open('/logs/verifier/reward.txt', 'w') as f:
    f.write(str(round(reward, 4)))
"
echo "Done. Written to $REWARD_FILE"
