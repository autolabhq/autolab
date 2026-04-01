#!/usr/bin/env bash
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
    printf '{"reward": %s, "correctness": %s, "metric": %s, "speedup": %s}\n' \
        "$1" "$2" "$3" "$4" > "$REWARD_FILE"
}

echo "=== Building ==="
if ! cargo build --release --manifest-path=/app/Cargo.toml 2>&1; then
    write_reward 0.0 false null null
    echo "0.0" > "/logs/verifier/reward.txt"
    exit 0
fi

echo "=== Correctness ==="
if ! python3 /tests/verify_correctness.py 2>&1; then
    write_reward 0.0 false null null
    echo "0.0" > "/logs/verifier/reward.txt"
    exit 0
fi

echo "=== Benchmark ==="
OUTPUT=$(/app/target/release/solve 2>/dev/null)
echo "$OUTPUT"
SCORE=$(echo "$OUTPUT" | sed -n 's/.*time=\([0-9.]*\).*/\1/p')
MATCHES=$(echo "$OUTPUT" | sed -n 's/.*matches=\([0-9]*\).*/\1/p')
CHECKSUM=$(echo "$OUTPUT" | sed -n 's/.*checksum=\([0-9]*\).*/\1/p')
if [[ -z "$SCORE" || -z "$MATCHES" || -z "$CHECKSUM" ]]; then
    write_reward 0.0 false null null
    echo "0.0" > "/logs/verifier/reward.txt"
    exit 0
fi

BASELINE=1.5
REFERENCE=0.37
RESULT=$(python3 - "$SCORE" "$BASELINE" "$REFERENCE" <<'PYEOF'
import math
import sys
score = float(sys.argv[1])
baseline = float(sys.argv[2])
reference = float(sys.argv[3])
speedup = baseline / score
ref_speedup = baseline / reference
reward = 0.0 if speedup <= 1.0 else min(1.0, 0.5 * math.log(speedup) / math.log(ref_speedup))
print(f"{round(reward,4)} {round(speedup,2)}")
PYEOF
)

REWARD=$(echo "$RESULT" | cut -d' ' -f1)
SPEEDUP=$(echo "$RESULT" | cut -d' ' -f2)
write_reward "$REWARD" true "$SCORE" "$SPEEDUP"
python3 -c "
reward = $REWARD
with open('/logs/verifier/reward.txt', 'w') as f:
    f.write(str(round(reward, 4)))
"
echo "Speedup: ${SPEEDUP}x Reward: ${REWARD}"
