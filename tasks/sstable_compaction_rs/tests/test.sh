#!/usr/bin/env bash
set -euo pipefail

REWARD_FILE="/logs/verifier/reward.json"
mkdir -p /logs/verifier

# ── 0. Build file integrity check ────────────────────────────────────────
ORIG_HASH=$(cat /orig/build_file.sha256 2>/dev/null || echo "")
CURR_HASH=$(sha256sum /app/Cargo.toml 2>/dev/null | cut -d' ' -f1 || echo "missing")
if [ -n "$ORIG_HASH" ] && [ "$ORIG_HASH" != "$CURR_HASH" ]; then
    echo "Build file was modified — scoring 0" >&2
    echo '{"reward": 0.0, "reason": "build_file_modified"}' > "/logs/verifier/reward.json"
    echo "0.0" > "/logs/verifier/reward.txt"
    exit 0
fi

write_reward() {
    local reward="$1" correctness="$2" metric="$3" speedup="$4"
    printf '{"reward": %s, "correctness": %s, "metric": %s, "speedup": %s}\n' \
        "$reward" "$correctness" "$metric" "$speedup" > "$REWARD_FILE"
    echo "$reward" > /logs/verifier/reward.txt
}

echo "=== Building ==="
# Ensure cargo detects source changes (cp/write may preserve timestamps)
touch /app/src/*.rs 2>/dev/null || true
if ! cargo build --release --manifest-path=/app/Cargo.toml 2>&1; then
    write_reward 0.0 false null null
    exit 0
fi

echo "=== Correctness + Benchmark ==="
OUTPUT=$(/app/target/release/solve 2>/dev/null)
echo "$OUTPUT"

VERIFY=$(echo "$OUTPUT" | sed -n 's/.*verify=\([^ ]*\).*/\1/p')
VLIVE=$(echo "$OUTPUT" | sed -n 's/.*verify_live=\([0-9]*\).*/\1/p')
VHASH=$(echo "$OUTPUT" | sed -n 's/.*verify_hash=\([0-9]*\).*/\1/p')
BLIVE=$(echo "$OUTPUT" | sed -n 's/.*bench_live=\([0-9]*\).*/\1/p')
BHASH=$(echo "$OUTPUT" | sed -n 's/.*bench_hash=\([0-9]*\).*/\1/p')
TIME=$(echo "$OUTPUT" | sed -n 's/.*time=\([0-9.]*\).*/\1/p')

if [[ "$VERIFY" != "ok" || -z "$TIME" ]]; then
    write_reward 0.0 false null null
    exit 0
fi

if [[ "$VLIVE" != "168064" || \
      "$VHASH" != "893799767553918716" || \
      "$BLIVE" != "619078" || \
      "$BHASH" != "8982827926715907049" ]]; then
    write_reward 0.0 false null null
    exit 0
fi

BASELINE=0.099
REFERENCE=0.041
RESULT=$(python3 - "$TIME" "$BASELINE" "$REFERENCE" <<'PY'
import math, sys
score = float(sys.argv[1])
baseline = float(sys.argv[2])
reference = float(sys.argv[3])
speedup = baseline / score
ref_speedup = baseline / reference
reward = 0.0 if speedup <= 1.0 else min(1.0, 0.5 * math.log(speedup) / math.log(ref_speedup))
print(f"{round(max(0.0, reward), 4)} {round(speedup, 2)}")
PY
)

REWARD=$(echo "$RESULT" | cut -d' ' -f1)
SPEEDUP=$(echo "$RESULT" | cut -d' ' -f2)
write_reward "$REWARD" true "$TIME" "$SPEEDUP"
