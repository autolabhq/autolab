#!/usr/bin/env bash
# tests/test.sh — verifier for gaussian_blur
#
# Writes a JSON reward to /logs/verifier/reward.json:
#   {"reward": <0.0–1.0>, "correctness": <bool>,
#    "metric": <seconds|null>, "speedup": <float|null>}
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
    echo '{"reward": 0.0, "reason": "build_file_modified"}' > "/logs/verifier/reward.json"
    echo "0.0" > "/logs/verifier/reward.txt"
    exit 0
fi

write_reward() {
    local reward="$1" correctness="$2" metric="$3" speedup="$4"
    printf '{"reward": %s, "correctness": %s, "metric": %s, "speedup": %s}\n' \
        "$reward" "$correctness" "$metric" "$speedup" \
        > "$REWARD_FILE"
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
echo ""
echo "=== Correctness check ==="
if ! python3 /tests/verify_correctness.py 2>&1; then
    echo "Correctness check FAILED."
    write_reward 0.0 false null null
    echo "0.0" > "/logs/verifier/reward.txt"
    exit 0
fi

# ─── Step 3: Prepare benchmark input ─────────────────────────────────────────
INPUT="/tests/benchmark_input.raw"

if [ ! -f "$INPUT" ]; then
    echo ""
    echo "=== Generating benchmark input (4096×4096, ~16 MB) ==="
    python3 /tests/generate_input.py > "$INPUT"
    echo "Input written: $(wc -c < "$INPUT") bytes"
fi

# ─── Step 4: Benchmark (median of 3 runs) ────────────────────────────────────
echo ""
echo "=== Benchmark: 5-pass 17×17 Gaussian blur on 4096×4096 image (3 runs) ==="

TIMES=()
for run in 1 2 3; do
    OUTPUT=$(/app/blur "$INPUT" - 4096 4096 2>/dev/null)
    T=$(echo "$OUTPUT" | sed 's/.*time=\([0-9.]*\).*/\1/')
    echo "  Run $run: ${T}s"
    TIMES+=("$T")
done

# Median of 3
MEDIAN=$(python3 - "${TIMES[@]}" <<'PYEOF'
import sys
times = sorted(float(x) for x in sys.argv[1:])
n = len(times)
median = times[n // 2] if n % 2 == 1 else (times[n//2-1] + times[n//2]) / 2
print(f"{median:.6f}")
PYEOF
)

echo "Median time: ${MEDIAN}s"

# ─── Step 5: Compute reward ───────────────────────────────────────────────────
BASELINE=12.0       # naive double-precision non-separable baseline (seconds)
REFERENCE=0.25     # separable + float + AVX2 + cache-optimal (seconds)

REWARD=$(python3 - "$MEDIAN" "$BASELINE" "$REFERENCE" <<'PYEOF'
import sys, math

score      = float(sys.argv[1])
baseline   = float(sys.argv[2])
reference  = float(sys.argv[3])

speedup     = baseline / score
ref_speedup = baseline / reference   # == 48

if speedup <= 1.0:
    reward = 0.0
else:
    reward = min(1.0, 0.5 * math.log(speedup) / math.log(ref_speedup))

reward    = round(max(0.0, reward), 4)
speedup_r = round(speedup, 2)
print(f"{reward} {speedup_r}")
PYEOF
)

REWARD_VAL=$(echo "$REWARD" | cut -d' ' -f1)
SPEEDUP_VAL=$(echo "$REWARD" | cut -d' ' -f2)

echo ""
echo "Speedup: ${SPEEDUP_VAL}×  |  Reward: ${REWARD_VAL}"

write_reward "$REWARD_VAL" true "$MEDIAN" "$SPEEDUP_VAL"
echo "$REWARD_VAL" > "/logs/verifier/reward.txt"
echo "Results written to $REWARD_FILE"
