#!/usr/bin/env bash
# tests/test.sh — verifier for flash_attention
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
    echo "0.0" > /logs/verifier/reward.txt
    exit 0
fi

# ─── Step 2: Correctness check (N=256, D=32 — fast) ──────────────────────────
echo ""
echo "=== Correctness check (N=256, D=32) ==="
if ! python3 /tests/verify_correctness.py 2>&1; then
    echo "Correctness check FAILED."
    write_reward 0.0 false null null
    echo "0.0" > /logs/verifier/reward.txt
    exit 0
fi

# ─── Step 3: Benchmark (N=4096, D=64) — correctness gate + median of 5 runs ──
echo ""
echo "=== Benchmark: N=4096, D=64 (5 runs) ==="

REF_CS=$(cat /tests/benchmark_checksum.txt)

TIMES=()
CHECKSUMS=()

for run in 1 2 3 4 5; do
    OUTPUT=$(/app/attn 2>/dev/null)
    T=$(echo  "$OUTPUT" | sed 's/.*time=\([0-9.]*\).*/\1/')
    CS=$(echo "$OUTPUT" | sed 's/.*checksum=\([0-9.eE+\-]*\).*/\1/')
    echo "  Run $run: ${T}s  checksum=${CS}"
    TIMES+=("$T")
    CHECKSUMS+=("$CS")
done

# Run 1 checksum must match the reference (correctness gate).
GATE_OK=$(python3 -c "
got = float('${CHECKSUMS[0]}')
ref = float('$REF_CS')
rel = abs(got - ref) / max(abs(ref), 1e-12)
print('ok' if rel < 1e-3 else 'fail')
")
if [ "$GATE_OK" != "ok" ]; then
    echo "Benchmark correctness gate FAILED (checksum mismatch)" >&2
    echo "  got=${CHECKSUMS[0]}  ref=$REF_CS" >&2
    write_reward 0.0 false null null
    echo "0.0" > /logs/verifier/reward.txt
    exit 0
fi

# All 5 checksums must agree to within 1% (detects non-deterministic code).
FIRST_CS="${CHECKSUMS[0]}"
for cs in "${CHECKSUMS[@]}"; do
    MATCH=$(python3 -c "
a = float('$FIRST_CS')
b = float('$cs')
denom = max(abs(a), 1e-100)
print('ok' if abs(a - b) / denom < 0.01 else 'mismatch')
")
    if [ "$MATCH" != "ok" ]; then
        echo "Non-deterministic output detected (checksum mismatch). reward = 0."
        write_reward 0.0 false null null
        echo "0.0" > /logs/verifier/reward.txt
        exit 0
    fi
done

# Compute median of 5 times.
MEDIAN=$(python3 - "${TIMES[@]}" <<'PYEOF'
import sys
t = sorted(float(x) for x in sys.argv[1:])
n = len(t)
print(f"{(t[n//2] if n % 2 else (t[n//2-1] + t[n//2]) / 2):.6f}")
PYEOF
)
echo "Median time: ${MEDIAN}s"

# ─── Step 4: Compute reward ───────────────────────────────────────────────────
BASELINE=0.75     # naive double-precision attention on reference hardware
REFERENCE=0.10   # Flash Attention + online softmax + AVX2 float32

RESULT=$(python3 - "$MEDIAN" "$BASELINE" "$REFERENCE" <<'PYEOF'
import sys, math

score      = float(sys.argv[1])
baseline   = float(sys.argv[2])
reference  = float(sys.argv[3])

speedup     = baseline / score
ref_speedup = baseline / reference   # ≈ 23.3×

if speedup <= 1.0:
    reward = 0.0
else:
    reward = min(1.0, 0.5 * math.log(speedup) / math.log(ref_speedup))

reward    = round(max(0.0, reward), 4)
speedup_r = round(speedup, 2)
print(f"{reward} {speedup_r}")
PYEOF
)

REWARD_VAL=$(echo "$RESULT" | cut -d' ' -f1)
SPEEDUP_VAL=$(echo "$RESULT" | cut -d' ' -f2)

echo ""
echo "Speedup: ${SPEEDUP_VAL}×  |  Reward: ${REWARD_VAL}"

write_reward "$REWARD_VAL" true "$MEDIAN" "$SPEEDUP_VAL"
echo "$REWARD_VAL" > /logs/verifier/reward.txt
echo "Results written to $REWARD_FILE"
