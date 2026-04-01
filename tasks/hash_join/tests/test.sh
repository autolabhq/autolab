#!/usr/bin/env bash
# tests/test.sh — verifier for hash_join
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

# ─── Step 2: Correctness check ───────────────────────────────────────────────
echo ""
echo "=== Correctness check (R=200, S=2000) ==="
if ! python3 /tests/verify_correctness.py 2>&1; then
    echo "Correctness check FAILED."
    write_reward 0.0 false null null
    echo "0.0" > /logs/verifier/reward.txt
    exit 0
fi

# ─── Step 3: Benchmark (median of 5 runs) ────────────────────────────────────
echo ""
echo "=== Benchmark: inner equi-join, R=80,000 rows × S=5,000,000 rows (5 runs) ==="

REF_MATCHES=$(cat /tests/benchmark_matches.txt)
REF_CHECKSUM=$(cat /tests/benchmark_checksum.txt)

TIMES=()
for run in 1 2 3 4 5; do
    OUTPUT=$(/app/join 2>/dev/null)
    T=$(echo "$OUTPUT" | sed 's/.*time=\([0-9.]*\).*/\1/')
    echo "  Run $run: ${T}s"
    TIMES+=("$T")

    # Correctness gate on run 1
    if [ "$run" -eq 1 ]; then
        GOT_MATCHES=$(echo  "$OUTPUT" | grep -oP 'matches=\K[0-9]+')
        GOT_CHECKSUM=$(echo "$OUTPUT" | grep -oP 'checksum=\K[0-9]+')
        if [ "$GOT_MATCHES" != "$REF_MATCHES" ] || [ "$GOT_CHECKSUM" != "$REF_CHECKSUM" ]; then
            echo "Benchmark correctness gate FAILED" >&2
            echo "  matches:  got=$GOT_MATCHES  ref=$REF_MATCHES" >&2
            echo "  checksum: got=$GOT_CHECKSUM ref=$REF_CHECKSUM" >&2
            write_reward 0.0 false null null
            echo "0.0" > /logs/verifier/reward.txt
            exit 0
        fi
    fi
done

# Median of 5
MEDIAN=$(python3 - "${TIMES[@]}" <<'PYEOF'
import sys
times = sorted(float(x) for x in sys.argv[1:])
n = len(times)
median = times[n // 2] if n % 2 == 1 else (times[n//2-1] + times[n//2]) / 2
print(f"{median:.6f}")
PYEOF
)

echo "Median time: ${MEDIAN}s"

# ─── Step 4: Compute reward ───────────────────────────────────────────────────
BASELINE=20.0      # naive nested-loop on R=20K × S=5M (reference hardware)
REFERENCE=0.04    # open-addressing hash join, multiplicative hash, SoA layout

REWARD=$(python3 - "$MEDIAN" "$BASELINE" "$REFERENCE" <<'PYEOF'
import sys, math

score      = float(sys.argv[1])
baseline   = float(sys.argv[2])
reference  = float(sys.argv[3])

speedup     = baseline / score
ref_speedup = baseline / reference  # ≈ 333×

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
echo "$REWARD_VAL" > /logs/verifier/reward.txt
echo "Results written to $REWARD_FILE"
