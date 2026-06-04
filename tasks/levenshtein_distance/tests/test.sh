#!/usr/bin/env bash
# tests/test.sh — verifier for levenshtein_distance
#
# Writes reward JSON to /logs/verifier/reward.json
# DO NOT MODIFY THIS FILE.

set -euo pipefail

REWARD_FILE="/logs/verifier/reward.json"
REWARD_TXT="/logs/verifier/reward.txt"
mkdir -p /logs/verifier

write_reward() {
    echo "{\"reward\": $1, \"correctness\": $2, \"metric\": $3, \"speedup\": $4}" \
        > "$REWARD_FILE"
}

write_zero() {
    local reason="$1"
    echo "{\"reward\": 0.0, \"correctness\": false, \"reason\": \"$reason\"}" \
        > "$REWARD_FILE"
    echo "0.0" > "$REWARD_TXT"
}

check_hash() {
    local path="$1"
    local name="$2"
    local hash_file="/orig/${name}.sha256"
    local orig_hash
    local curr_hash

    orig_hash=$(cat "$hash_file" 2>/dev/null || echo "")
    curr_hash=$(sha256sum "$path" 2>/dev/null | cut -d' ' -f1 || echo "missing")

    if [ -z "$orig_hash" ]; then
        echo "Missing original hash for $name — scoring 0" >&2
        write_zero "missing_${name}_hash"
        exit 0
    fi

    if [ "$orig_hash" != "$curr_hash" ]; then
        echo "$name was modified — scoring 0" >&2
        write_zero "${name}_modified"
        exit 0
    fi
}

# ── 0. Read-only file integrity checks ───────────────────────────────────────
check_hash /app/Makefile Makefile
check_hash /app/main.c main.c
check_hash /app/solve.h solve.h

# ── Step 1: Build agent's solve ──────────────────────────────────────────────
echo "=== Building ==="
if ! make -C /app 2>&1; then
    echo "Build failed."
    write_zero "build_failed"
    exit 0
fi

# ── Step 2: Correctness ──────────────────────────────────────────────────────
echo "=== Correctness tests ==="
if ! python3 /tests/verify_correctness.py 2>&1; then
    echo "Correctness FAILED."
    write_zero "correctness_failed"
    exit 0
fi

# ── Step 3: Build trusted reference for expected (checksum, fingerprint) ─────
echo "=== Building trusted reference ==="
TRUSTED_BIN="/tmp/lev_compute_expected"
if ! gcc -O2 -std=c11 -o "$TRUSTED_BIN" /tests/compute_expected.c 2>&1; then
    echo "Trusted reference build failed."
    write_zero "trusted_build_failed"
    exit 0
fi

# ── Step 4: Generate fresh per-run seed and compute expected fingerprint ─────
# 64-bit unsigned random integer (decimal). /dev/urandom + od is portable.
LEV_SEED=$(od -An -N8 -tu8 /dev/urandom | tr -d ' \n')
export LEV_SEED
echo "Per-run LEV_SEED=${LEV_SEED}"

if ! EXPECTED_OUT=$("$TRUSTED_BIN" "$LEV_SEED" 2>&1); then
    echo "Trusted reference run failed: $EXPECTED_OUT"
    write_zero "trusted_run_failed"
    exit 0
fi
EXPECTED_CHECKSUM=$(printf "%s\n" "$EXPECTED_OUT" | sed -n 's/.*checksum=\([0-9][0-9]*\).*/\1/p' | tail -n1)
EXPECTED_FINGERPRINT=$(printf "%s\n" "$EXPECTED_OUT" | sed -n 's/.*fingerprint=0x\([0-9a-fA-F][0-9a-fA-F]*\).*/\1/p' | tail -n1 | tr 'A-F' 'a-f')
if [ -z "$EXPECTED_CHECKSUM" ] || [ -z "$EXPECTED_FINGERPRINT" ]; then
    echo "Trusted reference output unparseable: $EXPECTED_OUT"
    write_zero "trusted_parse_failed"
    exit 0
fi
echo "Expected: checksum=${EXPECTED_CHECKSUM} fingerprint=0x${EXPECTED_FINGERPRINT}"

# ── Step 5: Benchmark — 3 outer invocations, each N_RUNS=3 median ────────────
echo "=== Benchmarking (N_RUNS=3, 3 invocations for stability) ==="
TIMES=()
for trial in 1 2 3; do
    if ! OUTPUT=$(LEV_SEED="$LEV_SEED" timeout 30s /app/solve 2>&1); then
        echo "Benchmark failed or timed out on trial $trial: $OUTPUT"
        write_zero "benchmark_failed_or_timeout"
        exit 0
    fi

    T=$(printf "%s\n" "$OUTPUT" | sed -n 's/.*time=\([0-9][0-9.]*\).*/\1/p' | tail -n1)
    RESULT=$(printf "%s\n" "$OUTPUT" | sed -n 's/.*result=\([^ ]*\).*/\1/p' | tail -n1)
    CHECKSUM=$(printf "%s\n" "$OUTPUT" | sed -n 's/.*checksum=\([0-9][0-9]*\).*/\1/p' | tail -n1)
    FINGERPRINT=$(printf "%s\n" "$OUTPUT" | sed -n 's/.*fingerprint=\(0x\)\{0,1\}\([0-9a-fA-F][0-9a-fA-F]*\).*/\2/p' | tail -n1 | tr 'A-F' 'a-f')

    if [ -z "$T" ] || [ -z "$RESULT" ] || [ -z "$CHECKSUM" ] || [ -z "$FINGERPRINT" ]; then
        echo "Benchmark output parse failed on trial $trial: $OUTPUT"
        write_zero "benchmark_parse_failed"
        exit 0
    fi

    if [ "$RESULT" != "ok" ] || \
       [ "$CHECKSUM" != "$EXPECTED_CHECKSUM" ] || \
       [ "$FINGERPRINT" != "$EXPECTED_FINGERPRINT" ]; then
        echo "WRONG ANSWER on trial $trial: $OUTPUT"
        echo "  expected checksum=${EXPECTED_CHECKSUM} fingerprint=0x${EXPECTED_FINGERPRINT}"
        write_zero "benchmark_wrong_answer"
        exit 0
    fi
    echo "  Trial $trial: ${T}s  checksum=${CHECKSUM} fingerprint=0x${FINGERPRINT} result=ok"
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

# ── Step 6: Reward ───────────────────────────────────────────────────────────
BASELINE=2.0845
REFERENCE=0.3107

RESULT=$(python3 - "$MEDIAN" "$BASELINE" "$REFERENCE" <<'PYEOF'
import sys, math
score       = float(sys.argv[1])
baseline    = float(sys.argv[2])
reference   = float(sys.argv[3])
speedup     = baseline / score
ref_speedup = baseline / reference
reward      = 0.0 if speedup <= 1.0 else min(1.0, 0.5 * math.log(speedup) / math.log(ref_speedup))
print(f"{round(max(0.0, reward), 4)} {round(speedup, 2)}")
PYEOF
)

REWARD=$(echo "$RESULT" | cut -d' ' -f1)
SPEEDUP=$(echo "$RESULT" | cut -d' ' -f2)

echo "Speedup: ${SPEEDUP}x  |  Reward: ${REWARD}"
write_reward "$REWARD" true "$MEDIAN" "$SPEEDUP"
python3 -c "
reward = $REWARD
with open('$REWARD_TXT', 'w') as f:
    f.write(str(round(reward, 4)))
"
echo "Done. Written to $REWARD_FILE"
