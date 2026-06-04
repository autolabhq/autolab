#!/usr/bin/env bash
# Verifier for the canonical Huffman decode CUDA task.
#
# Scores the hidden --benchmark-verify path -- the same run checks both
# byte-exact correctness AND emits the __VERIFIER_SCORE__ sentinel we
# grep for.  The public --benchmark path is not scored.
#
set -euo pipefail

LOG=/logs/verifier
mkdir -p "$LOG"

write_reward() {
    local reward="$1"
    local json="$2"
    python3 - "$reward" "$json" <<'PY'
import json, sys
reward = float(sys.argv[1])
obj = json.loads(sys.argv[2])
with open('/logs/verifier/reward.json','w') as f:
    json.dump(obj, f, indent=2)
with open('/logs/verifier/reward.txt','w') as f:
    f.write(str(reward))
PY
}

# -- Step 0: build-file integrity --------------------------------------
echo "=== Integrity ===" >&2

check_sha256 () {
    local path="$1"
    local ref="$2"
    local exp
    exp=$(cat "$ref" 2>/dev/null || echo "")
    local got
    got=$(sha256sum "$path" 2>/dev/null | cut -d' ' -f1 || echo "missing")
    if [ -n "$exp" ] && [ "$exp" != "$got" ]; then
        echo "INTEGRITY FAIL: $path was modified (got=$got exp=$exp)" >&2
        return 1
    fi
    return 0
}

if ! check_sha256 /app/Makefile /orig/makefile.sha256; then
    write_reward 0.0 '{"reward":0.0,"error":"makefile_modified"}'
    exit 0
fi
if ! check_sha256 /app/main.cu  /orig/main_cu.sha256;  then
    write_reward 0.0 '{"reward":0.0,"error":"main_cu_modified"}'
    exit 0
fi
if ! check_sha256 /app/solve.h  /orig/solve_h.sha256;  then
    write_reward 0.0 '{"reward":0.0,"error":"solve_h_modified"}'
    exit 0
fi

# -- Step 0b: strict /app whitelist ------------------------------------
echo "=== /app whitelist ===" >&2
EXPECTED_FILES=$(cat <<'EOF'
/app
/app/Makefile
/app/huffman_decode
/app/main.cu
/app/solve.cu
/app/solve.h
EOF
)
ACTUAL_FILES=$(find /app -mindepth 0 \( -type f -o -type d \) 2>/dev/null | sort)
EXPECTED_SORTED=$(echo "$EXPECTED_FILES" | sort)
UNEXPECTED=$(comm -23 <(echo "$ACTUAL_FILES") <(echo "$EXPECTED_SORTED") || true)
if [ -n "$UNEXPECTED" ]; then
    echo "UNEXPECTED files under /app:" >&2
    echo "$UNEXPECTED" >&2
    write_reward 0.0 '{"reward":0.0,"error":"unexpected_files"}'
    exit 0
fi

# -- Step 1: constraint checks on solve.cu -----------------------------
echo "=== Constraints ===" >&2

# Banned external headers.
if grep -qiE '#\s*include\s*[<"](thrust|cub/|cusparse|cublas|cusolver|cufft|cutlass|nccl|cudnn|nvcomp|zlib|zstd)' /app/solve.cu 2>/dev/null; then
    echo "BANNED external library include in solve.cu" >&2
    write_reward 0.0 '{"reward":0.0,"error":"banned_library"}'
    exit 0
fi

# Banned symbol patterns.  These would either pull in a prebuilt
# Huffman/decompression backend, or read system data the harness keeps
# hidden.
if grep -qE '\b(nvcomp[A-Za-z_]*|ZSTD_[A-Za-z_]+|inflate[A-Za-z_]*|huff_decode_table|HuffmanDecoder)\b' /app/solve.cu 2>/dev/null; then
    echo "BANNED: solve.cu references a forbidden decompression symbol" >&2
    write_reward 0.0 '{"reward":0.0,"error":"banned_symbol"}'
    exit 0
fi

# Hidden seed / sentinel must not be referenced.
if grep -qE 'HUFFMAN_DECODE_SEED' /app/solve.cu 2>/dev/null; then
    echo "BANNED: solve.cu references the hidden seed env" >&2
    write_reward 0.0 '{"reward":0.0,"error":"reads_hidden_seed"}'
    exit 0
fi
if grep -qE '__VERIFIER_SCORE__' /app/solve.cu 2>/dev/null; then
    echo "BANNED: solve.cu emits the verifier sentinel" >&2
    write_reward 0.0 '{"reward":0.0,"error":"emits_sentinel"}'
    exit 0
fi
if grep -qE '(\bgetenv\s*\(|\bsecure_getenv\s*\(|\b__environ\b|"/proc/self/environ"|"/proc/[0-9]+/environ")' /app/solve.cu 2>/dev/null; then
    echo "BANNED: solve.cu reads environment / /proc/self/environ" >&2
    write_reward 0.0 '{"reward":0.0,"error":"reads_environ"}'
    exit 0
fi
if grep -qE '(\bfopen\s*\(|\bopen\s*\(.*O_RDONLY|std::ifstream|ifstream)' /app/solve.cu 2>/dev/null; then
    echo "BANNED: solve.cu opens files" >&2
    write_reward 0.0 '{"reward":0.0,"error":"opens_files"}'
    exit 0
fi
if grep -qE '(\bsystem\s*\(|\bpopen\s*\(|\bexecv?e?\s*\(|\bfork\s*\()' /app/solve.cu 2>/dev/null; then
    echo "BANNED: solve.cu uses subprocess/exec" >&2
    write_reward 0.0 '{"reward":0.0,"error":"banned_syscall"}'
    exit 0
fi

# -- Step 2: build -----------------------------------------------------
echo "=== Build ===" >&2
make -C /app clean || true
if ! make -C /app > "$LOG/build.txt" 2>&1; then
    echo "BUILD FAILED" >&2
    cat "$LOG/build.txt" >&2 || true
    write_reward 0.0 '{"reward":0.0,"error":"build_failed"}'
    exit 0
fi

if [ ! -x /app/huffman_decode ]; then
    echo "BUILD: binary missing after make" >&2
    write_reward 0.0 '{"reward":0.0,"error":"no_binary"}'
    exit 0
fi

# -- Step 3: public correctness ----------------------------------------
echo "=== Verify (public, small) ===" >&2
VERIFY_OUT=$(/app/huffman_decode --verify 2>&1) || true
echo "$VERIFY_OUT" | tee "$LOG/verify.txt" >&2
if ! echo "$VERIFY_OUT" | grep -q '^PASS'; then
    write_reward 0.0 '{"reward":0.0,"error":"verify_failed"}'
    exit 0
fi

# -- Step 4: scored benchmark (hidden seed, correctness + timing) -------
echo "=== Benchmark-verify (hidden seed) ===" >&2
export HUFFMAN_DECODE_SEED=$(python3 -c "import os; print(int.from_bytes(os.urandom(8),'big'))")

BENCH_OUT=$(/app/huffman_decode --benchmark-verify 2>&1) || true
echo "$BENCH_OUT" > "$LOG/bench.txt"
echo "$BENCH_OUT" >&2

# Sentinel must appear exactly once.
N_SENT=$(echo "$BENCH_OUT" | grep -cE '^__VERIFIER_SCORE__=' || true)
if [ "$N_SENT" != "1" ]; then
    echo "SENTINEL count = $N_SENT (expected 1)" >&2
    write_reward 0.0 '{"reward":0.0,"error":"sentinel_count"}'
    exit 0
fi

SCORE=$(echo "$BENCH_OUT" | grep -oE '__VERIFIER_SCORE__=[0-9.]+' | head -1 | sed 's/^__VERIFIER_SCORE__=//') || true
if [ -z "$SCORE" ]; then
    echo "could not parse sentinel" >&2
    write_reward 0.0 '{"reward":0.0,"error":"parse_sentinel"}'
    exit 0
fi

# -- Step 5: reward ----------------------------------------------------
BASELINE=56.0
REFERENCE=3.11152

python3 - "$SCORE" "$BASELINE" "$REFERENCE" <<'PY'
import json, math, sys
score     = float(sys.argv[1])
baseline  = float(sys.argv[2])
reference = float(sys.argv[3])
speedup     = baseline / score if score > 0 else 0.0
ref_speedup = baseline / reference
if speedup <= 1.0:
    reward = 0.0
else:
    reward = min(1.0, 0.5 * math.log(speedup) / math.log(ref_speedup))
reward = round(max(0.0, reward), 4)
result = {
    "reward":   reward,
    "score":    score,
    "speedup":  round(speedup, 2),
    "baseline": baseline,
    "reference": reference,
}
with open('/logs/verifier/reward.json','w') as f:
    json.dump(result, f, indent=2)
with open('/logs/verifier/reward.txt','w') as f:
    f.write(str(reward))
print(f"score={score} ms  speedup={speedup:.2f}x  reward={reward:.4f}")
PY

exit 0
