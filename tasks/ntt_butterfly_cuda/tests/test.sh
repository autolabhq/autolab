#!/usr/bin/env bash
# Verifier for the NTT butterfly CUDA task.
#
# Scores the hidden --benchmark-verify run. The harness checks correctness on the timed workload
# (full equality + late-slice probe at a hidden offset) and only emits __VERIFIER_SCORE__ on
# success.
set -euo pipefail

LOG=/logs/verifier
mkdir -p "$LOG"

write_reward() {
    local reward="$1"
    local payload="$2"
    python3 - "$reward" "$payload" <<'PY'
import json, sys
reward = float(sys.argv[1])
obj = json.loads(sys.argv[2])
with open('/logs/verifier/reward.json','w') as f:
    json.dump(obj, f, indent=2)
with open('/logs/verifier/reward.txt','w') as f:
    f.write(str(reward))
PY
}

fail_zero() {
    local reason="$1"
    echo "$reason -- scoring 0" >&2
    write_reward 0.0 "{\"reward\":0.0,\"reason\":\"$reason\"}"
    exit 0
}

# -- Step 0: build-file integrity --
echo "=== Integrity ===" >&2
if ! sha256sum -c /orig/protected.sha256 --quiet 2>/dev/null; then
    sha256sum -c /orig/protected.sha256 >&2 || true
    fail_zero "protected_file_modified"
fi

# -- Step 0b: strict /app whitelist (recursive) --
echo "=== /app whitelist ===" >&2
EXPECTED=$(cat <<'EOF'
/app
/app/Makefile
/app/main.cu
/app/ntt_butterfly
/app/solve.cu
/app/solve.h
EOF
)
ACTUAL=$(find /app -mindepth 0 \( -type f -o -type d -o -type l \) 2>/dev/null | sort)
EXPECTED_SORTED=$(echo "$EXPECTED" | sort)
UNEXPECTED=$(comm -23 <(echo "$ACTUAL") <(echo "$EXPECTED_SORTED") || true)
if [ -n "$UNEXPECTED" ]; then
    echo "Unexpected files under /app:" >&2
    echo "$UNEXPECTED" >&2
    fail_zero "app_tree_modified"
fi

# -- Step 1: constraint checks on solve.cu --
echo "=== Constraints ===" >&2
SOLVE=/app/solve.cu

# Banned library / framework includes (third-party kernels, FFT, ZK / FHE libs).
if grep -qiE '#[[:space:]]*include[[:space:]]*[<"](thrust|cub/|cutlass|cublas|cublasLt|cudnn|curand|cusparse|cusolver|cufft|nccl|torch|ATen|tensorflow|xformers|flash_attn|gmp|mpfr|ntl|flint|seal/|HEXL|hexl)' "$SOLVE" 2>/dev/null; then
    fail_zero "banned_library_include"
fi

# Banned high-level API tokens.
if grep -qiE '\b(thrust::|cub::|cutlass::|cublas[A-Za-z_]*|cudnn[A-Za-z_]*|cufft[A-Za-z_]*|cufftExec[A-Za-z_]*|curand[A-Za-z_]*|nvcuda::wmma|torch::|at::|np\.fft|numpy\.fft)\b' "$SOLVE" 2>/dev/null; then
    fail_zero "banned_high_level_api"
fi

# Banned memory / copy APIs (workspace is pre-supplied).
if grep -qE '\b(cudaMemcpy|cudaMemcpyAsync|cudaMemcpy2D|cudaMemcpyToSymbol|cudaMemcpyFromSymbol|cudaMalloc|cudaMallocAsync|cudaMallocManaged|cudaHostAlloc|cudaHostRegister|cudaFree|cudaFreeHost|malloc|calloc|realloc|free)[[:space:]]*\(' "$SOLVE" 2>/dev/null; then
    fail_zero "banned_memory_or_copy_api"
fi

# Banned process-introspection APIs (so the agent cannot read the hidden NTT_BENCH_SEED env).
if grep -qE '(\bgetenv[[:space:]]*\(|\bsecure_getenv[[:space:]]*\(|\benviron\b|__environ|/proc/self|/proc/[0-9]+|cmdline|\bargv\b|\bargc\b|\bfopen[[:space:]]*\(|\bfreopen[[:space:]]*\(|\bopen[[:space:]]*\(|\bopenat[[:space:]]*\(|std::ifstream|std::ofstream|\bifstream\b|\bofstream\b|\breadlink[[:space:]]*\(|\bdlopen[[:space:]]*\(|\bdlsym[[:space:]]*\(|\bsystem[[:space:]]*\(|\bpopen[[:space:]]*\(|\bfork[[:space:]]*\(|\bexecv|\bexecl|\bsocket[[:space:]]*\(|\bconnect[[:space:]]*\(|\bclock_gettime[[:space:]]*\(|\bgetpid[[:space:]]*\(|/tmp/|/tests)' "$SOLVE" 2>/dev/null; then
    fail_zero "banned_io_or_process_api"
fi

# Reference to the hidden seed env name.
if grep -qE 'NTT_BENCH_SEED' "$SOLVE" 2>/dev/null; then
    fail_zero "reads_hidden_seed"
fi

# Including main.cu textually to pull file-local reference symbols.
if grep -qE '#[[:space:]]*include[[:space:]]*["<][^">]*main\.cu' "$SOLVE" 2>/dev/null; then
    fail_zero "main_cu_include_detected"
fi

# Sentinel spoofing.
if grep -qE '__VERIFIER_' "$SOLVE" 2>/dev/null; then
    fail_zero "sentinel_spoof_attempt"
fi

# -- Step 2: build --
echo "=== Build ===" >&2
make -C /app clean || true
if ! make -C /app > "$LOG/build.txt" 2>&1; then
    cat "$LOG/build.txt" >&2 || true
    fail_zero "build_failed"
fi

if [ ! -x /app/ntt_butterfly ]; then
    fail_zero "missing_binary"
fi

# -- Step 3: public correctness --
echo "=== Verify (public, small) ===" >&2
if ! timeout 240 /app/ntt_butterfly --verify > "$LOG/verify.txt" 2> "$LOG/verify_err.txt"; then
    cat "$LOG/verify.txt" >&2 || true
    cat "$LOG/verify_err.txt" >&2 || true
    fail_zero "correctness_failed"
fi
cat "$LOG/verify.txt" >&2 || true
cat "$LOG/verify_err.txt" >&2 || true

VERIFY_SENTINELS=$(grep -c '^__VERIFIER_CORRECTNESS__=PASS$' "$LOG/verify_err.txt" || true)
if [ "$VERIFY_SENTINELS" != "1" ]; then
    fail_zero "correctness_sentinel_count"
fi

# -- Step 4: scored benchmark (hidden seed, correctness + timing) --
echo "=== Benchmark-verify (hidden seed) ===" >&2
export NTT_BENCH_SEED=$(python3 -c "import os; print(int.from_bytes(os.urandom(8),'big'))")

if ! timeout 700 /app/ntt_butterfly --benchmark-verify > "$LOG/bench.txt" 2> "$LOG/bench_err.txt"; then
    cat "$LOG/bench.txt" >&2 || true
    cat "$LOG/bench_err.txt" >&2 || true
    fail_zero "timed_path_failed"
fi
cat "$LOG/bench.txt" >&2 || true
cat "$LOG/bench_err.txt" >&2 || true

RESULT=$(sed -n 's/^result=\([a-z]*\).*/\1/p' "$LOG/bench.txt" | tail -1)
if [ "$RESULT" != "ok" ]; then
    fail_zero "timed_result_not_ok"
fi

CORR_COUNT=$(grep -c '^__VERIFIER_CORRECTNESS__=PASS$' "$LOG/bench_err.txt" || true)
BENCH_COUNT=$(grep -c '^__VERIFIER_BENCHMARK__=PASS$' "$LOG/bench_err.txt" || true)
SCORE_COUNT=$(grep -c '^__VERIFIER_SCORE__=' "$LOG/bench_err.txt" || true)
if [ "$CORR_COUNT" != "1" ] || [ "$BENCH_COUNT" != "1" ] || [ "$SCORE_COUNT" != "1" ]; then
    echo "sentinel counts correctness=$CORR_COUNT timed=$BENCH_COUNT score=$SCORE_COUNT" >&2
    fail_zero "sentinel_count_mismatch"
fi

SCORE=$(sed -n 's/^__VERIFIER_SCORE__=\([0-9.]*\)$/\1/p' "$LOG/bench_err.txt" | tail -1)
if [ -z "$SCORE" ]; then
    fail_zero "score_parse_failed"
fi

# -- Step 5: reward --
# Calibrated H100 values; keep in sync with task.toml.
BASELINE=109.8
REFERENCE=1.28

python3 - "$SCORE" "$BASELINE" "$REFERENCE" <<'PY'
import json, math, sys
score     = float(sys.argv[1])
baseline  = float(sys.argv[2])
reference = float(sys.argv[3])
speedup     = baseline / score if score > 0 else 0.0
ref_speedup = baseline / reference if reference > 0 else 0.0
if speedup <= 1.0 or ref_speedup <= 1.0:
    reward = 0.0
else:
    reward = min(1.0, 0.5 * math.log(speedup) / math.log(ref_speedup))
reward = round(max(0.0, reward), 4)
result = {
    "reward": reward,
    "score": score,
    "metric": "runtime_ms",
    "speedup": round(speedup, 4),
    "baseline": baseline,
    "reference": reference,
    "correctness": True,
}
with open('/logs/verifier/reward.json','w') as f:
    json.dump(result, f, indent=2)
with open('/logs/verifier/reward.txt','w') as f:
    f.write(str(reward))
print(f"score={score} ms  speedup={speedup:.3f}x  reward={reward:.4f}")
PY

exit 0
