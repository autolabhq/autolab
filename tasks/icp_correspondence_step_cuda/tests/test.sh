#!/usr/bin/env bash
set -euo pipefail

LOG=/logs/verifier
mkdir -p "$LOG"

write_reward() {
    local reward="$1"
    local payload="$2"
    python3 - "$reward" "$payload" <<'PY'
import json
import sys

reward = float(sys.argv[1])
payload = json.loads(sys.argv[2])
with open("/logs/verifier/reward.json", "w") as f:
    json.dump(payload, f, indent=2)
with open("/logs/verifier/reward.txt", "w") as f:
    f.write(str(reward))
PY
}

fail_zero() {
    local reason="$1"
    echo "$reason -- scoring 0" >&2
    write_reward 0.0 "{\"reward\":0.0,\"reason\":\"$reason\"}"
    exit 0
}

# ── 1. Integrity of read-only files ─────────────────────────────────────
echo "=== Integrity ===" >&2
if ! sha256sum -c /orig/protected.sha256 --quiet 2>/dev/null; then
    sha256sum -c /orig/protected.sha256 >&2 || true
    fail_zero "protected_file_modified"
fi

# ── 2. Strict /app whitelist ────────────────────────────────────────────
echo "=== /app whitelist ===" >&2
EXPECTED=$(cat <<'EOF'
/app
/app/Makefile
/app/icp_corr
/app/kdtree.h
/app/main.cu
/app/solve.cu
/app/solve.h
EOF
)
ACTUAL=$(find /app -mindepth 0 \( -type f -o -type d -o -type l \) 2>/dev/null | sort)
UNEXPECTED=$(comm -23 <(echo "$ACTUAL") <(echo "$EXPECTED" | sort) || true)
if [ -n "$UNEXPECTED" ]; then
    echo "Unexpected:" >&2
    echo "$UNEXPECTED" >&2
    fail_zero "app_tree_modified"
fi

# ── 3. Constraint checks on solve.cu ────────────────────────────────────
echo "=== Constraints ===" >&2
SOLVE=/app/solve.cu

# 3a. Banned library headers (vendor / framework / external NN libs).
if grep -qiE '#[[:space:]]*include[[:space:]]*[<"](thrust|cub/|cutlass|cublas|cublasLt|cudnn|curand|cusparse|cusolver|cutensor|cuTensor|nvjitlink|nvJitLink|nccl|torch|ATen|tensorflow|xformers|flash_attn|pcl/|pcl\.|nanoflann|flann|open3d|libnabo)' "$SOLVE" 2>/dev/null; then
    fail_zero "banned_library_include"
fi

# 3b. Banned high-level / vendor APIs by symbol or namespace.
if grep -qiE '\b(thrust::|cub::|cutlass::|cublas[A-Za-z_]*|cublasLt[A-Za-z_]*|cudnn[A-Za-z_]*|curand[A-Za-z_]*|cusparse[A-Za-z_]*|cutensor[A-Za-z_]*|nvjit[A-Za-z_]*|torch::|at::|pcl::|open3d::|nanoflann::|flann::|DeviceRadixSort|DeviceSegmentedRadixSort|DeviceSelect|sort_by_key|cuvsKnn|raft::)\b' "$SOLVE" 2>/dev/null; then
    fail_zero "banned_high_level_api"
fi

# 3c. No process introspection / file IO / subprocesses / hidden side channels.
if grep -qE '(\bgetenv[[:space:]]*\(|\bsecure_getenv[[:space:]]*\(|\benviron\b|__environ|/proc/|cmdline|\bargv\b|\bargc\b|\bfopen[[:space:]]*\(|\bfreopen[[:space:]]*\(|\bopenat[[:space:]]*\(|std::ifstream|std::ofstream|\breadlink[[:space:]]*\(|\bdlopen[[:space:]]*\(|\bdlsym[[:space:]]*\(|\bsystem[[:space:]]*\(|\bpopen[[:space:]]*\(|\bfork[[:space:]]*\(|\bexecv|\bexecl|\bsocket[[:space:]]*\(|\bconnect[[:space:]]*\(|/tmp/|/tests)' "$SOLVE" 2>/dev/null; then
    fail_zero "banned_io_or_process_api"
fi

# 3d. Cannot include main.cu textually.
if grep -qE '#[[:space:]]*include[[:space:]]*["<][^">]*main\.cu' "$SOLVE" 2>/dev/null; then
    fail_zero "main_cu_include_detected"
fi

# 3e. Cannot spoof verifier sentinels.
if grep -qE '__VERIFIER_' "$SOLVE" 2>/dev/null; then
    fail_zero "sentinel_spoof_attempt"
fi

# ── 4. Build ────────────────────────────────────────────────────────────
echo "=== Build ===" >&2
make -C /app clean || true
if ! make -C /app > "$LOG/build.txt" 2>&1; then
    cat "$LOG/build.txt" >&2 || true
    fail_zero "build_failed"
fi
if [ ! -x /app/icp_corr ]; then
    fail_zero "missing_binary"
fi

# ── 5. Public correctness ───────────────────────────────────────────────
echo "=== Correctness ===" >&2
if ! timeout 240 /app/icp_corr --verify > "$LOG/verify.txt" 2> "$LOG/verify_err.txt"; then
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

# ── 6. Hidden-seeded scored timed run ───────────────────────────────────
echo "=== Timed correctness path ===" >&2
if ! timeout 700 /app/icp_corr --benchmark-verify > "$LOG/bench.txt" 2> "$LOG/bench_err.txt"; then
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

# ── 7. Reward computation (calibrated on H100, keep in sync with task.toml) ──
BASELINE=64.65
REFERENCE=0.28

python3 - "$SCORE" "$BASELINE" "$REFERENCE" <<'PY'
import json
import math
import sys

score = float(sys.argv[1])
baseline = float(sys.argv[2])
reference = float(sys.argv[3])

speedup = baseline / score if score > 0.0 else 0.0
ref_speedup = baseline / reference if reference > 0.0 else 0.0
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
with open("/logs/verifier/reward.json", "w") as f:
    json.dump(result, f, indent=2)
with open("/logs/verifier/reward.txt", "w") as f:
    f.write(str(reward))
print(f"score={score} ms  speedup={speedup:.3f}x  reward={reward:.4f}")
PY

exit 0
