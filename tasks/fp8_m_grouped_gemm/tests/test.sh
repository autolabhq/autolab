#!/usr/bin/env bash
# AutoLab verifier entry-point for fp8_m_grouped_gemm
# Runs inside the Docker container with /app mounted.

set -euo pipefail

LOG_DIR="/logs/verifier"
OUTPUT_FILE="${LOG_DIR}/reward.json"
APP_DIR="/app"
TESTS_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "${LOG_DIR}"

echo "=== fp8_m_grouped_gemm verifier ==="
echo "App dir : ${APP_DIR}"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Python  : $(python3 --version)"
echo "PyTorch : $(python3 -c 'import torch; print(torch.__version__)')"
echo "Triton  : $(python3 -c 'import triton; print(triton.__version__)')"
echo ""

# ── sanity: solve.py must exist ───────────────────────────────────────────────
if [ ! -f "${APP_DIR}/solve.py" ]; then
    echo "ERROR: /app/solve.py not found"
    python3 -c "
import json, os
os.makedirs('${LOG_DIR}', exist_ok=True)
json.dump({'reward':0.0,'correct':False,'agent_tflops':0.0,
           'deepgemm_tflops':0.0,'ratio':0.0,
           'message':'solve.py not found'}, open('${OUTPUT_FILE}','w'))
"
    exit 1
fi

# ── syntax check ─────────────────────────────────────────────────────────────
echo "--- syntax check ---"
python3 -m py_compile "${APP_DIR}/solve.py" || {
    echo "ERROR: solve.py has syntax errors"
    python3 -c "
import json, os
os.makedirs('${LOG_DIR}', exist_ok=True)
json.dump({'reward':0.0,'correct':False,'agent_tflops':0.0,
           'deepgemm_tflops':0.0,'ratio':0.0,
           'message':'solve.py failed syntax check'}, open('${OUTPUT_FILE}','w'))
"
    exit 1
}
echo "OK"

# ── run benchmark ─────────────────────────────────────────────────────────────
echo ""
echo "--- benchmark (warmup=5, repeat=10) ---"
python3 "${TESTS_DIR}/run_benchmark.py" \
    --warmup 5 \
    --repeat 10 \
    --output "${OUTPUT_FILE}" \
    --app-dir "${APP_DIR}" 2>&1

EXIT_CODE=$?
echo ""
echo "--- result ---"
cat "${OUTPUT_FILE}"
echo ""

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "Benchmark exited with code ${EXIT_CODE}"
    # reward.json was already written by run_benchmark.py
fi

exit ${EXIT_CODE}
