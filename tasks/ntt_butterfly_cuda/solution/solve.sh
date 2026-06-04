#!/usr/bin/env bash
set -euo pipefail

cp /solution/solve_optimized.cu /app/solve.cu
make -C /app clean
make -C /app
/app/ntt_butterfly --verify
