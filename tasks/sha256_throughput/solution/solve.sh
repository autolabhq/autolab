#!/usr/bin/env bash
set -euo pipefail
cp /solution/solve_optimized.c /app/solve.c
make -C /app
