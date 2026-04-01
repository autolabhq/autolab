#!/usr/bin/env bash
# Reference solution: replace solve.c with the AES-NI optimized implementation.
set -euo pipefail
cp /solution/solve_optimized.c /app/solve.c
make -C /app
