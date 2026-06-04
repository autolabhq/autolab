#!/usr/bin/env bash
# solution/solve.sh - apply the reference optimized implementation.
set -euo pipefail

cp /solution/solve_optimized.py /app/retriever.py
python3 -m py_compile /app/retriever.py
echo "Reference solution installed."
