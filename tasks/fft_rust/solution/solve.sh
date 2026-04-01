#!/usr/bin/env bash
# solution/solve.sh — apply the reference optimised implementation.
# Called by Harbor's Oracle agent to validate task feasibility.
set -euo pipefail
cp /solution/optimized_solve.rs /app/src/solve.rs
echo "Reference solution installed at /app/src/solve.rs"
