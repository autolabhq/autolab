#!/usr/bin/env bash
# solution/solve.sh — apply the reference optimized solution
# Run from the /app directory inside the container.
set -euo pipefail
cp /solution/optimized_program.s /app/program.s
echo "Applied optimized_program.s"
