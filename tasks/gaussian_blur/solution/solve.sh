#!/usr/bin/env bash
# solution/solve.sh — apply the reference optimised implementation.
# Called by the Harbor Oracle agent to verify task feasibility.
set -e
cp /solution/solve_optimized.c /app/solve.c
make -C /app
echo "Reference solution installed."
