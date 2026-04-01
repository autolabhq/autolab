#!/bin/bash
set -euo pipefail

# Reference solution: pre-capture CUDA graphs for all common batch sizes
# and increase prefill chunk budget.
#
# The baseline lazily captures CUDA graphs on first use of each batch size.
# This causes latency spikes during inference when a new batch size appears.
# Pre-capturing at startup eliminates these spikes.

python3 /solution/apply_optimizations.py

echo "Running benchmark..."
python3 /app/benchmark.py
