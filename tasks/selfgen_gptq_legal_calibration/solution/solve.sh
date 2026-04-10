#!/usr/bin/env bash
set -euo pipefail

# Reference run inside task container.
# This script is intentionally minimal and mirrors common task style.
python3 /app/benchmark.py --mode eval --seeds "42,314,999" --enforce-legal 1
