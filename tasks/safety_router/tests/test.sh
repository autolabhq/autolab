#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

APP_DIR="/app"
if [[ ! -f "$APP_DIR/train.py" ]]; then
  APP_DIR="$TASK_DIR/environment"
fi

PRIVATE_DATA="/tests/data/test_private.npz"
if [[ ! -f "$PRIVATE_DATA" ]]; then
  PRIVATE_DATA="$SCRIPT_DIR/data/test_private.npz"
fi

LOG_DIR="${LOG_DIR:-/logs/verifier}"
mkdir -p "$LOG_DIR"

MODEL_OUT="${MODEL_OUT:-/tmp/smallest_safety_router_model.npz}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" "$APP_DIR/train.py" \
  --train-path "$APP_DIR/data/train.npz" \
  --val-path "$APP_DIR/data/val.npz" \
  --output "$MODEL_OUT"

"$PYTHON_BIN" "$SCRIPT_DIR/verify.py" \
  --app-dir "$APP_DIR" \
  --private-data "$PRIVATE_DATA" \
  --model-path "$MODEL_OUT" \
  --log-dir "$LOG_DIR"
