#!/usr/bin/env bash
set -euo pipefail

APP_DIR=/app
SOL_DIR=/solution

if [ ! -d "$APP_DIR" ]; then
    APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
    SOL_DIR="$APP_DIR/solution"
fi

cp "$SOL_DIR/optimized_tokenize.go" "$APP_DIR/tokenize.go"
cp "$SOL_DIR/optimized_rank.go" "$APP_DIR/rank.go"
cp "$SOL_DIR/optimized_engine.go" "$APP_DIR/engine.go"
cp "$SOL_DIR/optimized_batch.go" "$APP_DIR/batch.go"
make -C "$APP_DIR"
"$APP_DIR/solve" -mode benchmark -runs 5
