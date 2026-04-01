#!/usr/bin/env bash
set -euo pipefail

APP_DIR=${APP_DIR:-/app}
SOLUTION_DIR=${SOLUTION_DIR:-/solution}

if [[ ! -d "$APP_DIR" ]]; then
    APP_DIR=$(pwd)
fi

if [[ ! -f "$SOLUTION_DIR/optimized_program.stk" ]]; then
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    SOLUTION_DIR="$SCRIPT_DIR"
fi

cp "$SOLUTION_DIR/optimized_program.stk" "$APP_DIR/program.stk"
make -C "$APP_DIR"
