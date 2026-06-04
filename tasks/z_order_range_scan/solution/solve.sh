#!/usr/bin/env bash
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SELF_DIR/solve_optimized.rs"
DST="/app/solve.rs"

if [ ! -f "$SRC" ]; then
    echo "ERROR: reference source not found at $SRC" >&2
    exit 1
fi

cp "$SRC" "$DST"
echo "Reference solution applied: $(wc -l < "$DST") lines in $DST" >&2
