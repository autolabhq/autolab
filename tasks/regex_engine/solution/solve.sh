#!/usr/bin/env bash
set -euo pipefail

cp /solution/solve_optimized.rs /app/src/solve.rs
cargo build --release --manifest-path=/app/Cargo.toml
