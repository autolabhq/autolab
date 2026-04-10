#!/usr/bin/env bash
set -euo pipefail

cp /solution/optimized_codec.rs /app/src/codec.rs
cp /solution/optimized_block.rs /app/src/block.rs
cp /solution/optimized_iter.rs /app/src/iter.rs
cp /solution/optimized_merge.rs /app/src/merge.rs
cp /solution/optimized_builder.rs /app/src/builder.rs
cp /solution/optimized_compaction.rs /app/src/compaction.rs

# Ensure cargo detects file changes (cp may preserve timestamps)
touch /app/src/*.rs
