# SSTable Compaction

Optimize the LSM-style SSTable compaction pipeline to be as fast as possible.

## Problem

The benchmark generates several deterministic sorted SSTables. Each table stores prefix-compressed records in length-delimited blocks. Records contain an 8-byte user key, a sequence number, an operation kind (Put or Delete), and a fixed-size value for Put. Compaction must merge the input tables, apply LSM visibility rules, drop shadowed versions, remove tombstoned keys, and emit a new sorted output SSTable. Correctness is verified by exact output checksum and live-entry count.

## Files

| File | Role |
|------|------|
| `src/codec.rs` | **Edit** — encoding/decoding |
| `src/block.rs` | **Edit** — block handling |
| `src/iter.rs` | **Edit** — table iteration |
| `src/merge.rs` | **Edit** — merge logic |
| `src/builder.rs` | **Edit** — output table building |
| `src/compaction.rs` | **Edit** — compaction orchestration |
| `src/main.rs` | Read-only |
| `src/types.rs` | Read-only |
| `Cargo.toml` | Read-only |

## Rules

- Do not modify `Cargo.toml` or `Cargo.lock`. Build configuration is fixed.
- Edit the files listed above only.
- Pure Rust only. No external crates.
- Single-threaded only.
- Output must be exactly correct.
- Time budget: 4 hours
