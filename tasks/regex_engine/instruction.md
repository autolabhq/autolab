# Regex Engine

Optimize `run_workload()` in `/app/src/solve.rs` to be as fast as possible.

## Problem

Compile a fixed set of regex patterns and search them against **100,000 haystacks** (HTTP paths, log messages, identifiers, emails, etc.). Both compilation and search are included in the measured time.

Supported syntax: concatenation, alternation (`|`), quantifiers (`*`, `+`, `?`), grouping (`(...)`), character classes (`[abc]`, `[a-z0-9]`), wildcard (`.`), anchors (`^`, `$`), and metacharacter escaping (`\`).

The function signature in `src/main.rs` is fixed and must not change.

## Files

| File | Role |
|------|------|
| `src/solve.rs` | **Edit this file only** |
| `src/main.rs` | Benchmark driver (read-only) |
| `Cargo.toml` | Build config (read-only) |

## Rules

- Do not modify `Cargo.toml` or `Cargo.lock`. Build configuration is fixed.
- Edit `/app/src/solve.rs` only. Do not modify `src/main.rs`, `Cargo.toml`, or anything under `/tests/`.
- No external crates. Pure Rust only.
- Single-threaded only.
- Output must remain exactly correct.
