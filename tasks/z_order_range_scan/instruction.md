# 2D Range Count

Implement a faster two-dimensional rectangular range count in `/app/solve.rs`.

## Problem

The input is a newline-separated coordinate stream. Each non-empty line holds two unsigned decimal integers separated by ASCII whitespace:

```text
12345 678
40000 40000
17 65535
```

Both values are in the inclusive grid range `[0, 65535]` (16 bits per axis). Coordinates may repeat: the input is a multiset, not a set. Lines that do not parse as `<x> <y>` should be ignored.

The runner builds the index once per case, then issues many rectangular count queries. `count_in(xlo, ylo, xhi, yhi)` returns the number of points `(x, y)` such that `xlo <= x <= xhi` *and* `ylo <= y <= yhi`, with both bounds inclusive. Bounds are themselves in `[0, 65535]`. If the rectangle is empty (`xlo > xhi` or `ylo > yhi`), return `0`.

Implement this API:

```rust
pub struct PointSet { /* private */ }

impl PointSet {
    pub fn new(input: &str) -> Self;
    pub fn count_in(&self, xlo: u32, ylo: u32, xhi: u32, yhi: u32) -> u32;
}
```

Per-query cost matters more than `new()` cost: construction is amortized over thousands of queries per case.

## Files

| Path | Role | Editable? |
|------|------|-----------|
| `/app/solve.rs` | Range-count implementation | **yes** |
| `/app/main.rs` | Task runner and correctness checks | no |
| `/app/Cargo.toml`, `/app/Cargo.lock` | Rust package files | no |
| `/app/target/` | Cargo build output | generated |

Every read-only file is checked against a host-side original during verification. Tampering scores 0.

## Rules

- Edit `/app/solve.rs` only. Do not create any other files under `/app/` (no extra `.rs` modules, no lookup tables, no `.bak`).
- `/app/` must contain only the files in the table above plus `/app/.task_seed` at verify time.
- Preserve the public `PointSet` API exactly.
- Use only the Rust standard library already available in the crate. No external crates (`rayon`, `tokio`, `crossbeam`, `serde`, `memmap`, …).
- No `unsafe`, FFI, inline assembly, generated-code escape, file I/O, process execution, environment access, threading, or networking.
- No global mutable state: `static mut`, `thread_local!`, `OnceCell` / `OnceLock`, `lazy_static`, `Mutex::new`, `RefCell::new`, `Atomic*::new` are all rejected.
- Output must match the expected results exactly.
- Time budget: 4 hours
