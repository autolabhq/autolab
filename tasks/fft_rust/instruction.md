# Fast Fourier Transform (Rust)

Optimize `fft()` in `/app/src/solve.rs` to be as fast as possible.

## Problem

Compute the DFT of a length-n real signal:

```
X[k] = Σ_{j=0}^{n-1}  x[j] · e^{−2πi·j·k/n}     k = 0, …, n−1
```

The benchmark uses **n = 32768**. The function signature must not change:

```rust
pub fn fft(signal: &[f64]) -> Vec<[f64; 2]>
```

`signal.len()` is always a power of two. `output[k]` must equal `[Re(X[k]), Im(X[k])]`.

## Files

| Path | Permission |
|------|-----------|
| `/app/src/solve.rs` | **Edit** |
| `/app/src/main.rs` | Read-only |
| `/app/Cargo.toml` | Read-only |
| `/app/Makefile` | Read-only |

## Rules

- Do not modify `Cargo.toml` or `Cargo.lock`. Build configuration is fixed.
- Edit `/app/src/solve.rs` only.
- Standard library (`std`) only — no external crates.
- Single-threaded only.
- Correctness verified via Parseval's theorem and spectral checksum (relative tolerance 1e-6).
