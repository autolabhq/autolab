# Flash Attention

Optimize `attention()` in `/app/solve.c` to be as fast as possible.

## Problem

Compute scaled dot-product attention:

```
output = softmax(Q K^T / sqrt(d)) V
```

with **n = 4096** (sequence length) and **d = 64** (head dimension), all float32.

The naive baseline allocates a full **n×n = 128 MB** score matrix (double precision), which never fits in CPU cache.

The function signature in `solve.h` is fixed and must not change.

## Files

| Path | Permission |
|------|-----------|
| `/app/solve.c` | **Edit** |
| `/app/solve.h` | Read-only |
| `/app/main.c` | Read-only |
| `/app/Makefile` | Read-only |

## Rules

- Do not modify `Makefile`. Build configuration is fixed.
- Edit `/app/solve.c` only.
- SIMD intrinsics (`<immintrin.h>`) are allowed.
- No external libraries.
- Single-threaded only.
- Wrong results score 0.
