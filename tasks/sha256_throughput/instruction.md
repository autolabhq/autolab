# SHA-256 Throughput

Optimize `sha256()` in `/app/solve.c` to hash a **512 MiB** buffer as fast as possible.

## Problem

Compute the SHA-256 digest of a 512 MiB pseudo-random buffer (deterministic pattern). The benchmark warms up once (not timed), then reports the median of 3 timed runs.

The function signature in `solve.h` is fixed and must not change:
```c
void sha256(const uint8_t *data, size_t len, uint8_t *digest);
```

## Files

| Path | Permission |
|------|-----------|
| `/app/solve.c` | **Edit** |
| `/app/main.c` | Read-only |
| `/app/solve.h` | Read-only |
| `/app/Makefile` | Read-only |

## Rules

- Do not modify `Makefile`. Build configuration is fixed.
- Edit `/app/solve.c` only.
- C standard library and `<immintrin.h>` are allowed. GCC extensions (`__attribute__`, `__builtin_*`, inline assembly) are allowed.
- Single-threaded only.
- Output must match Python's `hashlib.sha256` exactly. Wrong results score 0.
