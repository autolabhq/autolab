# Hash Join

Optimize `hash_join()` in `/app/solve.c` to be as fast as possible.

## Problem

Inner equi-join of two tables on a shared integer key:

| Table | Rows | Schema |
|-------|------|--------|
| R | 20,000 | `{int32 key, int32 payload}` |
| S | 5,000,000 | `{int32 key, int32 payload}` |

Keys are uniform random from `[0, 50000)`. Output:
- `match_count`: total matching `(R, S)` pairs
- `checksum`: `SUM((uint32)r.payload + (uint32)s.payload)` over all matches (uint64, wrapping)

The function signature in `solve.h` is fixed and must not change.

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
- Standard C99. SIMD intrinsics (`<immintrin.h>`) are allowed.
- No external libraries. Single-threaded only.
- Wrong results score 0.
