# Levenshtein Distance Batch

Optimize `levenshtein()` in `/app/solve.c` to compute edit distances for **1,000,000 string pairs** as fast as possible.

## Problem

The Levenshtein edit distance between two strings is the minimum number of single-character insertions, deletions, and substitutions needed to transform one into the other.

The benchmark generates 1M deterministic pairs of lowercase ASCII strings (lengths 16–64). It warms up once (not timed), then reports the median of 3 timed runs over all 1M pairs.

The function signature in `solve.h` is fixed and must not change:
```c
int levenshtein(const char *a, int la, const char *b, int lb);
```

During benchmarking, `la` and `lb` are both guaranteed to be in `[16, 64]`. Correctness tests may include empty strings and lengths in `[0, 64]`; inputs are lowercase ASCII `a`–`z`.

## Files

| Path | Permission |
|------|-----------|
| `/app/solve.c` | **Edit** |
| `/app/main.c` | Read-only |
| `/app/solve.h` | Read-only |
| `/app/Makefile` | Read-only |

## Rules

- Do not modify read-only files. `/app/Makefile`, `/app/main.c`, and `/app/solve.h` are hash-protected.
- Edit `/app/solve.c` only.
- C standard library, `<stdint.h>`, and `<immintrin.h>` are allowed. GCC extensions (`__attribute__`, `__builtin_*`, inline assembly) are allowed.
- Single-threaded only.
- Results must be mathematically correct for all inputs. Wrong results score 0.
- Time budget: 2 hours
