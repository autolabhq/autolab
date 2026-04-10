# Integer Sort

Optimize `radix_sort()` in `/app/solve.c` to sort **50 million random 32-bit unsigned integers** as fast as possible.

## Problem

The benchmark generates 50M random `uint32_t` values from a fixed seed, calls `radix_sort()` five times on fresh copies, and reports the median wall-clock time.

The function signature in `solve.h` is fixed and must not change:
```c
void radix_sort(uint32_t *arr, size_t n);
```

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
- C standard library (`<stdlib.h>`, `<string.h>`, `<stdint.h>`) is allowed. GCC extensions (`__builtin_*`, `restrict`, `__attribute__`) are allowed.
- Single-threaded only.
- Sorted output must be exactly correct. Wrong output scores 0.
- Time budget: 2 hours
