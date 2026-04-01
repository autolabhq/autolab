# AES-128-CTR Encryption Throughput

Optimize `aes128_ctr_encrypt()` in `/app/solve.c` to encrypt 256 MiB of data as fast as possible.

## Problem

The function implements AES-128 in CTR (Counter) mode. It must produce output matching NIST SP 800-38A test vectors — wrong answers score 0.

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
- The function signature in `solve.h` must remain unchanged.
- C11 only. No external libraries beyond the C standard library, `<immintrin.h>`, and `<wmmintrin.h>`.
- GCC extensions and intrinsics are allowed.
- Single-threaded only.
