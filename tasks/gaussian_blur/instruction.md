# Gaussian Blur

Optimize `blur_image()` in `/app/solve.c` to be as fast as possible.

## Problem

Apply a **17×17 Gaussian blur kernel** (σ = 3.0) with clamp-to-edge border handling to a **4096 × 4096** grayscale image **5 times in sequence**, where each pass feeds into the next. The image is stored as one byte per pixel, row-major.

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
- C99 only. C standard library and `<immintrin.h>` are allowed. No other external libraries.
- Single-threaded only.
- Output must remain correct. Max allowed per-pixel deviation from the reference: **±4**.
