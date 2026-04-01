# Gaussian Blur — Reference

## Background
Gaussian blur is a fundamental image processing primitive used in computational photography, video pipelines, edge detectors (Canny, LoG), and neural network preprocessing. On large images, the naive 2-D convolution dominates runtime and is a classic target for SIMD and cache optimization.

## Baseline Approach
The unoptimized implementation applies a full non-separable 17×17 kernel in double precision to every pixel, with per-pixel border-clamping branches inside the inner loop. This performs 289 double-precision MAC operations per pixel, per pass — roughly 24 billion FLOPs total across 5 passes on a 4096×4096 image. No vectorization is possible due to the branches and double-precision width.

## Possible Optimization Directions
1. **Separable filter decomposition** — replace the 17×17 2-D kernel with two 1×17 1-D passes (horizontal then vertical); reduces MACs from 289 to 34 per pixel — 8.5× fewer arithmetic operations
2. **float instead of double** — AVX2 holds 8 floats vs 4 doubles, doubling SIMD width; scalar float is also faster; the ±4 tolerance easily accommodates float rounding across 5 passes
3. **Branch-free interior loop** — split each row into left border, interior (no clamping needed), and right border; the branch-free interior is auto-vectorizable by the compiler
4. **Transposed vertical pass** — after the horizontal pass, transpose the float buffer so the vertical pass becomes a sequential horizontal pass on transposed data; use 64×64 cache-oblivious tiling for the transpose (~L2 working set)
5. **AVX2 SIMD intrinsics** — use `_mm256_fmadd_ps` to process 8 output pixels per FMA instruction; combined with separable decomposition, ~20–40× over baseline
6. **Integer fixed-point kernel** — scale the 1-D kernel to sum to 1024 or 4096, store as `int16_t`, accumulate in `int32_t`, right-shift at the end; enables `_mm256_maddubs_epi16` / `_mm256_add_epi32` which are faster than float FMAs

## Reference Solution
Applies all four foundational layers: (1) separable 1-D float32 passes reducing MACs 8.5×, (2) branch-free interior loops enabling compiler auto-vectorization, (3) cache-friendly vertical pass via double transposition (horizontal blur on transposed buffer eliminates column-striding cache misses), and (4) 64×64 tiled transpose. SIMD intrinsics and integer fixed-point are left as further headroom for agents.

## Source
- EasyPerf Performance Challenge #4 (Canny edge detection, `gaussian_smooth` bottleneck, Oct 2019): https://easyperf.net/blog/2019/10/05/Performance-analysis-and-tuning-contest-4
- MIT 6.172 Performance Engineering of Software Systems, Fall 2018 (image convolution labs): https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/
