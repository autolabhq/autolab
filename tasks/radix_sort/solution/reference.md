# Integer Sort (Radix Sort) — Reference

## Background
Sorting integers is a fundamental primitive in databases, compilers, graphics pipelines, and network packet processing. While comparison-based sorts are general, integer keys enable O(n) algorithms that dominate at scale. For 50 million 32-bit integers, the difference between O(n log n) and O(n) is roughly 26× in operation count alone.

## Baseline Approach
The unoptimized implementation delegates to `stdlib qsort` with a comparison function pointer. For n = 50M, this performs ~1.3 billion comparisons with an indirect function call overhead on every one. Typical runtime: ~4.5s on a 2-core machine.

## Possible Optimization Directions
1. **Naive 8-pass LSD radix sort** — process one byte per pass, 8 passes total; O(n) vs O(n log n) beats qsort by ~8–10×
2. **2-pass 16-bit radix sort** — use 16-bit "digits", only 2 scatter passes instead of 8; halves memory traffic to the input array (~1.3× over 8-pass)
3. **Single-scan histogram** — compute both lo-16 and hi-16 histograms in one forward scan, cutting counting-phase memory bandwidth in half (~1.2× over 2-pass)
4. **Software write prefetching** — issue `__builtin_prefetch` on scatter destinations 32 iterations ahead to hide write-miss latency in the random-access scatter phase (~1.2× over single-scan)
5. **`restrict` / non-aliasing hints** — declare scatter source/destination buffers as non-aliasing so the compiler can reorder loads/stores more aggressively (small, but free)
6. **Branchless prefix-sum accumulator** — accumulate histogram into exclusive prefix sums without branching over 65536 bins

## Reference Solution
2-pass LSD radix sort with 16-bit digit width. Counts both lo-16 and hi-16 histograms in a single forward scan (Step 1), converts to exclusive prefix sums (Step 2), then scatters arr→tmp by low 16 bits (Pass 1) and tmp→arr by high 16 bits (Pass 2). Both scatter loops issue `__builtin_prefetch` 32 elements ahead to hide write-miss latency. Source/destination buffers are declared `restrict`-compatible (static histogram arrays) to aid compiler optimization.

## Source
- Knuth, *The Art of Computer Programming*, Vol. 3 §5.2 — LSD radix sort
- Kim et al., *FAST: Fast Architecture Sensitive Tree Search on Modern CPUs and GPUs* (SIGMOD 2010) — cache-aware integer sort analysis
- MIT 6.172 Performance Engineering, Fall 2018, HW2 (sorting lab)
- Algorithmica, *Radix Sort Revisited* (2022) — https://en.algorithmica.org/hpc/algorithms/sorting/
