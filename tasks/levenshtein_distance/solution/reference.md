# Levenshtein Distance — Reference

## Background

The Levenshtein edit distance (minimum insertions, deletions, and substitutions to transform one string into another) is fundamental to spell-checking, DNA sequence alignment, fuzzy text search, and plagiarism detection. Batch evaluation over millions of pairs is the bottleneck in bioinformatics pipelines, search engines, and data deduplication tools.

## Baseline Approach

The unoptimized implementation uses the classic two-row rolling DP: for each pair of strings with lengths `la × lb`, it allocates two `int[65]` arrays on the stack and fills them with `O(la × lb)` scalar comparisons and `min3()` evaluations. With 1M pairs averaging roughly 40 × 40 characters, this amounts to ~1.6 billion scalar operations with no bit-level parallelism.

## Possible Optimization Directions

1. **Eliminate memcpy** — replace the row copy with pointer swapping or an explicit `prev` variable to track the diagonal, saving one memory pass per outer iteration (~5–10% improvement).
2. **Early termination** — once all cells in a row exceed `max(la, lb)`, the result cannot improve; abort early for nearly-identical or completely-dissimilar strings.
3. **Myers bit-vector algorithm** — represent each DP row as a pair of bit-planes (`vp`/`vm`) and reduce the inner loop to ~8 branchless 64-bit operations per column regardless of `la`. For `la ≤ 64` (exactly our bound) a single `uint64_t` covers all rows, cutting per-pair work from `O(la × lb)` integer ops down to `O(lb)` bitwise ops. This is the dominant optimization: **~8–12× speedup** on this workload.
4. **SIMD across pairs (SSE2/AVX2)** — process 2–4 independent pairs simultaneously by packing their `vp`/`vm` words into SIMD registers. Each SIMD lane tracks one pair independently. Requires refactoring the call site into a batched inner loop.
5. **Profile-guided loop unrolling** — manually unroll the outer `lb` loop in groups of 4 or 8, enabling the compiler to keep multiple `(vp, vm, eq)` triples in registers simultaneously.

## Reference Solution

Myers O(m) bit-vector algorithm using a single `uint64_t` per pair for strings of length ≤ 64. The character profile array `pm[256]` (2 KB) is built once per pair and fits entirely in L1 cache. For each column `j` (character `b[j]`), seven branchless 64-bit operations (`&`, `|`, `^`, `+`, `<<`, `~`) update the `vp`/`vm` bit-planes and adjust the running edit distance by ±1 based on the delta at the last row.

Inputs with `la > lb` are swapped before entering the algorithm so the pattern (rows) always fits in one word.

## Sources

- Myers, G., "A Fast Bit-Vector Algorithm for Approximate String Matching Based on Dynamic Programming", *Journal of the ACM*, 46(3):395–415, 1999.
- Hyyrö, H., "Explaining and Extending the Bit-parallel Approximate String Matching Algorithm of Myers", University of Tampere Technical Report A-2001-10.
- Hyyrö, H., "A Bit-Vector Algorithm for Computing Levenshtein and Damerau Edit Distances", *Nordic Journal of Computing*, 10(1):29–39, 2003.
