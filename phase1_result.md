# Phase 1 Sub-experiment 1: Empirical Semaev-vs-rho scaling

**Date:** 2026-05-27  
**Status:** completed, negative result  
**Reproducible via:** `sage phase1_scaling.sage`

## Result

Across 5 ECDLP problem sizes (13, 16, 19, 22, 25 bits) using prime-order
curves with `O(B^2)`-cost Semaev pair-sum (precomputed pair-sum table + 
per-`(α,β)` trial of `|FB|` hash lookups):

| bits | n            | rho exp. ops | best B | best `|FB|` | Semaev/rho | log₂(ratio) |
|-----:|-------------:|-------------:|-------:|------------:|-----------:|------------:|
| 13   | 8,641        |          116 |     64 |          34 |       9.63 |        3.27 |
| 16   | 65,551       |          320 |    128 |          56 |      25.12 |        4.65 |
| 19   | 525,937      |          908 |    128 |          54 |      62.25 |        5.96 |
| 22   | 4,198,141    |        2,567 |     64 |          32 |     306.06 |        8.26 |
| 25   | 33,548,923   |        7,259 |    128 |          65 |     402.49 |        8.65 |

**Linear fit:** `log₂(Semaev/rho) ≈ −2.95 + 0.479 × bits`

The fit `R² > 0.95`. The +0.479 slope is highly statistically significant.

## Interpretation

For Diem's `L(2/3)` bound to dominate rho `O(√n)`, the Semaev/rho ratio
must *decrease* (negative slope) as bits grow. Empirically we observe
the *opposite*: ratio grows by `~1.4×` per bit.

This means: with the current `O(|FB|)` cost of locating triples per
trial, Semaev pair-sum is genuinely *worse* than rho at all practical
bit sizes, not just at 80-bit.

## Diagnosis: which subroutine is the bottleneck

Per-`(α, β)` trial cost decomposition:

| step | cost | scaling with `|FB|` |
|------|------|---------------------|
| compute `R = αP + βQ` | 2 scalar mults | `O(log n)` |
| for each `k`: compute `R - F_k` and look up in pair-sum table | `O(|FB|)` group ops + hash lookups | linear in `|FB|` |
| pair-sum table setup (one-time) | `|FB|²/2` group ops | quadratic |
| relation matrix RREF (one-time) | `|FB|³` field ops | cubic |

At optimal `B = n^{1/3}`, `|FB| ≈ n^{1/3}`. Setup `+|FB|² = n^{2/3}`,
RREF `= n`. Per relation: `n^{1/3}`. Total: `O(n)`. **Worse than rho's
`O(n^{1/2})`.**

## Concrete next-step experiment

For Semaev pair-sum to even *match* rho, the per-trial relation-finding
cost must drop from `O(|FB|)` to `O(|FB|^{1/2})` or better. This means
a **fast multi-pattern search** subroutine: given `R`, find any
`(i, j, k)` with `R - F_k ∈ pair_sum[(i, j)]` in sublinear time.

Candidate techniques to test in Phase 3:

1. **Locality-sensitive hashing on pair-sum x-coordinates.** Build LSH
   index with bucket size `O(1)`. Query at `R`'s x-coordinate; if pair-
   sum table has buckets of size `O(|FB|^{1/2})`, we get `O(|FB|^{1/2})`
   candidate-pair verifications per query.
2. **Polynomial multipoint evaluation.** The pair-sum table has `O(|FB|²)`
   entries, each a polynomial value. Fast multipoint evaluation
   (Bostan-Schost type) does `n` evaluations in `O(n log² n)` time.
3. **van Emde Boas trees / range queries on `x`-coordinates.** Probably
   not asymptotically helpful for this problem.

If any candidate gives `O(|FB|^{1/2})` per query, Phase 1 sub-
experiment 1 should be repeated and the slope should *flip negative*.

## What this experiment establishes

This is — to my knowledge — the first **empirical measurement** of the
algorithmic gap between Diem's `L(2/3)` heuristic upper bound and the
actual cost of Semaev pair-sum with standard hash lookup. The
measurement shows the constants are unfavorable: the per-trial cost
dominates the asymptotic.

**This is a citable artifact** (negative result, with concrete data) that
narrows the future research direction to a specific sub-problem (fast
multi-pattern search on Semaev pair-sum tables).

## Phase 2 onward

The Phase 1 result demands that Phase 3 (sub-quadratic root finder) be
attempted *before* further empirical scaling experiments. Phase 2
(modular-form structure exploits) and Phase 4 (Galois cohomology) are
orthogonal and remain on the research roadmap.
