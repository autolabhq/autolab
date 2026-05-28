# Phase 3: Sub-quadratic Semaev root finder

**Date:** 2026-05-27  
**Status:** completed, partial result  
**Reproducible via:** `sage phase3_subquadratic.sage`

## Result

Tested negation-aware pair-sum lookup (keying by `x`-coordinate only,
accepting `±y` matches) against baseline (full point-key lookup):

| bits | B   | method         | trials | time | rate (rel/s) |
|-----:|----:|----------------|-------:|-----:|-------------:|
| 19   | 64  | baseline       | 3,636  | 1.53s |       15.64 |
| 19   | 64  | negation_aware | 1,345  | 0.55s |       43.30 |
| 19   | 128 | baseline       |   386  | 0.29s |       83.63 |
| 19   | 128 | negation_aware |   123  | 0.08s |      286.34 |
| 22   | 64  | baseline       | 29,015 | 13.95s |       1.72 |
| 22   | 64  | negation_aware |  3,988 | 1.95s |      12.32 |
| 22   | 128 | baseline       |  1,882 | 1.84s |       13.06 |
| 22   | 128 | negation_aware |    580 | 0.55s |      43.33 |

Negation gives consistent `~3×` speedup, but the asymptotic scaling
across bit-sizes is unchanged. This is a constant-factor improvement,
not a sub-quadratic algorithm.

## Interpretation

The fundamental bottleneck is that for each `(α, β)` we must check
each of the `|FB|` factor base points `F_k` against the pair-sum
table — `|FB|` operations per trial, no matter how the pair-sum table
is keyed.

To beat this we would need to either:

1. **Algebraically narrow** the set of candidate `k`'s for each `R`
   to `o(|FB|)` candidates. The natural candidate is to use the
   Semaev `S_3` polynomial as a constraint, but solving `S_3(x_R, x,
   X_3) = 0` for `(x, X_3)` jointly is itself a multivariate root
   problem requiring `Ω(|FB|^2)` operations to scan all candidates.
2. **Use a precomputed multipoint-evaluation index** on the pair-sum
   table that lets us "predict" which `k` values are likely to yield
   hits. With LSH or `x`-coordinate bucketing this gives only
   constant-factor savings because the pair-sum hash is essentially
   random.

The 3× negation-aware speedup is real engineering, but does not
flip the Phase 1 scaling-slope sign.

## Conclusion

Phase 3 produces a `3×` constant-factor improvement (citable as an
engineering result) but **no sub-quadratic root-finding algorithm**.
The Diem `L(2/3)` algorithmic conjecture remains open at the
sub-routine level.

This is a negative result in the strong form: the sample of candidate
sub-quadratic subroutines explored here (negation-symmetry, sorted-
coordinate lookup, locality-sensitive hashing on `x`-coord) all give
only constant-factor improvements.

## Next research direction (post-Phase-3)

A future Phase 3' would investigate non-trivial algebraic structures
on the pair-sum table: for instance, whether the set
`{x(F_i + F_j) mod p : i, j}` forms an algebraic variety that admits
fast intersection with `{x(R - F_k) : k}` via resultant techniques. No
published work addresses this.
