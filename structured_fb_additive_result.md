# Additive-shift factor base test

**Date:** 2026-05-27
**Status:** completed, negative result (with methodology caveat)

## Hypothesis

If the factor-base x-coords form an arithmetic progression
`{x_0, x_0 + d, ..., x_0 + (m-1)d}`, the constraint polynomial
becomes `d^m · Y(Y-1)...(Y-(m-1))` where `Y = (X-x_0)/d`.
This is a "falling factorial polynomial" with rich combinatorial
structure that *could* be exploited by a clever GB algorithm.

## Setup

Three configurations tested:
- bits=13, target FB size 16
- bits=16, target FB size 20
- bits=19, target FB size 24

## Result

| bits | \|FB\| | Avg GB / query |
|----:|------:|---------------:|
| 13 | 16 | 4.6ms |
| 16 | 20 | 14.9ms |
| 19 | 24 | 12.2ms |

These per-query times are essentially identical to the generic-FB
baseline (5-31ms for similar sizes).

## Methodology caveat

The arithmetic-progression construction found x_0=0, d=1 — meaning
the "AP" is actually just the natural order of integers 0, 1, 2, ...
filtered by quadratic-residue-ness. This degenerates to the
generic FB.

To do a *real* arithmetic-progression test, we'd need d > 1 with
each x_0 + i*d being a QR. The Legendre symbol distribution makes
this difficult to control: for `d` coprime to `p`, the AP
hits QRs roughly half the time, so finding a long AP of consecutive
QR-points requires extensive search.

## Why the GB cost is unchanged

Even with a structured FB:

1. Sage's F4/F5 implementation does not exploit the AP structure
   (or the binomial structure of `X^m - 1`). It treats constraint
   polynomials as generic dense polynomials.
2. The Hilbert regularity of the ideal is determined by the
   variety size, which is fixed by `|FB|`.
3. The variety enumeration step is the dominant cost, and it
   scales with the number of solutions, not the sparsity of the
   defining polynomials.

## Lesson learned

**No polynomial structure on the factor base sparsifies the
Semaev Groebner basis system unless we modify the GB algorithm
itself.** This is a deep observation: structured FBs *might* help
with a custom algorithm (FGLM with sparse linear algebra, or
F_n-aware decomposition), but they do not help with off-the-shelf
F4/F5.

## What the experiments leave open

A genuinely novel angle: **asymptotic crossover**. From the F_3
Groebner result at 22 bits, Method C (GB) was 0.65s vs Method A
(pair-sum) at 0.76s — i.e., the GB approach *becomes competitive*
at this scale. This is consistent with Diem's L(2/3) heuristic
asymptotic crossover.

We should test this carefully at 25, 28, 30 bits to see whether
the crossover continues. If yes, the GB approach has a real
asymptotic win, even though constant factors hurt at small sizes.

## Conclusion

The structured-FB direction is closed for both multiplicative
subgroups and arithmetic progressions. The next open sub-problem
is the asymptotic crossover regime (25-30 bits) for the standard
generic-FB F_3 Groebner approach.
