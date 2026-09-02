# P1553 genus-one abc product-line R53 independent review

- Review date: 2026-07-20.
- Reviewer execution: independent `xhigh` subagent
  `019f8177-d5b3-74e1-99df-fe6165c5603c`.
- Write scope: read-only review; the reviewer edited no files.
- Final verdict after correction replay: `PASS`.
- Claim class: theorem-scope gate only; `toy`, `model-bound`,
  `novelty-unverified`, and no breakthrough claim.

## Checks reconstructed

The reviewer independently checked:

1. Cancelling the common divisor makes the residual zero divisors pairwise
   disjoint and identifies them as the zero, one, and pole fibers of
   `[a:c]:X->P^1`.
2. For a nonconstant separable degree-`m` map, Riemann-Hurwitz and the local
   different inequalities give `m<=n+2g-2`, including in positive
   characteristic with wild ramification.
3. `0<m<p` implies separability because any nontrivial inseparable degree is
   a positive power of `p` dividing the total degree.
4. The pinned R52 report gives line-bundle degree nine, common divisor degree
   six, moving degree three, and three disjoint reduced residual fibers of
   three points each. Thus the R52 arithmetic is `3<=9` with slack six.
5. The primitive reduced degree-nine control is `9<=27` with slack eighteen.
6. The conclusion is scoped to the single three-fiber inequality applied
   after gcd cancellation. Auxiliary factor-sensitive `abc`/S-unit arguments
   remain open.

## Corrections required and resolved

The first review returned `PASS_WITH_CORRECTIONS` and required:

- an explicit `m>0` nonconstant-map hypothesis;
- replacing absolute ordinary-`abc` language by the narrower claim about the
  single three-fiber inequality;
- loading the pinned R52 report instead of repeating its summary values; and
- explicit assertions that the R52 base divisor has degree six and its
  original line bundle has degree nine.

All four corrections were applied. The R53 JSON report was regenerated and
matched a fresh deterministic replay byte-for-byte. A final narrow re-review
returned exactly `PASS`.

## Limits

This review does not classify trisecants of the multiplication image, prove a
one-varying-factor theorem, construct an asymptotic pencil, or supply target
location, R10 outputs, rank, logarithms, descent, a Shoup-bound improvement,
or an ECDLP breakthrough.
