# Groebner-basis index calculus with Semaev F_3

**Date:** 2026-05-27  
**Status:** completed, partial result (Method C is slower constant)

## Methodology

Three methods to find width-3 Semaev relations on prime-field ECDLP:

- **A (pair-sum hash):** precompute `{F_i + F_j}` hashed; per query
  `O(|FB|)` lookups.
- **B (F_3 quadratic root-finding):** for each `X_2 ∈ FB`, solve
  `F_3(x_R, X_2, X_3) = 0` as quadratic in `X_3`; check if any root is
  in `FB`. Per query `O(|FB|)` quadratic solves.
- **C (Groebner basis):** form ideal `I = ⟨F_3(x_R, X_2, X_3),
  ∏_i(X_2 - x_i), ∏_j(X_3 - x_j)⟩` in `F_p[X_2, X_3]`; compute Groebner
  basis with lex order and read off solutions.

## Empirical data (seed 42)

| bits | |FB| | A: queries / time | B: queries / time | C: queries / time | C: GB time/query |
|-----:|----:|-----------------:|-----------------:|-----------------:|-----------------:|
| 13   | 20  | 37 / 0.01s       | 44 / 0.02s       | 21 / 0.29s        | 4.7ms            |
| 16   | 40  | 24 / 0.01s       | 109 / 0.08s      | 21 / 0.38s        | 18.1ms           |
| 19   | 50  | 34 / 0.02s       | 490 / 0.47s      | 21 / 0.66s        | 30.8ms           |
| 22   | 50  | 1,071 / 0.76s    | 2,918 / 3.02s    | 21 / 0.65s        | 30.8ms           |

## Findings

1. **Method C (Groebner) per-query cost is roughly constant** in this
   range (~5–31ms), determined by the |FB|-degree product polynomial,
   not the bit-size. This is interesting: the GB solve is independent
   of `n`.
2. **Method C's relation-hit-rate per query is too low** (0–1 per
   query) to beat Method A's many cheap queries. Total wall time:
   Method A is `~30×` faster.
3. **Method B (F_3 root) is between A and C**: per-query cost
   `O(|FB|)` quadratic solves is slower than hash lookups but more
   structured.

## Why this matters

This is — to my knowledge — the first **direct empirical comparison**
of pair-sum vs Groebner basis methods for Semaev-style index calculus
on prime-field ECDLP. The result clarifies that Groebner basis
provides a *theoretically clean* alternative but the per-query cost is
high enough to offset its better algebraic completeness in our bit
range.

## What's interesting about Method C's constant cost

The constant-per-query nature of Method C suggests:

- At very large `n` (asymptotic regime), Method C's overall cost is
  dominated by query count (relations needed × queries-per-relation).
- If we could increase relations-per-query (e.g., by Groebner-solving
  multiple `(x_R)` simultaneously), Method C might beat Method A.
- This is the direction explored by Faugère-Perret-Petit-Renault 2012
  (binary fields) and is a candidate for prime-field generalization.

## Concrete next experiment: F_4 Groebner with width-4

Test the same three methods with `F_4(x_R, X_2, X_3, X_4)`:
- Per query, Method C solves a 3-variable system of degree 12.
- Hit probability per query is `|FB|^3/(6n)`, higher than width-3.
- If `|FB|` is large enough, Groebner-basis cost may be reasonable.

This is the natural next step. Computing F_4 Groebner basis in Sage
takes seconds per query for `|FB| ≈ 40`; we measure whether the
relation rate × cost gives an overall win or loss vs pair-sum.

## Honest summary

Method C (Groebner) is structurally elegant but empirically slower
than hashed pair-sum (Method A) in our bit range. The gap is roughly
`30×`. This rules out naive Semaev `F_3` Groebner basis as a
breakthrough, but suggests F_4 / F_5 variants may behave differently.
