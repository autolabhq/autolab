# Multiplicative-subgroup factor base test

**Date:** 2026-05-27
**Status:** completed, negative result

## Hypothesis

If the factor-base x-coordinates lie in a multiplicative subgroup
`H ⊂ F_p^*` of order `m`, then the constraint polynomial is
`X^m - 1` — only 2 monomials, vs `m + 1` for a generic
`∏(X - x_i)`. This should dramatically simplify the Groebner basis
of the ideal

```
I = ⟨F_3(x_R, X_2, X_3), X_2^m - 1, X_3^m - 1⟩
```

## Setup

Find `p` with `(p-1) | m` so that an order-`m` subgroup exists. For
each `(bits, m)`:
- `bits = 13, m = 16, p = 8273, |FB ∩ H| = 10`
- `bits = 16, m = 32, p = 67489, |FB ∩ H| = 16`
- `bits = 19, m = 64, p = 524353, |FB ∩ H| = 31`

## Result

| bits | m | \|FB\| | Avg GB+variety / query |
|----:|--:|------:|--------:|
| 13 | 16 | 10 | **12.5ms** |
| 16 | 32 | 16 | **17.3ms** |
| 19 | 64 | 31 | **74.7ms** |

Compare to F_3 Groebner with *generic* factor base (Phase 16):

| bits | \|FB\| | Avg GB / query |
|----:|------:|---------------:|
| 13 | 20 | 4.7ms |
| 16 | 40 | 18.1ms |
| 19 | 50 | 30.8ms |

**The sparse `X^m - 1` constraint does NOT speed up the F_3 Groebner
solve.** At bits=19, it is actually 2.4× slower than the dense
generic constraint.

## Why

Sage's Groebner-basis algorithm (F4/F5) computes a normal-form
reduction modulo the ideal. The algorithm depends on the
*Hilbert regularity* of the system, not the monomial sparsity
of generators. Both `X^m - 1` and `∏(X - x_i)` have the same
degree, and the Frobenius automorphism over `F_p` cannot exploit
the `μ_m`-structure without `p` being in a CM-like configuration.

The variety enumeration step iterates all common roots of the
ideal. With `X^m - 1`, this includes *all* `m`-th roots of unity
(of which only a fraction are factor-base x-values, requiring
post-filtering). With the dense constraint, the variety
contains exactly the factor-base solutions. Net cost is similar.

## What this rules out

Multiplicative-subgroup factor bases (the most algebraically
natural structured FB on prime fields) do not provide the
sparsity reduction needed to make the Semaev F_3/F_4 Groebner
attack practical.

## What still might work

- **Additive-shift FB** (arithmetic progression): constraint
  polynomial = falling factorial; tested next.
- **Two-variable subgroup product** (FB = H_1 × H_2 in some
  isogeny structure): no obvious construction on random curves.
- **Specialized GB algorithms** beyond F4/F5 that exploit
  monomial sparsity directly: e.g. FGLM with sparse-matrix
  arithmetic. Theoretical, not in Sage.

## Conclusion

The "subgroup factor base sparsifies the constraint" hypothesis
is empirically false in Sage's F4/F5 implementation. The
sparsity of the constraint polynomial does not propagate to
the F_n polynomial, so the joint Groebner basis stays expensive.

This adds to the body of negative results: pair-sum scaling,
F_3 Groebner, F_4 Groebner, mod-ℓ FB partitioning, and now
multiplicative-subgroup FB sparsification are all empirically
ruled out as breakthrough algorithmic improvements on
prime-field ECDLP in our compute envelope.
