# Phase 15: F_p-isogenous curve enumeration

**Date:** 2026-05-27  
**Status:** completed, negative result

## Result

Enumerated isogenies of degree 2, 3, 5, 7 from each of the four
precomputed-target curves at their first precomputed prime:

| Curve | ℓ=2 | ℓ=3 | ℓ=5 | ℓ=7 |
|-------|----:|----:|----:|----:|
| `67.a1` | 0 | 0 | 0 | 0 |
| `21175.bc1` | 0 | 0 | 0 | 1 |
| `23232.cr1` | 0 | 1 | 0 | 0 |
| `114224.v1` | 0 | 1 | 1 | 0 |

Where an isogeny exists, the codomain curve has a *different*
j-invariant but the *same* group order `n`.

## Why this doesn't help

By Tate's theorem (1966), isogenous abelian varieties over a finite
field have the same characteristic polynomial of Frobenius, hence the
same number of `F_p`-points. So:

```
#E'(F_p)  =  #E(F_p)  =  n     for any F_p-isogenous E'
```

The discrete-log problem on `E'` has identical difficulty to the
original. The isogeny `φ: E → E'` can be applied to transport
`(P, Q)` to `(φ(P), φ(Q))` without changing the discrete log: `k ·
φ(P) = φ(kP) = φ(Q)`, so `log_{φ(P)}(φ(Q)) = log_P(Q) = k`.

The volcano structure (Kohel 1996) places isogenous curves at
different "altitudes" in the `ℓ`-isogeny graph, with endomorphism
rings of decreasing conductor as we descend. But all conductors give
orders in the same imaginary quadratic field `Q(π_p)` of class number
`~10^{11}` (Phase 14). **No isogenous curve has a smaller class
number**, so no GLV exploit emerges.

## Conclusion

Phase 15 closes another candidate attack vector. The isogeny graph
is non-trivial for 3 of the 4 curves but provides no cryptanalytic
leverage.
