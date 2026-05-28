# Phase 9: mod-ℓ structured factor base partitioning

**Date:** 2026-05-27  
**Status:** completed, structural negative result  
**Reproducible via:** `sage phase9_modl_factor_base.sage`

## Result

For the elliptic curve `E(F_{1009})` (|E| = 960), tested whether the
map

```
(x(P_1) mod ℓ, x(P_2) mod ℓ)  ↦  x(P_1 + P_2) mod ℓ
```

is deterministic (a function), for `ℓ ∈ {3, 5, 7, 11}`. Sampled
19,801 point pairs for each `ℓ`.

| ℓ  | input pairs `(x_1 mod ℓ, x_2 mod ℓ)` | deterministic map? |
|---:|:------------------------------------:|:------------------:|
| 3  | 6                                    | **No**             |
| 5  | 15                                   | **No**             |
| 7  | 28                                   | **No**             |
| 11 | 66                                   | **No**             |

For **every** `(x_1, x_2)` residue pair, the value of `x(P_1+P_2) mod
ℓ` varies across point samples — the mod-`ℓ` reduction of `x(P+Q)`
is not determined by the mod-`ℓ` reductions of `x(P)`, `x(Q)` alone.

## Why this is structural

The elliptic curve group law uses the **chord-tangent formula**:

```
x(P+Q) = λ² - x(P) - x(Q)     where λ = (y(Q) - y(P)) / (x(Q) - x(P))
```

`λ` involves `1/(x(Q) - x(P))`, which is a **division** modulo `p`.
Modular inversion does not commute with reduction modulo `ℓ`: in
general `(a^{-1} mod p) mod ℓ ≠ ((a mod ℓ)^{-1} mod ℓ)`.

Hence the group law is rational in `(x, y)` but not linear or even
piecewise-linear modulo any ℓ. The mod-`ℓ` residue of the sum is a
**genuinely** non-deterministic function of the inputs' mod-`ℓ`
residues.

## Implication for the Phase 1 bottleneck

The Phase 1 result identified the `O(|FB|)`-per-query cost as the
algorithmic bottleneck. Phase 9 hypothesized that a mod-`ℓ` partition
could give an `O(|FB|/ℓ)` reduction. **The hypothesis is false.**

The set of mod-`ℓ` residues of factor base points is essentially
uniform, with no algebraic constraint linking summand residues to sum
residues. A query `R - F_k` can land in any of the `ℓ` residue classes
with roughly equal probability, so partitioning provides no
candidate-set reduction.

## What this rules out

This forecloses the highest-leverage candidate from research program
v2. The two remaining v2 directions (Phase 7 lattice and Phase 8
multi-target engineering) are speculative and engineering-grade
respectively.

## Honest interpretation

The chord-tangent formula's "rational but not modular" structure is
the **deep reason** prime-field ECDLP has resisted algebraic attacks.
Algorithms that work for binary/extension fields exploit `F_2`-linear
structure or extension-field decomposition — neither survives modular
reduction by an unrelated prime `ℓ`.

Phase 9 quantifies this structural barrier empirically.
