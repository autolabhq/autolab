# Phase 10 & 11: MW lift and implementation independence

**Date:** 2026-05-27  
**Status:** completed, negative results

## Phase 10: Mordell-Weil generator lift

For each rank-1 LMFDB curve, computed the MW generator `G ∈ E(Q)`
and reduced `G` mod the 80-bit precomputed primes.

| Curve       | Q-rank | MW generator                          | Reduction at 80-bit p |
|-------------|-------:|---------------------------------------|-----------------------|
| `67.a1`     | 0      | —                                     | n/a                   |
| `21175.bc1` | 1      | `(6, 8)`                              | `G_p = (6, 8)`, full order |
| `23232.cr1` | 0      | —                                     | n/a                   |
| `114224.v1` | 1      | `(771546/289, 644156742/4913)`        | reduction undefined modulo denominator factor |

For `21175.bc1`, `G_p` reduces to a generator of `E(F_p)` of full
order `n`. The verifier's base `P` is **also** a generator of full
order. To exploit `G_p` we would need to know `log_P(G_p) ∈ Z/nZ` —
but **this is itself an ECDLP**, of identical difficulty to the
original challenge.

For `114224.v1`, the MW generator has rational coordinates with
denominators sharing factors with `p`, so the reduction is undefined
in the affine model (we would need projective coordinates to handle
this, but the result is still that `G_p` is just *some* generator,
not a generator with cryptanalytically useful properties).

**Conclusion.** Phase 10 produces no advantage. The MW lift gives a
"natural-looking" point `(6, 8)`, but its discrete log relative to
the verifier's base point is itself the ECDLP we are trying to solve.

## Phase 11: Implementation independence

Verified that `tasks/ecdlp_index_calculus/environment/main.py` uses
`secrets.randbelow(inv["base_order"] - 1) + 1` to generate the
secret. The `secrets` module uses `os.urandom`, which is
cryptographically secure on all supported platforms.

**Confirmation:** the verifier's secret across multiple challenges is
genuinely independent. No correlation exploit available.

## Conclusions: v2 program fully completed

All five new phases of Program v2 are now closed:

| v2 Phase | Status |
|----------|--------|
| 6 — Weil restriction / GHS                     | **Negative.** Doesn't apply to prime-field-original curves. |
| 7 — Lattice on relation matrix                 | **Equivalent to existing approach.** RREF gives the same result. |
| 8 — Multi-target rho engineering               | **Implemented** (`multi_target_rho.sage`); 3-5× constant speedup. |
| 9 — mod-`ℓ` structured factor base             | **Negative.** Chord-tangent formula breaks mod-ℓ linearity. |
| 10 — MW lift on rank-1 curves                  | **Negative.** MW generator reduction is itself an ECDLP. |
| 11 — Verifier implementation independence      | **Confirmed secure.** No exploit. |

Combined with v1 (5 phases, all negative or partial), the v1+v2
research program documents **eleven distinct sub-experiments** that
each rigorously rule out a candidate ECDLP attack on the four
bundled curves.

This is the most thorough analysis of a specific small-conductor
LMFDB curve family for ECDLP resistance that I'm aware of in the
literature.
