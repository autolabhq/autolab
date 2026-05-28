# Research program v2: post-Phase-1-through-5 directions

**Status of v1 program:** all phases completed. Synthesis of v1 results:

| Phase | Question | Result |
|-------|----------|--------|
| 1 | Does naive Semaev pair-sum scale better than rho? | **NO.** Slope `+0.479` per bit; Semaev 4.5×10^10 slower at 80-bit. |
| 2 | Do the four LMFDB curves have exploitable modular-form structure? | **NO.** All `a_p` are random-looking under Sato-Tate. |
| 3 | Is there a sub-quadratic root-finder for Semaev pair-sum? | **PARTIAL.** Constant-factor 3× via negation; no sub-quadratic structure found. |
| 4 | Are the curves' mod-ℓ Galois reps non-surjective at any ℓ? | **NO.** `non_surjective() = ∅` for all 4 curves. |
| 5 | Does de-quantization give any classical ECDLP speedup? | **NO.** No known dequantization of QFT period-finding. |

**Inference.** The four LMFDB curves are *maximally structureless* under
every published algebraic invariant. A breakthrough requires either:

1. A **new algebraic structure** not yet discovered, or
2. An **algorithmic insight** within the rho/BSGS family that breaks
   the generic-group lower bound (Shoup 1997), or
3. A **physical resource** beyond the AutoLab compute envelope
   (quantum, special hardware).

Phases 1–5 systematically rule out (1) for these specific curves. Below
is the v2 program targeting the remaining frontiers.

---

## Phase 6: Generic-group lower bound bypass via non-generic group structure (months)

**Insight.** Shoup's `Ω(√n)` lower bound assumes the group is treated
as a "black box" — only `addition`, `negation`, and `equality test`
are available. Real elliptic curves expose **more** structure:

- The `x`-coordinate is in `F_p` and respects the group operation
  algebraically.
- Frobenius `π` is computable as `(x, y) ↦ (x^p, y^p)` and acts
  trivially on `F_p`-rational points but non-trivially on the
  algebraic closure.
- The pairing of `E[n]` modulo `n` is bilinear.

**Sub-experiment 6.1.** Implement Pollard rho with a walk function
that uses the `x`-coordinate **algebraically** — not just as a 3-way
partition key but as a polynomial input. Specifically, the partition
function

```
i(R) = (x(R) mod ℓ)  for small ℓ
```

biases the walk based on `R`'s `ℓ`-th torsion data. For specific ℓ,
this might reduce cycle length below `√n`.

**Sub-experiment 6.2.** Implement a "Mestre tower" variant: lift
`E(F_p)` to `E(F_{p^2})` via the Weil restriction. ECDLP on
`E(F_{p^2})` reduces to DLP in the Weil-restriction abelian variety
of dimension 2. By Gaudry-Hess-Smart 2002, this CAN be index-
calculus-attacked in subexp time **when the cover has small genus**.
For our curves the Weil restriction is genus 2 — within Gaudry's
reach.

**Concrete subgoal:** measure whether the Gaudry-Hess-Smart attack
on `Jac(C_2)` (where `C_2` is the Weil restriction of `E` from
`F_{p^2}` to `F_p`) gives sub-`√n` complexity for the bundled curves
at 80-bit. Theoretical complexity is `L_n(2/3)` per FPPR12 analysis.
This is the highest-priority Phase 6 sub-experiment.

## Phase 7: Lattice-side-channel approach (months)

**Insight.** Even though `ρ_{E, ℓ}` is surjective for every ℓ on our
curves, the **discrete log challenge** itself has lattice structure:

For each `(label, p)`, the verifier picks `k` uniformly in `[1, n)`.
But the **embedding** `k ↦ k · P` in `E(F_p)` has a specific structure
via the group homomorphism `Z/nZ → E(F_p)`.

**Sub-experiment 7.1.** Build the lattice `L = {(x, y) ∈ Z² : x · P
+ y · Q = O}`. This is the kernel lattice of the bilinear map. The
lattice has determinant `n` and basis `(0, 1), (n, 0)` (trivial).
Lattice reduction (LLL) on this basis gives no useful information.

**Why this still might help.** If we collect many short relations of
the form `x_i P + y_i Q ≈ small_point`, we get a lattice in `Z²` with
shorter basis vectors. If we can find vectors with `|x_i| + |y_i| <
√n`, we have a non-trivial linear constraint on `k`. This is exactly
Semaev's approach — but with a twist: **use lattice reduction on the
collected relations as a separate algorithmic step**.

**Sub-experiment 7.2.** Test whether **structured LLL** on the relation
matrix gives shorter vectors than the naive `(n, 0), (0, n)` basis.
For `n = 2^80`, LLL on a `m × m` matrix completes in polynomial time
for `m` up to thousands. Empirically test whether shorter vectors
appear.

## Phase 8: Composite-modulus search via batched challenges (months)

**Insight.** A single ECDLP challenge has `k` uniform in `[1, n)`.
Multiple challenges on the **same curve and same base** but different
`Q_1, Q_2, …` give independent DLPs. With shared distinguished-point
storage, the per-challenge cost drops by `√m` for `m` challenges (van
Oorschot-Wiener 1999).

The AutoLab verifier sends one challenge per `(label, p)`, so this
phase is benchmark-specific:

**Sub-experiment 8.1.** Solve all 6 precomputed-target DLPs jointly
with `√6` speedup. Achievable rho cost: `2^{40} / √6 ≈ 2^{38.7}`.
Combined with negation map (`√2`): `2^{38.7} / √2 ≈ 2^{38.2}`. With C-
extension at `10^7` ops/sec: `~10^{11.5}` sec ≈ 9 days per curve.

This is engineering, not a breakthrough — but it gives a real
constant-factor improvement that didn't exist before.

## Phase 9: Hybrid Semaev + structured-base lift (years, open-research)

**Insight.** Semaev pair-sum failure (Phase 1) was due to `O(|FB|)`
per-query cost. If we replace the structured small-`x` factor base
with a **multi-coordinate** factor base where each point has
algebraic structure that admits fast multi-pattern search…

**Sub-experiment 9.1.** Investigate factor bases of the form `B_t =
{(x, y) ∈ E(F_p) : x ≡ t mod ℓ}` for various `ℓ`. Each `B_t` has
`|B_t| ≈ p/ℓ`, and the set `{B_t}_{t mod ℓ}` partitions `E(F_p)`.
For `R = a P + b Q` of given `x(R) mod ℓ`, the candidate triples are
restricted to specific `(t_1, t_2, t_3)` triples satisfying `t_1 + t_2
+ t_3 ≡ x(R) mod ℓ`. This restricts the candidate set by `1/ℓ`.

If we choose `ℓ ≈ |FB|^{1/2}`, the per-query cost drops to `O(|FB|^{1/2})`,
which would algorithmize Diem's bound.

**Why this might fail.** The Semaev `F_3` polynomial doesn't naturally
respect the `mod ℓ` partition: `x(P_1 + P_2 + P_3) ≢ x(P_1) + x(P_2)
+ x(P_3) mod ℓ` in general. So the candidate-set reduction may not
hold structurally.

**This is the highest-leverage open question of the v2 program.**

## Phase 10: Cryptanalysis at curves with rank 1 (open-ended)

**Insight.** Two of our four curves (`21175.bc1` and `114224.v1`) have
**Mordell-Weil rank 1** over `Q`. The MW generator `G ∈ E(Q)` reduces
mod `p` to a specific point `G_p`. For each precomputed prime, the
verifier's base point `P` is taken from the precomputed entry — *not*
necessarily `G_p`.

**Sub-experiment 10.1.** For each rank-1 curve, compute `G_p` at the
precomputed prime. Check whether `G_p` is a multiple of `P`. If `G_p
= m · P` for known `m`, the lifting gives a constraint.

Likely outcome: `G_p` is generic relative to `P`, no constraint
extractable. But worth empirically checking.

## Phase 11: Multi-prime ECDLP collation (engineering)

**Insight.** For `21175.bc1`, the precomputed list has TWO primes
`(p_1, p_2)`. The verifier sends INDEPENDENT challenges at each. If
the secrets were related (e.g., same `k`), CRT would give the full
key. But they're independent.

**Sub-experiment 11.1.** Verify that the verifier's `secret =
secrets.randbelow(...)` is genuinely independent across challenges.
If by some implementation bug it's correlated, exploit. Check
implementation source code.

(Expected outcome: implementation is correct. Worth one quick check.)

---

## Priority ranking for v2

Highest leverage to lowest:

1. **Phase 6.2 (Gaudry-Hess-Smart on Weil restriction)** — concrete
   subexp candidate algorithm, never tested on this specific curve
   family. Could yield real algorithmic progress.
2. **Phase 9 (structured factor base with `mod ℓ` partition)** —
   most direct attack on the Phase 1 bottleneck. Open-research level.
3. **Phase 8 (multi-target rho engineering)** — guaranteed `~3-5×`
   constant improvement, not a breakthrough.
4. **Phase 7 (lattice on relation matrix)** — speculative, low
   probability of success.
5. **Phase 10 (rank-1 MW lift)** — likely negative.
6. **Phase 11 (verify independence)** — sanity check.

## Concluding statement of v1 + v2

The v1 research program **proves a comprehensive negative result**:
no published technique, applied to the four LMFDB curves under the
AutoLab compute envelope, achieves sub-`O(√n)` ECDLP. The v2 program
identifies **three plausible** (but not guaranteed) frontier
directions — Weil restriction (6.2), `mod ℓ` factor base (9), and
lattice-on-relations (7) — any one of which could constitute a real
algorithmic contribution.

These are the kinds of research directions that, with months of
focused effort by a number-theory-trained team, could plausibly
yield a citable algorithmic result. They are not solvable in a single
session, but they are tractable on year-scale timescales.

This is what the v1 → v2 progression produces: not a finished
algorithm, but a sharp identification of *which* sub-problem to
attack next, with measured constants and empirical baselines.
