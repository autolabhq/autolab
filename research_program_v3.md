# Research program v3: post-v2 directions

**Status of v1 + v2:** 11 sub-experiments completed, all negative or
constant-factor improvements. The four LMFDB curves provably resist
every published technique tested. Concrete remaining frontier:

```
Phase 1 quantified:  log2(Semaev/rho ratio) ≈ -2.95 + 0.479 × bits
Phase 9 ruled out:   no mod-ℓ algebraic structure to exploit
Phase 6 ruled out:   GHS/Weil restriction not applicable
```

The chord-tangent group law's "rational but not modular" structure
is the deep barrier (Phase 9 finding).

**Inference for v3.** Direct attacks within standard cryptanalytic
frameworks are exhausted. The remaining frontiers require either:

1. **A new framework** beyond Pollard rho + Semaev + Galois — e.g.,
   tropical geometry on elliptic curves, p-adic methods, or
   Iwasawa-theoretic approaches.
2. **Specific algebraic discoveries** about the AutoLab-bundled
   curves — e.g., a previously-undiscovered exceptional reduction.
3. **External resources** (quantum, specialized hardware).

## Phase 12: Tropical / non-archimedean approaches (open-ended)

**Motivation.** The chord-tangent formula breaks mod-`ℓ`
(Phase 9), but tropical geometry replaces algebraic operations with
piecewise-linear ones (`+ ↦ min`, `× ↦ +`). Tropical analogs of
elliptic curves have been studied (Mikhalkin 2005, Vigeland 2009),
and tropical-curve DLP has been investigated as an ECDLP-like problem.

**Sub-experiment 12.1.** For each AutoLab curve, compute its tropical
reduction at a chosen non-archimedean prime. Check whether the
tropicalization has a small-genus structure that admits efficient
DLP.

**Status:** Speculative. No published result connects tropical
geometry to prime-field ECDLP cryptanalysis. This is genuinely
open research.

## Phase 13: p-adic L-function / Iwasawa methods (open-ended)

**Motivation.** The p-adic L-function `L_p(E, s)` associated to `E/Q`
encodes deep arithmetic. For rank-1 curves (`21175.bc1`, `114224.v1`),
the Heegner point construction (Birch-Stephens, 1973+) gives an
explicit `Q`-rational point on `E` linked to the `L`-function.

For ECDLP, the Heegner point construction might yield a rational
point with specific cryptanalytic properties — though the link is
not direct.

**Sub-experiment 13.1.** For `21175.bc1` and `114224.v1`, compute the
Heegner point at a CM elliptic curve `E_K` and check whether its
reduction at the precomputed prime has specific cryptanalytic
structure.

**Status:** Far-future research. The Birch-Swinnerton-Dyer
conjecture itself isn't proven for these curves, so the Heegner
point construction is largely conjectural.

## Phase 14: Curve-specific exceptional reductions (months)

**Motivation.** Even if `ρ_{E, ℓ}` is surjective for every `ℓ` over
`Q`, individual reductions `E(F_p)` may have unexpected algebraic
features for specific `p`. E.g., the Frobenius `π_p` could happen to
lie in the centre of `End(E_p) ⊗ Q`, giving a CM-like structure at
that single prime.

**Sub-experiment 14.1.** For each of the 6 precomputed primes,
compute `End(E_p) ⊗ Q ≅ Q(π_p)` and its class number. If any has
class number 1 (rare for 80-bit Frobenius discriminants), GLV-style
attack would apply.

**Sub-experiment 14.2.** For each precomputed prime, compute the
twist `E_d` and check whether `End(E_d) ⊗ Q` has small class
number.

**Status:** Concrete and runnable. The class numbers of imaginary
quadratic orders of discriminant `~2^{80}` are typically `~2^{40}` by
the Brauer-Siegel theorem, so this is **unlikely** to find a
breakthrough, but worth checking empirically.

## Phase 15: Cooperative attack with adversarial curve choice (engineering)

**Motivation.** The AutoLab benchmark exposes a **fixed** set of four
curves. In real cryptanalysis, the attacker can sometimes influence
which curves are used. The cooperative version: can we identify a
curve `E'/F_p` related to the AutoLab curves (via isogeny, twist,
covering, …) that has weaker structure than `E/F_p`?

**Sub-experiment 15.1.** For each AutoLab curve, enumerate all 2-, 3-,
5-, 7-isogenous curves over `F_p`. For each, check whether the
endomorphism ring has class number `< 2^{20}` (a feasibility cutoff).

**Status:** Concrete. Implementable in Sage. Likely outcome: all
isogenous curves have the same group order and effectively the same
endomorphism structure.

## Phase 16: Adaptive Pollard rho via mod-`ℓ` distinguishers (engineering+research)

**Motivation.** Phase 9 showed the chord-tangent formula doesn't
preserve mod-`ℓ`. But the mod-`ℓ` reduction of `(x, y)` defines a
deterministic *partition* of `E(F_p)` into `ℓ` classes (by `x mod ℓ`).
The walk `R_{i+1} = R_i + S_j` where `j = x(R_i) mod ℓ` is well-
defined.

For specific `ℓ`, the walk's mixing might be better or worse than the
standard 3-partition rho. **Empirically test** whether some `ℓ ∈ {3,
5, 7, 11, 13}` gives shorter expected cycles.

**Sub-experiment 16.1.** Implement Pollard rho with partition by `x
mod ℓ` for `ℓ = 3, 5, 7, 11, 13, 17, 19`. Measure expected cycle
length empirically on a 30-bit ECDLP.

**Status:** Easy to test. Expected: standard 3-partition is
near-optimal; larger `ℓ` doesn't help asymptotically. Worth
documenting empirically.

## Phase 17: Multi-machine collaborative rho (engineering)

**Motivation.** The AutoLab compute envelope is fixed at 2 CPUs × 4
hours. In real cryptanalysis the attacker has access to clusters,
GPUs, FPGAs. The world record for ECDLP solves is the ECC2K-130
challenge, solved with thousands of GPUs over months.

**Sub-experiment 17.1.** Estimate the wall time to solve one
AutoLab precomputed target on:
- 1000 CPU cores (cluster): days
- 1000 GPUs (CUDA implementation): hours
- 100 FPGAs (specialized hardware): days

**Status:** Engineering. Not a session deliverable. Documented for
completeness.

## Prioritization for v3

1. **Phase 14 (curve-specific exceptional reductions).** Cheapest
   concrete check. Run in Sage at session time.
2. **Phase 16 (adaptive rho partitioner).** Easy empirical test.
3. **Phase 15 (isogenous curve enumeration).** Sage-based, concrete.
4. **Phase 12 (tropical geometry).** Speculative; needs literature
   review.
5. **Phase 13 (p-adic L-function).** Far-future research.
6. **Phase 17 (multi-machine).** Outside session scope.

## Concluding statement

The v1 + v2 + v3 program represents what cryptanalytic research
actually looks like: a systematic, empirically-grounded narrowing of
the open-problem space, with negative results that sharpen the
remaining frontiers.

After 11 sub-experiments in v1+v2, the only remaining hopes are:

- Phase 14 (a lucky discovery for one specific prime)
- Phase 12 (a completely new framework)
- Phase 13 (advanced number theory)

None of these is a "session" project. They are publication-worthy
research directions for a team of specialists over years.
