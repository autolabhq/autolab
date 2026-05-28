# Research Program v6: Beyond the Yokoyama lower bound

**Date:** 2026-05-27
**Status:** planned for next round
**Motivation:** All "naive" cryptanalytic directions have been empirically
and theoretically closed (see `final_synthesis.md`). The genuinely open
problems remain. This program targets them directly.

## Scope

Four genuinely-open sub-problems, each with concrete experiments:

### Phase 18: Non-naive index calculus

**Hypothesis:** The Yokoyama et al. 2020 lower bound applies only to
*naive* Buchberger/F4/F5 Groebner basis algorithms. A non-Buchberger
algebraic technique might escape this bound.

**Concrete experiments:**

**Phase 18.1: Wu's characteristic-set method on the Semaev system**

Wu's method (Ritt-Wu) decomposes a polynomial variety into a union
of irreducible components, each described by a *characteristic set*.
The complexity profile is different from Buchberger.

Test:
- Implement Wu's method for `I = ⟨F_3(xR, X_2, X_3), ∏(X_2 - x_i), ∏(X_3 - x_i)⟩`
- Measure per-query cost at 13-22 bits
- Compare to F4/F5 baseline

If Wu's method shows different scaling, this is a non-naive technique
that escapes the Yokoyama bound.

**Phase 18.2: Quasi-subfield in artificially-embedded prime field**

Huang et al. 2020 constructs factor bases from roots of
`X^{q^{n'}} - λ(X)` over `F_{q^n}`. For our prime field `F_p`,
artificially embed into `F_{p^2}` (degree-2 extension), then apply
the construction.

Test:
- Base-change `E/F_p` to `E/F_{p^2}`; `#E(F_{p^2}) = (p+1-t)(p+1+t) - 2p = ...`
- Construct quasi-subfield polynomial of small degree splitting in `F_{p^2}`
- Solve point decomposition problem (PDP) in `E(F_{p^2})` via the FB
- Use trace map `Tr: E(F_{p^2}) → E(F_p)` to descend relations

If this works, we'd have an L(2/3) attack on prime-field ECDLP via
artificial extension. Major if successful.

### Phase 19: Lattice / p-adic attacks

**Hypothesis:** A specifically-designed lattice or p-adic structure
encodes ECDLP in a way that LLL/BKZ can recover.

**Concrete experiments:**

**Phase 19.1: LLL on canonical lift coordinates**

For ordinary `E/F_p`, the canonical lift `Ẽ/Z_p` exists. Points
lift uniquely. The p-adic valuations of lifted point coordinates
encode information about the discrete log.

Test:
- Lift `P, Q ∈ E(F_p)` to `Ẽ(Z_p)` with precision 50-100
- Build lattice from lifted coordinates + p-adic constraints
- LLL-reduce; check if short vector reveals `k`

This is speculative — no known result, but worth testing.

**Phase 19.2: p-adic L-function evaluation**

The L-function `L(E, s)` has p-adic analogs `L_p(E, s)`. By the
Birch-Swinnerton-Dyer conjecture (conditionally), `L(E, 1) ≠ 0` for
rank-0 curves and the *p-adic regulator* relates to torsion. For
our LMFDB curves, this gives a specific p-adic number we can compute.

Test:
- Compute `L_p(E, 1)` mod `p^k` for `k = 5, 10, 20`
- Check if a known relation between `L_p(E, 1)` and ECDLP exists
- (Likely speculative; significant if it works)

### Phase 20: Isogeny-graph navigation (ordinary curves over F_p)

**Hypothesis:** While isogenies preserve order, they DO change other
invariants (j-invariant, endomorphism conductor). Some isogenous
curve might have algebraic structure that makes ECDLP easier.

**Concrete experiments:**

**Phase 20.1: ℓ-isogeny walk for small ℓ**

Walk the 2-, 3-, 5-isogeny graph from each LMFDB benchmark target.
At each neighbor, check:
- Trace (preserved — sanity check)
- j-invariant — does it ever hit `{0, 1728, supersingular j's}`?
- Endomorphism ring (computed via Cornacchia + 2-adic Frobenius)
- Mordell-Weil rank over Q of the lifted curve

If any neighbor has a special property, exploit it.

**Phase 20.2: Volcano crater exploration**

The ℓ-isogeny graph (volcano) has a crater of curves with isomorphic
End ring. Walk the crater for one curve; check for any "thinner"
points where Frobenius eigenvalues have special form.

Concretely: compute Hilbert class polynomial at discriminant `D`
modulo `p`; its roots are j-invariants of curves in the crater.

### Phase 21: C-extension constant-factor engineering

**Hypothesis:** A C implementation of pair-sum is ~10× faster than
Python/Sage. Combined with negation + multi-target + l=16 partition,
this brings the effective speedup to ~50×, possibly closing the gap
on the 80-bit benchmark.

**Concrete deliverable:**

**Phase 21.1: C-implemented Pollard rho**

Write a parallel C program implementing:
- Multi-target Pollard rho (4 targets simultaneously)
- Negation map (folds -P to P)
- Partition l=16 (Brent's improvement)
- Multi-threaded with shared distinguished-point table
- Linked to Python via cffi/ctypes for orchestration

Target: 2^37 group ops/CPU-hour. With 2 CPUs × 4 hours = 2^39 ops.
Compare to 80-bit instance requiring ~2^40 ops.

**Phase 21.2: GMP-accelerated point arithmetic**

Use GMP (libgmp) for big-integer modular arithmetic. Each point
operation is ~3 multiplications + 1 inverse. With Montgomery
arithmetic and Barrett reduction, ~50ns per operation. 2^40 ops
in 2^40 × 50ns = 18 hours.

Within 2-CPU × 4-hour budget: 2 × 4 × 3600 / 50e-9 = 5.76e11 ops =
2^39 ops. **One 80-bit target becomes borderline-solvable in 4 hours
on 2 CPUs**.

## Priority order

For the next research round, in order of likely impact / effort:

1. **Phase 21 (C-extension)**: highest impact, lowest research risk.
   ~10× concrete speedup, brings 80-bit ECDLP into-reach territory.
2. **Phase 18.2 (quasi-subfield artificial embedding)**: highest
   theoretical interest. If successful, L(2/3) algorithmic attack.
3. **Phase 20.1 (small-ℓ isogeny walk)**: concrete, computable.
   Expected negative but worth confirming.
4. **Phase 19.1 (LLL on canonical lift)**: speculative but novel.
5. **Phase 18.1 (Wu's method)**: significant implementation effort.
6. **Phase 19.2 (p-adic L-function)**: most speculative.

## Success criteria

A "breakthrough" in this program would mean:
- Phase 21: 80-bit ECDLP solvable in benchmark budget on one target
- Phase 18.2: empirical L(2/3) scaling beating O(√n)
- Phase 20: isogenous curve with anomalous/MOV structure
- Phase 19: lattice recovery of `k` from p-adic constraints

A "negative" result would extend the resistance map further. Either
outcome advances the empirical understanding of prime-field ECDLP.

## What this round will NOT pursue (and why)

- **Quantum attacks**: Shor would solve ECDLP but is hardware-bound.
- **Specialized hardware**: ASICs / FPGAs out of session scope.
- **New mathematical frameworks (e.g., tropical, motivic)**: too
  speculative without prior literature anchor.

## Next concrete step

Start with **Phase 21.1 (C-extension Pollard rho)**: highest concrete
impact, well-defined deliverable, immediate testability against the
AutoLab benchmark.
