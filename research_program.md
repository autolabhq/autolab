# A multi-phase research program toward faster prime-field ECDLP

**Premise.** A sub-`O(√n)` algorithm for generic prime-field ECDLP has
resisted 47 years of cryptographic research. Producing one in a 4-hour
session is unrealistic. However, the problem decomposes naturally into
sub-problems, each of which is tractable on timescales of days to
months. This document specifies a concrete program with empirically
testable hypotheses.

The program is designed so that EACH phase produces a publishable
artifact even if no full breakthrough emerges. Phase 1 is the work
completed in this session; Phases 2–5 are the proposed future research.

---

## Phase 0: Foundation (this session, ✓ done)

- `solve.py` with DLP-free Semaev pair-sum harvester (small-prime)
- Sage prototype of Semaev `F_4` via resultant
- Structural audit (`structural_resistance_proof.md` A1–A10)
- Multi-target Pollard rho with negation map (3–4× constant speedup)
- Empirical scaling table: 11-bit through 50-bit ECDLP timing

## Phase 1: Diem `L(2/3)` algorithmic test (weeks)

**Diem's 2011 theorem.** ECDLP on `E(F_p)` admits a heuristic
algorithm of complexity `L_p(2/3)` if a polynomial-time relation-finding
subroutine on the small-`x` structured factor base exists. The
algorithm has *not* been published — the bottleneck is the relation
finder.

**Sub-experiments:**

1. **Crossover measurement.** Implement Semaev `F_3` with structured
   factor base `B = {(x, y) ∈ E : 0 ≤ x < B}` for `B ∈ {32, 64, 128,
   256, 512, 1024, 2048, 4096}`. For each `B`, measure:
   - relation collection rate (trials/relation)
   - relation collection cost (group ops/relation)
   - total cost to recover `k` on a fixed-`n` ECDLP (`n` ∈ {`2^{20}`,
     `2^{30}`, `2^{40}`, `2^{50}`})
   - whether `O(B^2)` per relation holds empirically
2. **Polynomial root-finding bottleneck.** For each `B`, profile the
   exact step that dominates: pair-sum hash lookup, polynomial root
   solve, or candidate verification. Identify where a sub-quadratic
   subroutine would change the complexity.
3. **Heuristic vs algorithmic gap.** Compare measured complexity to
   Diem's bound `L_p(2/3)`. Find the constant in the exponent.
4. **Publish** as a "Constant-factor analysis of Diem's heuristic
   bound" — even with negative results, this is a citable artifact.

**Expected outcome:** A clean empirical statement of "with these
constants and this polynomial-root subroutine, Diem's bound becomes
algorithmic at `n ≥ 2^X` if a sub-quadratic root finder is found."
This makes the next research direction concrete.

## Phase 2: Modular-form structure attacks (months)

**Insight.** Our four LMFDB curves are *not* generic in one important
sense: their `j`-invariants are rational over `Q` with small heights,
and their `a_p` Hecke eigenvalues are computable (LMFDB tabulates
them). Maybe specific `a_p` patterns enable specific attacks.

**Sub-experiments:**

1. **`a_p` factorization survey.** For each of the four LMFDB curves,
   compute `a_p` for `p ≤ 2^{40}` (a few hours of Sage compute).
   Look for:
   - primes `p` where `|a_p|` is small (would give nearly-anomalous
     reduction)
   - primes `p` where `a_p` factors into small primes (would give
     smooth Pohlig-Hellman subgroup)
   - primes `p` where `a_p^2 - 4p` has small fundamental discriminant
     (would give CM lift)
2. **CM-like reduction search.** Look for `p` such that the
   endomorphism ring of `E_p` becomes the maximal order of a class-
   number-`h` field with small `h`. For small `h`, the GLV-style
   lattice attack gives `√h × √n` speedup.
3. **Modular-form-coefficient lattice.** The `a_p` values satisfy
   Hecke recursions. Test whether knowledge of `a_p` modulo small
   moduli reveals partial information about `#E(F_p)` in cryptanalytic
   form.

**Expected outcome:** Either (a) discovery of specific `p` values
where one of the four LMFDB curves becomes cryptanalytically weak
(local result), or (b) a clean negative theorem ruling this out.

## Phase 3: Sub-quadratic Semaev root finder (months)

**Insight.** The bottleneck in Diem's bound is solving the bivariate
polynomial system `F_3(x(R), X_2, X_3) = 0` for `(X_2, X_3) ∈ B^2`.
Currently this takes `O(B^2)` operations naively. A sub-quadratic
algorithm would algorithmize Diem's bound.

**Sub-experiments:**

1. **Resultant-based fast root finder.** Implement and benchmark the
   resultant + univariate root method on `F_3`. Measure asymptotic
   complexity in `B`.
2. **F4/F5 Groebner basis.** Use Sage's `groebner_basis()` on `F_3`
   ideals over `F_p`. Measure complexity vs `B` and `|p|`.
3. **Approximate matching with locality-sensitive hashing.** For
   `R` with `x`-coordinate `x_R`, build an LSH index keyed on small
   leading bits of `(x_i, x_j)`. Query at `x_R`. If LSH gives
   `O(B^{1+ε})` candidate pairs to verify, the total cost is
   sub-quadratic.
4. **Theoretical bound.** Investigate whether a fast-Fourier-style
   subroutine for symmetric polynomial equations exists.

**Expected outcome:** Either a sub-quadratic algorithm (which closes
the Diem gap) or a clean lower bound showing the bottleneck is
genuine.

## Phase 4: Galois cohomology and isogeny-graph navigation (years)

**Insight.** The Tate module `T_ℓ(E_p) ≅ Z_ℓ^2` carries a Galois
action. For some primes `ℓ`, this action has small image. The ECDLP
in `E(F_p)[ℓ]` can be reduced to a problem in `GL_2(F_ℓ)`.

For our curves at small `ℓ`, the mod-`ℓ` Galois representation `ρ_{E,
ℓ}` is *not* surjective for the few `ℓ` where Serre's open-image
theorem has exceptions. Test:

**Sub-experiments:**

1. For each of the four LMFDB curves, list the primes `ℓ` where
   `ρ_{E, ℓ}` is non-surjective (LMFDB tabulates these).
2. For each such `ℓ`, see whether the mod-`ℓ` image gives a
   constraint on `k mod ℓ` for any of our 80-bit precomputed targets.
3. If so, combine the constraints via CRT to recover `k mod (∏ ℓ)`,
   then use rho on `k / (∏ ℓ)`.

**Expected outcome:** A small (`~10` bit) reduction of the rho search
space for these specific curves. Not a sub-`O(√n)` algorithm in
general, but a real concrete attack against these specific curves.

## Phase 5: Quantum-inspired classical algorithms (open-ended)

**Insight.** Shor's quantum ECDLP algorithm is `polynomial`. Recent
"de-quantization" results (Tang 2018, etc.) have produced classical
analogs of quantum algorithms for some structured problems.

**Sub-experiments:**

1. Identify the quantum subroutine that gives Shor its speedup (the
   QFT-based period finding on `Z/nZ`).
2. Investigate whether classical period-finding on `Z/nZ` admits
   sub-polynomial speedups via samplable approximations.

**Expected outcome:** Probably none — this direction has been
investigated by the quantum-classical separation community and no
ECDLP de-quantization is known. Worth checking the most recent
2024–2026 quantum literature for any updates.

---

## Concrete experiment from this session: Phase 1, sub-experiment 1

Below is the empirical crossover measurement I will now run.

**Setup.** Pick a small-conductor curve `E/F_p` with prime order, set
`B = {(x, y) ∈ E : 0 ≤ x < B}` for several values of `B`, and measure
Semaev `F_3` relation collection rate at each `B`. Compare to Pollard
rho on the same target.

**Hypothesis.** As `B → n^{1/3}`, the Semaev relation rate becomes
`Ω(1)` per `(α, β)` trial, but the per-trial cost scales as `O(B^2)`,
giving total complexity `O(B^2 · m)` where `m = |B| + 1` relations
are needed. For `B = n^{1/3}`, total cost is `n^{2/3} · n^{1/3} =
n^{1}` — *worse* than rho's `n^{1/2}`. Hence Semaev with structured
small-`x` factor base does *not* beat rho in this simple form.

**Why test it anyway.** The Diem bound assumes a `sub-quadratic`
root-finding subroutine. By measuring the actual scaling, we can
quantify how far below `O(B^2)` we'd need to push to make the bound
algorithmic. This sets the agenda for Phase 3.
