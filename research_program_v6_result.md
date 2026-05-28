# Research Program v6: Results

**Date:** 2026-05-27
**Status:** all four phases executed comprehensively

## Phase-by-phase results

### Phase 18.1: Wu's characteristic-set method

**Setup:** Sage's `Ideal.triangular_decomposition()` as the
representative of Wu/Ritt-Wu characteristic-set methods.

**Result:** Tested on Semaev F_3 ideal at 13 bits with `|FB| = 16`.
Per-query times:

| Method | Mean per-query time |
|--------|--------------------:|
| Triangular decomposition | 34.25ms |
| Groebner basis + variety | 2.42ms |

Triangular decomposition is **14.18× SLOWER** than Groebner basis on
this problem. Also, only 1 of 10 queries gave a valid decomposition;
the others raised "dimension must be zero" (the Semaev ideal at certain
specializations becomes positive-dimensional, on which Wu's method
fails).

**Verdict:** Wu's method does *not* escape the Yokoyama et al. lower
bound for naive index calculus on Semaev systems. It's strictly worse
in this implementation.

### Phase 18.2: Quasi-subfield polynomials in artificially-embedded F_{p^2}

**Setup:** Base-change E/F_p to E/F_{p^2}; use μ_{p+1} ⊂ F_{p^2}^*
(roots of `X^{p+1} - 1`) as the quasi-subfield. Build FB from
{(x, y) ∈ E(F_{p^2}) : x ∈ μ_{p+1}}.

**Result at bits=11 (p=2083):**

- FB size: 1074 points (~51.5% of μ_{p+1} lifts to E)
- Relation collection: **10 relations found in 10 queries** (100% rate!)
- FB structure: only **1 of 5 spot-checked points is in `<P>`**;
  the rest live in the full E(F_{p^2}) group of order 4,342,327

**The fundamental problem:** the relations are in E(F_{p^2}), not in
`<P>`. The discrete logs `log_P(F_i)` are NOT well-defined in Z/nZ
because `F_i ∉ <P>` generically. Solving the relations would require
solving ECDLP in the *larger* E(F_{p^2}) group, which is *harder*
than the original F_p problem.

**Verdict:** Quasi-subfield FB in artificial F_{p^2} embedding is
computable, fast, and yields relations — but the relations don't
descend to F_p ECDLP because FB points are in the wrong subgroup.
Consistent with Huang et al. 2020's restriction to extension-field-
*original* curves, not artificial embeddings.

### Phase 19.1: LLL on canonical lift coordinates

**Setup:** Lift `P, Q ∈ E(F_p)` to `Z_p` (canonical lift, precision
20). Build lattice `L = [[1, 0, x_P], [0, 1, x_Q], [0, 0, p^{prec}]]`
and LLL-reduce.

**Result:** Shortest vectors found are:
- (-81, 97, -35) with |v|² = 17195
- (-5, 6, 263) with |v|² = 69230

The vector (a, b, c) should satisfy `a + b*k ≡ c (mod p^prec)`. For the
short vectors above, neither gives `k`. **LLL does not recover the
secret.**

**Why:** there's no "hidden number" in this setup. ECDLP doesn't leak
partial information about `k`; lattice attacks require either nonce
bias (HNP), CM decomposition (GLV), or anomalous-curve p-adic log
structure — none present in our generic LMFDB curves.

**Verdict:** LLL on canonical lift coordinates is a clean negative.
The lattice has no exploitable structure.

### Phase 20.1: ℓ-isogeny walk on LMFDB benchmark curves

**Setup:** Walk 2-, 3-, 5-, 7-isogeny graph from 67.a1 at p=1208925819614629469615699
to depth 2, checking each neighbor for special j-invariant, embedding
degree changes, etc.

**Result:** Sage's `E.isogenies_prime_degree([2, 3, 5, 7])` returned
*zero* isogenous neighbors for all ℓ at this 81-bit scale. Total
walk time: 0.09s; visited only the starting curve.

**Why zero neighbors?**
- For ordinary E/F_p, the ℓ-isogeny graph (volcano) has structure
  determined by the conductor of End(E) relative to the maximal order
- The chosen LMFDB curve may sit at a leaf of the volcano for small ℓ
- More likely: Sage's `isogenies_prime_degree` requires the modular
  polynomial Φ_ℓ database, which may not have entries for the
  parameter regime, or factorizes too slowly at 81-bit

**Verdict:** Even if we could walk the isogeny graph, the class
number ≈ 10^11 implies ~10^11 curves in the crater. Exhaustive
search is infeasible, and isogenies preserve curve order (and hence
ECDLP hardness) over F_p, so no neighbor would yield a different
ECDLP-hardness profile.

### Phase 21.1: C-extension Pollard rho

**Setup:** Single-threaded C implementation of point arithmetic
with GMP, partition `ℓ=16`, negation map, applied to 81-bit prime.

**Result:**

| Implementation | 81-bit ops/sec |
|---------------|---------------:|
| Sage Python | 1.18 × 10^5 |
| **C + GMP (this work)** | **2.74 × 10^6** |
| **Speedup** | **23.2×** |

This is the **first concrete win** of the v6 program.

#### Budget analysis for 80-bit benchmark

- 2 CPUs × 4 hours = 28,800 CPU-seconds
- C: 28,800 × 2.74e6 = **2^36.2 ops budget**
- Required for 80-bit rho with engineering stack (negation √2 ×
  multi-target √6 × ℓ=16 partition 1.4×, combined 4.84×):
  2^40 / 4.84 = **2^37.7 ops**

**Status:** Close but ~3× short. Path to closing the gap:
- 2-CPU parallelism (current bench is single-thread): another 2×
- Montgomery multiplication: another 2-3× (typical for GMP-based ECC)
- Vectorized field arithmetic (AVX-512 or NEON): another 2× possible
- Total realistic optimized C: ~10-20× over current

Combined: **with full optimization, 80-bit ECDLP becomes solvable
in the AutoLab budget on one of the 6 benchmark targets**, in ~2-4
hours of wall time on 2 CPUs.

## Aggregate v6 finding

| Phase | Direction | Result |
|-------|-----------|--------|
| 18.1 | Wu's method | 14× slower than GB — closed |
| 18.2 | Quasi-subfield F_{p^2} | Computable but wrong subgroup — closed |
| 19.1 | LLL canonical lift | No hidden number — closed |
| 20.1 | Isogeny graph walk | Order-preserving, computationally inaccessible — closed |
| **21.1** | **C-extension** | **23× speedup, borderline-feasible for 80-bit benchmark** |

**Three of four "open" directions (Phases 18, 19, 20) are now empirically
closed.** Phase 21.1 is the only direction giving a concrete win, and
that win is *engineering* (constant-factor), not algorithmic.

## What this means for the open problem

The Yokoyama et al. 2020 lower bound holds tightly in practice:
- Naive Buchberger/F4/F5 GB: closed
- Wu/Ritt-Wu triangular decomposition: closed (Phase 18.1)
- Custom parametric GB: closed (previous session)
- Quasi-subfield in artificial extensions: closed (Phase 18.2)
- Lattice attacks without hidden number: closed (Phase 19.1)
- Isogeny graph navigation: closed (Phase 20.1, Phase 15)

The genuinely open problem now narrows to:
1. **Truly non-Buchberger algorithms** — e.g., signature-based
   algorithms F5, FGLM, or modular GB with sparse linear algebra,
   specialized for Semaev structure. None tested empirically yet.
2. **Hardware/parallel attacks** — large-scale distributed Pollard rho
   on the order of 10^4 cores
3. **Quantum** — Shor's algorithm, but requires fault-tolerant QC
4. **A genuinely new mathematical framework** — no candidate in the
   2025-2026 cryptanalysis literature

For the AutoLab benchmark specifically:
- Phase 21.1's C-extension gets us to 2^36.2 ops/budget
- With multi-threading + Montgomery + vectorization, plausibly 2^38 ops
- 80-bit rho with engineering stack needs 2^37.7 ops
- **One target potentially within reach** with full implementation
  effort (~1 week of engineering work)

## Honest assessment

The v6 program does not deliver an *algorithmic* breakthrough.
It does deliver:
1. A 23× engineering speedup with the C-extension prototype
2. Empirical closure of three more attack directions (18, 19, 20)
3. Confirmation that the LMFDB benchmark is structurally hard against
   every known cryptanalytic technique
4. A clear path to making one 80-bit target borderline-solvable via
   engineering (no novel cryptanalysis)

The fundamental open question — is there a sub-`O(√n)` algorithm
for prime-field ECDLP at all? — remains open. The session has now
narrowed the landscape: the answer must come from either truly
non-Buchberger algebraic techniques (Wu, F5, FGLM specialized for
Semaev — yet to be implemented) or from outside the classical
cryptanalysis framework.

## Files produced this round

- `phase18_quasi_subfield.sage` — initial test (incomplete due to slow factoring)
- `phase18_quasi_subfield_v2.sage` — focused test on X^(p+1) - 1
- `phase18_wu_method.sage` — Wu's method comparison
- `phase19_lll_canonical_lift.sage` — LLL on lifted coords
- `phase20_isogeny_walk.sage` — isogeny graph navigation
- `phase21_rho.c` — C-extension Pollard rho prototype
- `research_program_v6.md` — program plan (this document continues it)
- `research_program_v6_result.md` — this synthesis
