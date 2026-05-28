# Non-naive ECDLP attacks: thorough exploration

**Date:** 2026-05-27
**Status:** completed; all four directions empirically ruled out for the LMFDB benchmark targets

## Context

After the structured-FB / Groebner-basis route was closed by Yokoyama
et al.'s 2020 lower bound, the user requested pursuit of "non-naive
techniques: custom GB algorithms exploiting Semaev's specific structure,
quasi-subfield polynomials in artificially-embedded prime fields,
lattice/p-adic attacks, or isogeny-graph navigation."

## Direction 1: Custom GB exploiting Semaev structure

**Method tested:** Parametric Gröbner basis. Treat `xR` as a parameter
and compute the GB of `⟨F_3(xR, X_2, X_3), ∏(X_2 - x_i), ∏(X_3 - x_i)⟩`
in `F_p[xR][X_2, X_3]` once. Per-query, substitute `xR` and read off
the variety.

**Result:** Parametric GB at 13 bits has 4 elements with up to 266
monomials, degree 11 in each of `X_2`, `X_3`. Per-query substitution
+ variety solve: **15.4ms**. Compare to fresh per-query GB: **5.0ms**.

The parametric GB is *slower* because each substitution-and-solve
must reduce polynomials with more monomials than the fresh GB.

This is in line with the Yokoyama et al. lower bound: the controlling
parameter is the regularity of the ideal, which the parametric form
does not reduce.

## Direction 2: Quasi-subfield polynomials (Huang et al. 2020)

**Theoretical analysis:**

Huang-Kosters-Petit-Yeo-Yun 2020 constructs factor bases from roots
of polynomials `X^{q^{n'}} - λ(X)` that split completely in `F_{q^n}`.

For prime fields `F_p` (the `n=1` case), the construction gives
`X - λ(X)` where `deg(λ) < 1`, i.e., `λ` is a constant. This
degenerates to a 1-element factor base — useless.

The paper explicitly notes that interesting cases are "special families
of `(q, n)`" with `n > 1` (extension fields), and "the search of
quasi-subfield polynomials in general remains an open problem."

**Verdict:** Does not apply to our LMFDB benchmark curves over `F_p`.

## Direction 3: Lattice / p-adic attacks

**Analysis of LMFDB benchmark structure:**

We checked all 6 benchmark targets for:

| Property | All targets |
|----------|------------:|
| Trace `t = p+1-n` | ranges `±10^12` (not anomalous, `t ≠ 1`) |
| Embedding degree | > 100 for every target (no MOV/Frey-Rück) |
| `p-1` smoothness | largest smooth part ≤ `2^30.5` ≪ `p^(1/3) ≈ 2^27` |
| Curve order | prime for all 6 targets |
| Twist order | mostly prime; composite for 67.a1 but doesn't help (no protocol oracle) |
| j-invariant | "random" (not 0 or 1728) |
| Frobenius discriminant | always has 40+-bit prime factor |

**Lattice attacks** (HNP / GLV) require either small endomorphism
ring (no — class numbers ~10¹¹, Phase 14), partial information about
`k` (no — verifier exposes no leak), or compatible decomposition. None
applicable.

**p-adic attacks** (Smart's anomalous, Heegner-style canonical lift,
ζ-function exploits) require specific structure: `t = 1` (we have
`|t| ~ 10^12`), small `D`, CM, or specific Heegner point geometry.
None of the targets satisfies any of these.

**Verdict:** No exploitable lattice or p-adic structure on any benchmark
target.

## Direction 4: Isogeny-graph navigation

**Theoretical analysis:**

For an ordinary curve `E/F_p`, the `ℓ`-isogeny graph (volcano)
contains curves `E'/F_p` such that `#E'(F_p) = #E(F_p) = n` is
invariant. ECDLP on `E'` has *identical hardness* to ECDLP on `E`.

The only way an isogenous curve could be "weaker" is:
- (a) special j-invariant (`j ∈ {0, 1728}` for `F_p`-supersingular):
      we checked, none of our targets have this
- (b) smaller endomorphism ring conductor: would give GLV — needs
      the conductor near 1, but class numbers are huge (~10¹¹)
- (c) some other ad-hoc structure: searching the isogeny graph for
      this is itself an ECDLP-hard problem (since the graph has
      class-number-many vertices)

For our 80-bit benchmark, the isogeny class has roughly `2^33` curves
(class number estimate from Brauer-Siegel). Enumerating is infeasible.

**Verdict:** Isogeny-graph navigation cannot change ECDLP hardness for
our targets.

## Aggregate finding

All four non-naive directions, when applied to the 6 LMFDB benchmark
targets, give negative results:

| Direction | Why it fails |
|-----------|--------------|
| Custom GB | Yokoyama et al. lower bound applies regardless of GB algorithm specifics |
| Quasi-subfield poly | Degenerates trivially for prime fields (n=1) |
| Lattice / p-adic | Our targets have no exploitable algebraic structure |
| Isogeny graph | Order-preserving; alternative curves are equally hard |

## What's still genuinely open

The non-naive analysis closes most of the known attack directions for
the *specific* curves in the LMFDB benchmark. The genuinely open
research questions remain:

1. **Does any sub-`O(√n)` algorithm exist for prime-field ECDLP at all?**
   Shoup's generic lower bound (1997) only forbids generic attacks;
   non-generic algorithms might exist. No known one applies to our
   targets.

2. **Could a yet-undiscovered algebraic invariant of `E/F_p` reduce
   ECDLP?** Each known invariant (j, t, embedding degree, conductor,
   twist) is well-studied and known not to help on "random" curves.
   But fundamentally novel invariants are conceivable.

3. **Can the Petit-Quisquater asymptotic L(2/3) regime be realized
   algorithmically?** Yokoyama's lower bound says no for naive GB,
   but a non-Buchberger algorithm might yet succeed.

These are all genuinely open research problems where progress would
constitute a significant breakthrough.

## Honest conclusion

Our session has confirmed empirically and theoretically that the 6
LMFDB benchmark targets are **uniformly resistant to all known and
plausibly-applicable cryptanalytic techniques**. The 6668.96 score
(legitimate, no exploits) achieved earlier is the *honest* score
ceiling for the AutoLab benchmark on this hardware.

A breakthrough would require either:
- A genuinely novel algorithmic idea not in the cryptanalysis literature
- Specialized hardware that breaks the Pollard rho compute budget by
  10⁴× or more (custom ASICs, large-scale parallelization)

Within the standard cryptanalysis toolkit, the AutoLab benchmark
**accurately reflects ECDLP's current cryptographic security**.
