# Phase 6.2: Weil restriction & Gaudry-Hess-Smart applicability

**Date:** 2026-05-27  
**Status:** completed, structural negative result

## The Gaudry-Hess-Smart (GHS) attack

For an elliptic curve `E/F_{q^n}` with `n ≥ 2`, the Weil restriction
`A = Res_{F_{q^n}/F_q}(E)` is an `n`-dimensional abelian variety over
`F_q`. If `A` contains a Jacobian of a hyperelliptic curve `C` of
small genus (specifically `g ≥ n/2`), Gaudry's index calculus on
`Jac(C)` gives `L_q(2/3)` subexponential ECDLP.

GHS works particularly well for:
- Binary fields `F_{2^n}` with composite `n` (Gaudry 2009, Diem 2003)
- Specific extension fields where the Weil restriction has small
  genus covers

## Applicability to AutoLab targets

The four LMFDB curves are defined over `Q` and the ECDLP is at
**prime-field reductions** `E(F_p)` where `p` is an 80-bit prime.
*There is no extension field structure to descend from.* The natural
question is whether we can artificially "lift" `E(F_p)` to
`E(F_{p^n})` and apply GHS to the lifted DLP.

**Sub-experiment 6.2.1.** Construct `E(F_{p^2})` for one of the
precomputed targets. The Weil restriction is `Res_{F_{p^2}/F_p}(E)`.
By Conrad's lecture notes ("Weil and Grothendieck approaches to
adelic points"), this is a 2-dimensional abelian variety over `F_p`
isogenous to `E × E_d` where `E_d` is a quadratic twist.

For our specifically chosen curves where the curve is *not* isomorphic
to its twist over `F_{p^2}`, the Weil restriction is a non-trivial
2-dim abelian variety. Its Jacobian decomposition determines whether
GHS applies.

**Critical structural fact.** ECDLP on `E(F_p)` of order `n` does NOT
become easier when embedded in `E(F_{p^2})`. The group `E(F_p) ⊂
E(F_{p^2})` is a subgroup, and the discrete log on the subgroup is
identical. So lifting to `F_{p^2}` doesn't change the DLP.

## What GHS actually needs

The GHS technique requires that the curve be **originally defined**
over an extension field `F_{q^n}` and that the ECDLP is on
`E(F_{q^n})`. The Weil restriction then maps to `Jac(C)/F_q` where
`Jac(C)` has a smaller relative group order than `E(F_{q^n})`.

For curves originally over `F_p` (our setting), Weil restriction
gives `E × E_d` over `F_p`. The DLP on `E × E_d` decomposes
componentwise: `(P, P') → (kP, kP')` for the same `k`. This is
**exactly** the original ECDLP on `E` — no advantage.

## Conclusion

Phase 6.2 produces a structural negative result: GHS does not apply
to prime-field-original elliptic curves. This is well-known to
specialists but worth documenting empirically.

The Weil restriction approach only helps when the curve is originally
over `F_{q^n}` for `n ≥ 2`. AutoLab uses prime-field-original curves,
so this attack vector is closed.

## What this rules out

This forecloses one of the most-cited subexponential ECDLP attack
vectors. The `L(q^n, 2/3)` result of Gaudry-Hess-Smart does not apply
to AutoLab's bundled curves.

The remaining Phase 6 sub-experiment (algebraic walk function via `x
mod ℓ`) is folded into Phase 9 (structured factor base) since they
share the same algebraic mechanism.
