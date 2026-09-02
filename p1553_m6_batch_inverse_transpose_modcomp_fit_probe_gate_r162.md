# P1553 M6 batch-inverse transpose/modular-composition fit gate R162

Date: 2026-08-01

## Scope

R162 separates the shareable linear denominator layer of the R161 signed-C3
translation from its nonlinear modular-composition and source-gcd layer. It
also distinguishes the strict R159 phase cap from the total cost actually
needed to beat Pollard rho.

This is an exact algebraic interface, literature-fit audit, and finite
correctness gate. It is not a source locator or a nonlinear lower bound.

## Exact batch inverse

Let `U` be the monic degree-`n` signed-C3 divisor polynomial. For a target
x-coordinate `u`, define

```text
Q_u(X) = (U(u)-U(X))/(u-X).
```

Then

```text
(u-X) Q_u(X) = U(u)-U(X).
```

If `U(u)` is nonzero, reduction modulo `U` gives

```text
(u-X)^(-1) = Q_u(X)/U(u) mod U.
```

If `U(u)=0`, this detects exactly the denominator-exception branch already
split by R161. All `U(u_j)` values are available through a subproduct and
remainder tree in soft-linear work in `n+N`.

## Transposed functional

Write `U=sum_k c_k X^k`. For one fixed coefficient functional `w=(w_r)`,

```text
<w,Q_u> = P_w(u),
[u^s] P_w = sum_r w_r c_(s+r+1).
```

The coefficients of `P_w` are one cross-correlation. Thus a constant number
of fixed linear functionals of all `Q_u` values can be computed by
soft-linear preprocessing and multipoint evaluation without materializing an
`n`-coefficient quotient for every target.

This does not compute `lambda_j^2`, `U(phi_j)`, `V(phi_j)`, a source gcd, or a
source backpointer. Those operations are nonlinear in the target-varying
inner map.

## Literature fit

Neiger, Rosenkilde, and Solomatov give a quasi-linear online modular-
composition result when both the modulus `M` and inner polynomial `A` are
available for precomputation and the input `f` then varies. R162 instead fixes
`M=U` and `f=U` while every target changes `A=phi_j`; the theorem as stated
does not share its precomputation across this batch.

Neiger, Salvy, Schost, and Villard accelerate one generic modular composition
to soft-`O(n^((omega+3)/4))` algebraic operations and recall the finite-field
Kedlaya-Umans near-linear bit bound. They do not state a many-varying-inner
algorithm with total soft-`O(n+N)` work.

Primary sources:

```text
https://arxiv.org/abs/2003.12468
local sha256 9fdce743a5183f544df2e2d640e30e6eb2cf2c647b6d89b04aea5faa21cdb488

https://arxiv.org/abs/2601.17422
local sha256 bfa0a9fb8f3ec6cd1d2aa95907a03df131d6a4ffb3abac56983bfee42c235866
```

Novelty of the batch inverse/functional formulation is unverified.

## Cost windows

With

```text
n = B^(9/4),  N = B^(5/4),  rho = B^(5/2),
```

a monomial batch cost `n^alpha N^beta` has B-exponent
`(9 alpha+5 beta)/4`. Therefore:

```text
strict R159 batch fit:  9 alpha + 5 beta <= 5
global below-rho fit:   9 alpha + 5 beta < 10.
```

One target-dependent soft-linear pass over `n` costs `B^(9/4)`. It misses the
strict `B^(5/4)` phase cap but remains below rho by exponent `B^(1/4)`.
Independent near-linear composition for every target costs `B^(7/2+o(1))`
and fails both gates.

The valid successor search is therefore wider than R161/V110 stated: an
aggregate `B^(9/4+o(1))` nonlinear pass would still give a below-rho total
algorithm if every other charged phase remains within the existing setup and
solve caps.

## Finite controls

Six public controls over three curves and two seeds verify:

- batched `U(u_j)` evaluation against direct evaluation;
- exact exceptional-root detection;
- every quotient and regular inverse identity;
- three deterministic transposed quotient and inverse functionals;
- absence of candidate DLP, root, count, marginal, rank, or source oracles.

Finite quotient materialization and functional checks receive no attack
credit.

## Admission

Eighteen of twenty-seven obligations pass. Admit the batch inverse identity,
fixed-functional transpose, literature fit audit, and global below-rho
`B^(9/4)` aggregate-pass window.

Do not admit target-batched nonlinear composition, source gcds, target-labeled
backpointers, deterministic hash-to-curve transfer, an unconditional attack,
a Pollard-rho or Shoup improvement, or an ECDLP breakthrough.

Disposition:

```text
ADMIT_BATCH_INVERSE_AND_FIXED_FUNCTIONAL_TRANSPOSE__STRICT_BATCH_B5O4_NOT_MET_BY_N_PASS__GLOBAL_B9O4_AGGREGATE_PASS_BELOW_RHO__NONLINEAR_COMPOSITION_GCD_BACKPOINTER_OPEN__NO_RHO__NO_SHOUP_IMPROVEMENT__NO_BREAKTHROUGH
```

## Next action

Construct an aggregate nonlinear signed-divisor operator with total charged
work strictly below `B^(5/2)`, preferably soft-linear in `n`. It must compute
the lambda-square/composition/gcd layer, label every degree-at-most-20 source
factor by its target, handle exceptional roots, and replay factor logs and
identical descent. Another denominator-inversion routine is not sufficient.
