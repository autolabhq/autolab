# P1553 5A5C Implicit Veronese Hyperplane Source Index Gate R94

## Claim boundary

No generic-prime-field ECDLP algorithm, Pollard-rho improvement, Shoup-bound
improvement, factor-log solve, or target descent is claimed.

Classification:

```text
IMPLICIT_VERONESE_QUERY_EQUALS_FERMAT_PROJECTOR__DIRECT_DYADIC_SOURCE_EXACT__STANDARD_MULTIPOINT_B12O5__CAP_SIZED_CONTRACTION_UNSUPPLIED
```

## Exact implicit query

R93 factors the quadratic resultant through a fixed rank-six Veronese
bilinear form `H`. Over `F_p`, its exact zero indicator is

```text
delta_0(H) = 1 - H^(p-1).
```

This is the restricted Fermat-projector trace/source router already screened
by R9, specialized to the R93 value kernel. It is not a new source-index
primitive. For `p=101`, the degree-100 univariate projector is exact and
minimal: any polynomial of degree below 100 that vanishes at all 100
nonzero field elements is identically zero and therefore cannot equal one at
zero.

## Finite source replay

On the frozen prime-order curve

```text
E/F_101: y^2=x^3+7x+4,
#E(F_101)=97,
```

deck sizes `4,6,8,10,12` produce unordered pair-chart counts

```text
10, 21, 36, 55, 78.
```

The Fermat projector agrees with the resultant-zero predicate at every
entry. Its matrix ranks are

```text
7, 18, 29, 53, 78.
```

The largest projector matrix is full rank despite the rank-six raw value
kernel. Exact direct range counting followed by dyadic descent returns a
signed affine source in every nonempty row with fewer than two full scans,
but this changes no asymptotic exponent.

These are finite exact controls, not an asymptotic rank lower bound.

## Charged routes

The smaller R84 root side has `B^(12/5)` source-bearing values. Therefore:

- materializing one constant-width feature row per source costs
  `B^(12/5)` persistent state, outside the `B^(9/4)` setup cap;
- direct projector counting and dyadic replay cost `B^(12/5)` fresh work,
  outside the `B^(5/4)` online cap;
- the bound all-output finite-field multipoint evaluator also costs
  `B^(12/5+o(1))` on `B^(12/5)` requested outputs.

The last item uses the output-sensitive upper bound from
Bhargava-Ghosh-Guo-Kumar-Umans, *Fast Multivariate Multipoint Evaluation
Over All Finite Fields*, arXiv:2205.00342v1. It is a matched upper-bound
control and does not lower-bound a zero-only or source-returning aggregate
algorithm.

## Projective boundary

The finite replay covers affine pair charts. Blind-zero branches, infinity,
nonreduced tangent multiplicity, and actual coupled five-A plus five-C source
unranking are not complete. Known-RHS relation rank, factor logs, and
identical fresh-target descent remain absent.

Ten of twenty-three obligations pass.

## Scope

R94 closes only materialized feature indices, direct implicit-projector
scans, and the bound all-output multipoint route for this split. It leaves
open an exact aggregate recurrence, character sum, or nonlinear
source-returning circuit that computes a projector count without evaluating
the individual `B^(12/5)` root-side values.

Disposition:

```text
REJECT_STANDARD_IMPLICIT_VERONESE_INDEX_ROUTES_ONLY__FERMAT_HYPERPLANE_PROJECTOR_EXACT__DIRECT_DYADIC_AFFINE_SOURCE_EXACT__PROJECTOR_FULL_RANK_AT_78__MATERIALIZED_AND_MULTIPOINT_ROUTES_B12O5__R9_TRACE_CONTRACTION_UNSUPPLIED__PROJECTIVE_AND_FULL_5A5C_SOURCE_INCOMPLETE__AGGREGATE_RECURRENCE_OPEN__NO_RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Construct or refute one exact aggregate recurrence for
`sum_box(1-H^(p-1))` using the rank-six Veronese form and the Cartesian A/C
source circuits. Every intermediate must fit `B^(9/4)` persistent state and
`B^(5/4)` fresh work/workspace, support one nonzero dyadic child and exact
coupled source unranking, and replay blind-zero, infinity, proper-subsum,
tangent, and multiplicity branches without individual `B^(12/5)`
evaluations, DLP labels, or verifier oracles.
