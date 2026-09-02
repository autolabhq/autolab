# P1553 5A5C Compact Elliptic Subfunction Map Gate R92

## Claim boundary

No generic-prime-field ECDLP algorithm, Pollard-rho improvement, Shoup-bound
improvement, factor-log solve, or target descent is claimed.

Classification:

```text
ELLIPTIC_RATIONAL_FIBER_ENDPOINT_TR_EXACT__THEOREM_STATE_MIN_B10O3__ONLINE_COMPATIBLE_MIN_B35O8__FIVE_A_FIVE_C_SOURCE_TR_ABSENT
```

## Bound framework

The locally bound source is Dinur and Golovnev,
*Improved Time-Space Tradeoffs for 3SUM-Indexing*,
`arXiv:2512.04258v2`, dated 2026-04-23:

```text
e56522544d9ae28ec542825fcd2e7238360a05306a79d0b757a910dda382420c
```

Theorem 4.1 assumes `D` functions `f_d:[L]->[L']`, public `MAP1`,
`MAP2`, and `TR`, and an auxiliary string. Its construction has

```text
S = soft-O(L^(3/2-delta) D + Aux + L^delta)
T = soft-O(L^delta)
0 <= delta <= 1.
```

The `L^delta` term is stored shared randomness after the theorem removes its
shared-randomness assumption. R92 charges that term to setup/state.

## Exact elliptic semantics

R92 fully enumerates

```text
E/F_101: y^2 = x^3 + 7x + 4.
```

The curve has exactly 97 projective points, prime group order 97, nonzero
discriminant, and no nontrivial two-torsion. Full addition-table closure and
the order of a nonidentity generator are checked directly.

Four frozen public maps instantiate `MAP1`, `MAP2`, `f_d`, and endpoint `TR`:

```text
h=x       : D=49, L=2
h=y       : D=63, L=3
h=x+y     : D=71, L=3
h=x+2y    : D=60, L=3.
```

For every projective target, including infinity, inversion of the selected
fiber returns exactly that endpoint. The maximum fiber is at most the rational
map degree, absent local coordinate values are rejected, and `D*L>=97` in
every case. Five deterministic random-bucket controls satisfy the same
coverage inequality.

This is a semantic positive control for identity endpoint inversion. Fiber
enumeration receives no asymptotic runtime credit.

## Coverage and charged exponents

Let the group endpoint count be `N=B^5`, the number of occupied `MAP1` fibers
be `D=B^d`, and the largest subfunction domain be `L=B^ell`. Every
deterministic endpoint partition obeys

```text
D*L >= N,
d+ell >= 5.
```

For a rational map `h:E->P1` of degree `r`, each rational fiber has at most
`r` points, giving the same inequality as
`|h(E(F_q))|*r >= |E(F_q)|`.

Substitution into the theorem's charged advice gives

```text
setup exponent >= max(5 + ell*(1/2-delta), ell*delta)
query exponent  = ell*delta.
```

Charging all theorem advice, the unconstrained optimum is

```text
ell=10/3, delta=1
S = B^(10/3)
T = B^(10/3).
```

Even deleting the stored `L^delta` term as an optimistic free-randomness
control leaves a best setup exponent of `B^(5/2)`, above `B^(9/4)`.

Imposing the fresh-work cap `ell*delta<=5/4` gives

```text
ell=5/4, delta=1
D = B^(15/4)
S = B^(35/8)
T = B^(5/4).
```

No theorem-certified endpoint-partition point meets setup, with or without
the online cap. Map descriptions, map evaluation, and auxiliary source state
are granted free in this optimization.

## Source translation

The rational-fiber interfaces return an endpoint, not a five-A plus five-C
source tuple. An optimistic finite control attaches one synthetic source to
each of the 97 endpoints and recovers it exactly through an explicit
dictionary. Its asymptotic analogue stores `B^5` entries, so it exceeds setup
and receives no candidate credit.

Actual compact source unranking, reduced and nonreduced Semaev branch replay,
signed provenance, rank, factor logs, and identical fresh-target descent
remain absent.

## Scope

This closes direct endpoint-partition instantiations of the bound Theorem 4.1
construction when each of `D` subfunctions receives independent generic
function-inversion advice. It is not a lower bound against a jointly compressed
family of overlapping correspondences, a non-Fiat-Naor data structure, or a
representation-changing summation-polynomial/FFE identity.

Twelve of twenty-two obligations pass.

Disposition:

```text
REJECT_DIRECT_ELLIPTIC_ENDPOINT_PARTITION_SUBFUNCTION_FRAMEWORK_ONLY__FOUR_PROJECTIVE_RATIONAL_MAP_CONTROLS_EXACT__D_TIMES_L_COVERAGE_EXACT__THEOREM_ALL_STATE_MIN_B10O3__ONLINE_COMPATIBLE_MIN_B35O8__EXPLICIT_SOURCE_TRANSLATOR_B5__ACTUAL_5A5C_SOURCE_TR_ABSENT__OVERLAPPING_SHARED_CORRESPONDENCE_OPEN__NO_RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Construct or refute one target-dependent overlapping `S3`/`S4` incidence
correspondence with a single shared semilinear operator across all charts. It
must prove joint state below `B^(9/4)`, fresh work and workspace below
`B^(5/4)`, and exact 5A+5C source unranking on every projective branch,
without `D` independent inversion tables, endpoint/source dictionaries, DLP
labels, verifier oracles, or omitted elimination cost.

## Primary reference

- Dinur and Golovnev, *Improved Time-Space Tradeoffs for 3SUM-Indexing*:
  <https://arxiv.org/abs/2512.04258v2>.
