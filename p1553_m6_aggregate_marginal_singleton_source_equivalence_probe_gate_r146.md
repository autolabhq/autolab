# P1553 M6 aggregate-marginal singleton source-equivalence gate R146

## Claim boundary

R146 proves that the R144 exact count-and-marginal output is
source-equivalent whenever a positive target fiber contains one canonical
`A`-multiset/`C`-multiset source pair. It independently replays R144 on all
eight actual controls and measures how often that condition holds.

It also records an occupancy-conservation identity and a random-hashing
occupancy model. The random model does not transfer to the structured factor
base. None of these statements is a computational lower bound or a count
index.

It claims no factor logs, identical descent, Pollard-rho improvement, Shoup
improvement, or breakthrough.

Classification:

```text
AGGREGATE_COUNT_AND_MARGINALS_INVERT_TO_CANONICAL_A_C_SOURCE_ON_SINGLETON_FIBERS__EIGHT_ACTUAL_SINGLETON_POSITIVE_FRACTIONS_98P18_TO_100_PERCENT__RANDOM_DENSE_M6_MODEL_CONDITIONAL_SINGLETON_PROBABILITY_1_MINUS_1_OVER_2_6FACTORIAL_SQUARED__SUPERCONSTANT_OCCUPANCY_TRADES_FOR_HIT_DENSITY__NO_COUNT_INDEX_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Canonical quotient

For a six-factor Cartesian relation, forget the pairing of `A` and `C`
atoms. A canonical source is one unordered six-`A` multiset and one unordered
six-`C` multiset. Any pairwise alignment of their ordered expansions has the
same target and the same meaningful log row because the public Cartesian
rectangle directions are redundant.

Let the unique canonical source in a target fiber have atom multiplicities
`r_A(a)` and `r_C(c)`. If its ordered expansion weight is

```text
c_T = w_A w_C,
```

then its exact aggregate marginals are

```text
d_A(a) = c_T r_A(a),
d_C(c) = c_T r_C(c).
```

Consequently,

```text
r_A(a) = d_A(a)/c_T,
r_C(c) = d_C(c)/c_T
```

over the integers. Exact count plus full marginals therefore recovers the
canonical atom source on every singleton fiber. It does not recover a
meaningless `A/C` pairing, which is unnecessary for the R144 log identity.

## Exact controls

The verifier independently enumerates canonical `A/C` source pairs on all
four R82 curve families and both offsets. The positive target counts equal
R144 exactly, and every sampled R144 ordered count and atom marginal replays.

The singleton positive-fiber fractions are:

```text
1.0000000000
1.0000000000
0.9818181818
1.0000000000
0.9999226904
1.0000000000
0.9920909091
0.9995746491
```

Maximum canonical occupancies range from one to three. Every singleton
marginal vector inverts to its exact `A/C` atom multiplicities.

BSGS labels group finite sources by target only in the independent verifier.
The theorem itself consumes no labels. Finite occupancy receives no
asymptotic or candidate credit.

## Random model

At the dense R115 vertex,

```text
M
  = binom(|A|+5,6) binom(|C|+5,6)
  = q/(6!)^2 (1+o(1))
```

canonical source pairs map into `q` targets. Under the explicitly
independent-uniform model, a target occupancy tends to

```text
Poisson(lambda),  lambda=1/(6!)^2=1/518400.
```

Conditioned on a positive target, the limiting singleton probability is

```text
lambda exp(-lambda)/(1-exp(-lambda))
  = 1-lambda/2+O(lambda^2)
  = 0.9999990355000988...
```

This is a model-bound prediction, not a theorem for the structured
factor-base image.

## Occupancy conservation

For any deterministic source-to-target map, let:

```text
M = canonical source count,
H = positive target support,
L = M/H = mean positive occupancy.
```

A uniform target succeeds with probability

```text
H/q = M/(Lq).
```

When `M=Theta(q)`, raising mean aggregate occupancy by `B^gamma` lowers
uniform-target hit density by `B^-gamma`. Thus a claimed compression from
superconstant fibers must charge the matching retry exponent unless it
constructs a targetable structured positive family. This is an accounting
identity, not an algorithmic lower bound.

## Admission

Fifteen of twenty-five obligations pass. Canonical source quotienting,
singleton integer inversion, occupancy conservation, random-model
calculation, all finite support replays, and all finite singleton inversions
are admitted. The random result is admitted only as model-bound evidence.

No implicit count/marginal index, targetable superconstant fiber family,
density-retry-avoiding selector, structured rank theorem, factor-log solve,
identical descent, or generic-prime algorithm is admitted.

Disposition:

```text
ADMIT_EXACT_SINGLETON_SOURCE_EQUIVALENCE_OF_COUNT_PLUS_MARGINALS__ADMIT_OCCUPANCY_DENSITY_ACCOUNTING__RETAIN_RANDOM_SINGLETON_RESULT_AS_MODEL_BOUND_ONLY__NO_COUNT_INDEX__NO_LOGS__NO_DESCENT__NO_GENERIC_TRANSFER__NO_RHO__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Treat count plus full marginals as source-equivalent on singleton canonical
fibers. Construct one targetable structured multi-fiber family whose
canonical occupancy grows as `B^gamma` while its hit-density retry, setup,
marginal output, rank, factor-log, and shifted-descent costs remain below the
R115 caps; or construct the implicit batched count/marginal index without
claiming that source elimination alone makes the query easier. It may use no
DLP, root, count, marginal, rank, or source oracle.
