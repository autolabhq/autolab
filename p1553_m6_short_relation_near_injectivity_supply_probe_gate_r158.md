# P1553 M6 short-relation near-injectivity and supply gate R158

## Claim boundary

R158 replaces R157's asymptotically false global coefficient-map
injectivity target with a quantitative near-injectivity theorem. It also
derives the exact singleton relation-row universe and proves concentration
of total projective row supply in the independent uniform cyclic-label
model.

Classification:

```text
GLOBAL_C6_INJECTIVITY_ASYMPTOTICALLY_REFUTED__BAD_SINGLETON_FRACTION_O1__EXACT_L1_5_OR_7_RELATION_ROW_UNIVERSE__PROJECTIVE_COMBINED_FORMS_PAIRWISE_INDEPENDENT__B3O4_LOG_B_DISTINCT_ROW_SUPPLY_CONCENTRATES__FULL_COVERAGE_RANK_HASH_TRANSFER_REVERSE_FFE_AND_DESCENT_OPEN__NO_SHOUP_BREAKTHROUGH
```

## Global injectivity is false

Let the signed C-log dimension be

```text
d = B^(3/4+o(1))
```

and the prime subgroup order be `q=B^(5+o(1))`. The feasible signed C6
coefficient universe has

```text
M = Theta(d^6) = B^(9/2+o(1))
```

vectors. For two distinct feasible vectors, equality of their group images
has probability `1/q`. The expected number of colliding pairs is therefore

```text
Theta(M^2/q) = B^(4+o(1)).
```

Global injectivity is not an asymptotically valid requirement.

## Near-injectivity

For a fixed `l1`-six source vector, the probability of colliding with any
other feasible vector is at most

```text
(M-1)/q = O(B^(-1/2+o(1))).
```

Thus the expected bad singleton fraction is `O(B^(-1/2))`. Markov gives

```text
bad fraction <= B^(-1/4)
```

except with probability `O(B^(-1/4))`.

A bad source vector can invalidate at most `2d` signed-target rows. Since
the complete row universe has size `Theta(d^7)`, an `o(1)` bad-source
fraction removes only an `o(1)` row fraction.

## Exact row universe

A public singleton relation row is

```text
r = v - sigma*e_j,
```

where `||v||_1=6`. Subtracting a signed unit changes the norm by one, so
every row has norm five or seven. Conversely, every integer vector with
norm five or seven admits such a representation. Therefore

```text
R_d = {r in Z^d : ||r||_1 in {5,7}}
```

exactly, with `|R_d|=Theta(d^7)=B^(21/4+o(1))`.

## Projective relation supply

A nonzero A6 coefficient vector has norm in `{2,4,6}`. A relation row has
norm in `{5,7}`. If two combined A/C forms are proportional modulo a prime
`q>84`, their small minors vanish over the integers, so they are rationally
proportional. The possible norm ratios intersect only at one. Canonicalizing
the first nonzero A coefficient removes the negative copy, leaving distinct
nonproportional forms.

Distinct nonproportional linear forms on independent uniform A/C labels have
pairwise-independent zero events. With `O(log B)` independent A batches:

```text
projective candidate forms = Theta(B^(23/4) log B)
expected relation events   = Theta(B^(3/4) log B)
variance                   = Theta(B^(3/4) log B)
```

Chebyshev gives constant-relative supply failure
`O(B^(-3/4)/log B)`.

Repeated event hits on the same row have expected total

```text
O(B^(-15/4) log(B)^2),
```

so event count and distinct projective row count agree with high
probability.

## Coverage boundary

A fixed column occurs in `R_d-R_(d-1)=Theta(d^6)` rows and receives
`Theta(log B)` expected relation events. Pairwise independence alone gives
only

```text
Pr[column uncovered] = O(1/log B).
```

This proves a vanishing expected uncovered fraction, but it does not prove
zero uncovered columns, residual full rank, or a usable factor-log system.
A higher-moment, dependency-graph, or direct rank argument remains required.

## Finite controls

The frozen grid contains:

```text
A-pair counts:       2,3
C-pair counts:       3,4,5
independent batches: 1,2,4
seeds:               15801,15802
```

All 36 controls apply C6 collision pruning before admitting relation rows.
Across the repeated grids, 10,578 theoretical rows are removed because no
singleton source survives. All 36 controls still cover every column and
reach full rank.

Observed relation events and distinct rows by batch count are:

```text
batches 1: 185 events, 183 distinct rows
batches 2: 323 events, 320 distinct rows
batches 4: 692 events, 673 distinct rows
```

These finite controls receive no asymptotic or attack credit.

## Cost and admission

Explicit C6 enumeration remains `B^(9/2)`, above the `B^(5/2)` rho proxy.
Sixteen of twenty-six obligations pass. R158 admits:

```text
global-injectivity refutation,
near-injectivity singleton-loss bound,
collision-pruning scale preservation,
exact l1-five/seven row universe,
projective nonproportionality,
pairwise-independent event supply,
second-moment concentration,
vanishing duplicate-row bound,
vanishing uncovered-fraction boundary.
```

It does not admit exact transfer to the conditioned hash-to-curve sampler,
full column coverage, full projective rank, a reverse signed FFE operator,
identical target descent, a generic-prime algorithm, rho improvement, or
Shoup improvement.

Disposition:

```text
ADMIT_NEAR_INJECTIVITY_AND_DISTINCT_ROW_SUPPLY__REFUTE_GLOBAL_INJECTIVITY_TARGET__DO_NOT_PROMOTE_PAIRWISE_SUPPLY_TO_FULL_COVERAGE_OR_RANK__REQUIRE_CONDITIONED_HASH_TRANSFER_HIGHER_MOMENTS_REVERSE_FFE_AND_DESCENT__NO_RHO__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Upgrade pairwise event concentration to zero uncovered columns and full
projective rank under the conditioned hash-to-curve sampler. Then construct
the reverse signed FFE operator and identical target descent within the
frozen caps.
