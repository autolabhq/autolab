# P1553 M6 static 3SUM-indexing tradeoff gate R148

## Claim boundary

R148 identifies the R147 occurrence-pair query with weighted static
3SUM-indexing and audits the applicable published upper bounds at the R115
caps. It rejects the unsupported assumption that a linear-state index
automatically gives square-root query time.

The published algorithms answer existence or return one witness. R144 needs
strictly stronger output: exact integer multiplicity and full `A/C` atom
marginals. No published count or marginal index is imported.

The tradeoff calculations close only the named standard data structures.
They are not lower bounds for the triple-convolution structure of the
occurrence divisor, elliptic arithmetic circuits, RAM, or cell probes.

Classification:

```text
OCCURRENCE_PAIR_QUERY_IS_WEIGHTED_STATIC_3SUM_INDEXING__LINEAR_STATE_TRIVIAL_QUERY_D_B9O4__FULL_BATCH_B7O2_N7O10__FIAT_NAOR_LINEAR_STATE_ENDPOINT_DOMINATED__DINUR_GOLOVNEV_IMPROVEMENT_REQUIRES_D3O2_STATE_AND_D2_PREPROCESSING__NO_PUBLISHED_EXACT_COUNT_OR_MARGINAL_INDEX__STRUCTURE_AWARE_ELLIPTIC_AUTOCORRELATION_OPEN__NO_LOWER_BOUND_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Exact reduction

Let the repeated ordered-`C3` endpoint list have length

```text
n = B^(9/4+o(1)).
```

For target `tau`, the R147 query asks for occurrence pairs whose Cayley sum
is `tau`. Ignoring multiplicities, this is exactly static 3SUM-indexing on
two identical lists in the elliptic Cayley group. Retaining multiplicities
turns it into a weighted exact-count query.

R144 additionally requires one marginal for each relevant `A` and `C` atom.
Decision or one-witness output does not provide those values.

## Finite controls

The verifier reuses all eighteen positive and empty R147 queries on each of
eight actual controls. The occurrence-list lengths are

```text
27, 27, 125, 125, 216, 216, 343, 343.
```

For each control it records exactly:

```text
linear state words       = n
scan operations/query    = n
sample-batch operations  = 18 n
full sumset table entries = n^2.
```

This is finite accounting only. It consumes no candidate root or DLP oracle
and receives no asymptotic credit.

## Published tradeoffs

Golovnev, Guo, Horel, Park, and Vaikuntanathan give the Fiat-Naor-derived
3SUM-indexing relation

```text
T S^3 = soft-O(n^6).
```

Primary source:

```text
https://arxiv.org/abs/1907.08355
references/golovnev_3sum_preprocessing_1907.08355.pdf
sha256 b9161a299ee5227bdf11be0bbfec1c58a9348deb8d261875b935d573b4112785
```

At the campaign state cap `S=n`, that curve gives `T=n^3`, which is
strictly dominated by the trivial `T=n` scan. It does not give a
square-root query.

Dinur and Golovnev improve the tradeoff to

```text
T S = soft-O(n^(5/2))
```

in the range `n^(3/2) << S << n^(7/4)`, with soft-`O(n^2)`
preprocessing.

Primary source:

```text
https://arxiv.org/abs/2512.04258
references/dinur_golovnev_3sum_indexing_2512.04258.pdf
sha256 e56522544d9ae28ec542825fcd2e7238360a05306a79d0b757a910dda382420c
```

With `n=B^(9/4)`, even the lower end of the improved state range is

```text
n^(3/2) = B^(27/8),
```

and its preprocessing is

```text
n^2 = B^(9/2).
```

Both exceed the `B^(9/4)` setup cap.

## Campaign charge

The two standard endpoints that fit or nearly answer the query are:

```text
linear state plus scan:
  setup B^(9/4)
  one query B^(9/4)
  B^(5/4) queries B^(7/2) = N^(7/10)

full sumset table:
  setup and state B^(9/2)
  one query soft-O(1).
```

The first exceeds rho over the complete query stream; the second exceeds
both setup and rho before querying. Neither supplies exact atom marginals.

This is an upper-bound audit, not an unconditional lower bound. It leaves
open algorithms that exploit the occurrence divisor as a triple elliptic
convolution before generic 3SUM abstraction.

## Admission

Fifteen of twenty-five obligations pass. The exact indexing reduction,
finite accounting, two primary-source tradeoff translations, and standard
route negatives are admitted.

No structure-aware shared autocorrelation, exact integer count index,
transposed atom marginals, rank theorem, factor logs, identical descent, or
generic-prime algorithm is admitted.

Disposition:

```text
ADMIT_WEIGHTED_STATIC_3SUM_REDUCTION__REJECT_LINEAR_STATE_SQRT_QUERY_ASSUMPTION__CLOSE_PUBLISHED_STANDARD_INDEXING_ROUTES_AT_R115_CAPS__PRESERVE_STRUCTURE_AWARE_ELLIPTIC_OPERATOR__NO_LOWER_BOUND__NO_LOGS__NO_DESCENT__NO_RHO__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Exploit the fact that the occurrence divisor is a triple elliptic
convolution of the compact `C` divisor before applying generic static
indexing. Build one structure-aware shared autocorrelation or marker
operator with `B^(9/4)` setup and `B^(5/4)` total query work that emits
exact integer counts and `A/C` marginals, then replay rank, factor logs, and
identical descent. It may use no DLP, root, 3SUM, count, marginal, rank, or
source oracle.
