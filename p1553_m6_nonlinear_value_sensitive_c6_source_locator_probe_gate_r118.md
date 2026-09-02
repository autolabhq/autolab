# P1553 M6 nonlinear value-sensitive C6 source-locator gate R118

## Claim boundary

R118 gives an exact one-atom branch reduction for the R117 six-`C` source
problem and rejects occurrence-materialized split tables plus several bound
standard FFE/indexing realizations at the frozen caps. It does not prove a
lower bound for an endpoint-compressed, output-sensitive nonlinear index.
It supplies no inside-cap `C5` predicate, relation rank, factor logs, target
descent, Pollard-rho improvement, Shoup improvement, or ECDLP breakthrough.

Classification:

```text
EXACT_ONE_C_BRANCH_REDUCES_ALLOWED_BATCH_TO_CONSTANT_QUERY_C5_SOURCE_INDEX__NO_EXPLICIT_OCCURRENCE_SPLIT_TABLE_MEETS_B9O4_SETUP_AND_B5O4_BATCH_CAPS__SETUP_COMPATIBLE_C3_SPLIT_BATCH_B11O4__ONLINE_COMPATIBLE_C5_TABLE_B15O4__DINUR_GOLOVNEV_K6_ZERO_SLACK_STATE_B33O8__STANDARD_C5_QUOTIENT_AND_GRID_B15O4__FINITE_PROJECTIVE_POSITIVE_REPEATED_EMPTY_BRANCH_REPLAY_EXACT__OUTPUT_SENSITIVE_NONLINEAR_C5_MEMBERSHIP_FFE_SOURCE_INDEX_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Exact branch reduction

At the frozen R115-R117 vertex,

```text
q = B^(5+o(1))
|A6| = B^(1/2+o(1))
|C| = B^(3/4+o(1))
C3 state = B^(9/4+o(1))
fresh A6 batch cap = B^(5/4+o(1)).
```

For every target `T`, ordered convolution gives

```text
mu_C^*6(T) = sum_(c in C) mu_C^*5(T-c).          (1)
```

Enumerating one distinguished `C` atom for every `A6` target costs

```text
B^(1/2) B^(3/4) = B^(5/4),                       (2)
```

which consumes the entire fresh-work cap. Therefore the remaining arbitrary
five-`C` membership/source query must cost only `B^(o(1))` field operations.
An exact negative answer is required for the overwhelmingly common empty
queries. Given such a predicate, a positive five-source can be isolated with
logarithmically many dyadic predicate calls, without changing the exponent.

The exact surviving interface is:

```text
setup: at most B^(9/4+o(1))
query: polylogarithmic field work
input: arbitrary projective target and the scalar-blind C deck
output: exact empty answer or five C occurrence backpointers.
```

## Occurrence split theorem

Suppose one side of a `C^6` split is materialized by occurrence. If `s`
atoms are stored, setup costs `B^(3s/4)` and complementary enumeration for
the entire `A6` target batch costs

```text
B^(1/2 + 3(6-s)/4).                              (3)
```

Setup compliance forces `s<=3`. Then the batch exponent is at least

```text
1/2 + 9/4 = 11/4.
```

Conversely, batch compliance forces `6-s<=1`, hence `s>=5` and occurrence
state at least

```text
B^(15/4).
```

No integer split meets both caps. The seven exact rows are:

| stored arity | setup | enumerated arity | A6 batch |
| ---: | ---: | ---: | ---: |
| 0 | `B^0` | 6 | `B^5` |
| 1 | `B^(3/4)` | 5 | `B^(17/4)` |
| 2 | `B^(3/2)` | 4 | `B^(7/2)` |
| 3 | `B^(9/4)` | 3 | `B^(11/4)` |
| 4 | `B^3` | 2 | `B^2` |
| 5 | `B^(15/4)` | 1 | `B^(5/4)` |
| 6 | `B^(9/2)` | 0 | `B^(1/2)` |

This theorem covers occurrence-materialized tables. It does not assert that
every endpoint support has occurrence size, and it does not rule out a
highly collision-compressed endpoint/source index.

## Current indexing and scan routes

Apply Dinur and Golovnev's `kSUM` indexing theorem to five-`C` membership:

```text
k = 6
n = |C| = B^(3/4+o(1))
S = soft-O(n^(k-1/2-delta))
T = soft-O(n^delta).
```

The outer branch in (2) leaves zero polynomial query slack, so `delta=0`.
The resulting state is

```text
S = n^(11/2) = B^(33/8+o(1)),
```

above setup. The integer-residue construction also has no scalar-blind
transfer to a generic prime-order elliptic group.

Storing `C3` fits setup. Answering a five-sum query by enumerating `C2` and
hashing complements into `C3` costs `B^(3/2)` per query. Composing it with
the one-atom branch gives `B^(11/4)` for the full batch. It returns sources
exactly but misses the fresh cap.

## Standard FFE representations

Let

```text
D_C(X) = product_(c in C) (X-x(c)).
```

The standard five-variable deck quotient

```text
A_C^tensor5,
A_C = F[X]/(D_C),
```

has dimension `|C|^5=B^(15/4+o(1))`. The relevant Semaev polynomial is
`S6(x1,...,x5,x_target)`, with fixed degree 16 in each source variable.
Its fixed degree has zero `B` exponent, but representing the full deck grid
does not.

The all-field multipoint theorem of Bhargava, Ghosh, Guo, Kumar, and Umans
runs in near-linear time in the represented coefficient body plus output
points. Applied to the full five-deck grid, that body already has exponent
`15/4`. This is a matched upper-bound control, not an output-sensitive
lower bound.

A standard `C2 | C3` coefficient/resultant query has explicit degree body
`B^(9/4)`. Moroz and Schost's constant-order truncated-resultant bound is
softly linear in that degree body, so applying it to all one-atom branch
queries costs `B^(7/2)`. Resultants, gcds, roots, cofactors, and source
recovery do not receive unit-cost oracle credit.

An `x`-only `S6` zero permits sign choices and is not by itself a
biconditional for the fixed signed deck points. At most 32 sign branches are
constant in the exponent, but every reported source still needs exact
projective point-sum verification.

These controls close only represented grid/quotient, all-output
multipoint, and explicit coefficient/subresultant grammars. They are not
general arithmetic-circuit or data-structure lower bounds.

## Finite projective controls

Two controls use the R82 prime-order subgroup `q=16427` with `C` deck
prefixes of sizes 2 and 3. They enumerate ordered endpoint measures
`C^1,...,C^6`.

For repeated-atom positive, another positive, empty, and identity targets:

- direct `C6` count equals the one-`C` plus `C5` branch count;
- every binary `C^r | C^(6-r)` split gives the same count;
- positive sources replay as six fixed signed projective points;
- empty branches return no source;
- finite enumeration receives no asymptotic credit.

## Admission

Ten of seventeen obligations pass:

- twelve immutable source bindings;
- inherited R117 nonclaim boundary;
- occurrence split-table cap theorem;
- exact one-atom branch reduction;
- exact finite one-atom and all-binary-split counts;
- positive, repeated, empty, and identity controls;
- exact finite source replay;
- charged current `k=6`, quotient, multipoint, and resultant routes;
- explicit exclusion of output-sensitive and general lower bounds.

The nonlinear `C5` membership predicate, `C5` source index, known-RHS rank,
factor logs, identical target descent, Shoup improvement, and breakthrough
remain open.

Disposition:

```text
ADMIT_EXACT_ONE_C_PLUS_C5_BRANCH_AND_FINITE_SOURCE_REPLAY_ONLY__REJECT_ALL_EXPLICIT_OCCURRENCE_SPLIT_TABLES_CURRENT_K6_INDEX_AND_STANDARD_REPRESENTED_FFE_RESULTANT_ROUTES_AT_FROZEN_CAPS__PRESERVE_OUTPUT_SENSITIVE_NONLINEAR_C5_MEMBERSHIP_SOURCE_INDEX__NO_LOCATOR__NO_RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Construct or refute one output-sensitive nonlinear `C5` membership/source
index over the original scalar-blind `C` deck. It may store the
`B^(9/4+o(1))` `C3` state, but it must:

- answer each arbitrary five-`C` target in polylogarithmic field work;
- reject empty targets exactly and return five occurrence backpointers;
- freeze `S6`, deck-domain, quotient/remainder, and subresultant dimensions;
- account for target-dependent branches, sign ambiguity, projective
  exceptions, false positives, and reverse source recovery;
- compose with (1) before receiving relation, rank, log, or descent credit.

## Primary sources

- Dinur and Golovnev, *Improved Time-Space Tradeoffs for
  3SUM-Indexing*, <https://arxiv.org/abs/2512.04258>.
- Bhargava, Ghosh, Guo, Kumar, and Umans, *Fast Multivariate Multipoint
  Evaluation Over All Finite Fields*, <https://arxiv.org/abs/2205.00342>.
- Moroz and Schost, *A Fast Algorithm for Computing the Truncated
  Resultant*, <https://arxiv.org/abs/1609.04259>.
- Semaev, *Summation polynomials and the discrete logarithm problem on
  elliptic curves*, <https://eprint.iacr.org/2004/031>.
