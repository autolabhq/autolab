# P1553 M6 singleton-relation hypergraph rank gate R155

## Claim boundary

R155 tests whether the R154 finite signed-rank transition is covered by a
published sparse random-matrix theorem.

It derives and replays the exact singleton relation rows, pins two primary
rank papers, and identifies both a coverage threshold and a row-dependency
gap. It does not transfer either theorem to the M6 convolution family.

Classification:

```text
SINGLETON_SIGNED_RELATION_ROWS_HAVE_WIDTH_AT_MOST_SEVEN__EXACT_NORMALIZATION_AND_TWO_CORE_CONTROLS__INDEPENDENT_SPARSE_SUPPORT_COMPARATOR_NEEDS_N_LOG_N_INCIDENCES__LOG_OVERSAMPLING_PRESERVES_B_EXPONENTS__PUBLISHED_RANK_THEOREMS_DO_NOT_APPLY_WITHOUT_CONVOLUTION_TANNER_CONTIGUITY__HASH_TO_CURVE_TRANSFER_REVERSE_FFE_LOGS_DESCENT_OPEN__NO_SHOUP_BREAKTHROUGH
```

## Singleton rows

Suppose target row `(s,a)` has exactly one unordered C6 source with
multiplicities `m_b`. The aggregate row contains its public A6 shift
multiplicity and ordered-permutation multiplicity. Dividing by both gives
the full row

```text
h_b = m_b - 1[b=a].
```

After the R154 sign quotient:

```text
hbar_j
  = m_(C_j) - m_(-C_j)
    - 1[a=C_j] + 1[a=-C_j].
```

Therefore a singleton row has at most seven nonzero signed columns: six
source inversion pairs and one target pair. All 48 controls normalize
exactly from aggregate rows. The maximum observed support is six.

## Finite dependency controls

The singleton matrices attain full rank in exactly the same 11 controls
as the aggregate R154 matrices. Thus multi-source aggregation does not
explain the finite transition.

Coverage is necessary but insufficient:

```text
full-rank singleton controls:       11
covered but rank-deficient controls: 17
```

Every singleton row set is closed under row negation from opposite-shift
symmetry. Deficient high-occupancy controls can contain dozens of rows but
only a small number of projectively distinct rows. Their 2-cores therefore
do not justify an independent-row model.

## Coverage comparator

For independent uniform supports of sizes `k_i` on `n` columns, a fixed
column is uncovered with probability

```text
product_i (1-k_i/n).
```

Exact inclusion-exclusion gives the finite all-column coverage
probability frozen in each control. For fixed width `k` and `m=c*n`,

```text
E[uncovered] ~ n exp(-c*k).
```

Constant occupancy therefore cannot give full column rank with high
probability in that comparator. The coverage scale is

```text
total incidence = n(log n + omega(1)).
```

With `n=B^(3/4+o(1))`, this introduces only a logarithmic factor. A
`B^(5/4)` reverse batch repeated `O(log B)` times and a conditional
`B^2 polylog(B)` solve preserve their B exponents.

## Literature boundary

R155 pins:

- Coja-Oghlan, Kang, Krieg, and Rolvien, *The k-XORSAT threshold
  revisited*, arXiv:2301.09287.
- Coja-Oghlan, Ergür, Gao, Hetterich, and Rolvien, *The rank of sparse
  random matrices*, arXiv:1906.05757.

The first uses independently uniform fixed-size row supports. The second
uses a random Tanner graph with prescribed degrees and associated moment
and simplicity hypotheses. M6 rows are correlated endpoint-collision
images of shared A/C decks. No independence, configuration-model
coupling, or contiguity theorem is supplied.

## Admission

Seventeen of twenty-eight obligations pass. The singleton row formula,
support bound, exact normalization, coverage comparator, two-core and
projective-row diagnostics, logarithmic oversampling charge, and
literature hypothesis gap are admitted.

No convolution-hypergraph contiguity theorem, random-rank concentration
theorem, hash-to-curve transfer, reverse signed FFE operator, candidate
factor logs, identical descent, generic-prime algorithm, rho improvement,
or Shoup improvement is admitted.

Disposition:

```text
ADMIT_SINGLETON_HYPERGRAPH_AND_N_LOG_N_COVERAGE_BOUNDARY__ADMIT_EXACT_ROW_DEPENDENCY_DIAGNOSTICS__DO_NOT_IMPORT_SPARSE_RANK_THEOREMS__REQUIRE_CONVOLUTION_TANNER_CONTIGUITY_HASH_TO_CURVE_TRANSFER_AND_REVERSE_SIGNED_FFE_OPERATOR__NO_LOGS__NO_DESCENT__NO_RHO__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Prove or refute contiguity between the logarithmically oversampled
singleton M6 Tanner graph and a prescribed-degree sparse random-matrix
model. Explicitly control shared-deck dependencies, repeated and opposite
rows, cancellations, and the 2-core. Then transfer the result to
hash-to-curve decks and instantiate the reverse signed FFE operator within
`B^(9/4)` setup and `B^(5/4+o(1))` batch work before attempting
exact-residual logs or identical descent.
