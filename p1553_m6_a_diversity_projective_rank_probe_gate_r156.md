# P1553 M6 A-diversity projective rank gate R156

## Claim boundary

R156 tests whether the R155 rank deficits are a structural convolution
invariant or a finite artifact of fixing only two A inversion pairs.

It projectively removes exact opposite rows and runs a preregistered
108-control grid with growing A diversity and logarithmic occupancy. The
finite transition is strong but receives no asymptotic or attack credit.

Classification:

```text
OPPOSITE_SINGLETON_ROWS_PROJECTIVELY_QUOTIENT_EXACTLY__A6_SHIFT_DIVERSITY_GROWS_AS_A_PAIR_COUNT_TO_SIXTH__PREREGISTERED_108_CONTROL_LOG_OVERSAMPLING_GRID__COVERAGE_PROJECTIVE_COUNT_AND_RESIDUAL_DEPENDENCY_SEPARATED__FINITE_A_DIVERSITY_RANK_TRANSITION_TESTED__ASYMPTOTIC_DIRECT_RANK_HASH_TO_CURVE_REVERSE_FFE_LOGS_DESCENT_OPEN__NO_SHOUP_BREAKTHROUGH
```

## Opposite-row quotient

Inversion symmetry maps a singleton row at `(s,a)` to its negative at
`(-s,-a)`. Multiplying each nonzero row by the inverse of its first
nonzero coefficient gives one canonical projective row. Deduplication
changes neither row span nor rank.

All 108 controls replay the relation identities and opposite-row pairing
exactly.

## A diversity

For `r` independent A inversion pairs, the generic signed A6 coefficient
support has size

```text
sum over even t <= 6
  sum over 1 <= j <= min(r,t)
    C(r,j) C(t-1,j-1) 2^j,
```

with the weight-zero term included. This is `Theta(r^6)`, matching the
selected `|A|^6=B^(1/2+o(1))` shift exponent. The fixed two-pair R155
controls therefore cannot represent the unbounded asymptotic A deck.

The frozen grid is:

```text
A inversion pairs:       2, 3, 4
C inversion pairs:       5, 6, 7, 8
logarithmic factors:      2, 4, 8
seeds:                    15601, 15602, 15603
```

For each cell, the occupancy multiplier is
`ceil(factor*ln(C-pair-count))`, and the prime modulus is selected before
labels or ranks.

## Finite transition

Across 36 controls per A-pair count, full-rank counts are:

```text
A pairs 2: 18
A pairs 3: 22
A pairs 4: 22
```

Growing A diversity helps modestly and then saturates.

Across 36 controls per logarithmic factor, full-rank counts are:

```text
factor 2:  6
factor 4: 21
factor 8: 35
```

At factor eight, only one of 36 controls remains deficient. The dominant
finite transition is therefore projective row supply under logarithmic
oversampling, not removal of a universal two-pair invariant.

Every deficit is decomposed into:

```text
uncovered columns,
too few projectively distinct rows,
residual dependency after projective deduplication.
```

This finite pattern supports a direct rank-theorem target but does not
prove concentration, contiguity, or hash-to-curve transfer.

## Cost envelope

The intended scaling remains:

```text
A-pair count:             B^(1/12+o(1))
A6 shift support:         B^(1/2+o(1))
signed C-log dimension:   B^(3/4+o(1))
relations:                B^(3/4+o(1)) log B
reverse row batch:        B^(5/4+o(1)) log B
conditional solve:        B^(2+o(1))
```

Projective opposite-row quotienting, A growth, and logarithmic
oversampling change no selected B exponent.

## Admission

Fifteen of twenty-six obligations pass. The projective quotient, signed
A6 support formula, frozen grid, exact finite controls, deficit
decomposition, monotone logarithmic transition, and cost ledger are
admitted.

No asymptotic A-diversity rank theorem, convolution-Tanner contiguity,
hash-to-curve transfer, reverse signed FFE operator, candidate factor
logs, identical descent, generic-prime algorithm, rho improvement, or
Shoup improvement is admitted.

Disposition:

```text
ADMIT_PROJECTIVE_OPPOSITE_ROW_QUOTIENT_AND_FINITE_LOG_OVERSAMPLING_TRANSITION__DO_NOT_TRANSFER_FINITE_LABEL_CONTROLS__REQUIRE_DIRECT_ASYMPTOTIC_RANK_HASH_TO_CURVE_TRANSFER_AND_REVERSE_SIGNED_FFE_OPERATOR__NO_LOGS__NO_DESCENT__NO_RHO__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Use the frozen transition to derive or refute a direct rank theorem after
projective opposite-row quotienting. The theorem must cover A-pair count
`B^(1/12)`, `B^(3/4)log(B)` singleton relations, shared-deck
dependencies, coverage, and residual nullity. Then transfer to
hash-to-curve decks and instantiate the reverse signed FFE operator within
`B^(9/4)` setup and `B^(5/4+o(1))` batch work before exact-residual logs
or identical descent.
