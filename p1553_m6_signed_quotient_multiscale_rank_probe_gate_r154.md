# P1553 M6 signed-quotient multiscale rank gate R154

## Claim boundary

R154 corrects the R153 rank target for the public identities

```text
ell(-C_j) = -ell(C_j)
```

and tests the resulting signed quotient on preregistered,
occupancy-calibrated prime-cyclic controls.

The quotient and rank formula are exact. The finite synthetic controls
show a rank transition, but they use verifier labels and receive no
asymptotic, hash-to-curve, factor-log, or attack credit.

Classification:

```text
PUBLIC_INVERSION_CONSTRAINTS_HALVE_C_LOG_DIMENSION__SIGNED_QUOTIENT_RANK_FORMULA_EXACT__ACTUAL_QUOTIENT_RANKS_ONE_OR_ZERO_AND_NONE_FULL__OCCUPANCY_CALIBRATED_MULTISCALE_FINITE_RANK_TRANSITION_TESTED__RANDOM_RANK_THEOREM_HASH_TO_CURVE_TRANSFER_REVERSE_FFE_OPERATOR_LOGS_DESCENT_OPEN__NO_SHOUP_BREAKTHROUGH
```

## Signed quotient

Choose one representative from every public inversion pair
`(C_j,-C_j)`. If a full relation row has coefficients `h_j` and
`h_(-j)`, substitution gives

```text
hbar_j = h_j - h_(-j).
```

The public sign constraints have rank `|C|/2`, and

```text
rank([sign constraints; H])
  = |C|/2 + rank(Hbar).
```

Thus full recovery of the inversion-closed C deck is equivalent to full
column rank of `Hbar`. The quotient changes the constant factor in the
unknown dimension but not its `B^(3/4+o(1))` exponent.

## Actual controls

All eight inherited R153 controls replay the signed relation identities,
the public sign pairing, and the combined-rank formula. Their signed
quotient ranks are

```text
1, 0, 0, 0, 0, 0, 0, 0.
```

None is full rank. The correction therefore sharpens the rank target but
does not recover any actual control logs.

## Multiscale controls

The synthetic design is frozen before labels or ranks are observed:

```text
A inversion pairs:       2
C inversion pairs:       4, 5, 6, 7
occupancy multipliers:   1, 2, 4, 8
seeds:                   15401, 15402, 15403
arity:                   6
```

For each scale, the prime order is the next prime above the maximum A6
signed coefficient support times the maximum C6 signed coefficient
support, divided by the frozen occupancy multiplier.

All 48 controls satisfy the signed relation identity and combined-rank
formula. Eleven controls have full signed quotient rank. By
`C-pair-count x occupancy-multiplier`, the full-rank trial counts out of
three are:

```text
4: 0, 0, 2, 2
5: 0, 0, 0, 2
6: 0, 0, 0, 2
7: 0, 0, 1, 2
```

The transition is concentrated at occupancy multipliers four and eight.
This supports the finite hypothesis that the R153 failures were
underdense. It is not a concentration theorem or a transfer to the
deterministic elliptic-curve decks.

## Cost and scope

The signed quotient preserves the campaign exponents:

```text
meaningful C-log dimension:  B^(3/4+o(1))
structured relation rows:    B^(5/4+o(1))
conditional reverse batch:   B^(5/4+o(1))
conditional matrix solve:    B^(2+o(1))
```

R154 supplies no hash-to-curve rank theorem, reverse-adjoint marker
operator, signed internal FFE DAG, factor logs without verifier labels,
or identical target descent.

## Admission

Sixteen of twenty-six obligations pass. The public sign quotient, exact
rank formula, actual rank deficit, preregistered multiscale design, and
finite synthetic rank transition are admitted.

No random-rank concentration theorem, hash-to-curve transfer, reverse FFE
operator, candidate factor logs, descent, generic-prime algorithm,
Pollard-rho improvement, or Shoup improvement is admitted.

Disposition:

```text
ADMIT_SIGNED_QUOTIENT_AND_FINITE_RANK_TRANSITION__DO_NOT_TRANSFER_VERIFIER_LABEL_CONTROLS__REQUIRE_RANDOM_RANK_THEOREM_HASH_TO_CURVE_TRANSFER_AND_REVERSE_SIGNED_FFE_OPERATOR__NO_LOGS__NO_DESCENT__NO_RHO__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Use the frozen signed quotient and occupancy controls to derive or refute
a random-deck concentration and full-rank theorem before claiming
hash-to-curve transfer. In the same bounded experiment, construct the
single reverse-adjoint signed marker FFE operator required by R153 within
`B^(9/4+o(1))` setup and `B^(5/4+o(1))` batch work. Freeze pivots, integer
counts, exact-residual factor logs, and identical descent without DLP,
root, count, marginal, rank, or source oracles.
