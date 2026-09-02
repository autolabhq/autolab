# P1553 5A+5C Theta-Addition Cancellation Network Gate R110

## Claim boundary

This receipt admits an exact finite-field target predicate for one supplied
canonical `5A+5C` source. It does not admit a relation locator, factor-log
solve, target descent, Pollard-rho improvement, Shoup-bound improvement, or
generic-prime ECDLP breakthrough.

## Exact construction

For eleven pairwise-distinct affine points `Q_i` on a short Weierstrass
elliptic curve, evaluate the basis

```text
1, x, y, x^2, xy, x^3, x^2y, x^4, x^3y, x^5, x^4y
```

of `L(11O)`. The resulting `11 x 11` alternant is singular exactly when
`sum_i Q_i = O`. Singularity supplies a nonzero section whose eleven zeros
exhaust its degree; the converse follows from the Abel-Jacobi principal
divisor criterion.

For source points `P_1,...,P_10` and target `T`, freeze position shifts
`S_1,...,S_11` with total sum `O`, and set

```text
Q_i = P_i + S_i
Q_11 = -T + S_11
```

The shifts make all eleven position domains pairwise disjoint and affine.
Therefore the alternant is zero exactly when `sum_i P_i = T`, even when the
original source repeats atoms. This removes the false zeros of the unshifted
determinant without confluent jets.

## Exact controls

- 16 actual and matched-random deck instances replay.
- Both known R105 double fibers replay with alternant rank 10.
- Every positive control has rank 10 and every sampled negative has rank 11.
- Every unshifted repeated-source control reproduces the expected false zero.
- All position-shift schedules have zero sum and pairwise-disjoint affine
  domains.
- The largest observed shift multiplier is one.
- A deterministic scan of at most `(5|A|+5|C|+1)^2+1` multipliers suffices:
  every forbidden affine point or inter-domain collision excludes at most
  one multiplier. Its `B^(6/5+o(1))` work remains below the `B^(5/4)` online
  cap.
- R108 weight 14400 and the R105 marker vectors are preserved.

## Cost boundary

The shifted row tables and determinant for one supplied source fit the online
cap. They do not locate a zero among the `B^5` canonical source body.

The universal field zero mask `1-det^(p-1)` restores a
`Theta(p)=B^(5+o(1))` row mode. On the finite deck, the post-mask balanced
incidence still has the R109 `B^(12/5)` obstruction. Linear and quadratic
determinant moments are exact value channels but do not identify zero
locations.

The construction is deduplicated against `ECDLP-IDEA-012`, including P1539
and the determinant-value-channel receipts. The new local contribution is
the exact zero-sum position-separation chart on the current `5A+5C` decks.

## Disposition

```text
ADMIT_POSITION_SEPARATED_ABEL_ALTERNANT_PREDICATE_ONLY
MERGE_ELLIPTIC_CAUCHY_VALUE_CHANNEL_WITH_IDEA012
REJECT_POINTWISE_SOURCE_SCAN_AND_GLOBAL_FERMAT_ROW_MODE
PRESERVE_FINITE_DECK_ANNIHILATOR_CONTRACTION
NO_RANK
NO_FACTOR_LOGS
NO_DESCENT
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

## Next action

Construct or refute one target-specialized finite-deck annihilator for the
position-separated Abel alternant. It must compute the exact zero count or
existence bit and return one coupled canonical source without evaluating the
`B^5` source body, materializing the `B^(12/5)` balanced incidence, or using
DLP labels. Charge target updates, integer lifting, weight 14400, markers,
multiplicity, rank, factor logs, and identical descent.
