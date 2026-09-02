# P1553 R108 Factored Elliptic Lambda-Ring/Chow Norm Gate

## Claim boundary

R108 admits an exact canonical-weight identity. It does not construct an
inside-cap scalar target-norm circuit.

No generic-prime ECDLP algorithm, Pollard-rho improvement, Shoup-bound
improvement, factor-log solve, target descent, or breakthrough is claimed.

## Degree-five cycle identity

In the elliptic group algebra, let `p_r(D)` apply `[r]` to every atom of
`D`, and let multiplication denote Pontryagin convolution. The complete
homogeneous fifth power satisfies

```text
120 h_5(D) =
  p_1^5
  + 10 p_1^3 p_2
  + 15 p_1 p_2^2
  + 20 p_1^2 p_3
  + 20 p_2 p_3
  + 30 p_1 p_4
  + 24 p_5.
```

The coefficients are the seven conjugacy-class sizes of `S_5` and sum to
120. Applying the identity independently to A and C gives a positive
49-term identity:

```text
14,400 H_A,5 * H_C,5 = sum_(lambda,mu) c_lambda c_mu P_lambda(A)*P_mu(C).
```

All eight actual and eight matched-random decks satisfy both side identities
and the full all-target identity exactly. Every finite coefficient divides
by 14,400, and 14,400 is invertible in every frozen field.

## Canonical marker gate

Expanding a cycle of length `r` as `r` copies of its selected atom index
preserves every R105 power-sum marker. Exhaustive source profiles for every
frozen A- and C-deck size show:

```text
each canonical side source has cycle weight 120
each canonical full source has cycle weight 14,400.
```

Thus the cycle identity removes R107's source-dependent repartition weights.
Conditionally, canonical count is aggregate jet order divided by 14,400.
The scalar term norms and generic marker factorization remain unsupplied.

## Cost gate

The 49-term ledger contains the coefficient-one identity-cycle term

```text
(lambda,mu) = ((1,1,1,1,1),(1,1,1,1,1)).
```

It is the original ordered `B^5` source body and retains the best
`B^(13/5)` binary resultant interface. All 49 coefficients are positive, so
termwise norm evaluation has no cancellation that removes this term.
Only 14 term source bodies fit setup and one fits online. Termwise
evaluation is rejected.

## Theorem-of-the-cube boundary

The theorem of the cube factors the pullback line-bundle class of an
addition map into one-body and pairwise pieces. In this campaign the
corresponding pair-table exponents are:

```text
A-A  B^(4/5)
A-C  B^1
C-C  B^(6/5)
```

All are inside the online cap. This is promising but conditional. A
line-bundle isomorphism does not assert that the pulled-back target section
is one pure tensor of those factors. No Poincare trivialization, theta
section factorization, bounded section rank, or scalar evaluator is
supplied.

Primary source:

- https://stacks.math.columbia.edu/tag/0BFE

## Disposition

```text
DEGREE_FIVE_CYCLE_INDEX_GIVES_EXACT_SEVEN_TERM_CANONICAL_DIVISOR
FORTY_NINE_TERM_5A5C_IDENTITY_AND_14400_MARKER_WEIGHT_EXACT
IDENTITY_CYCLE_TERM_RETAINS_B5_BODY_AND_B13O5_ROOT_INTERFACE
THEOREM_OF_CUBE_FACTORS_LINE_BUNDLE_NOT_TARGET_SECTION
POINCARE_THETA_SECTION_FACTORIZATION_OPEN
```

The lane is not admitted: 17 of 30 obligations pass.

## Exactly one next action

Construct or refute one Poincare/theta target-section factorization of the
ten-input elliptic sum pullback. Freeze trivializations and addition
identities before outcomes; prove the section, not only its line-bundle
class, has a bounded-rank one-body/pairwise tensor representation whose deck
contractions fit both caps and preserve the 14,400 canonical marker weight,
projective charts, generic multiplicity, and integer lifting.
