# P1553 R109 Poincare/Theta Target-Section Rank Gate

## Claim boundary

R109 closes regular finite-valued pure pairwise products and uniform
two-block separated target sections below `B^(12/5)`. It does not prove a
lower bound for arbitrary arithmetic circuits or rational theta networks
with pole cancellation.

No generic-prime ECDLP algorithm, Pollard-rho improvement, Shoup-bound
improvement, factor-log solve, target descent, or breakthrough is claimed.

## Zero-cylinder theorem

Consider a proposed scalar factorization

```text
s_T(P1,...,P10) =
  product_i u_i(P_i) product_(i<j) b_ij(P_i,P_j)
```

whose zero set is exactly `P1+...+P10=T`. If the product vanishes, one unary
or pairwise factor vanishes, so every completion of that fixed unary or
pairwise assignment also vanishes. The target equality fiber contains no
such cylinder: with at least one free coordinate, changing that coordinate
changes the sum in the prime-order group.

All eight actual and eight matched-random decks replay this statement for
all 56 subsets of size zero, one, or two. Therefore a regular finite-valued
pure product of the one-body and pairwise Poincare tables cannot be the
target section.

This does not cover rational factors whose poles cancel or bounded sums of
products.

## Translate-rank theorem

Over `F_(p^2)(E)`, use the signed rational coordinate

```text
kappa(P)=x(P)+w*y(P).
```

It has a pole of order three at the identity. For distinct `L_i`, the
translate

```text
R -> kappa(L_i+R)-kappa(T)
```

has its unique order-three pole at `R=-L_i`. At that pole no other
translate has a pole, so the translates are linearly independent.

The balanced generic split has `B^(12/5)` distinct endpoints on its smaller
side and `B^(13/5)` on its larger side. Any uniform two-block separated
section therefore needs rank and state at least `B^(12/5)`, above the
`B^(9/4)` setup cap. Scalar extension for FFE does not lower rank.

This is a function-field separated-rank theorem, not an arithmetic-circuit
lower bound.

## Finite controls

Both balanced actual side images are injective on all 16 actual and matched
instances. Signed `Fp2` section matrices on square samples up to dimension
64 have full rank throughout; the smallest complete sample is `24 x 24`
with rank 24.

The finite point-key verifier reserves zero for the projective identity,
whereas `kappa` has a function-field pole. No homogeneous projective
trivialization is credited.

## Cost boundary

The theorem-of-the-cube pair tables remain individually small:

```text
A-A  B^(4/5)
A-C  B^1
C-C  B^(6/5)
```

R109 proves that these tables cannot simply be multiplied into the target
section, and that a uniform separated sum needs over-cap rank. An implicit
high-rank theta-addition circuit with exact rational cancellation remains
open.

## Disposition

```text
TARGET_EQUALITY_FIBER_HAS_NO_UNARY_OR_PAIRWISE_ZERO_CYLINDER
REGULAR_PURE_POINCARE_SECTION_PRODUCT_REFUTED
TRANSLATED_SIGNED_SECTIONS_HAVE_B12O5_UNIFORM_FLATTENING_RANK
ACTUAL_AND_MATCHED_BALANCED_IMAGES_INJECTIVE_AND_SAMPLE_RANKS_FULL
RATIONAL_THETA_CANCELLATION_NETWORK_OPEN
```

The lane is not admitted: 13 of 26 obligations pass.

## Exactly one next action

Construct or refute one exact finite-field theta-addition cancellation
network for the target section. Freeze the theta basis, rational poles,
projective trivializations, additions, bond dimensions, and deck
contraction order; allow high flattening rank only when generated implicitly
inside both caps; require exact zero biconditional, the R108 weight-14,400
marker lift, generic multiplicity/integer lifting, rank, logs, and identical
descent.
