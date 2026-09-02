# P1553 M6 generalized target-divisor Weil-reciprocity swap gate R167

Date: 2026-08-01

## Scope

R167 asks whether the arbitrary target list in R166 can first be compressed as
one principal divisor and then moved through Weil reciprocity. The answer is
algebraically yes, but the standard evaluation orders remain above rho.

The finite producer uses dense Riemann-Roch nullspaces and explicit endpoint
evaluation. Those controls prove the identities below and receive no
asymptotic attack credit.

## Compact target divisor

Let the retained targets be `T_1,...,T_N`, and write

```text
S_T = sum_j T_j,  C = -S_T.
```

Choose distinct auxiliary points `R_1,...,R_(N-1)` and set

```text
A = S_T - sum_i R_i.
```

The two degree-`N+1` point lists

```text
T_1,...,T_N,C
A,R_1,...,R_(N-1),C
```

both sum to `O`. Riemann-Roch therefore supplies functions `F_num` and
`F_den` with those zeros and pole order `N+1` at `O`. Their quotient

```text
h = F_num / F_den
```

has divisor

```text
div(h) = sum_j [T_j] - [A] - sum_i [R_i].
```

The common zero `C` and the equal poles at `O` cancel. A generalized Miller
line-function straight-line program can represent this principal divisor in
`O(N)` line merges. This corrects the narrower R165 intuition that an
arbitrary target set has no compact scalar-chain-like representation.

It does not yet give a fast way to restrict the represented function to all
roots of `U`.

## Reciprocity identity

Let

```text
f_0(Q) = U(x(Q)),
div(f_0) = S + (-S) - 2n[O],
h_P(Q) = h(Q+P).
```

For disjoint support, Weil reciprocity gives

```text
product_j f_0(T_j-P)
  = f_0(A-P) product_i f_0(R_i-P)
    product_(Q in S union -S) h(Q+P) / h(P)^(2n).
```

The auxiliary points are selected so that every correction and `h(P)` is a
unit on the selected signed divisor. Consequently, after denominator
clearing, the swapped expression has exactly the same roots modulo `U` as the
R166 Kummer translate product.

At a candidate root, the functions have common support and the literal
disjoint-support statement does not apply. The denominator-cleared identity
extends by specialization: a target factor and a corresponding translated
`h` factor both vanish. R167 records those rows separately instead of
silently calling them disjoint-support evaluations.

## Finite controls

Each inherited batch contains one `denominator_exception` target equal to a
selected endpoint. R167 drops exactly those six targets so that the finite
rational-function control never assigns an artificial value to `f_0(O)`.

The retained target counts are five, seven, and eight across the three curve
families. For both seeds, the producer constructs one-dimensional
Riemann-Roch nullspaces with full pole order, verifies every prescribed zero,
and evaluates `h(O)` from leading local coefficients.

Across six controls:

```text
selected endpoints checked:          202
candidate-zero specializations:      140
disjoint-support rows:                 62
direct target evaluations:          1,486
raw swapped h evaluations:         17,844
```

Every direct product equals its corrected reciprocity value. The numerator
and denominator point lists both sum to `O`; all auxiliary corrections and
all `h(P)` values are units. No DLP, root, count, marginal, rank, source, or
resultant oracle is consumed.

## Resultant interface

The swapped product is an elliptic-resultant or tame-symbol interface,
schematically

```text
Res_E(f_0(Q), h(Q+P)),
```

up to the explicit `h(P)` and auxiliary unit corrections. This is a sharper
successor interface than an undifferentiated target product: the target set is
now a degree-`N` generalized Miller SLP, and the desired output is only its
denominator-cleared resultant restriction modulo `U`.

No fast operator for that restriction is supplied.

## Full cost boundary

At the campaign caps,

```text
n = B^(9/4),  N = B^(5/4),  rho = B^(5/2).
```

The compact target-divisor SLP has `B^(5/4+o(1))` state. Direct evaluation of
the original target product uses

```text
nN = B^(7/2)
```

values. Direct evaluation after the reciprocity swap uses

```text
2n^2 = B^(9/2)
```

values and is one exponent worse than the original table. A standard
represented elliptic resultant has `Theta(nN)=B^(7/2)` divisor or
representation scale. Both standard routes remain above rho.

The surviving primitive is an output-sensitive SLP elliptic resultant or
tame-symbol resultant modulo `U`, preferably in `B^(9/4+o(1))` work and in all
cases strictly below `B^(5/2)`, without an `nN` table, an `n^2` table, or a
degree-`Theta(nN)` represented function.

## Deduplication

R166 leaves the arbitrary-target Kummer translate-product remainder open.
R167 replaces the target list by a compact principal-divisor witness and
proves the exact swap, but does not solve the remainder problem.

R165 correctly excludes one ordinary scalar chain for unrelated targets.
R167 refines that boundary with generalized Miller line merges; it does not
claim that compact divisor state implies fast batch evaluation.

Eagen's primary paper supplies principal-divisor interpolation, Weil
reciprocity, and elliptic-resultant interfaces. Miller supplies line-function
divisor arithmetic and compact evaluation chains. Neither source claims this
ECDLP complexity improvement, and R167 attributes none to them.

R113 studies ordinary orbit/product-tree recurrence. Its recurrence is not
reused as an output-sensitive arbitrary-target resultant.

## Admission

Twenty-four of thirty-two obligations pass. Admit the compact generalized
target-divisor witness, exact divisor identity, exact Weil-reciprocity swap,
candidate-zero specialization, unit corrections, and six finite controls.

Do not admit an output-sensitive elliptic-resultant constructor, avoidance of
the degree-`nN` representation barrier, deterministic hash-to-curve transfer,
an unconditional generic-prime algorithm, Pollard-rho or Shoup improvement,
or an ECDLP breakthrough.

Disposition:

```text
ADMIT_COMPACT_GENERALIZED_TARGET_DIVISOR_WITNESS__EXACT_WEIL_RECIPROCITY_SWAP_ON_SIX_FINITE_CONTROLS__CANDIDATE_ZERO_ROWS_BY_SPECIALIZATION__AUXILIARY_CORRECTIONS_UNITS__RAW_SWAP_B9O2__STANDARD_RESULTANT_B7O2__OUTPUT_SENSITIVE_RESULTANT_MOD_U_OPEN__NO_RHO_SHOUP_BREAKTHROUGH
```

## Next action

Construct or refute an SLP elliptic-resultant or tame-symbol remainder operator
modulo `U` directly from the degree-`N` target-divisor witness, in less than
`B^(5/2)` total work and preferably `B^(9/4+o(1))`. Test quotient-ring,
transposed modular-composition, and half-GCD formulations for a reusable
low-displacement operator, while forbidding the `nN`, `n^2`, and represented
degree-`Theta(nN)` intermediates.
